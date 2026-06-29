# Demographics Inference Layer Architecture

## Current System Understanding

The repository is organized as a small event-driven video analytics pipeline:

- `detection/` owns person detection. `detection_engine.detect(frame)` validates `frame_id`, `timestamp`, and `image`, runs YOLO for person class only, clamps bounding boxes, and currently returns detections containing `detection_id`, `frame_id`, `timestamp`, `bbox`, `confidence`, and a preprocessed crop under `image`.
- `embed/` owns re-identification embedding. `embed_engine.embed` consumes the detection crop in `detection["image"]`, runs OSNet, normalizes the feature vector, and returns `detection_id` plus `embedding`.
- `build_observation.py` maps detection plus embedding into the tracking observation contract: `detection_id`, `timestamp`, `center`, `bbox`, `confidence`, and `embedding`.
- `track/` owns TrackV2. It consumes timestamp-bucketed observations, performs deterministic matching, maintains `RuntimeTrackV2` state, and returns both tracks and an `assignment_map` from detection IDs to runtime track IDs.
- `events/` owns line-crossing event derivation. `detect_events(tracks, line_config)` reads stable track state and emits deterministic event candidates containing `event_id`, `runtime_track_id`, `timestamp`, `event_type`, `direction`, and supporting positions.
- `test_tracking_v2_pipeline.py` is the current executable integration pipeline. It reads frames, calls detection, embedding, observation building, tracking, event detection, rendering, and contact-sheet generation.
- `mobilenetv3/` contains an embedded MobileNetV3 codebase and pretrained checkpoint files, but no demographics-facing wrapper or explicit sex/race/age heads currently exist in the visible pipeline.

The required demographics layer should be added without modifying `detection/`, `track/`, or stable event logic. It should sit beside these modules and consume their public outputs.

## Proposed Architecture

Introduce a new top-level `demographics/` module that enriches event candidates with a deterministic demographics prediction for exactly one detection per event.

The demographics layer should be event-driven, not frame-global. It should not classify every detection by default. Instead, it should run only when a new event is emitted, select the single highest-confidence detection associated with that event, extract a fresh crop from the original frame using that detection bounding box, run MobileNetV3-based inference, and attach a forced-argmax classification result:

- `sex`: one of `M` or `F`
- `race`: one of `L`, `X`, or `D`
- `age`: one configured age bucket from bucket indices `0` through `5`

No `UNKNOWN`, null, or abstain state should be emitted by the classifier. If input validation fails, the pipeline should fail explicitly or emit a structured error outside the demographics labels; it should not invent an unknown label.

## Proposed Data Flow

```text
Video frame packet
  {frame_id, timestamp, image}
        |
        v
Detection module (unchanged)
  detect(frame) -> detections[]
  each detection: {detection_id, frame_id, timestamp, bbox, confidence, image}
        |
        +--------------------------+
        |                          |
        v                          v
Embedding path                 Event evidence cache
  embed(detection image)          stores raw frame reference/copy metadata and
  build_observation()             detection metadata needed for later event lookup
        |                          |
        v                          |
TrackV2.update()                  |
  observations_by_ts -> tracks, assignment_map
        |                          |
        v                          |
Event module (unchanged)          |
  detect_events(tracks, line_config)
        |                          |
        v                          |
New event finalizer/enricher <----+
  for each newly emitted event:
    1. resolve runtime_track_id to supporting detection IDs
    2. select highest-confidence detection for that event
    3. crop original frame by selected detection bbox
    4. run demographics classifier
    5. attach demographics result to finalized event
        |
        v
Final event output
  {event fields..., demographics: {sex, race, age, scores/probs, source_detection_id}}
```

## Where Crop Extraction Should Occur

Crop extraction for demographics should occur inside the new `demographics/` module, after event selection and before MobileNetV3 preprocessing.

This is intentionally separate from the existing crop embedded in detection results:

1. The detection module must not be modified. Reusing or changing its crop contract risks coupling demographics to detection internals.
2. The detection crop is currently preprocessed for OSNet embedding: BGR-to-RGB conversion, resize to `128x256`, normalization to `[0, 1]`. That is not necessarily the MobileNetV3 demographics input contract.
3. Demographics should classify one event-selected detection, not every detection. Extracting crops after event selection avoids unnecessary MobileNetV3 inference and keeps the system event-driven.
4. A demographics cropper can be deterministic and model-specific: clamp bbox, validate non-empty crop, optional configurable padding, color conversion, resize, normalization, and tensor conversion in one place.
5. Keeping crop extraction in `demographics/` makes the MobileNetV3 preprocessing contract testable without affecting tracking, embedding, or event behavior.

The event pipeline will therefore need access to the original frame image, or to a frame cache from which the original frame can be retrieved by `frame_id`/`timestamp`, plus the selected detection metadata. The cache should live outside `detection/` and should store only the bounded history needed to finalize events.

## MobileNetV3 Integration Strategy

MobileNetV3 should be integrated through a demographics-specific wrapper, not by directly importing and calling model internals throughout the pipeline.

Recommended integration pattern:

- Treat `mobilenetv3/` as a vendored backbone source.
- Add `demographics/model.py` or `demographics/engine.py` that owns model construction, checkpoint resolution, device placement, inference mode, preprocessing, and postprocessing.
- Add explicit classification heads or load a checkpoint that already includes demographics heads. The wrapper should expose a small stable API, for example `predict(crop) -> DemographicsPrediction`.
- If the existing MobileNetV3 checkpoints are ImageNet or generic classification checkpoints, implementation must add/load trained demographics heads before this can produce meaningful labels.
- The wrapper must force argmax for every output head:
  - `sex = SEX_LABELS[argmax(sex_logits)]`
  - `race = RACE_LABELS[argmax(race_logits)]`
  - `age = AGE_BUCKETS[argmax(age_logits)]`
- Confidence scores/probabilities can be included as diagnostics, but they must not gate label emission into `UNKNOWN`.

## Proposed `/demographics/` Module Breakdown

```text
demographics/
  __init__.py
    Public exports only.

  models.py
    Typed result contracts:
    - DemographicsPrediction
    - DemographicsScores
    - DemographicsEventResult
    - CropSource / DetectionEvidence metadata

  labels.py
    Canonical label constants:
    - SEX_LABELS = ["M", "F"]
    - RACE_LABELS = ["L", "X", "D"]
    - AGE_BUCKETS = [0, 1, 2, 3, 4, 5]
    Also owns validation that no UNKNOWN label can be emitted.

  crop.py
    Deterministic bbox clamping and crop extraction from original frames.
    Owns optional padding policy and MobileNetV3 image preprocessing.

  evidence.py
    Bounded event evidence cache keyed by frame_id, timestamp, detection_id,
    and runtime_track_id assignment. Stores enough information to select the
    highest-confidence detection for an event without reaching into detection.

  selector.py
    Selects exactly one detection per event. Default policy:
    among detections associated with event.runtime_track_id and event support
    window, choose highest confidence; break ties deterministically by timestamp,
    frame_id, and detection_id.

  engine.py
    MobileNetV3 demographics wrapper. Owns model load, device, inference,
    forced argmax postprocessing, and stable `predict(crop)` API.

  enricher.py
    Event-facing orchestration:
    `enrich_event(event, evidence_cache, demographics_engine)` or
    `enrich_events(events, ...)` returns finalized event records with demographics.

  config.py
    Paths, input size, normalization, crop padding, device preference, cache size,
    checkpoint path, and optional batch size.

  tests/
    Unit tests for crop clamping, deterministic selection, no UNKNOWN outputs,
    argmax mapping, cache eviction, and event enrichment behavior.
```

## Dependencies and Coupling Risks

### Detection Coupling

Risk: demographics accidentally depends on `detection["image"]`, which is an OSNet-preprocessed crop rather than raw image data.

Mitigation: use only public detection metadata (`detection_id`, `frame_id`, `timestamp`, `bbox`, `confidence`) plus the original frame held by the orchestrating pipeline or evidence cache. Do not modify `detection/`.

### Tracking Coupling

Risk: TrackV2 stores `detection_history` but not a per-detection timestamp, bbox, confidence, or frame reference. Event candidates only expose track ID and supporting positions, not exact detection IDs.

Mitigation: maintain an external evidence cache populated during the integration pipeline using `assignment_map` from TrackV2. The cache maps runtime track IDs back to detection metadata. Avoid adding demographics fields to `RuntimeTrackV2` unless a later design explicitly changes track contracts.

### Event Coupling

Risk: event IDs and event behavior are stable and deterministic; adding demographics inside `events/` could alter or destabilize event emission.

Mitigation: keep `events/` unchanged. Add a post-event enrichment/finalization layer that consumes event candidates and produces a separate final event representation.

### MobileNetV3 Coupling

Risk: direct dependency on vendored `mobilenetv3/` internals can make future model replacement difficult.

Mitigation: isolate MobileNetV3 imports in `demographics.engine`. The rest of the pipeline should depend only on a demographics predictor interface.

### Determinism Risk

Risk: GPU nondeterminism, model dropout, unordered dictionary iteration, and tie handling could produce inconsistent outputs.

Mitigation: always use `eval()` and inference mode, define deterministic tie-breakers in selection, sort event processing, and configure deterministic backend behavior if strict bitwise reproducibility is required.

### Privacy and Compliance Risk

Risk: demographics inference involves sensitive attributes and may be regulated or policy-constrained depending on jurisdiction and product use.

Mitigation: require product/legal validation before implementation. Store only necessary outputs and provenance, document retention limits, and make confidence diagnostics auditable.

### Model Validity Risk

Risk: the repository currently includes MobileNetV3 checkpoints, but it is not clear that they are trained for sex/race/age prediction or include the required heads.

Mitigation: block implementation until trained demographics weights, label definitions, input preprocessing, and evaluation metrics are provided.

## Open Questions / Missing Information

1. What exact age buckets do indices `0` through `5` represent?
2. What do race labels `L`, `X`, and `D` stand for, and are those labels product-approved?
3. Which MobileNetV3 checkpoint should be used for demographics inference?
4. Are trained heads/checkpoints available for all three outputs, or must the implementation add heads and load separate weights?
5. What MobileNetV3 input size and normalization should demographics use?
6. Should the selected detection be chosen from all detections assigned to the runtime track, or only from detections within the event supporting-position window?
7. If the highest-confidence detection crop is too small, blurred, or invalid after bbox clamping, should the system fail, skip the event, or select the next-best detection? The labels still must not become `UNKNOWN`.
8. What is the final event output contract and persistence destination?
9. Should demographics inference run synchronously before emitting an event, or asynchronously with deterministic eventual finalization?
10. What retention policy applies to raw frame crops and evidence cache entries?
11. Is batching allowed when multiple events occur in the same frame, or must inference be strictly one event at a time?
12. Are there latency budgets for event finalization?
13. Are there audit requirements for logits/probabilities, model version, checkpoint hash, or crop provenance?

## Recommended Implementation Plan After Design Validation

1. Confirm label semantics, age bucket definitions, checkpoint paths, MobileNetV3 input preprocessing, and final event output schema.
2. Add the `demographics/` package with typed models, labels, config, cropper, selector, evidence cache, engine wrapper, and enricher interfaces.
3. Implement deterministic crop extraction from original frames in `demographics.crop`, including bbox clamping and invalid-crop behavior approved in design review.
4. Implement evidence-cache population in the integration/orchestration layer using detections, original frames, and TrackV2 `assignment_map` without changing detection or tracking modules.
5. Implement event-to-detection selection in `demographics.selector`, including explicit tie-breakers.
6. Implement the MobileNetV3 demographics engine wrapper with trained demographics heads/checkpoints and forced argmax postprocessing.
7. Add unit tests for crop extraction, selection policy, label mapping, no-UNKNOWN guarantees, cache eviction, and event enrichment.
8. Add integration tests around the existing pipeline that verify demographics enrichment occurs only for events and does not alter detection, tracking, or event IDs.
9. Add model provenance metadata to finalized event output: model name, checkpoint hash/version, source detection ID, frame ID, timestamp, and bbox.
10. Run performance profiling with representative videos and decide whether event-level batching or async finalization is needed.
11. Validate privacy/compliance requirements before enabling demographics inference in production.
12. Document operational configuration, model artifact management, and rollback procedure.
