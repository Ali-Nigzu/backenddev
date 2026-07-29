# MiVOLO Demographics Rebuild and Tracking Harness Extension Plan

Status: planning only. This document does not change production demographics, tracking, Event, or harness behavior.

## 1. Repository baseline and local checkpoint validation

- Planning branch: `codex/plan-mivolo-demographics-and-harness`, intended base/PR target: `codex/demographic-mivolo-foundation`.
- The repository currently has no established `docs/` planning convention, so this plan uses `docs/plans/mivolo_demographics_and_tracking_harness_plan.md`.
- Checkpoint assembly command run: `bash demographics/assemble_weights.sh`.
- Checkpoint validation commands run:
  - `sha256sum demographics/demographicweights.pth` -> `cc279b6914b3ee8be6a58139c06ecb24ca95751233cf6c07804b93184614eb17`.
  - `ls -lh demographics/demographicweights.pth` -> `105M`.
- `demographics/demographicweights.pth` is generated from `demographics/model_parts/demographicweights.pth.part-00` and `demographics/model_parts/demographicweights.pth.part-01`; keep it uncommitted.
- Do not edit `demographics/model_parts/demographicweights.pth.part-00`, `demographics/model_parts/demographicweights.pth.part-01`, `demographics/assemble_weights.sh`, or `.gitignore` for this rebuild unless a later checksum failure proves one is broken.

## 2. Current repository contracts verified

### 2.1 Conceptual target versus actual code

Target pipeline remains:

```text
Detect(Frame) -> DetectionBatch
Track(TrackingState, DetectionBatch) -> TrackingState
Event(TrackingState, LineConfig) -> EventBatch
Demographic(EventBatch, FrameBatch) -> DemographicsBatch
Assemble(EventBatch, DemographicsBatch) -> OutputBatch
```

Actual implementation differences:

- `Detect` is a stateful class in `detect/detect.py`; public import is `from detect import Detect` via `detect/__init__.py`.
- `Track` is a function in `track/tracker.py`; public import is `from track import Track` via `track/__init__.py`. It mutates `tracking_state` in place and returns it.
- `Event` is a function in `events/event.py`; public import is `from events import Event` via `events/__init__.py`. It does not mutate inputs.
- There is no production `Demographic(event_batch, frame_batch)` stage yet. Current `demographics` exports legacy `DemographicsEngine`, which accepts a single `person_image` and returns FairFace-derived `sex`, `race`, and age-bucket index values.
- There is no `Assemble` implementation in this repository; this plan must not create one.

### 2.2 Frame and DetectionBatch contract

`Detect.__call__(frame)` requires a mapping with:

```python
{
    "frame_id": str,
    "timestamp": float | int,
    "image": np.ndarray,  # contiguous uint8 [H, W, 3]
}
```

`Detect` validates `image.dtype == np.uint8`, `image.ndim == 3`, `image.shape[2] == 3`, non-zero height/width, and C-contiguity. The harness converts OpenCV BGR to RGB before calling `Detect`, so the practical `Frame.image` convention is RGB.

`Detect` returns one DetectionBatch per frame:

```python
{
    "frame_id": frame_id,
    "timestamp": float(timestamp),
    "detections": [
        {
            "detection_id": f"{frame_id}:det:{detection_index}",
            "bbox": {"x1": float, "y1": float, "x2": float, "y2": float},
            "centre": {"x": float, "y": float},
            "confidence": float,
        }
    ],
}
```

### 2.3 TrackingState contract

`Track(tracking_state, detection_batch)` requires `tracking_state == {"tracks": list}` and each existing track to contain:

```python
{
    "track_id": str,
    "path": [{"timestamp": float, "centre": {"x": float, "y": float}}, ...],
    "best_crop": {"frame_id": str, "bbox": {"x1": float, "y1": float, "x2": float, "y2": float}},
    "best_crop_confidence": float,
}
```

`Track` updates `best_crop` only when a matched detection has confidence greater than the current `best_crop_confidence`. Therefore `best_crop` is already a deterministic best-confidence body crop per track.

### 2.4 EventBatch contract to preserve

`Event(tracking_state, line_config)` returns exactly:

```python
{
    "events": [
        {
            "track_id": str,
            "timestamp": float,
            "event_type": int,
            "best_crop": {
                "frame_id": str,
                "bbox": {"x1": float, "y1": float, "x2": float, "y2": float},
            },
        }
    ]
}
```

Policies verified in code/tests:

- `event_type == 1` means entry; `event_type == 0` means exit.
- Event timestamp is the first timestamp of the stable opposite-side run, not inference time, not confirmation-frame output time, and not final track timestamp.
- `best_crop_confidence` is required in `TrackingState` but intentionally excluded from each Event record.
- `Event` copies `track["best_crop"]`; it does not include event IDs, directions, supporting positions, run counts, or confidence fields.
- Multiple events for one track all receive the same `track["best_crop"]` because `_make_event()` copies the final track-level `best_crop`. Current tests assert identical `best_crop` for entry/exit events from the same track.

Demographic must consume Event output by reading only `event["track_id"]`, `event["best_crop"]["frame_id"]`, and `event["best_crop"]["bbox"]`. It must not recalculate timestamps, infer crossing direction, detect crossings, replace event timestamps, mutate Event records, or choose a new crossing point.

## 3. MiVOLO checkpoint and official-source findings

Inspection commands used:

- `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` for local checkpoint introspection only.
- `python - <<'PY' ... torch.load('demographics/demographicweights.pth', map_location='cpu') ... PY`.
- `git clone --depth 1 https://github.com/WildChlamydia/MiVOLO.git /tmp/MiVOLO` and source inspection of `/tmp/MiVOLO/mivolo/model/mi_volo.py`, `/tmp/MiVOLO/mivolo/data/misc.py`, and `/tmp/MiVOLO/mivolo/data/dataset/age_gender_dataset.py`.

Checkpoint details:

- Top-level keys: `min_age`, `max_age`, `avg_age`, `no_gender`, `with_persons_model`, `state_dict`.
- Values: `min_age=1`, `max_age=95`, `avg_age=48.0`, `no_gender=False`, `with_persons_model=True`.
- `state_dict` length: `290`.
- `pos_embed` shape: `[1, 14, 14, 384]`, so official MiVOLO computes `input_size = 14 * 16 = 224`.
- State-dict key structure includes separate person/face patch stems: `patch_embed.conv1.*` and `patch_embed.conv2.*`; heads include `aux_head.weight`/`bias` and `head.weight`/`bias`, both with output dimension `3`.
- Expected model architecture: official `mivolo_d1_224` created with `num_classes=3`, `in_chans=6`, `checkpoint_path=...`, and `filter_keys=["fds."]`.
- Official source/version: use `WildChlamydia/MiVOLO` repository current main source compatible with MiVOLO v1 `mivolo_d1_224`; pin by commit in the implementation PR after vendoring or dependency evaluation.
- Official preprocessing: `class_letterbox(img, new_shape=(224, 224))`, then `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`, divide by `255.0`, normalize by timm/Imagenet default mean/std, convert HWC to contiguous CHW float32 tensor.
- Tensor layout: batch-first `NCHW`.
- Official model input for person+face model is `torch.cat((faces_input, person_input), dim=1)`, producing `N x 6 x 224 x 224`. For body-only operation with this face+body checkpoint, pass a zero normalized face image for the first 3 channels and the body crop tensor for the last 3 channels; this matches official `prepare_classification_images(None)` behavior for missing crops.
- RGB/BGR: official MiVOLO crop preprocessing expects BGR arrays because it converts BGR to RGB internally. This repository's `Frame.image` is RGB for detection. The new preprocessing must therefore either (recommended) accept repository RGB crops and skip `cv2.COLOR_BGR2RGB`, or convert RGB to BGR before reusing official helper. Do not accidentally double-swap channels.
- Age output: when `no_gender=False`, official `output[:, 2]` is normalized continuous age. Official de-normalization is `age = raw * (max_age - min_age) + avg_age`; official display rounds to two decimals.
- Sex/gender output: official `output[:, :2].softmax(-1).topk(1)`; official mapping is index `0 -> "male"`, index `1 -> "female"`. Public repository mapping must be `male -> 1`, `female -> 0`.
- CPU/CUDA: official code accepts a `device` string, moves model/input to that device, disables half precision on CPU, calls `model.eval()`, and uses no-grad inference. New code should use `torch.inference_mode()` instead of `torch.no_grad()`.
- Required Python dependencies: `torch`, `timm`, `numpy`, OpenCV (`cv2`) if reusing letterbox/color code, and the minimal MiVOLO model modules. Existing legacy code also imports `torchvision`, but the new MiVOLO path should not depend on FairFace/torchvision ResNet.
- Checkpoint loading should be once per `Demographic` instance, not once per track. Construction validates checkpoint existence/checksum/compatibility, builds the model, moves it to selected device, and calls `eval()`.
- Checkpoint incompatibilities: this checkpoint is not a single raw state dict; loaders must read metadata and pass `state_dict` through the MiVOLO/timm model creation path. Implementation must not try to load it into ResNet or any FairFace structure.

Recommended MiVOLO source strategy:

- Do not import production code directly from `/tmp/MiVOLO` or require a live git clone.
- Prefer adding a pinned dependency if packaging/license review confirms it is acceptable. If packaging is unsuitable, vendor the minimal official model/preprocessing files under an internal-only path such as `demographics/_mivolo/` with attribution, preserving only code required for `mivolo_d1_224` inference.
- Production modules outside `demographics` must never import MiVOLO internals directly.

## 4. New Demographic stage contract

Recommended public import:

```python
from demographics import Demographic
```

Recommended public type: class with `__call__`, not a stateless function, because model loading must occur once per instance/process and model/device/checkpoint state must be retained.

Recommended future signature:

```python
class Demographic:
    def __init__(self, checkpoint_path: str | Path | None = None, device: str | None = None) -> None: ...
    def __call__(self, event_batch: dict, frame_batch: dict) -> dict: ...
```

Result contract:

```python
{
    "results": [
        {"track_id": str, "age": int, "sex": int}
    ]
}
```

- `sex`: `1` for male, `0` for female.
- `age`: estimated exact age integer; no age buckets.
- No `race`, `ethnicity`, FairFace label, confidence, or MiVOLO internal field in public output.
- Deterministic result ordering: first by earliest Event timestamp for each track, then `track_id`. This keeps outputs stable and aligned to event chronology without relying on positional joins.

### 4.1 Minimal proposed package structure

```text
demographics/
    __init__.py              # replace export with Demographic only
    demographic.py           # public class, event/frame validation, per-track orchestration, result construction
    model.py                 # internal MiVOLO loading/inference adapter, checkpoint validation, device choice
    preprocessing.py         # internal frame/crop validation, clipping, letterbox, tensor construction
    exceptions.py            # optional DemographicError/DemographicInputError/DemographicModelError
    assemble_weights.sh      # retain unchanged
    model_parts/...          # retain unchanged
    demographicweights.pth   # generated local file, uncommitted
```

Ownership:

- `demographic.py`: validates EventBatch and FrameBatch shape, selects one crop per unique track, calls model adapter once per track or as one batch, and constructs `{"results": ...}`.
- `model.py`: checkpoint path resolution (`demographics/demographicweights.pth` by default), checksum compatibility guard, MiVOLO model creation, `model.eval()`, `torch.inference_mode()`, CPU fallback.
- `preprocessing.py`: source-frame lookup helpers, crop clipping, channel convention, letterboxing/normalization, tensor shape validation.
- `exceptions.py`: stage-specific exceptions. If omitted, define exceptions in `demographic.py`.

### 4.2 Once-per-track inference policy

Implementation steps:

1. Validate `event_batch` is `{"events": list}`.
2. If the events list is empty, return `{"results": []}` without requiring frames or loading/running the model. If eager model loading in `__init__` remains, this still returns no results.
3. Collect event records by `track_id` in a dictionary.
4. For each track, verify all duplicate events are valid Event records. Current Event emits the same track-level `best_crop` for each event for a track; still compare crop references defensively.
5. Crop selection rule:
   - If all events for a track have identical `best_crop`, use that crop.
   - If future Event changes expose different crop references and no `best_crop_confidence` is present, fail with `DemographicInputError` explaining that EventBatch lacks crop-quality information. Do not choose first/last/random. If Event later adds explicit crop quality, select highest quality, tie-break by earliest event timestamp then frame_id.
6. Sort selected track crops by earliest event timestamp then `track_id`.
7. Resolve frames by `best_crop.frame_id`, crop bodies, preprocess, and batch inference.
8. Run the model exactly once per unique track. Tests should inject a fake backend and assert call count equals unique track count or batch size equals unique track count.
9. Return one result per unique track. Downstream and harness joins must use `track_id`, never list position.

## 5. FrameBatch contract and validation

Recommended FrameBatch shape:

```python
{
    "frames": [
        {"frame_id": str, "timestamp": float, "image": np.ndarray}
    ]
}
```

Use the same `Frame.image` convention as `Detect`: contiguous `np.uint8` `[H, W, 3]` RGB. This is repository-specific evidence from `test_tracking_v2_pipeline.py`, which converts BGR video frames to RGB before building `frame` for `Detect`.

Resolution flow:

```text
Event.best_crop.frame_id -> frame map -> Frame.image -> Event.best_crop.bbox -> clipped body crop -> MiVOLO preprocessing -> age/sex inference
```

Validation rules:

- Missing `frame_id`: fail with `DemographicInputError` containing `track_id` if known.
- Duplicate frame IDs in FrameBatch: fail before inference.
- Missing image data: fail.
- Malformed bbox: require object with finite numeric `x1`, `y1`, `x2`, `y2` and `x2 > x1`, `y2 > y1` after clipping.
- Floating bbox: accept finite floats; clip to image boundaries; convert crop coordinates deterministically with `floor(x1/y1)` and `ceil(x2/y2)` before slicing so the full detection extent is included.
- Out-of-bounds bbox: clip to `[0, width] x [0, height]` and continue if non-zero area remains.
- Zero-area crop after clipping: fail with `track_id`, `frame_id`, and bbox.
- Unsupported image shapes: require 3D HWC with channel count `3`.
- Grayscale images: reject; do not expand silently.
- BGR versus RGB: production Demographic accepts repository RGB frames; if official MiVOLO helper is reused, wrap it so channel conversion is correct exactly once.
- Non-contiguous arrays: either copy to contiguous in preprocessing or fail. Recommendation: copy crop tensors to contiguous explicitly after slicing because slicing often creates non-contiguous views; still require original FrameBatch image be a valid uint8 array.
- Empty EventBatch: return `{"results": []}`.
- Empty FrameBatch with non-empty EventBatch: fail because required source pixels are missing.

Fail-fast is recommended for production because current `Detect`, `Track`, and `Event` all raise `ValueError` for malformed required input rather than fabricating defaults. Empty EventBatch is a valid no-work case.

## 6. Age and sex conversion rules

Age conversion from MiVOLO output:

1. Require raw model output shape `N x 3` for this checkpoint (`no_gender=False`).
2. Use `raw_age = output[:, 2]`.
3. Validate each raw value is finite.
4. Convert continuous age: `age_float = raw_age * (max_age - min_age) + avg_age`, with metadata from checkpoint (`min_age=1`, `max_age=95`, `avg_age=48.0`).
5. Validate converted age is finite.
6. Clamp to `[min_age, max_age]` because the checkpoint metadata defines the trained age range and model regression can overshoot.
7. Convert to integer with Python `round()` semantics? Recommendation: avoid banker's rounding ambiguity by using half-up integer rounding: `int(math.floor(age_float + 0.5))` after clamping. This is deterministic and approximates official display rounding to nearest while producing the required integer exact-age estimate.
8. Do not produce age buckets in Demographic.
9. NaN, infinity, or unexpected output shape raises `DemographicModelError`.

Sex conversion from MiVOLO output:

1. Require raw model output shape `N x 3`.
2. Use `gender_logits = output[:, :2]`.
3. Validate both logits are finite.
4. Select `gender_index = argmax(gender_logits, dim=1)`; softmax is unnecessary for argmax but may be used for diagnostics only.
5. Verified official mapping: index `0 -> "male"`, index `1 -> "female"` in `MiVOLO.fill_in_results()`; training dataset parses `M` as `0` and non-`M`/`F` as `1`.
6. Public mapping: if index `0`, output `sex=1`; if index `1`, output `sex=0`.
7. Any other output shape/index raises `DemographicModelError`.

## 7. Legacy demographics audit

Current `demographics` files:

| File | Current role | Classification | Reason |
| --- | --- | --- | --- |
| `demographics/__init__.py` | Exports `DemographicsEngine` | replace | Public API must become `Demographic`; no FairFace shim unless a production import is found. |
| `demographics/demographics_engine.py` | FairFace ResNet34 loader, preprocessing, `predict(person_image)` | delete/replace | Contains FairFace checkpoint path, race output, age-bucket index logic, and wrong public contract. |
| `demographics/assemble_weights.sh` | Reconstructs MiVOLO checkpoint | retain unchanged | Validated checksum and size. |
| `demographics/model_parts/demographicweights.pth.part-00` | checkpoint part | retain unchanged | Source part for generated checkpoint. |
| `demographics/model_parts/demographicweights.pth.part-01` | checkpoint part | retain unchanged | Source part for generated checkpoint. |
| `demographics/demographicweights.pth` | generated checkpoint | retain local uncommitted | Runtime artifact, not source. |

Legacy public exports: `DemographicsEngine` only.

FairFace-specific code in `demographics/demographics_engine.py`:

- Class/docstring says FairFace.
- `CHECKPOINT_PATH = fairface_alldata_4race_20191111.pt`.
- `torchvision.models.resnet34(weights=None)` and `model.fc = nn.Linear(..., 18)`.
- `RACE_SLICE`, `SEX_SLICE`, `AGE_SLICE` over 18 FairFace outputs.
- `transforms.Resize((224, 224))`, PIL conversion, torchvision normalization.
- `predict()` returns `race`, `sex`, and `age` as argmax class indices.

Race-related fields/labels: `RACE_SLICE`, `race = int(...)`, return key `"race"`, FairFace `4race` checkpoint name.

Age-bucket logic: `AGE_SLICE = slice(9, 18)` and `argmax` over nine FairFace age classes; replace with continuous MiVOLO age conversion.

Repository references/imports: `rg` found no active production imports of `DemographicsEngine` outside `demographics/__init__.py`; no compatibility shim is justified.

Dependency entries: there is no requirements/pyproject file in the repository. Implementation PR must add dependency documentation or dependency files if the project establishes them later.

Documentation references: no existing docs found.

Tests/scripts: no current demographics tests; `test_tracking_v2_pipeline.py` currently stops after Event summary.

## 8. `test_tracking_v2_script` current path and extension plan

Exact path: `test_tracking_v2_pipeline.py`. This is the existing `test_tracking_v2_script` harness; do not create a disconnected replacement.

Current behavior:

- CLI: positional `input` defaults to `videoplayback.mp4`; optional `--output` for annotated replay path.
- Output video default: `test_tracking_v2_pipeline.py` sibling `tracking_replay.mp4`.
- Video loading: `cv2.VideoCapture(str(video_path))`.
- FPS: uses `CAP_PROP_FPS`, falling back to `30.0` if invalid/outside `0..240`.
- Frame ID: `frame-{frame_index}`.
- Timestamp: raw frame index as float, not seconds.
- Detector: `detect = Detect()`, `detection_batch = detect(frame)`.
- Observation construction: no separate Observation object; DetectionBatch is passed directly to `Track`.
- Tracker: `tracking_state = Track(tracking_state, detection_batch)`.
- Line config: module constant `LINE_CONFIG` with points `(100,300)` and `(700,300)`.
- Visualization: draws track boxes/labels, summary overlay, and line overlay on BGR frame.
- Console output: progress counter, birth/unmatched diagnostics, track summary, event summary with raw timestamp and elapsed seconds.
- Saved artifacts: annotated MP4 only.

Future harness call flow using actual interfaces:

```python
event_batch = Event(tracking_state, LINE_CONFIG)
frame_batch = build_minimal_frame_batch(video_path, selected_event_crops)
demographics_batch = Demographic()(event_batch, frame_batch)
enriched = build_enriched_events(event_batch, demographics_batch)
```

The harness must not implement crossing logic, infer timestamps, or mutate `event_batch` to add demographics.

### 8.1 FrameBatch strategy decision

Options:

- Option A keep all frames: simplest but memory grows with video length.
- Option B two-pass source-video resolution: after Event, reopen video and load only frames referenced by selected per-track crops. Bounded memory, requires deterministic `frame-{index}` mapping.
- Option C existing cache: not available.

Recommendation: Option B. The current harness uses deterministic `frame-{frame_index}` IDs and raw frame-index timestamps, making re-reading by source frame index straightforward. Build a minimal FrameBatch containing only unique frames referenced by selected per-track crops. Convert reread BGR frames to contiguous RGB exactly like the first pass.

### 8.2 Enriched output contract

Recommended JSON path: `output/events_with_demographics.json` unless a future harness CLI adds an override such as `--events-output`.

Schema:

```json
{
  "events": [
    {
      "track_id": "track_0007",
      "timestamp": 14.633,
      "event_type": 1,
      "age": 32,
      "sex": 1
    }
  ]
}
```

Ordering: `timestamp` ascending, then `track_id`, then `event_type`.

Join validation:

- Build `demographics_by_track = {result["track_id"]: result for result in demographics_batch["results"]}`.
- Duplicate demographic result for one track: fail.
- Event without demographic result: fail.
- Demographic result without matching Event: fail.
- No events: success with `{"events": []}`.

Timing semantics:

- Use exactly `event["timestamp"]` from Event.
- Never use wall-clock inference time, final video timestamp, track closure time, or frame processing completion time.
- If console output wants seconds, calculate presentation seconds as existing code does: `event["timestamp"] / fps`. Machine-readable `timestamp` remains Event's value.

Harness error behavior:

- Model checkpoint missing: fail early with `Run: bash demographics/assemble_weights.sh`.
- Invalid checksum/incompatible checkpoint: fail before video processing where possible.
- Invalid crop/missing frame: fail immediately with `track_id`, `frame_id`, and bbox.
- Prefer fail-fast over accumulating errors because current harness raises immediately for video/writer failures and production stages validate fail-fast.

## 9. Testing plan

### 9.1 Unit tests

Create `test_demographics.py`:

- Empty EventBatch returns `{"results": []}`.
- One event/track returns one result.
- Multiple events for one track run inference once and return one result.
- Multiple tracks return multiple results.
- Deterministic ordering by earliest event timestamp then track_id.
- Frame lookup by `best_crop.frame_id`.
- Missing frame ID raises stage-specific error.
- Duplicate FrameBatch frame IDs raise.
- Malformed bbox raises.
- Out-of-bounds bbox clips.
- Zero-area crop after clipping raises.
- RGB/BGR conversion policy verified with a color-sentinel crop.
- Expected tensor shape is `N x 6 x 224 x 224` for body-only face+body checkpoint.
- Age conversion tests for finite values, half-up rounding, bounds, NaN/inf, shape errors.
- Sex mapping tests verify logits `[high, low, age] -> sex=1` and `[low, high, age] -> sex=0`.
- No race output.
- CPU inference backend path.
- Mocked model outputs avoid loading 105 MB checkpoint.
- Model-load failure and checkpoint incompatibility.

Create or extend `test_events.py` only if Event contract must be locked further; do not change Event behavior.

### 9.2 Integration tests

Create `test_event_demographic_integration.py`:

- EventBatch with one completed crossing feeds Demographic with matching FrameBatch.
- Two crossings for same track produce one demographic result reused for both enriched events.
- Event timestamp preserved exactly.
- Join by track ID.
- Result order independent of DemographicsBatch input order.

### 9.3 Real-checkpoint smoke test

Create `test_demographics_real_checkpoint.py` with a slow/integration marker:

- Skip if `demographics/demographicweights.pth` does not exist.
- Verify checksum.
- Load MiVOLO on CPU.
- Run one valid synthetic or fixture body crop.
- Assert output types and finite values only; do not assert specific age/sex for arbitrary synthetic imagery.

### 9.4 Harness tests

Create `test_tracking_v2_harness_demographics.py` or extend existing harness tests:

- Zero-event video writes `{"events": []}`.
- One entry.
- One exit.
- Entry then exit for one track shares one demographic result.
- Multiple tracks.
- JSON file creation at `output/events_with_demographics.json` or CLI override.
- Exact output schema and integer encodings.
- Deterministic event ordering.
- Preserved Event timestamps.
- Existing annotated-video behavior remains operational.
- Checkpoint missing fails with assemble command.
- Frame-resolution failure includes context.

Manual validation command after implementation:

```bash
python test_tracking_v2_pipeline.py videoplayback.mp4 --output output/tracking_replay.mp4
```

## 10. Dependency-aware implementation sequence

1. Lock contracts.
   - Modify/create: `test_demographics.py`, possibly `test_event_demographic_integration.py`.
   - Public interfaces: none yet.
   - Completion: failing tests express EventBatch, FrameBatch, and DemographicsBatch contracts.
2. Remove/isolate FairFace.
   - Delete/replace: `demographics/demographics_engine.py`.
   - Modify: `demographics/__init__.py`.
   - Completion: no public `DemographicsEngine`, no `race`, no FairFace checkpoint path.
3. Establish MiVOLO model loading.
   - Create: `demographics/model.py`; maybe `demographics/_mivolo/` if vendoring.
   - Completion: checksum/metadata validation, CPU load, `eval()`.
4. Implement preprocessing.
   - Create: `demographics/preprocessing.py`.
   - Completion: RGB crop -> `6x224x224` tensor with zero-face channels and body channels.
5. Implement per-track crop selection.
   - Modify: `demographics/demographic.py`.
   - Completion: unique track IDs and deterministic crop selection/failure if conflicting crops lack quality.
6. Implement once-per-track inference.
   - Modify: `demographics/demographic.py`, `demographics/model.py`.
   - Completion: fake backend call count equals unique tracks.
7. Build DemographicsBatch.
   - Modify: `demographics/demographic.py`.
   - Completion: `{"results": [{"track_id", "age", "sex"}]}` only.
8. Add unit tests.
   - Create/modify test files above.
   - Completion: unit suite passes without checkpoint.
9. Add real-checkpoint smoke test.
   - Create: `test_demographics_real_checkpoint.py`.
   - Completion: skips when checkpoint absent; passes CPU smoke when assembled.
10. Extend harness.
    - Modify: `test_tracking_v2_pipeline.py` only; do not create separate script.
    - Public interface: add optional JSON output CLI if desired.
    - Completion: current annotated-video diagnostics retained and JSON emitted.
11. Add Event-to-Demographic integration tests.
    - Create: `test_event_demographic_integration.py`.
    - Completion: timestamps preserved and joins use track_id.
12. Validate full video workflow.
    - Run manual command and inspect JSON/MP4.
13. Document execution commands.
    - Modify: future docs/README if repository adds docs convention.

Files to delete in implementation PR: `demographics/demographics_engine.py` unless migration chooses to replace in place. Files to retain unchanged: weight parts and assembly script.

## 11. Non-goals

No race/ethnicity inference, face recognition, identity recognition, re-identification, second detector, MiVOLO YOLO detection, line-crossing changes, unrelated tracker changes, database/BigQuery/Cloud Run work, Assemble implementation, age buckets, business IDs, production windowing, live ingestion, GUI, or optimization beyond avoiding repeated model loads.

## 12. Required decision table

| Decision | Recommended choice | Evidence | Files affected |
| --- | --- | --- | --- |
| Public Demographic API | `from demographics import Demographic`; `Demographic(event_batch, frame_batch)` via `__call__` | Existing packages export narrow callable names: `Detect`, `Track`, `Event` | `demographics/__init__.py`, `demographics/demographic.py` |
| Function vs class | Class with `__call__` | Model lifecycle/checkpoint/device state must load once like `Detect` | `demographics/demographic.py`, `demographics/model.py` |
| MiVOLO source strategy | Pinned dependency if acceptable; otherwise vendor minimal official v1 inference internals under `demographics/_mivolo/` | Checkpoint requires `mivolo_d1_224`, `in_chans=6`, metadata-aware loader | `demographics/model.py`, optional `demographics/_mivolo/*` |
| Frame resolution | Demographic accepts FrameBatch with RGB `np.uint8` frames; harness uses two-pass minimal frames | `Detect` Frame contract and harness deterministic `frame-{index}` IDs | `demographics/demographic.py`, `demographics/preprocessing.py`, `test_tracking_v2_pipeline.py` |
| Per-track crop selection | Use identical Event `best_crop`; if conflicting crops lack quality, fail | Current Event copies one final track-level `best_crop` and omits confidence | `demographics/demographic.py` |
| Age conversion | `floor(clamp(raw*(max-min)+avg, min, max)+0.5)` | Official MiVOLO uses output column 2 and metadata de-normalization | `demographics/model.py` or `demographics/demographic.py` |
| Sex mapping | Argmax logits `output[:2]`; index 0 -> male -> `sex=1`, index 1 -> female -> `sex=0` | Official `fill_in_results()` maps index 0 male and index 1 female; dataset parses `M` as 0 | `demographics/model.py` |
| Failure policy | Fail-fast with stage-specific exception; empty events return empty results | Existing `Detect`, `Track`, `Event` validate malformed inputs and raise | `demographics/exceptions.py`, `demographics/demographic.py`, harness |
| Output ordering | Demographic results by earliest event timestamp then track_id; enriched events by timestamp then track_id then event_type | Determinism and no positional joins | `demographics/demographic.py`, `test_tracking_v2_pipeline.py` |
| Harness JSON path | `output/events_with_demographics.json` | No existing JSON convention; requested default is suitable | `test_tracking_v2_pipeline.py` |

## 13. Unresolved issues before implementation approval

1. Dependency/license decision: confirm whether adding MiVOLO/timm as dependencies is acceptable or whether minimal vendoring is required.
2. Checkpoint checksum enforcement location: decide whether production `Demographic.__init__` always checks SHA-256 or only the harness/slow test does. Recommendation: production checks checksum by default with an opt-out only for tests.
3. Device CLI/config: decide whether harness should expose `--demographics-device` or rely on deterministic default `cuda` if available else `cpu`. Recommendation: support optional CLI with default auto.
4. Public exception names: decide whether to add `demographics.exceptions` or keep exceptions private and raise `ValueError`/`RuntimeError`. Recommendation: add stage-specific public exception classes only if callers will catch them.

## 14. Completion-report facts for this planning PR

- Planning document path: `docs/plans/mivolo_demographics_and_tracking_harness_plan.md`.
- Existing production files inspected: `detect/__init__.py`, `detect/detect.py`, `track/__init__.py`, `track/tracker.py`, `track/matcher.py`, `track/config.py`, `events/__init__.py`, `events/event.py`, `demographics/__init__.py`, `demographics/demographics_engine.py`, `demographics/assemble_weights.sh`, `test_tracking_v2_pipeline.py`, `test_events.py`, `pytest.ini`, `.gitignore`.
- Legacy demographics files discovered: `demographics/__init__.py`, `demographics/demographics_engine.py`, `demographics/assemble_weights.sh`, `demographics/model_parts/demographicweights.pth.part-00`, `demographics/model_parts/demographicweights.pth.part-01`, generated local `demographics/demographicweights.pth`.
- Exact `test_tracking_v2_script` path: `test_tracking_v2_pipeline.py`.
- Recommended future signature: `Demographic().__call__(event_batch: dict, frame_batch: dict) -> {"results": [{"track_id": str, "age": int, "sex": int}]}`.
- Recommended enriched-event output schema: `{"events": [{"track_id": str, "timestamp": float, "event_type": int, "age": int, "sex": int}]}` joined by `track_id`.
- Recommended FrameBatch strategy: harness Option B, two-pass source-video reread of only unique selected event-crop frames.
- Confirmation: this PR should commit only this planning document.
