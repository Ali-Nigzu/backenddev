# Track Module V2 Transformation Plan

## Purpose

This document is an investigation and migration plan for converting the current Track module into the strict V2 state-transition architecture. It intentionally does **not** implement the refactor.

Target contract:

```text
Track(tracking_state, observation_batch) -> tracking_state
```

Track must become one deterministic transition from `State(t)` plus `ObservationBatch(t)` to `State(t+1)`. It must not own video reading, frame iteration, detection, embedding, observation construction, persistence, window management, output rendering, demographics, event assembly, or other pipeline orchestration.

## Existing V2 Architecture Context

The locked V2 pipeline is:

```text
Frame
  ↓
Detect
  ↓
Embed
  ↓
Observe
  ↓
Track
  ↓
Event
  ↓
Demographic
  ↓
Assemble
  ↓
Output
```

The currently completed direction is:

```text
DetectionBatch + EmbeddingBatch
            |
            v
        Observe()
            |
            v
     ObservationBatch
```

`Observe` already returns the expected V2 shape: `frame_id`, `timestamp`, and ordered `observations`, where each observation has `detection_id`, `bbox`, `center`, `embedding`, and `confidence`.

## Current Track Folder Structure

```text
track/
  __init__.py      Public exports for TrackV2, TrackV2Config, RuntimeTrackV2.
  config.py        Thresholds and lifecycle/motion configuration.
  lifecycle.py     Track creation, matchable-track selection, promotion, stale closure.
  matching.py      Motion gate, cosine embedding similarity, candidate building, assignment.
  models.py        RuntimeTrackV2 mutable runtime dataclass.
  motion.py        Prediction, velocity, distance helpers.
  tracker.py       Stateful TrackV2 wrapper and update loop.
```

Related current runner/test harness:

```text
test_tracking_v2_pipeline.py
```

Despite its name, `test_tracking_v2_pipeline.py` is a full runnable pipeline script: it opens video, calls detection, calls embedding, builds observations, updates the tracker, detects events, predicts demographics, renders video overlays, writes contact sheets, and prints summaries.

## Current Track Execution Flow

```text
External script / runner
  |
  | owns video capture loop, FPS, frame index, output writer, overlay, summaries
  v
For each video frame:
  |
  | build frame_packet
  v
Detect(frame_packet) or legacy detect(frame_packet)
  |
  v
For each detection:
  |
  | call embedder.embed(...)
  | call build_observation(...)
  v
observations_by_ts[timestamp] = [observation, ...]
  |
  v
tracker.update(observations_by_ts)
  |
  | TrackV2.update owns an internal timestamp loop over sorted(observations_by_ts.keys())
  | TrackV2 owns persistent mutable state:
  |   - self.tracks
  |   - self.pending_births
  |   - self.frame_index
  |   - self.last_update_timestamp
  |   - debug counters
  v
For each timestamp inside update:
  |
  | eligible_tracks = ACTIVE + TENTATIVE tracks
  v
assign_matches(eligible_tracks, observations, config)
  |
  | build_candidates(...)
  |   - motion_gate(...)
  |   - predict_center(...)
  |   - distance(...)
  |   - embedding_similarity(track.last_embedding, observation.embedding)
  |
  | greedy assignment:
  |   - sort by motion distance, track index, observation index
  |   - resolve motion ambiguity with embedding similarity
  v
Matched tracks updated in place
  |
  | update velocity, center, bbox, embedding, timestamps, hit/miss counts,
  | detection history, center history, lifecycle state
  v
close_stale_tracks(...)
  |
  | increment miss counts for unmatched ACTIVE/TENTATIVE tracks
  | close tentative/active tracks according to age and miss thresholds
  v
Unmatched observations processed
  |
  | skip if gated to an existing track but not chosen
  | skip if recent unmatched track remains plausible
  | otherwise record/advance pending birth buffer
  | create RuntimeTrackV2 when birth buffer threshold is reached
  v
Prune pending births, update last timestamp, increment internal frame_index
  |
  v
Return (list(self.tracks), assignment_map)
  |
  v
External script consumes tracks and assignment_map for events, display IDs,
contact-sheet crops, best detections, demographics, overlays, and summaries.
```

## Where Key Things Happen Today

| Concern | Current location | Notes |
| --- | --- | --- |
| Entry point | `TrackV2.update(observations_by_ts, current_timestamp=None)` | Accepts a timestamp-to-observations map, not a single `ObservationBatch`. |
| Frame loop ownership | `test_tracking_v2_pipeline.py` | Not inside `track/`, but tightly coupled to current tracker return shape. |
| Timestamp loop ownership | `TrackV2.update` | Tracker can process multiple timestamps in one call, which conflicts with one-batch V2 transition. |
| Observations enter | `observations_by_ts[timestamp]` | Current observations are legacy dicts using list-shaped `bbox`/`center`, not the locked V2 dict-shaped `bbox`/`center`. |
| Matching happens | `track/matching.py` | Valuable algorithmic core. |
| Motion helpers | `track/motion.py` | Valuable deterministic helper logic. |
| Lifecycle handling | `track/lifecycle.py` and `TrackV2.update` | Valuable but currently mutates hidden object state. |
| State lives | `TrackV2` instance and `RuntimeTrackV2` objects | Hidden state violates strict V2 stateless module principle. |
| Outputs produced | `TrackV2.update` plus runner | Tracker returns tracks and `assignment_map`; runner derives events, overlays, crops, demographics, display IDs. |

## Responsibility Map

### KEEP: Tracking Intelligence to Preserve

These behaviours should survive unless founder decisions explicitly change them:

- Cosine embedding similarity for appearance comparison.
- Motion prediction based on current center, velocity, and elapsed time.
- Distance-based motion gating.
- Maximum speed rejection.
- Candidate construction across matchable tracks and current observations.
- Deterministic greedy assignment ordered by motion distance, track index, and observation index.
- Ambiguity resolution that uses embedding similarity when motion distances are close.
- ACTIVE/TENTATIVE/CLOSED lifecycle semantics.
- Promotion from TENTATIVE to ACTIVE after enough hits.
- Miss counting and stale closure for unmatched tracks.
- Minimum lifetime before closure.
- Pending-birth buffering for unmatched detections.
- Plausibility check for recently unmatched tracks to avoid premature new-track creation.
- Velocity smoothing.
- Threshold/config behaviour in `TrackV2Config`.

### MOVE: Responsibilities Belonging Elsewhere

These should be moved out of the Track module contract or owned by an orchestrator/test harness:

- Video reading and frame iteration.
- FPS normalization and timestamp generation from frame index.
- Detection calls.
- Embedding calls.
- Observation construction.
- Event detection invocation after tracking.
- Demographic prediction.
- Best detection/crop collection.
- Runtime display-ID mapping for visualization.
- Video writer and overlay rendering.
- Contact-sheet generation.
- Summary printing and progress display.
- Multi-timestamp batching/window management currently accepted by `TrackV2.update`.
- Persistence or loading/saving of tracking state, if added later.

### DELETE or Redesign: V2-Violating Shape

These should not remain in the V2 Track interface:

- Hidden persistent analytics state inside a `TrackV2` instance.
- `TrackV2.update(observations_by_ts)` as the primary contract.
- Multiple timestamp transitions inside one Track call.
- Implicit synthetic timestamp fallback when no observations are supplied.
- Random UUID generation inside the deterministic transition unless seeded or replaced by state-owned deterministic ID allocation.
- Debug counters as hidden mutable instance fields.
- Legacy observation shape assumptions: `center[0]`, `center[1]`, and list bbox values.
- Returning `assignment_map` as an undeclared side output if the non-negotiable contract remains strictly `TrackingState` only.

## Current State Inventory

The current tracker preserves these values between frame updates:

### Tracker-level State

- `tracks`: list of all runtime tracks, including CLOSED tracks.
- `pending_births`: buffered unmatched observations with center, observation, first/last seen frame, and count.
- `frame_index`: monotonically incremented internal counter.
- `last_update_timestamp`: timestamp of last processed update.
- `last_new_tracks_created`: debug/diagnostic counter.
- `last_debug_report`: debug counters for new tracks, continuations, rejected gates, and forced continuations.
- `config`: thresholds and policy values.

### Per-track State

- `runtime_track_id`.
- `state`: `TENTATIVE`, `ACTIVE`, or `CLOSED`.
- `current_center`.
- `current_bbox`.
- `velocity`.
- `first_seen_timestamp`.
- `last_seen_timestamp`.
- `hit_count`.
- `miss_count`.
- `detection_history`.
- `center_history`.
- `last_embedding`.
- `closed_timestamp`.
- `created_frame_index`.
- `last_matched_frame_index`.
- `last_unmatched_frame_index`.

## Proposed V2 Track Architecture

### High-level Shape

```text
ObservationBatch(t)
        |
        v
Track(tracking_state, observation_batch)
        |
        | pure/explicit transition using config + helper functions
        v
TrackingState(t+1)
```

Track should be a stateless callable or function. It may have immutable configuration, but no per-camera/per-run/per-frame analytics state in the module object.

Recommended public shape for implementation discussion:

```python
class Track:
    def __init__(self, config: TrackConfig | None = None):
        self.config = config or TrackConfig()

    def __call__(self, tracking_state: dict, observation_batch: dict) -> dict:
        ...
```

The callable may mutate and return `tracking_state` only if the contract explicitly allows `TrackingState` mutation. Otherwise it should return a copied/updated state. The V2 architecture says the only modified analytics object is `TrackingState`, so in-place mutation can be acceptable if documented and tested.

### Internal Engine Extraction

Extract the current algorithm from `TrackV2.update` into an engine function that processes exactly one timestamp/frame:

```text
transition_one_batch(state, observation_batch, config) -> state
```

The existing helpers should remain recognizable:

- `embedding_similarity` stays algorithmically unchanged.
- `motion_gate` stays algorithmically unchanged, after adapting observation accessors to V2 shape.
- `assign_matches` stays algorithmically unchanged, after adapting data access.
- `compute_velocity`, `predict_center`, and `distance` stay unchanged or receive small type-normalization wrappers.
- `promote_if_ready`, `close_track`, and stale closure logic stay unchanged in behaviour, but operate on explicit state-held track records.
- Pending-birth and recent-unmatched plausibility logic move from hidden `self` fields into explicit `TrackingState` fields.

## Proposed TrackingState Contract

The locked architecture currently states a minimal public `TrackingState`:

```text
TrackingState:
  tracks: Track[]

Track:
  track_id: string
  path: Points[]
  best_crop: BestCrop
```

That public shape is not enough to preserve the current tracking algorithm. The current algorithm requires runtime fields for matching, lifecycle, IDs, velocity, misses, hits, and pending births. There are two viable contract directions for founder decision.

### Option A: Expand TrackingState to Include Runtime Tracking Fields

Recommended if behaviour preservation is the priority and Track must return only `TrackingState`.

```text
TrackingState:
  tracks: RuntimeTrack[]
  pending_births: PendingBirth[]
  frame_index: int
  last_update_timestamp: float | null
  next_track_index: int or id_allocator state
  diagnostics: TrackingDiagnostics | optional

RuntimeTrack:
  track_id: string
  lifecycle_state: TENTATIVE | ACTIVE | CLOSED
  current_center: Point2D
  current_bbox: BoundingBox
  velocity: Point2D-like vector
  first_seen_timestamp: float64
  last_seen_timestamp: float64
  hit_count: int
  miss_count: int
  detection_history: string[]
  path: Points[]
  last_embedding: FeatureVector | null
  best_crop: BestCrop | null
  closed_timestamp: float64 | null
  created_frame_index: int
  last_matched_frame_index: int
  last_unmatched_frame_index: int | null

PendingBirth:
  center: Point2D
  observation: Observation
  first_seen_frame: int
  last_seen_frame: int
  count: int

TrackingDiagnostics optional:
  last_assignment_map: { detection_id: track_id }
  last_new_tracks_created: int
  last_debug_report: dict
```

Why each field is needed:

- `tracks`: persistent identities and histories.
- `pending_births`: preserves current delayed-birth logic.
- `frame_index`: required by pending-birth pruning and max association gap behaviour.
- `last_update_timestamp`: currently used for empty updates and useful for monotonic validation.
- `next_track_index` or equivalent: needed to replace nondeterministic UUIDs.
- `current_center`, `current_bbox`, `velocity`, `last_embedding`: matching and updates.
- `first_seen_timestamp`, `last_seen_timestamp`, `created_frame_index`, `last_matched_frame_index`, `last_unmatched_frame_index`: lifecycle and plausibility logic.
- `hit_count`, `miss_count`, `lifecycle_state`, `closed_timestamp`: lifecycle semantics.
- `detection_history` and `path`: regression/debug/event semantics.
- `best_crop`: required by downstream Event/Demographic contracts.
- diagnostics/assignment map: needed only if downstream consumers require per-observation track assignment after the call.

### Option B: Keep Public TrackingState Minimal and Add Private Runtime State

This aligns with the existing locked text more closely but creates a conflict: V2 says persistent analytics memory must live only in `TrackingState`, so a private runtime state outside `TrackingState` would violate the stateless-module principle unless it is still part of `TrackingState` under a namespaced internal field.

A compromise is:

```text
TrackingState:
  tracks: Track[]
  runtime: TrackRuntimeState
```

This keeps public downstream track data clear while explicitly carrying algorithmic runtime state in the only allowed state object.

## ObservationBatch Requirements

The provided expected contract is sufficient for current matching if Track normalizes V2-shaped fields:

```text
ObservationBatch:
  frame_id: string
  timestamp: float64
  observations:
    detection_id: string
    bbox: {x1, y1, x2, y2}
    center: {x, y}
    embedding: FeatureVector
    confidence: float32
```

Additional fields may be required only for the following decisions:

- `frame_id` should be retained in `best_crop` updates.
- A raw crop/image should **not** be added to ObservationBatch unless the locked `BestCrop` contract requires Track to own actual crop pixels. Current locked text only names bbox, confidence, embedding, and frame id.
- If lifecycle should be frame-count based, `TrackingState.frame_index` is enough. If lifecycle should be timestamp-only, frame index can eventually be removed after behaviour changes are approved.

## Output Semantics

The non-negotiable target says Track returns `TrackingState`, not `(TrackingState, TrackOutput)`. However, current downstream code depends on `assignment_map` to label observations and collect crops for the current frame.

Possible resolutions needing founder decision:

1. Store `last_assignment_map` inside `TrackingState.diagnostics` or `TrackingState.last_assignments`.
2. Return a second object, e.g. `TrackOutput`, despite the stated target.
3. Derive current assignments from updated tracks' latest detection history, if sufficient and deterministic.

Do not implement until this is decided.

## Existing Tests and Regression Value

Current test coverage is limited:

- `testv2/test_detect_v2.py` validates Frame -> Detect -> Embed -> Observe and confirms the ObservationBatch shape.
- `test_tracking_v2_pipeline.py` is not a deterministic unit test for Track. It is a full video pipeline runner and visual/regression harness.

### Algorithm Validation to Preserve/Add

The current code lacks focused deterministic Track unit tests. Add tests that pin existing algorithm behaviour before or during extraction:

- Same person across frames keeps one track ID.
- Motion gate rejects impossible jumps.
- Embedding similarity breaks ambiguous motion ties.
- Tentative track promotes after configured hit count.
- Unmatched tentative/active tracks close after configured misses and minimum lifetime.
- Pending birth creates a track only after buffer threshold.
- Recent unmatched plausible track suppresses premature birth.
- Velocity smoothing updates predictably.
- Closed tracks are not matchable.

### Architecture Validation to Add

Add tests for strict V2 shape:

- `Track` accepts `TrackingState` and one `ObservationBatch`.
- `Track` returns the same `TrackingState` object if in-place mutation is chosen, or a new state if immutability is chosen.
- No hidden state survives in the Track instance between calls.
- Two independent states processed by the same Track instance do not influence each other.
- Empty ObservationBatch advances lifecycle according to the approved timestamp/frame policy.
- ObservationBatch order produces deterministic assignment results.

## Deterministic State-transition Test Strategy

Do not rely on full videos for primary Track validation. Test transitions directly:

```text
previous TrackingState
+ ObservationBatch
+ config
= expected next TrackingState
```

Proposed tests:

1. **First person appears**
   - Initial empty state.
   - One observation.
   - Depending on pending-birth config, assert pending birth or new TENTATIVE track.

2. **Person moves**
   - Existing ACTIVE/TENTATIVE track with center, velocity, embedding.
   - New observation within motion gate.
   - Assert same track ID, updated center/bbox/path, hit count, miss reset, velocity smoothing.

3. **Multiple people**
   - Two tracks and two observations.
   - Assert deterministic one-to-one assignment independent of incidental object identity.

4. **Missing observation**
   - Existing active track, empty observations.
   - Assert miss count increments and no new track is created.

5. **New person appears**
   - Existing track plus far unmatched observation.
   - Assert birth buffering first, then new track creation after threshold.

6. **Matching failure**
   - Observation outside motion/speed gate.
   - Assert old track is not updated and unmatched observation follows birth policy.

7. **Lifecycle transitions**
   - TENTATIVE to ACTIVE after hits.
   - TENTATIVE to CLOSED after misses.
   - ACTIVE to CLOSED after misses.
   - CLOSED tracks ignored by matcher.

8. **Timestamp behaviour**
   - Monotonic timestamp updates accepted.
   - Negative `dt` rejected or raises a contract error, depending on founder decision.
   - Large timestamp gaps interact with motion gate and max speed as expected.

9. **Embedding ambiguity**
   - Two candidates with close motion distances.
   - Stronger embedding match wins only when threshold delta is met.

10. **Deterministic IDs**
    - Given identical initial `TrackingState.next_track_index`, identical observations produce identical new track IDs.

### Role of Existing Video Tests

Full-video tests should remain as higher-level regression tests after deterministic unit tests exist. They are useful for catching integration drift in detection/embedding/observe/track/event/demographic interactions, but they should not be the only proof of Track correctness because video tests are slow, model-dependent, environment-sensitive, and difficult to diagnose.

## Migration Steps

1. **Freeze current behaviour with tests**
   - Add deterministic tests around `track/matching.py`, `track/lifecycle.py`, and `TrackV2.update` using synthetic observations.

2. **Define V2 state schema**
   - Resolve founder decisions on runtime fields, assignment output, ID allocation, closed-track retention, config ownership, and lifecycle timing.

3. **Add adapters without behaviour change**
   - Create small normalization helpers from V2 `ObservationBatch` dicts to the current internal observation access pattern.
   - Do not rewrite scoring or matching.

4. **Extract one-batch engine**
   - Move the body of the per-timestamp loop from `TrackV2.update` into a function that accepts explicit state, one timestamp, observations, and config.

5. **Introduce V2 `Track` facade**
   - `Track(tracking_state, observation_batch) -> tracking_state` delegates to the extracted engine.
   - Ensure the Track object owns only immutable config/resources.

6. **Make state explicit**
   - Move `tracks`, `pending_births`, `frame_index`, `last_update_timestamp`, debug counters, and ID allocator out of `self` and into `TrackingState`.

7. **Replace nondeterministic IDs**
   - Replace UUID creation with deterministic IDs derived from `TrackingState` allocator policy.

8. **Backwards compatibility layer**
   - Keep `TrackV2.update` temporarily as an adapter over the new engine if existing scripts need it.
   - Mark it legacy and do not let new V2 tests depend on it.

9. **Wire with Observe output**
   - Add architecture tests using `Observe()` output directly as Track input.

10. **Retire or move pipeline-runner responsibilities**
    - Keep full-video harness outside Track as integration/regression tooling.

## Risks and Hidden Dependencies

- Current UUID generation makes exact deterministic replay impossible.
- Current observations are list-shaped, while V2 observations are dict-shaped for `bbox`, `center`, and `embedding` values.
- `TrackV2.update` processes multiple timestamps per call; extracting one timestamp can subtly alter frame-index increments if callers pass multiple timestamps today.
- Empty updates synthesize timestamps today; V2 should likely require explicit `ObservationBatch.timestamp`.
- `closed_track_cooldown_sec` exists in config but is not currently used by the inspected tracking flow; changing or removing it may break expected future behaviour.
- `unmatched_track_indices` is returned by `assign_matches` but not used in `TrackV2.update`; refactor should avoid accidentally changing closure semantics.
- `gated_observation_indices` counts observations with at least one allowed candidate, not rejected pairs; debug metric naming may be misleading.
- Pending births store the full observation, including embedding; memory growth is bounded by pruning but still must be explicit in V2 state.
- `last_embedding` stores whatever object the embedder returns; V2 FeatureVector is a dict with `values`, so current `_iter_values` may need adaptation or `embedding_similarity` will return zero.
- Event detection currently expects runtime fields such as `runtime_track_id`, `last_seen_timestamp`, and `center_history`, not the minimal locked `Track` shape.
- Downstream visualization currently needs `assignment_map`; if Track only returns state, that dependency must be redesigned.
- Frame-count lifecycle fields and timestamp lifecycle fields coexist. Changing from frame counts to timestamp-only lifecycle can change behaviour.
- Best-crop ownership is not implemented in current `RuntimeTrackV2`; adding it during Track refactor could alter downstream demographics/event responsibilities.
- In-place mutation of `TrackingState` must be consistent and documented; accidental mutation of input `ObservationBatch` should be forbidden.

## Open Questions Requiring Founder Decisions

1. Should `Track` return strictly `TrackingState`, or may it return `TrackingState + TrackOutput` for current-frame assignments?
2. If the return must be only `TrackingState`, should `last_assignment_map` live inside `TrackingState`?
3. Should `TrackingState.tracks` be the minimal locked public track shape, or should it include runtime fields required by the algorithm?
4. If runtime fields are included, should they be top-level track fields or nested under a `runtime` namespace?
5. Should CLOSED/dead tracks remain in `TrackingState`, and for how long?
6. Where should configuration live: immutable `Track` constructor config, explicit function argument, or inside `TrackingState`?
7. Should track IDs remain globally unique, per process, per camera, or per `TrackingState`?
8. What deterministic ID format should replace UUIDs?
9. Should lifecycle timing be driven by frame counts, timestamps, or both during compatibility migration?
10. Should `Track` own lifecycle timing, or should another module decide when tracks are retired from state?
11. Is `pending_births` an approved part of `TrackingState`, or should new-track creation be immediate in V2?
12. Should `best_crop` be selected inside Track from observation confidence, or owned by Event/Demographic/Assemble?
13. Should `ObservationBatch` ever carry crop pixels, or only bbox/frame references and embeddings?
14. Should negative timestamps raise errors, close no tracks, or simply reject matching candidates as today?
15. Should existing `TrackV2.update` remain as a compatibility adapter after V2 `Track` is introduced?

## Recommended Next Action

Before implementation, resolve the founder decisions above, then add deterministic regression tests around the current algorithm. After tests pin behaviour, extract a one-batch explicit-state engine and introduce a strict V2 `Track` facade over it.
