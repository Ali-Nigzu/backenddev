# Analytics Engine V2 — Development Architecture Plan

## 1. Purpose

This document is the V2 development architecture plan for building a new production-grade Analytics Engine pipeline in parallel with the existing implementation.

The previous architecture file has been deleted intentionally. No architectural language, module list, ownership model, state model, pipeline definition, diagnostics model or output model from the previous architecture is carried forward.

V2 development must use only:

1. The locked V2 Contract Matrix.
2. This V2 development architecture plan.

If any existing code, deleted architecture text, comments, tests, scripts or historical assumptions conflict with the locked V2 Contract Matrix, the locked V2 Contract Matrix wins.

This is not an implementation document. It defines how future implementation work must be structured so that a new V2 pipeline can be built beside the current pipeline without changing existing production behaviour until V2 is deliberately adopted.

## 2. Development Strategy: Parallel V2 Pipeline

V2 must be developed as a separate parallel pipeline rather than by mutating the current pipeline in place.

The existing implementation may be inspected for algorithms and business logic, but it must not be treated as the V2 design. Existing modules may not silently become V2 modules unless they are rebuilt or wrapped to satisfy the locked V2 contracts exactly.

Development rules:

- Build V2 under explicit V2 ownership and naming.
- Keep the current runnable pipeline intact until a future cutover task explicitly changes runtime routing.
- Do not modify current production behaviour while creating V2 contracts or modules.
- Do not reuse old architecture decisions.
- Do not add compatibility fields to V2 contracts merely because current code has them.
- Do not hide state inside module instances.
- Do not create extra architectural stages outside the locked V2 pipeline.

## 3. Required V2 Pipeline

The V2 pipeline is exactly:

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

No stage may be added, removed, renamed, merged or split at the architecture level.

Internal helper functions are allowed during implementation, but they are not architectural stages and must not change module contracts.

## 4. Production Engineering Principles

### 4.1 Deterministic Execution

For identical inputs, initial `TrackingState`, configuration and model versions, V2 must produce identical `OutputBatch` and updated `TrackingState`.

Execution order must be explicit. Output ordering must be deterministic. Object identity rules must be deterministic. Modules must not depend on hidden state, ambient mutable globals, uncontrolled clocks or filesystem side effects for contract-level behaviour.

### 4.2 Stateless Modules

All V2 modules are architecturally stateless. The only stateful pipeline object is `TrackingState`.

A module may hold non-analytics runtime resources such as loaded weights or compute handles, but those resources must not contain hidden per-frame, per-track, per-camera or per-run analytics state.

### 4.3 Explicit Ownership

Every object has exactly one owner at a time. Ownership transfer happens only through the module boundaries defined by the locked V2 Contract Matrix.

Inputs must be treated as immutable unless the contract explicitly says the object is modified. The only modified analytics object in V2 is `TrackingState`.

### 4.4 Bounded Memory

Temporary objects must have short, obvious lifetimes. Persistent analytics memory must live only in `TrackingState`.

V2 implementations should prefer:

- minimal allocation;
- predictable object lifetime;
- early release of temporary batches;
- no unnecessary copies of images or vectors;
- bounded growth of track history according to future tracking policy;
- explicit ownership of any retained `BestCrop` data.

### 4.5 Predictable Compute

V2 must support high-throughput production workloads across many cameras by keeping compute boundaries explicit.

Batching is encouraged inside modules where it does not alter contracts. Parallelism is allowed only when deterministic outputs and required ordering are preserved.

### 4.6 Maintainability

Each module must have one clear responsibility. Modules must not invent new behaviours outside their contracts. Cross-module coupling must happen only through declared contract objects.

## 5. Locked Contract Objects

This section restates the locked V2 objects for implementation planning only. It does not redesign them.

### 5.1 Frame

`Frame` is owned by the orchestrator and contains:

- `frame_id : string`
- `timestamp : float64`
- `image : Image`

`Image` contains:

- `dtype : uint8`
- `shape : uint32[3]` with `[H, W, 3]`
- `channel_order : "RGB"`
- `memory_layout : "contiguous_row_major"`
- `value_range : [0, 255]`
- `color_space : "sRGB"`

### 5.2 DetectionBatch

`Detect()` creates `DetectionBatch`:

- `frame_id : string`
- `timestamp : float64`
- `detections : Detection[]`

Each `Detection` contains:

- `detection_id : string`
- `bbox : BoundingBox`
- `confidence : float32`

Each `BoundingBox` contains `x1`, `y1`, `x2`, `y2` as `float32`.

### 5.3 EmbeddingBatch

`Embed()` creates `EmbeddingBatch`:

- `frame_id : string`
- `timestamp : float64`
- `embeddings : Embedding[]`

Each `Embedding` contains:

- `detection_id : string`
- `vector : FeatureVector`

Each `FeatureVector` contains:

- `dtype : float32`
- `shape : uint32[1]`
- `values : float32[]`

### 5.4 ObservationBatch

`Observe()` creates `ObservationBatch` from `DetectionBatch` and `EmbeddingBatch`.

The required invariant is:

```text
DetectionBatch.detections[i].detection_id == EmbeddingBatch.embeddings[i].detection_id
```

`ObservationBatch` contains:

- `frame_id : string`
- `timestamp : float64`
- `observations : Observation[]`

Each `Observation` contains:

- `detection_id : string`
- `bbox : BoundingBox`
- `center : Point2D`
- `embedding : FeatureVector`
- `confidence : float32`

Each `Point2D` contains `x` and `y` as `float32`.

### 5.5 TrackingState

`TrackingState` is the only stateful analytics object:

- `tracks : Track[]`

Each `Track` contains:

- `track_id : string`
- `path : Points[]`
- `best_crop : BestCrop`

Each `Points` contains:

- `timestamp : float64`
- `center : Point2D`

Each `BestCrop` contains:

- `frame_id : string`
- `bbox : BoundingBox`
- `best_crop_confidence : float32`
- `best_crop_embedding : FeatureVector`

### 5.6 EventBatch

`Event()` creates `EventBatch` from `TrackingState` and `LineConfig`:

- `events : Event[]`

Each `Event` contains:

- `track_id : string`
- `timestamp : float64`
- `event_type : INT`, where `ENTRY = 1` and `EXIT = 0`
- `best_crop : BestCrop`

### 5.7 DemographicsBatch

`Demographic()` creates `DemographicsBatch` from `EventBatch`:

- `results : DemographicsResult[]`

Each `DemographicsResult` contains:

- `track_id : string`
- `age : int`
- `sex : int`, where `MALE = 1` and `FEMALE = 0`

### 5.8 OutputBatch

`Assemble()` creates `OutputBatch`:

- `rows : OutputRow[]`
- `model_id : Int`

Each `OutputRow` contains:

- `event_id : string`
- `event : int`
- `timestamp : float64`
- `sex : int`
- `age_bucket : int`

### 5.9 RunConfig and LineConfig

`Analyse()` consumes `RunConfig`:

- `source_id : Int`
- `model_id : Int`
- `frames : Frame[]`
- `line : LineConfig`
- `TrackingState : object`

`LineConfig` contains:

- `point_a : Point2D`
- `point_b : Point2D`

## 6. Module Development Contracts

### 6.1 Detect

Input owner: orchestrator.

Input: `Frame`.

Output owner: `Detect()`.

Output: `DetectionBatch`.

Objects created: `DetectionBatch`.

Objects modified: none.

Development requirements:

- V2 `Detect()` must not crop for downstream embedding as an architectural output.
- V2 `Detect()` must not produce embeddings, observations, tracks, events, demographics or output rows.
- Any model-specific preprocessing must remain internal and must not alter the contract.
- `DetectionBatch.frame_id` and `DetectionBatch.timestamp` must match the input frame.

### 6.2 Embed

Input owner: `Detect()`.

Input: `DetectionBatch`.

Output owner: `Embed()`.

Output: `EmbeddingBatch`.

Objects created: `EmbeddingBatch`.

Objects modified: none.

Development requirements:

- V2 `Embed()` must produce embeddings keyed by `detection_id`.
- V2 `Embed()` must not assign track IDs.
- V2 `Embed()` must not update `TrackingState`.
- V2 `Embed()` must preserve deterministic ordering aligned to `DetectionBatch`.

### 6.3 Observe

Input owners: `Detect()` and `Embed()`.

Inputs: `DetectionBatch` and `EmbeddingBatch`.

Output owner: `Observe()`.

Output: `ObservationBatch`.

Objects created: `ObservationBatch`.

Objects modified: none.

Development requirements:

- V2 `Observe()` must enforce the detection/embedding index-alignment invariant.
- V2 `Observe()` must compute `center` from `BoundingBox`.
- V2 `Observe()` must not create or modify tracks.
- V2 `Observe()` must not run event or demographic logic.

### 6.4 Track

Input owners: `Observe()` and `Analyse()`.

Inputs: `ObservationBatch` and `TrackingState`.

Output owner: `Track()`.

Output: `TrackingState`.

Objects modified: `TrackingState.tracks`.

Development requirements:

- V2 `Track()` is the only module allowed to modify `TrackingState`.
- V2 `Track()` must not retain hidden analytics state inside a tracker instance.
- V2 `Track()` must update `Track.path` and `Track.best_crop` according to future implementation policy while preserving the locked object shape.
- V2 `Track()` must return the updated `TrackingState` explicitly.
- V2 `Track()` must not emit events or demographics.

### 6.5 Event

Input owners: `Track()` and `Analyse()`.

Inputs: `TrackingState` and `LineConfig`.

Output owner: `Event()`.

Output: `EventBatch`.

Objects created: `EventBatch`.

Objects modified: none.

Development requirements:

- V2 `Event()` must derive events only from `TrackingState` and `LineConfig`.
- V2 `Event()` must not mutate `TrackingState`.
- V2 `Event()` must emit `event_type` using the locked integer values.
- V2 `Event()` must include `BestCrop` from the related track.

### 6.6 Demographic

Input owner: `Event()`.

Input: `EventBatch`.

Output owner: `Demographic()`.

Output: `DemographicsBatch`.

Objects created: `DemographicsBatch`.

Objects modified: none.

Development requirements:

- V2 `Demographic()` must produce results keyed by `track_id`.
- V2 `Demographic()` must not mutate events or tracking state.
- V2 `Demographic()` must use `MALE = 1` and `FEMALE = 0`.
- V2 `Demographic()` must preserve deterministic result ordering for `Assemble()`.

### 6.7 Assemble

Input owners: `Event()` and `Demographic()`.

Inputs: `EventBatch` and `DemographicsBatch`.

Output owner: `Assemble()`.

Output: `OutputBatch`.

Objects created: `OutputBatch`.

Objects modified: none.

Development requirements:

- V2 `Assemble()` must join events and demographic results by aligned `track_id`.
- V2 `Assemble()` must emit only the locked `OutputBatch` shape.
- V2 `Assemble()` must not compute new tracks, events or demographics.
- V2 `Assemble()` must attach `model_id` from `RunConfig` through `Analyse()`.

## 7. Analyse() Development Plan

`Analyse()` is the V2 orchestration boundary.

Input owner: orchestrator.

Input: `RunConfig`.

Output owner: `Analyse()`.

Outputs:

- `OutputBatch`
- updated `TrackingState`

Development requirements:

- Accept `RunConfig` with `source_id`, `model_id`, `frames`, `line` and input `TrackingState`.
- Iterate frames in deterministic order.
- Invoke modules in the exact V2 pipeline order.
- Pass `TrackingState` explicitly into `Track()`.
- Use the `TrackingState` returned by `Track()` as the only state for subsequent processing.
- Pass `TrackingState` and `LineConfig` into `Event()`.
- Pass `EventBatch` into `Demographic()`.
- Pass `EventBatch`, `DemographicsBatch` and `model_id` context into `Assemble()`.
- Return `OutputBatch` and updated `TrackingState`.
- Destroy or release temporary batches as soon as their last consumer has completed.

`Analyse()` must not become a hidden state container. It coordinates ownership transfer and lifetime only.

## 8. Ownership and Lifetime Plan

| Object | Owner at Creation | Last Required Consumer | Lifetime Rule |
| --- | --- | --- | --- |
| `Frame` | Orchestrator | `Detect()` | Owned by orchestration input; not modified by V2 modules. |
| `DetectionBatch` | `Detect()` | `Observe()` | Released after embedding and observation construction no longer need it. |
| `EmbeddingBatch` | `Embed()` | `Observe()` | Released after observation construction unless copied into `BestCrop`. |
| `ObservationBatch` | `Observe()` | `Track()` | Released after tracking update. |
| `TrackingState` | `Analyse()` / `Track()` transition | Returned by `Analyse()` | The only persistent analytics state. |
| `EventBatch` | `Event()` | `Assemble()` | Released after demographic and assembly work. |
| `DemographicsBatch` | `Demographic()` | `Assemble()` | Released after assembly. |
| `OutputBatch` | `Assemble()` | Caller | Returned from `Analyse()`. |

## 9. State Rules

- `TrackingState` is the only persistent analytics state.
- `TrackingState` must be passed in and returned explicitly.
- `Track()` is the only module that modifies `TrackingState.tracks`.
- `Detect()`, `Embed()`, `Observe()`, `Event()`, `Demographic()` and `Assemble()` must be stateless with respect to analytics state.
- Multi-camera deployments must hold separate explicit `TrackingState` objects per orchestration context.
- No V2 module may rely on current-process singleton analytics state.

## 10. Parallel Build Plan

Future implementation tasks should proceed in this order:

1. Create V2 contract datatypes or schemas without modifying current production modules.
2. Add V2 module interfaces matching the locked contracts.
3. Implement `Detect()` V2 behind the V2 interface.
4. Implement `Embed()` V2 behind the V2 interface.
5. Implement `Observe()` V2 as a deterministic contract join.
6. Implement `Track()` V2 with explicit `TrackingState` input and output.
7. Implement `Event()` V2 from `TrackingState` and `LineConfig`.
8. Implement `Demographic()` V2 from `EventBatch`.
9. Implement `Assemble()` V2 to produce `OutputBatch`.
10. Implement `Analyse()` V2 as the orchestrator.
11. Add V2-only tests for each module contract.
12. Add end-to-end V2 tests.
13. Only after V2 passes acceptance tests, perform a separate cutover task.

The current pipeline must remain available during steps 1 through 12.

## 11. Acceptance Rules for Future V2 Work

A future V2 implementation task is acceptable only if:

- it preserves the locked contract shape;
- it does not introduce hidden analytics state;
- it keeps ownership explicit;
- it includes tests for module inputs and outputs;
- it does not alter non-V2 production behaviour unless the task is explicitly a cutover task;
- it does not add architectural stages;
- it documents any algorithm borrowed from existing code as implementation detail, not architecture.

## 12. Explicit Non-Goals for This Document

This document does not:

- implement V2;
- modify production code;
- refactor current modules;
- choose final model implementations;
- add tests;
- define deployment routing;
- perform cutover from the current pipeline to V2;
- preserve any deleted architecture authority.
