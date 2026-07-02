# Analytics Engine Architecture Constitution

## 0. Authority, Scope, and Interpretation

This Constitution is the authoritative architectural specification for the analytics engine. It defines what the analytics engine is, what objects exist within it, how those objects relate, which modules may own them, and which invariants every conforming design must preserve.

This document governs all future analytics engine modules, contracts, interfaces, review decisions, and redesigns. When any future design conflicts with this Constitution, this Constitution is correct.

This document is intentionally independent of any particular computation method, model family, framework, runtime, storage engine, hardware target, or source-code layout. A conforming implementation may change any internal method provided every externally observable contract, ownership rule, lifetime rule, and invariant in this Constitution remains true.

### 0.1 Normative Language

The terms **MUST**, **MUST NOT**, **SHALL**, **SHALL NOT**, **REQUIRED**, and **PROHIBITED** are binding architectural requirements. The terms **MAY** and **OPTIONAL** identify permitted choices that remain valid only when all binding requirements are satisfied.

### 0.2 Architectural Boundary

The analytics engine begins at the point where a validated `Frame` object is accepted for analysis and ends when deterministic `AnalyticsOutput` objects have been emitted to downstream consumers. External frame acquisition, external persistence, user interface rendering, deployment packaging, and operator workflow are outside the engine boundary unless explicitly represented as input or output contracts in this Constitution.

### 0.3 Implementation Exclusion

This Constitution specifies contracts, responsibilities, dependencies, ownership, lifetimes, state, data flow, and invariants. It does not specify how any score, label, association, movement, event, crop, or demographic attribute is computed.

## 1. Architectural Philosophy

### 1.1 Deterministic Execution

For identical input contracts, configuration contracts, initial state contracts, and schema versions, the analytics engine MUST produce identical output contracts. Determinism applies to object identity assignment, ordering, timestamps, event emission, state transitions, and diagnostic metadata.

### 1.2 Explicit Contracts

Every module boundary MUST be expressed through named contracts. A module MUST NOT rely on undocumented fields, inferred side channels, global mutable objects, ambient runtime state, or caller-specific conventions.

### 1.3 Stateless Pipeline, Explicit State

Pipeline modules MUST be stateless with respect to analytics state. Analytics state MUST exist only in explicit state objects that are passed into a state transition boundary and returned as updated state objects. Hidden module-owned analytics state is prohibited.

### 1.4 Single Ownership

Every object MUST have exactly one owner at every moment in its lifetime. Shared reading is permitted only through immutable borrowing rules. Shared writing is prohibited.

### 1.5 Immutable Interfaces

Input contracts received by a module MUST be treated as immutable. A module that needs to create modified data MUST produce a new output contract or an explicitly updated state object for which it is the owner.

### 1.6 Bounded Memory

The engine MUST define bounded lifetimes for all transient and persistent objects. No object may accumulate without a defined owner, purpose, retention scope, and destruction condition.

### 1.7 Predictable Execution

Each module MUST have a finite, declared responsibility. A module MUST NOT initiate unrelated work, alter unrelated state, or produce undocumented side effects.

### 1.8 Separation of Responsibility

Detection, crop selection, embedding, observation construction, tracking, event derivation, demographic attribution, aggregation, diagnostics, and output assembly are separate architectural responsibilities. A module MAY combine responsibilities only behind an adapter that preserves the constitutional module contracts externally.

### 1.9 Reproducibility

Every output MUST be traceable to input frame identity, timestamps, configuration identity, schema version, and upstream object identities sufficient to reproduce the output under the same constitutional contracts.

### 1.10 Scalability by Contract

The engine MUST scale by preserving clear ownership, bounded lifetimes, index alignment, explicit state transfer, and module independence. Scalability MUST NOT depend on hidden coupling between modules.

### 1.11 Data Consistency

All object references MUST resolve to objects within the same processing window, state snapshot, or explicitly named historical scope. Cross-window references MUST be represented by stable identifiers, not by borrowed transient objects.

## 2. Complete End-to-End Pipeline

The constitutional analytics pipeline consists of the following ordered stages:

1. `Frame Intake Boundary`
2. `Frame Validation Module`
3. `Detection Module`
4. `Crop Selection Module`
5. `Embedding Module`
6. `Observation Module`
7. `Tracking State Transition Module`
8. `Event Derivation Module`
9. `Demographics Module`
10. `Analytics Assembly Module`
11. `Output Boundary`

A conforming system MAY omit an optional module only by emitting a contract that explicitly marks the corresponding result as unavailable. Omission MUST NOT alter required contracts for other modules.

### 2.1 Frame Intake Boundary

The intake boundary accepts `Frame` objects from outside the engine. Ownership of the accepted `Frame` transfers to the engine for the frame lifetime. The boundary MUST preserve frame identity, timestamp, image dimensions, image format metadata, and source metadata.

### 2.2 Frame Validation Module

The validation module verifies that the `Frame` contract is complete and internally consistent. It produces a `ValidatedFrame`. It MUST NOT infer missing identity, modify timestamp meaning, alter image content, or create analytics outputs.

### 2.3 Detection Module

The detection module consumes a `ValidatedFrame` and emits a `DetectionBatch`. Each detection represents one localized candidate subject in the frame. The module owns the `DetectionBatch` until it is transferred to the next boundary. It MUST NOT own tracking state, event state, demographic state, or final output state.

### 2.4 Crop Selection Module

The crop selection module consumes a `ValidatedFrame` and `DetectionBatch` and emits a `CropBatch`. Each crop is aligned by index and detection identity to exactly one detection. The module MUST NOT change detection identity or detection geometry.

### 2.5 Embedding Module

The embedding module consumes a `CropBatch` and emits an `EmbeddingBatch`. Each embedding is aligned by index and detection identity to exactly one crop and one detection. Embeddings are descriptors of detected subjects and MUST NOT contain tracking identity.

### 2.6 Observation Module

The observation module consumes `DetectionBatch`, `EmbeddingBatch`, selected frame metadata, and optional crop metadata. It emits an `ObservationBatch`. Each observation is the canonical per-frame subject measurement used by stateful downstream modules.

### 2.7 Tracking State Transition Module

The tracking transition module consumes the previous `TrackingState`, the current `ObservationBatch`, and the applicable `TrackingConfig`. It emits a new `TrackingState`, a `TrackSnapshotBatch`, and an `ObservationAssignmentBatch`. It MUST NOT mutate the previous state object after accepting it. The emitted state is the sole tracking state for the next transition.

### 2.8 Event Derivation Module

The event derivation module consumes `TrackSnapshotBatch`, `EventConfig`, and optional prior `EventState`. It emits `EventBatch` and an updated `EventState` when event de-duplication or historical event context is required. It MUST NOT modify tracking state.

### 2.9 Demographics Module

The demographics module consumes subject imagery represented by `CropBatch` or crop references plus subject identity context represented by assignments or tracks. It emits `DemographicsBatch`. Demographic outputs are descriptive attributes with explicit confidence and taxonomy metadata. The module MUST NOT affect tracking identity, event identity, or observation validity.

### 2.10 Analytics Assembly Module

The assembly module consumes all selected outputs from prior modules and emits `AnalyticsOutput`. It joins objects only through declared identifiers and index-alignment guarantees. It MUST NOT compute new detection, embedding, tracking, event, or demographic facts except for packaging metadata and consistency diagnostics.

### 2.11 Output Boundary

The output boundary transfers immutable `AnalyticsOutput` objects to downstream consumers. After transfer, the engine MUST NOT mutate the emitted output object.

## 3. Module Definitions

### 3.1 Frame Validation Module

**Purpose:** Establish a canonical valid frame contract.

**Responsibilities:** verify required frame fields; preserve identity; preserve timestamp; preserve image metadata; emit validation diagnostics.

**Allowed responsibilities:** structural validation, schema validation, metadata normalization to constitutional types.

**Forbidden responsibilities:** subject detection, crop creation, descriptor creation, state transition, event creation, demographic attribution, output assembly.

**Inputs:** `Frame`, `FrameValidationConfig`.

**Outputs:** `ValidatedFrame`, `ValidationDiagnosticBatch`.

**Upstream producers:** frame intake boundary.

**Downstream consumers:** detection module, crop selection module, diagnostics consumers.

**Dependencies:** global data type standards and frame schema only.

### 3.2 Detection Module

**Purpose:** Produce frame-local subject candidates.

**Responsibilities:** emit one `Detection` per candidate subject; assign unique detection identities within the engine identity namespace; preserve frame reference; provide geometry and detection confidence.

**Allowed responsibilities:** candidate localization, candidate confidence reporting, candidate class taxonomy reporting.

**Forbidden responsibilities:** track creation, track continuation, event emission, demographic classification, final output packaging, mutation of frames.

**Inputs:** `ValidatedFrame`, `DetectionConfig`.

**Outputs:** `DetectionBatch`, `DetectionDiagnosticBatch`.

**Upstream producers:** frame validation module.

**Downstream consumers:** crop selection module, observation module, analytics assembly module.

**Dependencies:** validated frame contract and detection configuration contract.

### 3.3 Crop Selection Module

**Purpose:** Create subject image regions required by downstream per-subject modules.

**Responsibilities:** emit one `Crop` or explicit unavailable marker for each detection requiring imagery; preserve detection identity; preserve index alignment.

**Allowed responsibilities:** region contract creation, crop metadata production, unavailable reason reporting.

**Forbidden responsibilities:** modifying detection geometry, assigning track identity, deriving events, deriving demographics, owning long-lived subject state.

**Inputs:** `ValidatedFrame`, `DetectionBatch`, `CropSelectionConfig`.

**Outputs:** `CropBatch`, `CropDiagnosticBatch`.

**Upstream producers:** frame validation module and detection module.

**Downstream consumers:** embedding module, demographics module, analytics assembly module.

**Dependencies:** frame contract, detection contract, crop selection configuration contract.

### 3.4 Embedding Module

**Purpose:** Produce per-detection descriptor vectors.

**Responsibilities:** emit one `Embedding` or explicit unavailable marker per required crop; preserve detection identity; declare descriptor schema identity.

**Allowed responsibilities:** descriptor production, descriptor validity reporting, descriptor metadata reporting.

**Forbidden responsibilities:** assigning tracks, altering observations, emitting events, deriving demographic labels, retaining analytics state between frames.

**Inputs:** `CropBatch`, `EmbeddingConfig`.

**Outputs:** `EmbeddingBatch`, `EmbeddingDiagnosticBatch`.

**Upstream producers:** crop selection module.

**Downstream consumers:** observation module, tracking state transition module through observations, analytics assembly module.

**Dependencies:** crop contract and embedding configuration contract.

### 3.5 Observation Module

**Purpose:** Build canonical frame-local subject observations.

**Responsibilities:** combine detection facts and descriptor facts into `Observation` objects; preserve detection identity; define canonical subject point, geometry, timestamp, and optional descriptor reference.

**Allowed responsibilities:** contract joining, field canonicalization, alignment verification, observation validity reporting.

**Forbidden responsibilities:** persistent state ownership, track identity assignment, event emission, demographic attribution, image mutation.

**Inputs:** `DetectionBatch`, `EmbeddingBatch`, `ObservationConfig`, selected `ValidatedFrame` metadata.

**Outputs:** `ObservationBatch`, `ObservationDiagnosticBatch`.

**Upstream producers:** detection module and embedding module.

**Downstream consumers:** tracking state transition module, analytics assembly module.

**Dependencies:** detection, embedding, and frame metadata contracts.

### 3.6 Tracking State Transition Module

**Purpose:** Convert prior tracking state and current observations into a new tracking state and externally visible track snapshots.

**Responsibilities:** own the state transition boundary; create, continue, close, or report tracks through explicit state contracts; emit observation-to-track assignments; preserve stable runtime track identifiers.

**Allowed responsibilities:** reading previous tracking state, producing next tracking state, producing track snapshots, producing assignment metadata, producing transition diagnostics.

**Forbidden responsibilities:** detecting subjects, producing crops, producing descriptor vectors, deriving demographic attributes, mutating previous state after transition, consuming event output as an input to tracking.

**Inputs:** `TrackingState`, `ObservationBatch`, `TrackingConfig`, `ProcessingClock`.

**Outputs:** next `TrackingState`, `TrackSnapshotBatch`, `ObservationAssignmentBatch`, `TrackingDiagnosticBatch`.

**Upstream producers:** previous tracking transition and observation module.

**Downstream consumers:** event derivation module, demographics module, analytics assembly module, next tracking transition.

**Dependencies:** observation contract, tracking state contract, tracking configuration contract, processing clock contract.

### 3.7 Event Derivation Module

**Purpose:** Derive semantic movement events from track snapshots and event configuration.

**Responsibilities:** emit event candidates or final events with stable event identities; preserve track identity references; maintain explicit event state only when required by the event contract.

**Allowed responsibilities:** event contract creation, event identity creation, event state transition, event diagnostics.

**Forbidden responsibilities:** modifying tracks, assigning tracks, detecting subjects, producing descriptors, producing demographic labels.

**Inputs:** `TrackSnapshotBatch`, `EventConfig`, optional `EventState`.

**Outputs:** `EventBatch`, optional next `EventState`, `EventDiagnosticBatch`.

**Upstream producers:** tracking state transition module and configuration boundary.

**Downstream consumers:** analytics assembly module and external event consumers.

**Dependencies:** track snapshot contract and event configuration contract.

### 3.8 Demographics Module

**Purpose:** Produce descriptive subject attribute observations.

**Responsibilities:** emit demographic attributes tied to detection identity and, when available, track identity; declare taxonomy version; declare confidence and availability.

**Allowed responsibilities:** demographic attribute production, unavailable reason reporting, taxonomy metadata reporting.

**Forbidden responsibilities:** track state mutation, event derivation, detection geometry mutation, output contract mutation after emission.

**Inputs:** `CropBatch`, `ObservationAssignmentBatch`, optional `TrackSnapshotBatch`, `DemographicsConfig`.

**Outputs:** `DemographicsBatch`, `DemographicsDiagnosticBatch`.

**Upstream producers:** crop selection module and tracking state transition module.

**Downstream consumers:** analytics assembly module and external attribute consumers.

**Dependencies:** crop, assignment, track snapshot, and demographics configuration contracts.

### 3.9 Analytics Assembly Module

**Purpose:** Produce the canonical output package for a processed frame or processing window.

**Responsibilities:** join module outputs by contract identifiers; preserve module-native facts; include schema metadata; include diagnostics; expose deterministic ordering.

**Allowed responsibilities:** packaging, consistency verification, schema version stamping, output-level metadata creation.

**Forbidden responsibilities:** changing module facts, filling missing values without explicit unavailable markers, changing identities, mutating state.

**Inputs:** selected batches from all prior modules, engine configuration metadata, schema registry metadata.

**Outputs:** `AnalyticsOutput`.

**Upstream producers:** all analytics modules.

**Downstream consumers:** output boundary and external consumers.

**Dependencies:** all emitted contract schemas and global standards.

## 4. Module Interface Rules

### 4.1 Universal Interface Requirements

Every module interface MUST define:

- input contract names and schema versions;
- output contract names and schema versions;
- owner of each input before call, during call, and after return;
- owner of each output at creation and after transfer;
- mutability of every object;
- lifetime scope of every object;
- ordering guarantees;
- index-alignment guarantees;
- identity preservation rules;
- error and diagnostic contract behavior.

### 4.2 Input Ownership

A module borrows input contracts immutably unless the interface explicitly transfers ownership of a state object. Borrowed inputs MUST remain valid for the duration of the module call. A borrowed input MUST NOT be retained beyond the call unless retained only as a stable identifier or immutable value copy permitted by the contract.

### 4.3 Output Ownership

A module owns every output it creates until the output is transferred to the orchestrating engine or downstream consumer. After transfer, the producing module MUST NOT mutate or destroy the output.

### 4.4 Ordering and Alignment

Batches MUST expose deterministic ordering. If batch `B` is declared index-aligned to batch `A`, then `B.items[i]` MUST correspond to `A.items[i]` for every valid index `i`, unless `B.items[i]` is an explicit unavailable marker for `A.items[i]`. Removing elements from an aligned batch is prohibited.

### 4.5 Error and Diagnostic Behavior

Invalid input contracts MUST produce deterministic diagnostics and MUST NOT produce partially valid outputs unless the output contract explicitly supports per-item unavailable markers. Diagnostics MUST reference the exact object identity, field name, and invariant violated.

## 5. Contract Definitions

### 5.1 `Frame`

A `Frame` is the raw engine input for one image-bearing observation from a source stream.

| Field | Type | Units / Format | Owner | Producer | Consumers | Lifetime | Mutability | Valid Range / Invariants | Purpose |
|---|---|---|---|---|---|---|---|---|---|
| `frame_id` | `EngineId` | string | intake boundary, then engine | external source or intake boundary | all frame consumers | frame lifetime | immutable | unique within source stream and processing session | identifies the frame |
| `source_id` | `EngineId` | string | intake boundary | external source | diagnostics, output assembly | engine lifetime value | immutable | non-empty | identifies source stream |
| `timestamp` | `Timestamp` | seconds | intake boundary | external source | all temporal consumers | frame lifetime | immutable | finite; monotonic per source unless marked otherwise | establishes temporal position |
| `frame_index` | `UInt64` | count | intake boundary | external source or intake boundary | ordering consumers | frame lifetime | immutable | monotonic per source | establishes deterministic order |
| `image` | `ImageBuffer` | declared image format | engine | external source | validation, detection, crop selection | frame lifetime | immutable | dimensions match metadata | carries visual data |
| `width` | `UInt32` | pixels | intake boundary | external source or validation | geometry consumers | frame lifetime | immutable | greater than zero | defines horizontal extent |
| `height` | `UInt32` | pixels | intake boundary | external source or validation | geometry consumers | frame lifetime | immutable | greater than zero | defines vertical extent |
| `image_format` | `ImageFormat` | enum | intake boundary | external source or validation | image consumers | frame lifetime | immutable | declared in global standards | defines interpretation of image buffer |
| `metadata` | `MetadataMap` | key-value | intake boundary | external source | diagnostics, output assembly | frame lifetime | immutable | keys are strings; values are schema-safe | preserves source context |

### 5.2 `ValidatedFrame`

`ValidatedFrame` is a `Frame` that satisfies all structural and type invariants. It preserves every `Frame` field and adds validation metadata.

| Field | Type | Owner | Producer | Consumers | Lifetime | Mutability | Invariants | Purpose |
|---|---|---|---|---|---|---|---|---|
| all `Frame` fields | as defined | validation module, then engine | validation module | downstream frame consumers | frame lifetime | immutable | identical semantic values to input frame | preserves source facts |
| `schema_version` | `SchemaVersion` | validation module | validation module | all downstream modules | frame lifetime | immutable | supported by engine | identifies frame schema |
| `validation_status` | `ValidationStatus` | validation module | validation module | diagnostics, output assembly | frame lifetime | immutable | `VALID` for normal downstream use | records validation result |

### 5.3 `Detection`

A `Detection` is a frame-local candidate subject.

| Field | Type | Units / Format | Owner | Producer | Consumers | Lifetime | Mutability | Valid Range / Invariants | Purpose |
|---|---|---|---|---|---|---|---|---|---|
| `detection_id` | `EngineId` | string | detection module, then engine | detection module | crop, observation, assembly | frame lifetime, identifier may persist by reference | immutable | unique within engine session | identifies candidate |
| `frame_id` | `EngineId` | string | detection module | detection module | all detection consumers | frame lifetime | immutable | references existing frame | binds detection to frame |
| `source_id` | `EngineId` | string | detection module | detection module | output consumers | frame lifetime | immutable | equals frame source | binds source stream |
| `timestamp` | `Timestamp` | seconds | detection module | detection module | observation, tracking | frame lifetime | immutable | equals frame timestamp | temporal alignment |
| `frame_index` | `UInt64` | count | detection module | detection module | ordering consumers | frame lifetime | immutable | equals frame index | deterministic ordering |
| `bbox` | `BoundingBoxXYXY` | pixels | detection module | detection module | crop, observation, output | frame lifetime | immutable | finite; `x_min < x_max`; `y_min < y_max`; coordinate standard applies | locates subject extent |
| `confidence` | `Float32` | unit interval | detection module | detection module | observation, output | frame lifetime | immutable | `[0.0, 1.0]` or unavailable marker | expresses detection certainty |
| `class_label` | `EnumValue` | taxonomy | detection module | detection module | output | frame lifetime | immutable | taxonomy declared | identifies subject class |
| `class_taxonomy_version` | `SchemaVersion` | version | detection module | detection module | output | frame lifetime | immutable | non-empty when class label exists | identifies class meaning |
| `metadata` | `MetadataMap` | key-value | detection module | detection module | diagnostics, output | frame lifetime | immutable | schema-safe | preserves module facts |

### 5.4 `DetectionBatch`

| Field | Type | Owner | Producer | Consumers | Lifetime | Mutability | Invariants | Purpose |
|---|---|---|---|---|---|---|---|---|
| `frame_id` | `EngineId` | detection module | detection module | all consumers | frame lifetime | immutable | all detections reference this frame | batch binding |
| `items` | array of `Detection` | detection module, then engine | detection module | crop, observation, output | frame lifetime | immutable | deterministic order; no duplicate detection IDs | detection collection |
| `schema_version` | `SchemaVersion` | detection module | detection module | all consumers | frame lifetime | immutable | supported version | schema identity |
| `producer_metadata` | `MetadataMap` | detection module | detection module | diagnostics | frame lifetime | immutable | schema-safe | traceability |

### 5.5 `Crop`

A `Crop` is a subject image region derived for a detection.

| Field | Type | Units / Format | Owner | Producer | Consumers | Lifetime | Mutability | Valid Range / Invariants | Purpose |
|---|---|---|---|---|---|---|---|---|---|
| `crop_id` | `EngineId` | string | crop selection module | crop selection module | embedding, demographics | frame lifetime unless referenced by output policy | immutable | unique within engine session | identifies crop |
| `detection_id` | `EngineId` | string | crop selection module | crop selection module | all crop consumers | frame lifetime | immutable | references exactly one detection | preserves alignment |
| `frame_id` | `EngineId` | string | crop selection module | crop selection module | output | frame lifetime | immutable | references source frame | provenance |
| `image` | `ImageBuffer` or `Unavailable` | declared image format | crop selection module | crop selection module | embedding, demographics | crop lifetime | immutable | present or explicit unavailable reason | carries subject imagery |
| `bbox_source` | `BoundingBoxXYXY` | pixels | crop selection module | crop selection module | diagnostics | frame lifetime | immutable | equals or references detection geometry per contract | provenance |
| `image_format` | `ImageFormat` | enum | crop selection module | crop selection module | image consumers | crop lifetime | immutable | declared in global standards | image interpretation |
| `width` | `UInt32` | pixels | crop selection module | crop selection module | image consumers | crop lifetime | immutable | greater than zero when image exists | image extent |
| `height` | `UInt32` | pixels | crop selection module | crop selection module | image consumers | crop lifetime | immutable | greater than zero when image exists | image extent |
| `availability` | `Availability` | enum | crop selection module | crop selection module | all crop consumers | frame lifetime | immutable | `AVAILABLE` or reasoned unavailable value | explicit missing data |

### 5.6 `CropBatch`

`CropBatch.items` MUST be index-aligned to `DetectionBatch.items`. `CropBatch.items[i].detection_id` MUST equal `DetectionBatch.items[i].detection_id` when a crop record is present.

### 5.7 `Embedding`

An `Embedding` is a per-detection descriptor vector.

| Field | Type | Units / Format | Owner | Producer | Consumers | Lifetime | Mutability | Valid Range / Invariants | Purpose |
|---|---|---|---|---|---|---|---|---|---|
| `embedding_id` | `EngineId` | string | embedding module | embedding module | observation, output | frame lifetime | immutable | unique within engine session | identifies descriptor |
| `detection_id` | `EngineId` | string | embedding module | embedding module | observation, tracking | frame lifetime | immutable | references exactly one detection | preserves subject linkage |
| `crop_id` | `EngineId` or `Unavailable` | string | embedding module | embedding module | diagnostics | frame lifetime | immutable | references crop when available | provenance |
| `vector` | `NumericArray<Float32>` or `Unavailable` | descriptor units | embedding module | embedding module | observation, tracking | frame lifetime, may be copied into state by explicit rule | immutable | dimension equals descriptor schema; finite values when available | subject descriptor |
| `descriptor_schema_id` | `SchemaId` | string | embedding module | embedding module | tracking, output | frame lifetime | immutable | non-empty when vector exists | defines vector meaning |
| `availability` | `Availability` | enum | embedding module | embedding module | observation, output | frame lifetime | immutable | explicit value | missing-data semantics |

### 5.8 `EmbeddingBatch`

`EmbeddingBatch.items` MUST be index-aligned to `CropBatch.items` and transitively to `DetectionBatch.items` whenever embeddings are requested for all detections. If embeddings are requested for a subset, the batch MUST include an explicit alignment map from detection identity to embedding identity.

### 5.9 `Observation`

An `Observation` is the canonical frame-local measurement of a subject.

| Field | Type | Units / Format | Owner | Producer | Consumers | Lifetime | Mutability | Valid Range / Invariants | Purpose |
|---|---|---|---|---|---|---|---|---|---|
| `observation_id` | `EngineId` | string | observation module | observation module | tracking, output | frame lifetime, identifier may persist by reference | immutable | unique within engine session | identifies observation |
| `detection_id` | `EngineId` | string | observation module | detection through observation module | tracking, output | frame lifetime | immutable | references exactly one detection | provenance |
| `frame_id` | `EngineId` | string | observation module | observation module | tracking, output | frame lifetime | immutable | references frame | temporal provenance |
| `source_id` | `EngineId` | string | observation module | observation module | tracking, output | frame lifetime | immutable | references source | stream provenance |
| `timestamp` | `Timestamp` | seconds | observation module | observation module | tracking, events | frame lifetime | immutable | equals source frame timestamp | temporal position |
| `frame_index` | `UInt64` | count | observation module | observation module | tracking, ordering | frame lifetime | immutable | equals source frame index | deterministic order |
| `subject_point` | `Point2D` | pixels | observation module | observation module | tracking, events | frame lifetime | immutable | finite; inside or explicitly related to bbox | canonical point |
| `bbox` | `BoundingBoxXYXY` | pixels | observation module | detection module through observation module | tracking, output | frame lifetime | immutable | same semantics as detection bbox | subject extent |
| `confidence` | `Float32` | unit interval | observation module | detection module through observation module | tracking, output | frame lifetime | immutable | `[0.0, 1.0]` or unavailable marker | observation certainty |
| `embedding_ref` | `EngineId` or `Unavailable` | string | observation module | observation module | tracking, output | frame lifetime | immutable | references embedding when present | descriptor linkage |
| `embedding_vector` | `NumericArray<Float32>` or omitted by reference policy | descriptor units | observation module | embedding module through observation module | tracking | frame lifetime unless copied into tracking state | immutable | schema-compatible when present | optional descriptor payload |
| `validity` | `ValidityStatus` | enum | observation module | observation module | tracking, output | frame lifetime | immutable | explicit valid or invalid reason | processing eligibility |

### 5.10 `ObservationBatch`

`ObservationBatch.items` MUST be deterministically ordered by frame order and detection order unless a stricter configuration-defined ordering is declared. Observations MUST NOT exist without a detection reference.

### 5.11 `TrackingState`

`TrackingState` is the complete state required to perform the next tracking transition.

| Field | Type | Owner | Producer | Consumers | Lifetime | Mutability | Invariants | Purpose |
|---|---|---|---|---|---|---|---|---|
| `state_id` | `EngineId` | tracking transition module | tracking transition module | next transition, diagnostics | transition lifetime until superseded | immutable after emission | unique | identifies state snapshot |
| `source_id` | `EngineId` | tracking transition module | tracking transition module | next transition | engine lifetime value | immutable | one state per source unless multi-source contract declares otherwise | stream binding |
| `tracks` | array of `TrackStateRecord` | tracking transition module | tracking transition module | next transition | engine or window lifetime | immutable after emission | unique track IDs | active historical state |
| `transition_index` | `UInt64` | tracking transition module | tracking transition module | next transition, output | engine lifetime | immutable | monotonic | state order |
| `last_timestamp` | `Timestamp` or `Unavailable` | tracking transition module | tracking transition module | next transition | state lifetime | immutable | monotonic per source | temporal state anchor |
| `schema_version` | `SchemaVersion` | tracking transition module | tracking transition module | next transition | state lifetime | immutable | supported version | state schema identity |

### 5.12 `TrackStateRecord`

| Field | Type | Units / Format | Owner | Producer | Consumers | Lifetime | Mutability | Valid Range / Invariants | Purpose |
|---|---|---|---|---|---|---|---|---|---|
| `runtime_track_id` | `EngineId` | string | tracking transition module | tracking transition module | events, demographics, output | track lifetime and external reference lifetime | immutable | unique within source and session | identifies tracked subject hypothesis |
| `track_status` | `TrackStatus` | enum | tracking transition module | tracking transition module | events, output | track lifetime | immutable within state snapshot | declared value | lifecycle state |
| `first_seen_timestamp` | `Timestamp` | seconds | tracking transition module | tracking transition module | output | track lifetime | immutable after initial assignment | finite | start time |
| `last_seen_timestamp` | `Timestamp` | seconds | tracking transition module | tracking transition module | events, output | state snapshot lifetime | immutable | not earlier than first seen | latest observation time |
| `first_frame_index` | `UInt64` | count | tracking transition module | tracking transition module | output | track lifetime | immutable | finite | start order |
| `last_frame_index` | `UInt64` | count | tracking transition module | tracking transition module | output | state snapshot lifetime | immutable | not earlier than first frame | latest order |
| `current_subject_point` | `Point2D` | pixels | tracking transition module | tracking transition module | events, output | state snapshot lifetime | immutable | finite when available | current canonical position |
| `current_bbox` | `BoundingBoxXYXY` | pixels | tracking transition module | tracking transition module | output | state snapshot lifetime | immutable | valid bbox when available | current extent |
| `observation_history_refs` | array of `EngineId` | identifiers | tracking transition module | tracking transition module | diagnostics, output | bounded track lifetime | immutable within state snapshot | ordered; references observations retained by policy | provenance |
| `subject_point_history` | array of `Point2D` | pixels | tracking transition module | tracking transition module | events | bounded track lifetime | immutable within state snapshot | ordered by time | event support |
| `descriptor_ref` | `EngineId` or `Unavailable` | string | tracking transition module | tracking transition module | next transition | state snapshot lifetime | immutable | references compatible descriptor when present | descriptor continuity |
| `closed_timestamp` | `Timestamp` or `Unavailable` | seconds | tracking transition module | tracking transition module | output | track lifetime | immutable once set | not earlier than first seen | closure time |

### 5.13 `TrackSnapshot`

`TrackSnapshot` is the externally visible representation of a track at a processing boundary. It contains a read-only projection of `TrackStateRecord` fields approved for downstream consumers. It MUST NOT expose mutable internal state containers.

### 5.14 `ObservationAssignment`

| Field | Type | Owner | Producer | Consumers | Lifetime | Mutability | Invariants | Purpose |
|---|---|---|---|---|---|---|---|---|
| `assignment_id` | `EngineId` | tracking transition module | tracking transition module | output | frame lifetime | immutable | unique | identifies assignment |
| `observation_id` | `EngineId` | tracking transition module | tracking transition module | demographics, output | frame lifetime | immutable | references observation | observation link |
| `detection_id` | `EngineId` | tracking transition module | tracking transition module | demographics, output | frame lifetime | immutable | matches observation detection | detection link |
| `runtime_track_id` | `EngineId` or `Unavailable` | tracking transition module | tracking transition module | demographics, output | frame lifetime | immutable | references track when assigned | track link |
| `assignment_status` | `AssignmentStatus` | enum | tracking transition module | tracking transition module | output | frame lifetime | immutable | explicit assigned or unassigned reason | interpretation |

### 5.15 `Event`

| Field | Type | Units / Format | Owner | Producer | Consumers | Lifetime | Mutability | Valid Range / Invariants | Purpose |
|---|---|---|---|---|---|---|---|---|---|
| `event_id` | `EngineId` | string | event derivation module | event derivation module | output, external consumers | event lifetime and external reference lifetime | immutable | stable for same event facts | identifies event |
| `runtime_track_id` | `EngineId` | string | event derivation module | event derivation module | output | event lifetime | immutable | references existing track | subject linkage |
| `event_type` | `EventType` | enum | event derivation module | event derivation module | output | event lifetime | immutable | declared taxonomy | event category |
| `direction` | `Direction` or `Unavailable` | enum | event derivation module | event derivation module | output | event lifetime | immutable | declared taxonomy | directional meaning |
| `timestamp` | `Timestamp` | seconds | event derivation module | event derivation module | output | event lifetime | immutable | within supporting track time span | event time |
| `supporting_positions` | array of `Point2D` | pixels | event derivation module | event derivation module | diagnostics, output | event lifetime | immutable | ordered; bounded | event evidence |
| `event_config_id` | `EngineId` | string | event derivation module | event derivation module | output | event lifetime | immutable | references event config | interpretation context |
| `confidence` | `Float32` or `Unavailable` | unit interval | event derivation module | event derivation module | output | event lifetime | immutable | `[0.0, 1.0]` when present | event certainty |

### 5.16 `DemographicsResult`

| Field | Type | Units / Format | Owner | Producer | Consumers | Lifetime | Mutability | Valid Range / Invariants | Purpose |
|---|---|---|---|---|---|---|---|---|---|
| `demographics_id` | `EngineId` | string | demographics module | demographics module | output | frame or track attribution lifetime | immutable | unique | identifies result |
| `detection_id` | `EngineId` | string | demographics module | demographics module | output | frame lifetime | immutable | references detection | detection provenance |
| `runtime_track_id` | `EngineId` or `Unavailable` | string | demographics module | demographics module | output | result lifetime | immutable | references assigned track when present | track attribution |
| `attributes` | array of `DemographicAttribute` | taxonomy values | demographics module | demographics module | output | result lifetime | immutable | taxonomy declared | descriptive facts |
| `taxonomy_version` | `SchemaVersion` | version | demographics module | demographics module | output | result lifetime | immutable | non-empty when attributes present | meaning of attributes |
| `availability` | `Availability` | enum | demographics module | demographics module | output | result lifetime | immutable | explicit value | missing-data semantics |

### 5.17 `AnalyticsOutput`

`AnalyticsOutput` is the immutable output package emitted by the engine for a frame or processing window.

| Field | Type | Owner | Producer | Consumers | Lifetime | Mutability | Invariants | Purpose |
|---|---|---|---|---|---|---|---|---|
| `output_id` | `EngineId` | assembly module, then output boundary | assembly module | external consumers | external retention lifetime | immutable | unique | identifies output package |
| `source_id` | `EngineId` | assembly module | assembly module | external consumers | external retention lifetime | immutable | references source | stream context |
| `frame_id` | `EngineId` or `WindowId` | assembly module | assembly module | external consumers | external retention lifetime | immutable | references processed scope | output scope |
| `timestamp` | `Timestamp` | assembly module | assembly module | external consumers | external retention lifetime | immutable | deterministic representative timestamp | temporal context |
| `detections` | `DetectionBatch` projection | assembly module | detection module through assembly | external consumers | external retention lifetime | immutable | preserves detection facts | detection output |
| `observations` | `ObservationBatch` projection | assembly module | observation module through assembly | external consumers | external retention lifetime | immutable | preserves observation facts | observation output |
| `tracks` | `TrackSnapshotBatch` projection | assembly module | tracking module through assembly | external consumers | external retention lifetime | immutable | preserves track facts | track output |
| `assignments` | `ObservationAssignmentBatch` projection | assembly module | tracking module through assembly | external consumers | external retention lifetime | immutable | references valid observations | association output |
| `events` | `EventBatch` projection | assembly module | event module through assembly | external consumers | external retention lifetime | immutable | stable ordering | event output |
| `demographics` | `DemographicsBatch` projection | assembly module | demographics module through assembly | external consumers | external retention lifetime | immutable | references detections/tracks | attribute output |
| `diagnostics` | `DiagnosticBatch` | assembly module | all modules through assembly | external consumers | external retention lifetime | immutable | deterministic ordering | auditability |
| `schema_versions` | map | assembly module | schema registry | external consumers | external retention lifetime | immutable | all emitted schemas declared | compatibility |

## 6. Data Model Relationships

### 6.1 Identity Graph

`Frame` is the root object for frame-scoped data. `Detection`, `Crop`, `Embedding`, and `Observation` MUST reference exactly one frame directly or transitively. `TrackStateRecord` MAY reference observations across frames through bounded history references. `Event` MUST reference at least one track. `DemographicsResult` MUST reference a detection and MAY reference a track when assignment exists. `AnalyticsOutput` MUST reference the processed frame or window and contain only objects from that scope or stable projections of explicitly referenced state.

### 6.2 Batch Relationships

- `DetectionBatch` belongs to one `ValidatedFrame`.
- `CropBatch` belongs to one `DetectionBatch` and MUST preserve detection alignment.
- `EmbeddingBatch` belongs to one `CropBatch` or declared detection subset.
- `ObservationBatch` belongs to one `DetectionBatch` and MAY reference one `EmbeddingBatch`.
- `TrackSnapshotBatch` belongs to one tracking transition.
- `ObservationAssignmentBatch` belongs to one `ObservationBatch` and one tracking transition.
- `EventBatch` belongs to one event derivation boundary.
- `DemographicsBatch` belongs to one demographic attribution boundary.

### 6.3 Index Rules

Indexes are zero-based unsigned integer positions within a batch. Indexes are not identities and MUST NOT be persisted as cross-boundary references unless accompanied by the batch identity and schema version. Stable identifiers MUST be used for cross-module and external references.

## 7. Data Type Standards

### 7.1 Integers

Unsigned counts, indexes, and dimensions MUST use `UInt64` for unbounded process counters and `UInt32` for image dimensions. Signed integers MUST be used only where negative values are meaningful and declared.

### 7.2 Floating Point Values

Coordinates, timestamps, confidences, and descriptor values MUST be finite floating point values unless represented by an explicit unavailable marker. `NaN`, positive infinity, and negative infinity are prohibited in emitted contracts.

### 7.3 Booleans

Booleans MUST represent strictly binary facts. A boolean MUST NOT encode unavailable, unknown, invalid, or not-applicable states. Such states MUST use enums.

### 7.4 Enums

Enums MUST have named values, schema versions, and declared unknown-value handling. External numeric enum encodings MUST NOT be interpreted without taxonomy metadata.

### 7.5 Timestamps

Timestamps MUST be numeric seconds on a declared time base. Per-source timestamps MUST be monotonic unless the source contract explicitly marks non-monotonic input and the output diagnostics record the violation.

### 7.6 Identifiers

Identifiers MUST be strings in the engine identity namespace. Identifier uniqueness scope MUST be declared for every identifier field. Identifiers are immutable after emission.

### 7.7 Coordinates

Image coordinates use a two-dimensional pixel coordinate system with origin at the top-left of the image, positive `x` to the right, and positive `y` downward. Coordinates are floating point pixel positions unless declared otherwise.

### 7.8 Bounding Boxes

`BoundingBoxXYXY` contains `[x_min, y_min, x_max, y_max]` in image pixel coordinates. `x_min < x_max` and `y_min < y_max` are required for valid boxes. Bounds relative to image extent MUST be declared by the producing contract.

### 7.9 Arrays

Arrays MUST declare element type, ordering, length constraints, and alignment rules. Empty arrays are valid only where explicitly allowed.

### 7.10 Tensor-Like Data

Tensor-like data is an architectural array contract with declared element type, rank, shape, axis meaning, value domain, and schema identity. No contract may require a particular in-memory tensor library.

### 7.11 Colour Spaces and Image Formats

Every image-bearing contract MUST declare colour space, channel order, element type, value range, width, height, and row/column axis meaning. Consumers MUST NOT infer image format from field name.

## 8. Object Ownership

### 8.1 Ownership Table

| Object | Creator | Owner After Creation | Readers | Writers | Destruction Point | Transfer Rules |
|---|---|---|---|---|---|---|
| `Frame` | intake boundary | engine orchestrator | validation, detection, crop selection | none after intake | end of frame lifetime | immutable borrow only |
| `ValidatedFrame` | validation module | engine orchestrator | detection, crop, assembly | none after emission | end of frame lifetime | immutable transfer from validation |
| `DetectionBatch` | detection module | engine orchestrator | crop, observation, assembly | none after emission | end of frame/window output construction | immutable transfer |
| `CropBatch` | crop selection module | engine orchestrator | embedding, demographics, assembly | none after emission | after dependent modules and output policy | immutable transfer |
| `EmbeddingBatch` | embedding module | engine orchestrator | observation, assembly | none after emission | after observation and output policy | immutable transfer |
| `ObservationBatch` | observation module | engine orchestrator | tracking, assembly | none after emission | after tracking transition and output policy | immutable transfer |
| `TrackingState` | tracking transition module | engine orchestrator until next transition | tracking transition only | next transition creates replacement | superseded state destruction after safe transfer | ownership transfer into transition; replacement returned |
| `TrackSnapshotBatch` | tracking transition module | engine orchestrator | events, demographics, assembly | none after emission | output policy | immutable transfer |
| `ObservationAssignmentBatch` | tracking transition module | engine orchestrator | demographics, assembly | none after emission | output policy | immutable transfer |
| `EventState` | event derivation module | engine orchestrator until next event boundary | event derivation only | next event boundary creates replacement | superseded state destruction after safe transfer | ownership transfer into event boundary; replacement returned |
| `EventBatch` | event derivation module | engine orchestrator | assembly | none after emission | output policy | immutable transfer |
| `DemographicsBatch` | demographics module | engine orchestrator | assembly | none after emission | output policy | immutable transfer |
| `AnalyticsOutput` | assembly module | output boundary then external consumer | external consumers | none after emission | external retention policy | immutable transfer |

### 8.2 Borrowing Rules

Borrowed objects MUST be read-only. A borrower MUST NOT retain references to frame-lifetime image buffers beyond the frame lifetime. If data must outlive its source object, a new object with explicit ownership and lifetime MUST be created.

### 8.3 Writer Rules

Only the owner may write to an object, and only while the object is in a mutable construction phase. After publication across a module boundary, the object becomes immutable.

## 9. Object Lifetime

### 9.1 Lifetime Classes

- **Process lifetime:** exists from engine process start until process end.
- **Engine lifetime:** exists from engine initialization until engine shutdown.
- **Source lifetime:** exists while a source stream is registered.
- **Window lifetime:** exists for a declared processing window.
- **Frame lifetime:** exists from frame intake until all frame-scoped consumers finish.
- **Transition lifetime:** exists during one state transition.
- **Track lifetime:** exists from track creation until closure plus configured retention scope.
- **Event lifetime:** exists from event emission through external retention scope.
- **Transient lifetime:** exists only during one module call.

### 9.2 Lifetime Requirements

Every object MUST declare one lifetime class. Objects MUST NOT be retained beyond their lifetime. State objects MUST define destruction or supersession points. Frame image buffers MUST NOT be retained in long-lived state unless represented by a separate explicit object with declared retention and ownership.

## 10. Memory Model

### 10.1 Persistent Memory

Persistent memory contains engine configuration, schema registry data, current state objects, and retained output objects. Persistent memory MUST be bounded by declared source, window, track, event, and retention policies.

### 10.2 Transient Memory

Transient memory contains module-local construction objects and borrowed inputs during a module call. Transient objects MUST be destroyed or transferred before the module call completes.

### 10.3 Immutable Memory

Published contracts are immutable memory. Immutable memory MAY be shared by multiple readers because no reader may write to it.

### 10.4 Mutable Memory

Mutable memory is permitted only during object construction or explicit state transition. Mutable memory MUST have one writer. Mutable state MUST NOT be exposed through public output contracts.

### 10.5 Reference Semantics

References across modules MUST be stable identifiers unless the referenced object is borrowed immutably for a call and cannot outlive the caller-owned scope. Long-lived state MUST store identifiers or explicitly owned value copies, not borrowed transient references.

### 10.6 Copy Semantics

Copies MUST create a new owned object or value with its own lifetime. Copying MUST preserve semantic values and MUST NOT silently alter schema, units, coordinate system, or identity.

### 10.7 Bounded History

Any history retained in state MUST declare maximum scope by count, time, window, or lifecycle. Unbounded history is prohibited.

### 10.8 Image Memory

Image buffers are large frame-scoped objects. Image memory MUST remain immutable after intake. Derived image regions MUST declare whether they own independent image memory or borrow from a parent image and MUST declare the maximum lifetime permitted by that relationship.

### 10.9 Descriptor Memory

Descriptor vectors are immutable after emission. A descriptor retained in tracking state MUST either be copied into state ownership or referenced by an identifier whose backing storage has a lifetime at least as long as the reference.

## 11. State Model

### 11.1 Definition of Analytics State

Analytics state is any information from prior processing boundaries that can influence future analytics outputs. Analytics state includes tracking state, event state when required, source processing clocks, schema compatibility state, and bounded retention indexes.

### 11.2 State Ownership

Analytics state is owned by the engine orchestrator between transitions and by the relevant transition module during a transition. No other module may own analytics state.

### 11.3 State Transfer

A state transition consumes a prior state object and emits a replacement state object. The prior state MUST NOT be mutated after replacement emission. Downstream modules receive snapshots or projections, not mutable state.

### 11.4 Prohibited State

The following MUST NOT become hidden module state: prior frames, prior detections, prior crops, prior embeddings, prior observations, prior assignments, prior events, demographic outputs, output packages, and diagnostics. If any such data must influence future outputs, it MUST be included in an explicit state object with declared ownership and retention.

## 12. Data Flow Trace

### 12.1 Frame Flow

Producer: intake boundary → Consumers: validation, detection, crop selection, assembly metadata → Destruction: end of frame lifetime after all frame-scoped consumers complete.

### 12.2 Detection Flow

Producer: detection module → Consumers: crop selection, observation, assembly → Destruction: after output assembly unless retention policy stores immutable projection.

### 12.3 Crop Flow

Producer: crop selection module → Consumers: embedding, demographics, assembly → Destruction: after dependent modules complete unless output policy stores immutable projection.

### 12.4 Embedding Flow

Producer: embedding module → Consumers: observation, tracking through observation, assembly → Destruction: after frame processing unless copied or referenced under an explicit state retention rule.

### 12.5 Observation Flow

Producer: observation module → Consumers: tracking transition, assembly → Destruction: after transition and output assembly unless referenced by bounded state history.

### 12.6 Tracking State Flow

Producer: tracking transition module → Consumer: next tracking transition → Destruction: when superseded and no valid immutable snapshot references remain.

### 12.7 Track Snapshot Flow

Producer: tracking transition module → Consumers: event derivation, demographics, assembly → Destruction: after output policy completes.

### 12.8 Assignment Flow

Producer: tracking transition module → Consumers: demographics, assembly → Destruction: after output policy completes.

### 12.9 Event Flow

Producer: event derivation module → Consumers: assembly and external event consumers → Destruction: external retention policy.

### 12.10 Demographics Flow

Producer: demographics module → Consumers: assembly and external attribute consumers → Destruction: external retention policy.

## 13. Pipeline Invariants

1. Contracts MUST NOT change schema during execution without explicit schema version transition.
2. Every object MUST have exactly one owner.
3. Published contracts MUST be immutable.
4. Input contracts MUST be read-only to consumers.
5. Batches declared as aligned MUST remain aligned for their entire lifetime.
6. Stable identifiers MUST NOT be reused within their declared uniqueness scope.
7. Per-source frame indexes MUST be monotonic.
8. Per-source timestamps MUST be monotonic unless explicitly diagnosed.
9. Outputs MUST be deterministic for identical inputs, configuration, state, and schema versions.
10. State MUST be explicit and transferable.
11. Hidden analytics state is prohibited.
12. A module MUST NOT mutate objects owned by another module.
13. A module MUST NOT depend on downstream modules.
14. Missing data MUST be explicit.
15. Invalid data MUST be explicit.
16. Diagnostics MUST identify the violated contract and field.
17. Output assembly MUST preserve module-native facts.
18. Image format MUST be declared wherever image data appears.
19. Coordinate system MUST be declared and consistent.
20. External consumers MUST receive immutable outputs.

## 14. Global Standards

### 14.1 Naming

Contract names MUST be singular for single objects and end with `Batch` for ordered collections. Identifier fields MUST end with `_id`. Timestamp fields MUST include `timestamp`. Index fields MUST include `index`.

### 14.2 IDs

IDs MUST be opaque strings. Consumers MUST NOT derive meaning from ID text. Every ID field MUST declare uniqueness scope and producer.

### 14.3 Ordering

Ordering MUST be deterministic and declared. If no domain-specific order is declared, objects MUST be ordered by source, frame index, timestamp, producer order, and identifier as needed to produce stable total ordering.

### 14.4 Sorting

Sorting MUST NOT alter identity, ownership, or alignment unless a new batch with a declared ordering and alignment map is produced.

### 14.5 Null Handling

Null values are prohibited in emitted contracts unless a field explicitly permits null. Missing, unknown, unavailable, invalid, and not applicable values MUST be represented by typed enums or unavailable marker objects.

### 14.6 Invalid Data

Invalid objects MUST NOT be silently dropped when downstream alignment requires positional preservation. They MUST be represented by invalid status or unavailable marker records.

### 14.7 Versioning

Every contract MUST include or be covered by a schema version. Schema versions MUST be present in output packages. A consumer MUST be able to determine the schema of every object from the output alone.

### 14.8 Schema Evolution

Schema evolution MUST be additive or explicitly versioned. Removing fields, changing units, changing coordinate meaning, changing enum semantics, or changing identity scope requires a new schema version.

## 15. Module Dependency Rules

### 15.1 Allowed Dependency Direction

Dependencies flow from upstream contracts to downstream modules only:

`Frame Validation → Detection → Crop Selection → Embedding → Observation → Tracking → Event Derivation → Demographics / Analytics Assembly → Output`

Demographics MAY depend on crop and assignment contracts. Event derivation MAY depend on track snapshots. Assembly MAY depend on all module outputs.

### 15.2 Prohibited Dependencies

- Detection MUST NOT depend on tracking, events, demographics, or assembly.
- Crop selection MUST NOT depend on tracking, events, or demographics.
- Embedding MUST NOT depend on tracking, events, demographics, or assembly.
- Observation MUST NOT depend on tracking, events, or demographics.
- Tracking MUST NOT depend on events, demographics, or assembly.
- Event derivation MUST NOT depend on demographics or assembly.
- Demographics MUST NOT mutate or influence tracking or event state.
- No module may depend on private data structures of another module.

### 15.3 Adapter Rule

Adapters MAY bridge external libraries or legacy code to constitutional contracts. Adapters MUST terminate at module boundaries and MUST NOT leak non-constitutional types across those boundaries.

## 16. Scalability Principles

1. Module contracts MUST permit independent replacement of modules.
2. State MUST be partitionable by declared source or window identity.
3. Batches MUST carry explicit scope so they can be processed independently without hidden global context.
4. Memory retention MUST be bounded by contract.
5. Output ordering MUST remain deterministic even when internal execution order differs.
6. Cross-source coupling is prohibited unless represented by an explicit multi-source contract.
7. Backpressure, partial availability, and skipped optional outputs MUST be represented by explicit contracts, not by missing objects.

## 17. Future Extensibility

A future module may be added only when it declares:

- purpose and responsibility;
- upstream producers;
- downstream consumers;
- input and output contracts;
- ownership and lifetime rules;
- state rules;
- dependency position;
- diagnostics;
- schema versioning;
- compatibility with all pipeline invariants.

A future module MUST NOT require changes to existing constitutional contracts unless a schema evolution process creates a new version while preserving compatibility rules for existing consumers.

## 18. Architectural Laws

1. The Constitution is the highest technical authority for analytics engine architecture.
2. The pipeline is stateless; analytics state exists only as explicit state contracts.
3. Every object has exactly one owner.
4. Every object has a declared lifetime and destruction point.
5. Every published contract is immutable.
6. Every module boundary is contract-based.
7. Every field exists for a documented purpose.
8. Every field has a producer, owner, consumer, type, lifetime, mutability rule, and invariant.
9. Every batch has deterministic ordering.
10. Every alignment relationship is explicit and preserved.
11. Every identifier has declared uniqueness scope.
12. Every timestamp has declared units and time base.
13. Every coordinate has declared coordinate system and units.
14. Every image buffer has declared format, colour space, shape, and value domain.
15. Missing data is explicit.
16. Invalid data is explicit.
17. Hidden analytics state is prohibited.
18. Shared mutable analytics objects are prohibited.
19. Downstream modules do not influence upstream modules.
20. Tracking state is updated only through the tracking transition boundary.
21. Event state is updated only through the event derivation boundary.
22. Demographic outputs do not affect tracking identity or event identity.
23. Output assembly preserves facts; it does not redefine them.
24. Diagnostics are part of the architecture, not optional side effects.
25. Schema versions are mandatory for durable interpretation.
26. A module may be replaced only if it preserves its constitutional contracts.
27. A contract may evolve only through explicit schema evolution.
28. No object may be retained without an ownership, lifetime, and memory rule.
29. No external type may cross a constitutional module boundary unless represented by a constitutional contract.
30. Deterministic outputs are required for identical inputs, configuration, state, and schema versions.
