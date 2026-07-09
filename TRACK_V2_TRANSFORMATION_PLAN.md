# Track Module V2 Architecture Redesign From Locked IO Contract

## 1. Confirmation of Locked IO Contract

This document is an investigation and architecture design plan only. It does **not** implement Track V2.

The locked V2 pipeline is:

```text
Frame
  |
Detect
  |
Embed
  |
Observe
  |
Track
  |
Event
  |
Demographic
  |
Assemble
  |
Output
```

The locked Track transition is:

```text
Track(TrackingState(t), ObservationBatch(t)) -> TrackingState(t+1)
```

The locked Track input/output state is:

```text
TrackingState:
    tracks: Track[]

Track:
    track_id: string
    path: Points[]
    best_crop: BestCrop
    best_crop_confidence: float32

BestCrop:
    frame_id: string
    bbox: BoundingBox

Points:
    timestamp: float64
    center: Point2D
```

The locked Track observation input is:

```text
ObservationBatch:
    frame_id: string
    observations: Observation[]

Observation:
    detection_id: string
    bbox: BoundingBox
    center: Point2D
    embedding: FeatureVector
    confidence: float32
```

Track returns only `TrackingState`. There is no assignment map, no secondary output, no debug output, and no hidden contract extension in the public return value.

Track is responsible only for assigning observations to tracks, creating tracks, updating track paths, updating representative crop references, and applying lifecycle policy that can be expressed through the locked state. Track must not own frame loading, frame iteration, detection, embedding generation, observation creation, event detection, demographics, output generation, or orchestration.

## 2. Current Track Implementation Summary

Current files:

```text
track/
  __init__.py      Exports TrackV2, TrackV2Config, RuntimeTrackV2.
  config.py        Motion, lifecycle, birth-buffer, and threshold configuration.
  lifecycle.py     Runtime track creation, promotion, closure, stale handling.
  matching.py      Motion gating, cosine embedding similarity, candidate assignment.
  models.py        Mutable RuntimeTrackV2 dataclass with runtime-only fields.
  motion.py        Prediction, velocity, and distance helpers.
  tracker.py       Stateful TrackV2 wrapper containing tracks, pending births, frame index, and update loop.
```

Current execution flow:

```text
External runner owns frame/video loop
  |
  v
Detection and embedding are produced outside Track
  |
  v
Legacy observations are grouped as observations_by_ts[timestamp]
  |
  v
TrackV2.update(observations_by_ts)
  |
  | internal mutable state:
  |   self.tracks
  |   self.pending_births
  |   self.frame_index
  |   self.last_update_timestamp
  |   self.last_debug_report
  v
For each timestamp sorted inside TrackV2.update:
  |
  v
matchable_tracks = ACTIVE + TENTATIVE runtime tracks
  |
  v
assign_matches(matchable_tracks, observations, config)
  |
  | motion_gate(track, observation)
  | embedding_similarity(track.last_embedding, observation.embedding)
  | greedy deterministic assignment
  v
Update matched tracks in place
  |
  | center, bbox, velocity, last_embedding, timestamps,
  | hit/miss counters, detection history, center history
  v
close_stale_tracks(...)
  |
  v
Process unmatched observations through pending-birth logic
  |
  v
Return (tracks, assignment_map)
```

Important mismatch with locked V2: the current implementation is a stateful tracker engine. V2 Track must become a deterministic transition over explicit `TrackingState` and one `ObservationBatch`.

## 3. State Sufficiency Analysis

The central question is whether the locked state:

```text
track_id + path[] + best_crop + best_crop_confidence
```

is sufficient to reconstruct the behaviours currently implemented with hidden runtime fields.

### 3.1 Identity Continuity

Identity continuity can be represented by `track_id`. If Track appends new path points to the same `track_id`, downstream modules can observe continuity without an assignment map.

Sufficiency: **yes**, for representing existing identities.

Limitation: deterministic creation of new `track_id` values is not specified. Random UUIDs would violate deterministic replay. A deterministic ID can be derived from locked state without adding a field, for example:

```text
next_id = "track-" + (1 + max numeric suffix already present)
```

This requires a defined ID format. If arbitrary existing `track_id` strings must be accepted, ID allocation needs founder-approved rules.

### 3.2 Distance Calculation

Given a track's latest path point and an observation center:

```text
latest = path[-1].center = (x_t, y_t)
obs = observation.center = (x_o, y_o)
```

distance is derivable:

```text
d = sqrt((x_o - x_t)^2 + (y_o - y_t)^2)
```

Sufficiency: **yes**.

### 3.3 Velocity Estimation and Motion Prediction

If a track has at least two path points:

```text
p1 = path[-2] = (timestamp t1, center (x1, y1))
p2 = path[-1] = (timestamp t2, center (x2, y2))
```

then velocity can be reconstructed as:

```text
vx = (x2 - x1) / (t2 - t1)
vy = (y2 - y1) / (t2 - t1)
```

For a current observation timestamp `t3`, predicted position is:

```text
predicted_x = x2 + vx * (t3 - t2)
predicted_y = y2 + vy * (t3 - t2)
```

Sufficiency: **conditionally yes**.

Conditions:

1. `path` must contain at least two points.
2. Consecutive path timestamps must differ by more than zero.
3. The current observation batch must provide the current timestamp.

Concrete limitation: the locked `Points` contract includes `timestamp`, but the locked `ObservationBatch` shown in the brief does **not** include `timestamp`. Without an input timestamp, Track cannot append a valid `Points(timestamp, center)` and cannot compute `dt = t3 - t2` for prediction. Therefore either:

- `ObservationBatch.timestamp` must be confirmed as part of the locked contract, as earlier V2 Observe work already used; or
- timestamp must be supplied by another locked field; or
- Track cannot produce contract-valid path points.

This is a proven technical limitation, not an optimization preference.

### 3.4 Smoothed Velocity

The current tracker stores smoothed velocity:

```text
v_smooth(t+1) = alpha * v_smooth(t) + (1 - alpha) * v_measured(t+1)
```

The locked state does not store `v_smooth(t)`. It stores only path points.

Can smoothed velocity be reconstructed exactly from the full path? Mathematically, yes **if** all historical path points are retained and the smoothing constant is fixed. One can replay the recurrence from the start of the path.

Practical implication:

- If full path history is retained, stored velocity is not mathematically required.
- If path history is bounded/truncated, exact smoothed velocity cannot be reconstructed unless velocity is stored.

Recommendation: prefer deriving velocity from the last two path points for V2 simplicity unless founder approval requires exact compatibility with current smoothed-velocity behaviour.

### 3.5 Motion Gating

Motion gating can be expressed using derived prediction:

```text
allowed_distance = base_motion_gate + max_speed_px_per_sec * dt
motion_distance = distance(predicted_center, observation.center)
allowed = motion_distance <= allowed_distance
```

Sufficiency: **yes**, if current timestamp exists and path has enough points. For one-point tracks, use a no-velocity fallback:

```text
predicted_center = path[-1].center
```

### 3.6 Track Matching

Motion-only matching can be performed from locked state:

1. For each track, derive latest center and optional velocity from `path`.
2. For each observation, compute predicted-position distance.
3. Reject impossible motion using gate thresholds.
4. Solve one-to-one assignment deterministically by sorted candidate cost.

Sufficiency: **yes for motion-only matching**.

Sufficiency for current behaviour including appearance matching: **no**, explained below.

## 4. Embedding Matching Requirements

Current tracker uses appearance continuity:

```text
similarity = cosine(track.last_embedding, observation.embedding)
```

The locked `ObservationBatch` exposes only current observation embeddings. Locked `TrackingState.Track` does not contain any previous embedding, average embedding, best-crop embedding, or embedding reference.

### 4.1 Can Previous Embedding Be Reconstructed?

No. Given only:

```text
track_id
path[]
best_crop.frame_id
best_crop.bbox
best_crop_confidence
```

there is no mathematical function that reconstructs a prior embedding vector. Many different embedding histories can produce the same path and best crop metadata.

Proof by counterexample:

```text
State A track path = [(t1, center1)]
State B track path = [(t1, center1)]
State A prior embedding = [1, 0]
State B prior embedding = [0, 1]
```

Both states are identical under the locked contract, but for a new observation embedding `[1, 0]`, cosine similarity differs:

```text
cos([1,0], [1,0]) = 1
cos([0,1], [1,0]) = 0
```

Therefore appearance matching cannot be reconstructed from the locked `TrackingState`.

### 4.2 Minimum Extension If Appearance Matching Is Required

If founder approval requires preserving embedding-based matching, the minimum state extension is one retained vector or vector reference per track:

```text
Track:
    matching_embedding: FeatureVector
```

or, if embedding should remain hidden from downstream public track shape:

```text
TrackingState:
    tracks: Track[]
    internal_track_memory:
        track_id -> matching_embedding
```

However, the second option is still a contract extension because all persistent analytics memory must be in `TrackingState` and must be serializable through the transition boundary.

Impact: the locked contract would need to change. Without this extension, V2 Track can only use embeddings from the current batch for tie-breaking between observations if no previous appearance state is needed; it cannot compare current observations to historical track appearance.

## 5. Lifecycle Reconstruction

Current tracker has hidden lifecycle fields:

```text
TENTATIVE / ACTIVE / CLOSED
hit_count
miss_count
first_seen_timestamp
last_seen_timestamp
closed_timestamp
created_frame_index
last_matched_frame_index
last_unmatched_frame_index
pending_births
```

The locked state has only `path[]` and crop metadata.

### 5.1 Derivable Lifecycle Values

From `path[]`:

```text
first_seen_timestamp = path[0].timestamp
last_seen_timestamp = path[-1].timestamp
hit_count = len(path)
age = current_timestamp - path[0].timestamp
gap_since_seen = current_timestamp - path[-1].timestamp
```

These are derivable if the current timestamp is available.

### 5.2 Not Derivable From Locked State

The following are not exactly derivable:

- `miss_count` measured in frames with no matched observation.
- Number of consecutive missed frames if empty frames do not append points.
- `TENTATIVE`, `ACTIVE`, and `CLOSED` as explicit states.
- `closed_timestamp` for closed tracks if the track remains in state without new path points.
- `last_unmatched_frame_index` used for recent-unmatched plausibility logic.
- Pending-birth counters for detections that have not yet become tracks.

Some of these can be approximated from timestamp gaps, but that changes semantics.

### 5.3 Lifecycle Designs Compatible With Locked State

To avoid contract extension, lifecycle must be represented implicitly:

- A track is considered **confirmed/active** if `len(path) >= min_hits`.
- A track is considered **stale for matching** if `current_timestamp - path[-1].timestamp > max_match_gap_sec`.
- A stale track can remain in `TrackingState.tracks` for downstream event/history purposes but is excluded from future matching.
- A track is effectively **closed** when it is too stale to match, without storing a closed state.

This is contract-compatible but not behaviour-identical to the current explicit miss-count lifecycle.

## 6. New Track Creation

Input:

```text
TrackingState(t) + unmatched observations at t
```

Output:

```text
TrackingState(t+1) with new Track objects appended
```

### 6.1 Immediate Creation

Contract-compatible logic:

```text
for each unmatched observation:
    create Track(
        track_id = deterministic_next_id(state),
        path = [{timestamp: current_timestamp, center: observation.center}],
        best_crop = {frame_id: observation_batch.frame_id, bbox: observation.bbox},
        best_crop_confidence = observation.confidence,
    )
```

Advantages:

- Requires no hidden pending-birth memory.
- Fully expressible through locked `TrackingState`.
- Deterministic.
- Simple to test.

Disadvantages:

- More false short-lived tracks may be created than the current pending-birth design.

### 6.2 Confirmation Period / Pending Births

The current tracker delays creation through `pending_births`. That requires storing unmatched detections that are not yet tracks.

Can this be represented in locked `TrackingState.tracks`? Only by creating tentative tracks immediately and interpreting tracks with `len(path) < min_hits` as unconfirmed. That is not identical to hidden pending births because tentative tracks become visible in output state.

A hidden `pending_births` list would violate the locked contract unless added to `TrackingState`.

Recommendation under locked IO: use immediate creation plus implicit confirmation by path length. Downstream modules can ignore short paths if their contracts allow, or Track can exclude too-young tracks from matching policy decisions only through derived `len(path)`.

## 7. Mathematical State Transition Model

Assume `ObservationBatch.timestamp` is confirmed. If it is not confirmed, Track cannot append valid `Points` and the transition is technically incomplete.

Let:

```text
S_t = { tracks: [T_1, ..., T_n] }
O_t = { frame_id, timestamp, observations: [O_1, ..., O_m] }
```

Each track:

```text
T_i = { track_id, path, best_crop, best_crop_confidence }
```

Each observation:

```text
O_j = { detection_id, bbox, center, embedding, confidence }
```

### 7.1 Candidate Generation

For each track `T_i`, if `T_i.path` is empty, it is invalid and should be rejected by contract validation.

Latest point:

```text
p_last = T_i.path[-1]
```

Velocity:

```text
if len(path) >= 2 and path[-1].timestamp > path[-2].timestamp:
    v = (path[-1].center - path[-2].center) / (path[-1].timestamp - path[-2].timestamp)
else:
    v = (0, 0)
```

Delta time:

```text
dt = O_t.timestamp - p_last.timestamp
```

Prediction:

```text
predicted = p_last.center + v * dt
```

Motion cost:

```text
cost_motion = distance(predicted, O_j.center)
```

Gate:

```text
allowed = dt >= 0 and cost_motion <= base_motion_gate + max_speed_px_per_sec * dt
```

Candidate:

```text
C_ij = (track_index=i, observation_index=j, cost=cost_motion)
```

### 7.2 Assignment

Sort candidates deterministically:

```text
(cost, track_id, observation.detection_id)
```

Greedily choose candidates where neither the track nor the observation has already been used.

This preserves the deterministic spirit of the current matcher without requiring hidden runtime fields.

### 7.3 State Update

For every matched pair `(T_i, O_j)`:

```text
T_i.path.append({timestamp: O_t.timestamp, center: O_j.center})

if O_j.confidence > T_i.best_crop_confidence:
    T_i.best_crop = {frame_id: O_t.frame_id, bbox: O_j.bbox}
    T_i.best_crop_confidence = O_j.confidence
```

For every unmatched observation `O_j`:

```text
create new Track with one path point and best_crop from O_j
```

For unmatched tracks:

```text
leave path and best_crop unchanged
```

For stale tracks:

```text
keep in TrackingState.tracks but exclude from future matching when timestamp gap exceeds policy
```

No assignment map is returned.

## 8. Can Track Operate Correctly Without Extending TrackingState?

Answer depends on the definition of "correctly".

### 8.1 Yes: A Clean V2 Motion-Based Tracker Is Possible

Using only locked state and a confirmed observation timestamp, Track can perform:

- identity continuity by `track_id`;
- path maintenance;
- latest-center distance matching;
- velocity-derived motion prediction from `path[-2:]`;
- motion gating;
- deterministic assignment;
- immediate new-track creation;
- best-crop reference updates;
- implicit lifecycle from path length and timestamp gaps.

### 8.2 No: The Existing Behaviour Cannot Be Fully Reconstructed

The current behaviour cannot be exactly reconstructed because locked state does not contain:

- previous embedding or appearance memory;
- explicit smoothed velocity unless replaying full path is guaranteed;
- miss counters;
- pending births;
- explicit tentative/active/closed lifecycle state;
- frame index;
- last unmatched frame index;
- closed timestamp.

The largest proven limitation is embedding continuity. Previous embeddings cannot be reconstructed from path and crop metadata.

## 9. Proposed Track V2 Folder Structure

```text
track/
  __init__.py
      Public export for Track and config.

  track.py
      Public stateless Track callable implementing
      Track(tracking_state, observation_batch) -> tracking_state.

  matching.py
      Candidate generation and deterministic one-to-one observation-track assignment.
      Uses only contract fields plus config.

  motion.py
      Path-derived velocity, prediction, and distance helpers.

  lifecycle.py
      Contract-compatible lifecycle policy:
      match eligibility from path length and timestamp gap;
      new track creation;
      stale-track exclusion rules.

  models.py
      Internal dataclasses/types for calculation only.
      These must not add persistent state outside TrackingState.

  config.py
      Thresholds such as base motion gate, max speed, min hits, max stale gap,
      and best-crop update policy.

  tests/
      Deterministic state-transition tests.
```

External contracts remain dict-compatible `TrackingState`, `Track`, `ObservationBatch`, `Observation`, `Points`, and `BestCrop`. Internal dataclasses may normalize these shapes during a call, but they must not become hidden persistent analytics state.

## 10. Proposed Internal Algorithm Flow

```text
Track.__call__(tracking_state, observation_batch)
  |
  v
Validate locked contract fields
  |
  | require tracking_state.tracks
  | require observation_batch.frame_id
  | require observation_batch.timestamp, or fail pending contract decision
  | require observation fields
  v
Normalize points, centers, bboxes, and confidences for calculation
  |
  v
Select match-eligible tracks
  |
  | path non-empty
  | not stale by timestamp gap
  | optionally len(path) >= min_hits for confirmed-only matching,
  | or include tentative one-point tracks for continuity
  v
Build motion candidates
  |
  | derive velocity from path
  | predict center
  | compute distance
  | apply motion gate
  v
Assign candidates deterministically
  |
  v
Update matched tracks
  |
  | append new path point
  | update best_crop when confidence improves
  v
Create tracks for unmatched observations
  |
  | deterministic track_id
  | one path point
  | best_crop from observation bbox and frame_id
  v
Leave unmatched existing tracks in state
  |
  | no hidden miss counter
  | future matching eligibility is derived from timestamp gap
  v
Return updated TrackingState only
```

## 11. Responsibility Map Under Locked IO

### KEEP

- Motion prediction concept.
- Distance calculation.
- Motion gating.
- Deterministic candidate assignment.
- Best-crop selection by confidence.
- Track creation and path updates.
- Threshold-driven behaviour through config.

### KEEP ONLY IF CONTRACT IS EXTENDED

- Historical embedding similarity.
- Exact smoothed velocity state.
- Exact miss-count lifecycle.
- Hidden pending-birth confirmation.
- Explicit tentative/active/closed states.

### MOVE OUTSIDE TRACK

- Video reading.
- Frame iteration.
- Detection.
- Embedding generation.
- Observation creation.
- Event detection.
- Demographics.
- Output assembly.
- Visualization.
- Persistence.

### DELETE FROM PUBLIC V2 TRACK API

- `TrackV2.update(observations_by_ts)` as the V2 entry point.
- Multiple timestamp updates inside one Track call.
- `(tracks, assignment_map)` return shape.
- Debug output return shape.
- Hidden instance-owned analytics state.
- Random UUID generation.

## 12. Testing Strategy

Primary tests must be deterministic transition tests:

```text
TrackingState(t) + ObservationBatch(t) = Expected TrackingState(t+1)
```

Required cases:

1. **Empty state + first observation**
   - Creates one track with one path point and best crop from the observation.

2. **Existing track + same person movement**
   - Appends a path point to the existing track.

3. **Multiple tracks + multiple observations**
   - Produces deterministic one-to-one assignment.

4. **Crossing trajectories**
   - Verifies motion prediction behaviour and documents where appearance matching would be needed if motion is ambiguous.

5. **Impossible movement rejection**
   - Observation outside motion gate creates a new track instead of updating the old one.

6. **New person creation**
   - Unmatched observation becomes a deterministic new track.

7. **Missing observation**
   - Existing tracks remain unchanged; stale eligibility is derived later from timestamp gap.

8. **Timestamp gaps**
   - Large gaps make old tracks ineligible or widen gate according to policy.

9. **Deterministic repeated execution**
   - Running the same transition from equal input state produces equal output state.

10. **Same input state produces identical output**
    - Confirms no hidden state in the Track callable.

Video tests can remain as integration regression tests only. They must not be the primary proof of Track correctness.

## 13. Decisions Requiring Founder Approval

1. The brief's `ObservationBatch` omits `timestamp`, while `Points` requires `timestamp`. Should `ObservationBatch.timestamp` be restored/confirmed as part of the locked contract?
2. Is a motion-only V2 tracker acceptable for the first strict-contract implementation?
3. If appearance continuity is required, may `TrackingState` be extended with per-track embedding memory?
4. Should V2 preserve exact current smoothed-velocity behaviour, or derive velocity from `path[-2:]`?
5. Should new tracks be created immediately, with confirmation represented by `len(path)`, or may `TrackingState` be extended with pending births?
6. Should lifecycle be implicit from path length and timestamp gap, or may explicit lifecycle state be added to the contract?
7. Should stale/closed tracks remain forever in `TrackingState.tracks`, or should another module/orchestrator prune them?
8. What deterministic `track_id` format should be required?
9. Should one-point tentative tracks be eligible for matching on the next frame?
10. Should best crop update strictly on higher confidence, or should crop quality use additional criteria later?

## 14. Recommended Architecture Decision

Implement the first V2 Track as a strict locked-contract, deterministic, motion-based state transition after confirming `ObservationBatch.timestamp`.

Do not add embeddings, miss counters, pending births, or explicit lifecycle fields unless founder approval accepts the contract extension. The investigation proves that historical embedding matching and exact current lifecycle behaviour cannot be reconstructed from the locked state alone, but a clean V2 tracker can still operate with path-derived motion, immediate track creation, best-crop updates, and implicit stale handling.
