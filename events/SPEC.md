# Card 9 Event Detection System — Engineering Specification

## Status and Scope

This document is the locked implementation specification for Card 9 (`events/`). It accompanies the production module and describes the contract the implementation must preserve.

Card 9 is a downstream, stateless consumer of Card 8 `RuntimeTrackV2` objects. It must not modify Card 5–8 modules and must not rely on detections, embeddings, frames, per-point timestamps, tracker internals, or velocity.

Allowed Phase B write scope:

- `events/` for the new event detection module.
- Root-level test files for Card 8 → Card 9 integration coverage.

Forbidden Phase B write scope:

- `track/`, `trackv2/`, `detection/`, `embed/`, and existing pipeline modules, except by explicit future product decision.

## A. Architecture Plan

### Module Layout

```text
events/
├── __init__.py        # public exports only
├── detector.py        # main pure detection entry point
├── geometry.py        # signed line-side and transition math
├── models.py          # RuntimeEventCandidate model and type constants
└── utils.py           # optional deterministic serialization / ID helpers
```

### Responsibilities

#### `events.models`

Defines the output contract:

```python
RuntimeEventCandidate = {
    "event_id": str,
    "runtime_track_id": str,
    "timestamp": float,
    "event_type": "ENTRY" | "EXIT",
    "direction": "IN" | "OUT",
    "supporting_positions": list[list[float]],
}
```

Implementation may use a frozen dataclass or a dict-compatible model, but public output from the detector must be JSON-like and exactly contain the fields above. No optional fields are allowed in Phase B.

#### `events.geometry`

Owns all geometric primitives:

- Normalize `LineConfig` into two points `a=[x, y]`, `b=[x, y]`.
- Validate line has non-zero length.
- Compute signed cross product for each center point:

```text
cross = (p_x - a_x) * (b_y - a_y) - (p_y - a_y) * (b_x - a_x)
```

- Convert cross product to side label:
  - `A` for negative side.
  - `B` for positive side.
  - `ON` for exactly on the line or within the configured epsilon.

The infinite oriented line is defined from `point_a` to `point_b`; no segment clipping or camera-space assumptions are permitted.

#### `events.detector`

Owns the public API and deterministic event construction:

```python
def detect_events(tracks: list[RuntimeTrackV2], line_config: dict) -> list[RuntimeEventCandidate]:
    ...
```

`detect_events` must:

- Read only `runtime_track_id`, `center_history`, and `last_seen_timestamp` from tracks.
- Never mutate track objects or nested `center_history` points.
- Emit every valid chronological crossing event contained in each supplied trusted track.
- Be stateless, deterministic, and free of random UUID generation.
- Return events sorted deterministically.

#### `events.utils` Optional Helpers

May contain helpers for:

- Stable event ID construction.
- Canonical float formatting.
- Defensive point copying.

No cache, global mutable state, persistence, or in-memory event registry is permitted.

### Boundaries

Card 9 is a pure geometric transition detector. It answers only: "Did this track cross the infinite oriented line?" It must not infer speed, exact crossing time, identity, confidence, tracking quality, or cause.

## B. Algorithm Definition

### Inputs

```python
tracks: list[RuntimeTrackV2]
line_config: {
    "point_a": [x, y],
    "point_b": [x, y],
}
```

Required `RuntimeTrackV2` fields:

- `runtime_track_id: str`
- `center_history: list[list[float]]`
- `last_seen_timestamp: float`

Ignored fields:

- `velocity`
- `detection_history`
- `embeddings` / `last_embedding`
- `current_center`, except insofar as Card 8 already included it in `center_history`
- track lifecycle state, unless a future contract explicitly filters states

### Constants

Phase B must define these constants in `events.detector` or `events.geometry`:

```python
LINE_SIDE_EPSILON = 1e-9
```

`LINE_SIDE_EPSILON` prevents floating-point noise near the line. Card 9 does not apply track-quality, confidence, or stability heuristics; Card 8 owns trajectory trustworthiness.

### Step 1 — Extract Trajectory

For each track, copy and normalize `center_history` into `list[list[float]]`.

Validation rules:

- Fewer than two usable non-`ON` positions cannot produce a geometric transition.
- Malformed points should fail fast with a `ValueError`; silent repair is forbidden because it would hide contract violations.

### Step 2 — Compute Side Sequence

For each point `p`, compute:

```text
side_raw = (p - a) × (b - a)
```

Then map to labels:

- `side_raw < -LINE_SIDE_EPSILON` → `A`
- `side_raw > LINE_SIDE_EPSILON` → `B`
- otherwise → `ON`

### Step 3 — Compress `ON` Points

`ON` points do not themselves define `A → B` or `B → A`. The detector must build a side sequence of non-`ON` points while retaining their original indices.

Example:

```text
A, A, ON, B, B  =>  A, A, B, B
```

If all points are `ON`, emit no event.

### Step 4 — Detect Every Geometric Transition

A valid geometric transition exists at compressed index `i` when:

```text
compressed_side[i - 1] != compressed_side[i]
```

Every such transition in the trusted trajectory must emit one event in chronological order. Card 9 must not reject transitions because a track is short, terminal, oscillatory, or lacks post-transition stability; those are track-quality concerns owned by Card 8.

Examples:

- `A, A, B, B` emits one `ENTRY`.
- `B, B, A, A` emits one `EXIT`.
- `A, B, A` emits `ENTRY`, then `EXIT`.
- `A, A, B, B, A, A, B, B` emits `ENTRY`, `EXIT`, `ENTRY`.
- `A, ON, B, ON, B` emits one `ENTRY` because `ON` points are ignored for side transitions.

### Step 5 — Determine Event Semantics

Line orientation is fixed by `point_a -> point_b`.

Side mapping:

- `A` = negative side of the oriented infinite line.
- `B` = positive side of the oriented infinite line.

Event mapping:

- `A → B` → `event_type="ENTRY"`, `direction="IN"`
- `B → A` → `event_type="EXIT"`, `direction="OUT"`

This mapping is contractual and must not be changed for camera-specific semantics.

### Step 6 — Timestamp

Because Card 9 does not receive timestamps per center point, Phase B must use:

```python
timestamp = float(track.last_seen_timestamp)
```

The crossing index may be used for event ID generation and supporting window selection, but not as a real time. The detector must not fabricate per-point timestamps.

### Step 7 — Supporting Positions

`supporting_positions` must contain the local transition window in original trajectory coordinates:

- The last non-`ON` point before transition.
- The first non-`ON` point after transition.
- The next same-side non-`ON` point after transition when one is immediately available before another transition.

For `A, A, B, B`, supporting positions are the original points corresponding to the second `A`, first `B`, and second `B`, preserving existing stable-crossing support context. For terminal or two-point crossings, supporting positions end at the transition point.

If `ON` points exist between these positions, they may be included only if they fall inside the transition window. The recommended locked behavior is to include the continuous original-index slice from previous non-`ON` index through the selected support end index, preserving order.

All supporting coordinates must be copied as floats.

### Step 8 — Deterministic Event ID

Event IDs must be deterministic and content-derived. Phase B must use this strategy unless superseded by a future product contract:

```text
event_id = "evt_" + sha256(
    runtime_track_id + "|" +
    line_a_x + "," + line_a_y + "|" +
    line_b_x + "," + line_b_y + "|" +
    transition_original_index + "|" +
    event_type + "|" +
    direction
).hexdigest()[:16]
```

Float values must be canonicalized with a stable representation, e.g. `format(float(value), ".12g")`, to avoid platform-dependent string drift. Random UUIDs, counters, memory addresses, object IDs, and current time are forbidden.

### Step 9 — Ordering Guarantees

Output events must be sorted by:

1. `timestamp` ascending.
2. `runtime_track_id` ascending.

The detector must append events for each track in chronological transition order before sorting. Python's stable sort preserves that chronological order for multiple events from the same track that share the track-level timestamp.

## C. Integration Plan

### Public Import Surface

`events/__init__.py` should export:

```python
from events.detector import detect_events
from events.models import RuntimeEventCandidate
```

Tests and downstream code should import from `events` where possible.

### Card 8 → Card 9 Flow

The integration path is:

```text
Mock Observations
      ↓
Card 8 TrackV2.update(observations_by_ts)
      ↓
RuntimeTrackV2 list
      ↓
events.detect_events(tracks, line_config)
      ↓
Assertions / downstream candidates
```

The test harness must use real `TrackV2` code and synthetic observations. It must not run video decoding, detection models, embedding models, or frame processing.

### Synthetic Observation Shape

Root-level tests should construct observations compatible with Card 8:

```python
{
    "detection_id": "det_001",
    "timestamp": 0.0,
    "center": [x, y],
    "bbox": [x - 5, y - 10, x + 5, y + 10],
    "confidence": 0.99,
    "embedding": [1.0, 0.0],
}
```

Embedding values exist only to satisfy the observation shape; Card 9 must not read them.

### Tracker Configuration for Tests

Tests should use a permissive deterministic `TrackV2Config` so synthetic motion remains a single runtime track:

```python
TrackV2Config(
    max_speed_px_per_sec=10000.0,
    base_motion_gate=10000.0,
    tentative_hits_to_activate=1,
    unmatched_detection_buffer_frames=1,
    min_track_lifetime_sec=0.0,
)
```

Because Card 8 currently creates UUID runtime track IDs, tests must not assert hard-coded track ID values. Instead, they must assert that emitted `runtime_track_id` equals the ID of the track produced by Card 8 for the crossing trajectory.

## D. Full Test Strategy

Add a root-level test file, recommended name:

```text
test_events_card9_pipeline.py
```

The suite must be runnable without model downloads or video files:

```bash
python -m pytest test_events_card9_pipeline.py
```

### Shared Test Helpers

The test file should include helpers:

- `make_observation(detection_id, timestamp, center)`
- `run_tracker_for_trajectory(points, detection_prefix="det")`
- `run_tracker_for_multi_track(trajectories)`
- `vertical_line_at_x(x)` or direct line configs

For a vertical line at `x=0`, define:

```python
line_config = {"point_a": [0.0, -10.0], "point_b": [0.0, 10.0]}
```

With the specified cross-product formula, points with `x < 0` are side `A` and points with `x > 0` are side `B`. Therefore, for this oriented vertical line:

- left-to-right (`x < 0` to `x > 0`) is `A → B`, `ENTRY / IN`.
- right-to-left (`x > 0` to `x < 0`) is `B → A`, `EXIT / OUT`.

Tests should choose trajectories accordingly.

### Test 1 — Single Crossing

Trajectory:

```python
[[-10, 0], [-5, 0], [5, 0], [10, 0]]
```

Line:

```python
{"point_a": [0, -10], "point_b": [0, 10]}
```

Assertions:

- Exactly one event.
- `event_type == "ENTRY"`.
- `direction == "IN"`.
- `runtime_track_id` equals the crossing track's Card 8 `runtime_track_id`.
- `supporting_positions` includes the transition window around `[-5, 0]`, `[5, 0]`, `[10, 0]`.

### Test 2 — No Crossing

Trajectory parallel to the line and staying on one side:

```python
[[10, -10], [10, 0], [10, 10], [10, 20]]
```

Assertions:

- Zero events.

### Test 3 — Oscillation Filter

Trajectory:

```python
[[-10, 0], [5, 0], [-10, 0]]
```

This is `A → B → A` for the vertical test line.

Assertions:

- Zero events.

### Test 4 — Multi-Track Independence

Trajectories:

```python
crossing = [[-10, 0], [-5, 0], [5, 0], [10, 0]]
non_crossing_1 = [[20, 20], [20, 25], [20, 30], [20, 35]]
non_crossing_2 = [[-20, -20], [-20, -25], [-20, -30], [-20, -35]]
```

Assertions:

- Exactly one event.
- The event references only the crossing track ID.
- Neither non-crossing track ID appears in emitted events.
- No cross-contamination of supporting positions from other tracks.

### Test 5 — Full Pipeline Validation

This is the critical integration test and must use real Card 8 `TrackV2.update`.

Steps:

1. Generate deterministic observations for a crossing trajectory.
2. Feed observations through `TrackV2.update(observations_by_ts)`.
3. Pass returned tracks into `detect_events(tracks, line_config)`.
4. Assert event count, event type, direction, track mapping, and supporting positions.
5. Assert output ordering is stable by comparing to `sorted(events, key=lambda e: (e["timestamp"], e["runtime_track_id"]))` if output is dict-based.

### Test 6 — Determinism

Use the same `tracks` object and same `line_config` repeatedly:

```python
events_1 = detect_events(tracks, line_config)
events_2 = detect_events(tracks, line_config)
events_3 = detect_events(list(reversed(tracks)), line_config)
```

Assertions:

- `events_1 == events_2`.
- Event IDs are identical.
- Reversed input order still returns the same sorted event list.
- No randomness is used in Card 9 output.

### Additional Unit Tests Recommended

Although the required suite is integration-first, Phase B should also include focused tests for:

- Invalid zero-length line raises `ValueError`.
- All points on the line emit no event.
- `ON` points between sides do not block a geometric crossing.
- `B → A` emits `EXIT / OUT`.
- Tracks with fewer than two usable non-`ON` points emit no event.

## E. Risk Analysis and Open Questions

### Deterministic Event ID Strategy

Resolved for Phase B by the SHA-256 content strategy above. The main risk is float canonicalization. Implementation must centralize canonical formatting and tests must assert exact repeatability.

### Oscillation Threshold Definition

Resolved for the geometry-only refactor as a pure transition rule: every `A → B` or `B → A` side change in the trusted compressed trajectory emits an event. Card 8 owns any track-quality filtering before Card 9 receives the trajectory.

### Stable Line Orientation Rules

Resolved for Phase B: `point_a -> point_b` defines orientation; `A` is negative cross-product side, `B` is positive cross-product side. The semantic mapping is fixed and not camera-dependent.

### Multiple Crossings Per Track

Resolved for the geometry-only refactor: Card 9 emits every valid chronological side transition in each trusted trajectory. Duplicate suppression and track trust remain outside Card 9.

### Ordering Guarantees

Resolved for the geometry-only refactor as sorted by `(timestamp, runtime_track_id)` with stable preservation of chronological transition order within each track.

### Timestamp Source

Resolved for Phase B as `track.last_seen_timestamp`. This is not the precise crossing time. The risk must be documented for downstream consumers because Card 9 does not receive per-point timestamps.

### Card 8 Runtime Track IDs

Card 8 currently generates UUID track IDs. Card 9 can still be deterministic for identical input track objects, but end-to-end test runs that instantiate a fresh tracker may receive different `runtime_track_id` values and thus different event IDs. Tests must assert determinism for repeated detection calls over the same Card 8 output, not across fresh tracker instantiations unless Card 8 later exposes deterministic IDs.

### Track Lifecycle State

The input contract includes `state`, but the brief does not define filtering by state. Phase B must not filter by `state`; all provided tracks are evaluated. If downstream consumers need active-only events, that must be specified outside Card 9 or by future contract change.

### Mutation Risk

Because `center_history` is a nested list, the detector must copy points before constructing `supporting_positions`. Tests should verify that track histories are unchanged after detection.
