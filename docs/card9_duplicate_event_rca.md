# Card 9 Duplicate Event Emission — Root Cause Analysis

## Root cause

The duplicate-looking Card 9 rows originate inside the Card 9 detector when a supplied `center_history` contains repeated side changes across the configured line. `detect_events()` is stateless and scans every supplied track from the beginning on every invocation. For each non-`ON` side transition in the compressed trajectory, it appends one event. It has no memory of previously emitted transitions and no lifecycle gating.

In the full video harness, `detect_events(tracks, line_config)` is invoked once per processed frame with the complete `tracker.tracks` list, not only once at the end and not only for newly closed tracks. The harness overwrites `latest_events` with the complete regenerated result each frame and prints only the final regenerated list. Therefore the final event table is not produced by appending old events across invocations; it is the current complete Card 9 interpretation of all track histories at the last frame.

The same timestamp on multiple rows is expected under the current Card 9 event model because `_events_for_track()` assigns every event for a track to `track.last_seen_timestamp`, not to the actual crossing point. Multiple events from one runtime track will share the track's last timestamp even when they correspond to different transition indices.

## Code path

1. `test_tracking_v2_pipeline.py::__main__` calls `run_contact_sheet_self_tests_section()`, `run_card9_event_scenario_tests_section()`, and then `main()`.
2. `main()` opens the video, builds a vertical center-line `line_config`, constructs `TrackV2`, and enters the frame loop.
3. Each frame is detected, embedded, converted to observations, and supplied to `tracker.update(observations_by_ts)`.
4. `TrackV2.update()` updates matched tracks by appending `track.current_center` to `track.center_history`, promotes tracks when hit thresholds are met, closes stale active/tentative tracks, and returns `self.tracks`.
5. In the same frame loop, the harness calls `latest_events = detect_events(tracks, line_config)` with the complete returned track list.
6. `detect_events()` normalizes the line, calls `_events_for_track()` for every supplied track, extends a local `events` list, sorts by timestamp and runtime track ID, and returns that list.
7. After the frame loop finishes, `print_event_table(latest_events)` prints exactly the last regenerated `latest_events` list.

## Evidence

### Invocation frequency and aggregation

`detect_events()` is called from the full pipeline inside the `while True` frame loop immediately after each `tracker.update(...)`. The target variable is reassigned, not appended. At end-of-run, `print_event_table(latest_events)` prints only that most recent regenerated result.

This proves Card 9 is called every frame with the full track list, and also proves the final table is not an accumulating event store in the harness.

### Runtime track lifecycle

Runtime tracks are created by `create_track()` with state `TENTATIVE`, a UUID `runtime_track_id`, `last_seen_timestamp`, and an initial one-point `center_history`. `TrackV2.update()` appends one center to `center_history` on every matched update and calls `promote_if_ready()`. `promote_if_ready()` changes `TENTATIVE` to `ACTIVE` when `hit_count >= tentative_hits_to_activate`. `close_stale_tracks()` changes unmatched tracks to `CLOSED` after configured miss/lifetime thresholds. Closed tracks remain in `self.tracks`, and the harness continues passing `self.tracks` to Card 9 on later frames.

### Transition discovery

`_events_for_track()` copies the full `center_history`, computes side values for every point relative to the configured line, discards `ON` points from the compressed side sequence, and emits an event every time adjacent compressed sides differ. It always starts at index 0 of the track history. It does not read or write any processed-transition marker. It intentionally supports multiple crossings by emitting one event for every side change.

Event IDs are stable for the same track ID, line, transition index, event type, and direction. Therefore repeated invocations over unchanged input regenerate the same event IDs; new rows with unique transition indices come from additional side changes in the supplied geometry, not from the printing loop.

### Geometry validation method

For the vertical line used by the full harness (`point_a=[frame_w/2, 0]`, `point_b=[frame_w/2, frame_h]`), Card 9 computes:

```text
cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
```

Because `bx == ax`, side is determined by the x coordinate relative to the line: `px < ax` is side `A`, `px > ax` is side `B`, and `px == ax` is `ON` within epsilon. A side sequence `A, B, A, B` produces `ENTRY, EXIT, ENTRY`; `A, B, A, B, A` produces `ENTRY, EXIT, ENTRY, EXIT`.

Thus, if the duplicated runtime track's `center_history` reconstructs as `A → B → A → B`, Card 9 is doing what its current logic specifies: emitting multiple geometric crossings for one runtime track. If the source video contains no such physical re-crossing, the false events are supported by the runtime track geometry but not by the real-world video, which assigns ownership to upstream track geometry/association quality rather than to event storage or printing.

### Reproduction trace

A minimal trace with one two-point crossing and then repeated closed-track invocations produced the same event ID on every invocation, while the harness-style `latest_events = ...` reassignment would retain only one row:

```text
frame 0 state [('ACTIVE', [[390.0, 0.0]])] events []
frame 1 state [('ACTIVE', [[390.0, 0.0], [410.0, 0.0]])] events [('ENTRY', 'evt_7bc0351be343aea9')]
frame 2 state [('ACTIVE', None)] events [('ENTRY', 'evt_7bc0351be343aea9')]
frame 3 state [('CLOSED', 3.0)] events [('ENTRY', 'evt_7bc0351be343aea9')]
frame 4 state [('CLOSED', 3.0)] events [('ENTRY', 'evt_7bc0351be343aea9')]
```

This proves repeated analysis of the same unchanged completed track regenerates identical results, but does not by itself create duplicate rows in the final table unless a caller appends those results. The checked full harness does not append them.

## Fault ownership

- **Card 9 detector:** owns the stateless full-history rescanning behavior. This is intentional per the current spec/tests and is required for deterministic repeated calls, but it means Card 9 cannot suppress previously consumed transitions.
- **Card 8 runtime tracking:** owns the geometry supplied in `center_history`. If a single runtime track's history crosses the line more times than the real person did, Card 9 will faithfully emit those crossings because it is a pure geometry consumer.
- **Test harness:** owns calling Card 9 every frame with all tracks, including active and closed tracks. However, the harness overwrites `latest_events` and prints only once at the end, so it is not the source of duplicated final rows in the inspected pipeline.
- **Event aggregation/storage/printing:** no persistent aggregation or storage layer was found in the full harness. Printing iterates the supplied list once. It does not duplicate events.

## Proposed resolution

Do not change Card 9 as part of this investigation. For a later fix, choose one architectural direction explicitly:

1. **Preferred if Card 9 remains purely geometric/stateless:** fix Card 8 or its input association so each runtime track's `center_history` represents only the actual person trajectory. Under this contract, every side transition in the history is a legitimate event.
2. **If the runtime system needs incremental event emission:** add an event-consumption layer outside the pure detector that records emitted `(runtime_track_id, transition_index/event_id)` values and publishes only newly discovered events. This should not change `detect_events()` semantics unless the Card 9 contract is revised.
3. **If events must be timestamped at crossings:** extend the upstream track history to include per-point timestamps and update the Card 9 contract to timestamp each event from the transition segment instead of using `last_seen_timestamp` for all events in a track.

## Commands used

- `rg "def detect_events|detect_events\(|Card 9|EVENT TABLE|center_history|RuntimeTrack|CLOSED|ACTIVE" -n .`
- `sed -n '1,220p' events/detector.py`
- `sed -n '580,760p' test_tracking_v2_pipeline.py`
- `sed -n '1,180p' track/tracker.py`
- `sed -n '1,120p' track/lifecycle.py`
- `python - <<'PY' ... minimal repeated-invocation trace ... PY`
