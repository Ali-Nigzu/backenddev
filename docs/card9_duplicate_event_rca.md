# Card 9 Duplicate Event Emission — Root Cause Analysis

## Follow-up proof status

The requested proof for runtime track `1ac37490-5e87-484e-a267-0ceaac7bea74` cannot be completed from the artifacts present in this checkout. The repository does not contain the source video (`videoplayback.mp4`), any persisted tracker trace, any event-table log containing that runtime track ID, or any saved `center_history` for that runtime track. The local environment also lacks `cv2`, so the full video harness cannot currently be executed even if the video were supplied.

This is an evidence boundary, not a conclusion about the offending geometry. The previous hypothesis that Card 8 produced an oscillating `center_history` remains plausible from code inspection, but it is not proven for the named runtime track without the missing runtime data.

Commands run to verify the boundary:

```text
find /workspace -iname 'videoplayback.mp4' -o -iname '*.mp4' -o -iname '*event*' -o -iname '*trace*'
python - <<'PY'
from pathlib import Path
print('video exists', Path('videoplayback.mp4').exists())
try:
 import cv2
 print('cv2 ok')
except Exception as e:
 print('cv2 unavailable', type(e).__name__, e)
PY
```

Observed output:

```text
/workspace/backenddev/events
/workspace/backenddev/docs/card9_duplicate_event_rca.md
/workspace/backenddev/test_events_card9_geometry.py
video exists False
cv2 unavailable ModuleNotFoundError No module named 'cv2'
```

## What is proven from the repository

### Root cause that can be proven statically

Card 9 is a stateless full-history geometric detector. For every invocation, `detect_events()` evaluates every supplied track from the beginning of its `center_history`. `_events_for_track()` computes the side of every point relative to the configured line, discards `ON` points, and emits one event every time adjacent compressed side values differ. It does not store or consult an already-emitted transition marker.

The full video harness calls `detect_events(tracks, line_config)` once per processed frame with the complete `tracker.tracks` list. It reassigns `latest_events` rather than appending to it, and it prints only the final regenerated list after the video loop finishes. Therefore the inspected harness does not create final duplicate rows by accumulating old event lists.

The same timestamp on multiple events from one track is expected under the current event model because Card 9 assigns every event for that track to `track.last_seen_timestamp`, not to a per-transition timestamp.

### Code path

1. `test_tracking_v2_pipeline.py::__main__` calls `main()` after self-tests.
2. `main()` opens the video, builds a vertical center-line config, constructs `TrackV2`, and enters the frame loop.
3. Each frame is passed through detection, embedding, observation building, and `tracker.update(observations_by_ts)`.
4. `TrackV2.update()` appends matched observation centers into `track.center_history`, promotes tracks, closes stale tracks, and returns `self.tracks`.
5. The harness calls `latest_events = detect_events(tracks, line_config)` inside the frame loop.
6. `detect_events()` calls `_events_for_track()` for every supplied track, extends a local list, sorts the list, and returns it.
7. After the loop, `print_event_table(latest_events)` prints the final regenerated event list once.

## Required proof that is still missing

The follow-up request asks for a complete reconstruction of the offending track:

- every `center_history` point;
- frame index and timestamp for every point;
- signed cross product, x coordinate, and side for every point;
- every transition index;
- every emitted event mapped to its transition;
- visual validation against the source frames.

Those deliverables require the actual offending runtime data. The current `RuntimeTrackV2` model does not store per-point frame indices or per-point timestamps in `center_history`; it stores only `[x, y]` points plus track-level timestamps. Per-point frame/timestamp reconstruction therefore requires tracing during the video run or a previously persisted trace. No such trace is present in this checkout.

## Production geometry needed for the reconstruction

For the vertical counting line reported in the follow-up (`x ≈ 315`), production Card 9 computes:

```text
cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
```

When the line is vertical (`ax == bx`) and `by > ay`, this reduces to side being determined by `px - ax`:

- `x < 315` => negative cross => side `A`;
- `x > 315` => positive cross => side `B`;
- `x == 315` within epsilon => side `ON`.

Therefore an emitted sequence `ENTRY, EXIT, ENTRY, EXIT` for a single track requires the compressed non-`ON` side sequence to contain at least five alternating side runs, for example:

```text
A -> B -> A -> B -> A
```

or the reverse sequence:

```text
B -> A -> B -> A -> B
```

This statement follows directly from Card 9 control flow, but it still does not prove that the named track contains that sequence.

## Minimal trace that proves repeated invocation is not enough

A synthetic trace with one unchanged crossing track was run to isolate invocation behavior. It shows repeated calls over an unchanged active/closed track regenerate the same event ID each time. Repeated invocation alone does not create multiple final rows unless a caller appends returned lists; the inspected full harness does not append.

```text
frame 0 state [('ACTIVE', [[390.0, 0.0]])] events []
frame 1 state [('ACTIVE', [[390.0, 0.0], [410.0, 0.0]])] events [('ENTRY', 'evt_7bc0351be343aea9')]
frame 2 state [('ACTIVE', None)] events [('ENTRY', 'evt_7bc0351be343aea9')]
frame 3 state [('CLOSED', 3.0)] events [('ENTRY', 'evt_7bc0351be343aea9')]
frame 4 state [('CLOSED', 3.0)] events [('ENTRY', 'evt_7bc0351be343aea9')]
```

## Answer to the follow-up question

Does runtime track `1ac37490-5e87-484e-a267-0ceaac7bea74` actually contain enough geometric evidence for:

```text
ENTRY
EXIT
ENTRY
EXIT
```

**Unknown from the available repository artifacts.**

The repository proves how Card 9 would generate those four events if the track history oscillated across `x ≈ 315`, and it proves the final table is not duplicated by the inspected harness's event-list append behavior. It does **not** contain the named track's complete `center_history` or source frames, so it cannot prove whether the named track actually oscillated around the counting line, whether Card 8 generated invalid geometry, or whether the visual source video contradicts the runtime geometry.

## Evidence required to complete the investigation

To complete the requested proof without modifying production behavior, capture a one-run trace with the following fields for every update of the offending runtime track:

```text
runtime_track_id
state
frame_idx
timestamp
center_history_index
x
y
last_seen_timestamp
last_matched_frame_index
last_unmatched_frame_index
closed_timestamp
event_id
event_type
direction
transition_index
supporting_positions
```

Then compute the production Card 9 side sequence from that trace and compare the transition frames against the source video frames. Without that data, assigning the final discrepancy to Card 8 invalid geometry versus Card 9 misinterpretation would be speculation.
