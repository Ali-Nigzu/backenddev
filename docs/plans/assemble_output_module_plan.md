# Assemble Output Module Implementation Plan

## 1. Verified baseline and planning scope

The planning work started from local branch `work` at commit
`d1893d6a4dbef683ee9c86a01111279194757cc3` (`Move DetectionBatch iteration
into Track`). `git fetch origin --prune` was attempted first, but this checkout
has no `origin` remote, so no newer remote head could be fetched. The local
history, working tree, and required pipeline files were then verified before
creating `codex/plan-assemble-output-module` from that commit. A future
implementation must repeat the fetch and baseline check in a checkout with the
Ali-Nigzu/backenddev remote and base its branch on the latest pipeline commit,
not blindly on the commit recorded here.

This document is the only deliverable. It plans, but does not implement,
Assemble, tests, fixtures, or runner changes.

## 2. Current pipeline and observed contracts

The production flow at the verified baseline is:

```text
video -> FrameBatch -> Detect -> DetectionBatch -> Track -> TrackingState
      -> Event -> EventBatch
EventBatch + FrameBatch -> Demographic -> DemographicsBatch
EventBatch + DemographicsBatch -> manual runner join
```

`events/event.py` iterates `TrackingState.tracks` in input order and appends each
track's crossings in path-discovery order. It returns the following exact shape:

```text
EventBatch = {"events": list[Event]}
Event = {
    "track_id": non-empty str,
    "timestamp": finite float,
    "event_type": int,  # 1 entry, 0 exit
    "best_crop": {
        "frame_id": non-empty str,
        "bbox": {"x1": float, "y1": float, "x2": float, "y2": float},
    },
}
```

The Event package requires these fields but does not reject additional mapping
keys. Event ordering is therefore meaningful and must not be changed downstream.

`demographics/demographic.py` validates the Event records, collapses them by
`track_id`, rejects conflicting crops for a repeated track, and sorts unique
tracks by `(timestamp, track_id)` for inference. It returns:

```text
DemographicsBatch = {"results": list[DemographicsResult]}
DemographicsResult = {
    "track_id": str,
    "age": int,
    "sex": int,  # 1 male, 0 female
}
```

Thus result order deliberately differs from Event order and there is one result
per unique event track, not necessarily one per Event. With no Events,
`Demographic.__call__` returns `{"results": []}` immediately.

The model rounds its finite converted prediction to an `int`, and validates that
checkpoint `max_age` exceeds `min_age`, but it neither clamps nor enforces a
public minimum/maximum on the resulting age. No current production age-bucket
mapping was found. Repository-wide searches also found no existing `event_id`,
`OutputBatch`, or `model_id` convention to reuse.

### Current manual join to remove

`test_tracking_v2_pipeline.py:171-184` defines
`join_events_and_demographics`. It indexes results by track ID, then emits an
`{"events": [...]}` object containing `track_id`, raw `age`, `sex`, `timestamp`,
and `event_type`. `main()` calls it at the current lines 211-214 and writes that
manual result. The helper silently overwrites duplicate demographics, raises an
uncontextualized `KeyError` for a missing result, and has no unused-result,
age-bucket, or event-ID handling. This whole helper—not merely its comprehension—
will be deleted in the implementation.

## 3. Locked Assemble API and output

The public API will be exactly:

```python
from assemble import Assemble

output_batch = Assemble()(
    event_batch,
    demographics_batch,
)
```

`Assemble()` takes no constructor arguments. Its instances retain no input or
mutable per-call state. `__call__` accepts an EventBatch and DemographicsBatch and
returns a fresh plain-Python mapping:

```python
{
    "rows": [
        {
            "event_id": str,
            "event": int,
            "timestamp": float,
            "sex": int,
            "age_bucket": int,
        }
    ]
}
```

`rows` is the only OutputBatch field, and the five shown keys are the only
OutputRow fields. In particular, `track_id`, raw `age`, `best_crop`, bbox,
frame ID, confidence, diagnostics, and source metadata are excluded.
`event_type` is renamed to `event`, without a duplicate field.

`model_id` is explicitly outside Assemble: it will not appear in construction,
calling, configuration, validation, OutputBatch, OutputRow, or the runner call.
It remains the responsibility of a future `Analyse()` layer.

## 4. Join, ordering, and cardinality design

`Assemble.__call__` will validate both batches, then `_index_demographics` will
build one `dict[str, Mapping]` keyed by `DemographicsResult["track_id"]` in a
single pass. Each Event is then resolved by
`demographics_by_track[event["track_id"]]`; list positions are never used as a
join key and no nested scan is allowed.

Events are enumerated exactly once in `EventBatch.events` input order. Assemble
does not sort, deduplicate, merge entry/exit pairs, or reorder from demographic
result order. Each valid Event appends exactly one row, so:

```text
len(OutputBatch.rows) == len(EventBatch.events)
```

If one track has repeated Events, every Event produces a distinct row in its
original relative position, while all those rows reuse that track's one sex and
computed age bucket. Identical-looking Events are not collapsed.

After all records are validated, track-set consistency is strict:

* a missing Event track raises `AssembleInputError` from `assemble/assemble.py`
  with `EventBatch.events[i] track_id <id> has no matching demographic result`;
  no partial OutputBatch is returned;
* `_index_demographics` rejects a second result for a track with
  `Duplicate demographic result for track_id <id>` rather than choosing first or
  last;
* any demographic track absent from the Event track set raises
  `Unused demographic result for track_id <id>`. This matches the current
  Demographic stage, which selects results exclusively from unique Event tracks,
  and catches cross-run/mismatched batches;
* empty Events plus empty results returns exactly `{"rows": []}`; empty Events
  plus non-empty results is rejected by the unused-result rule. This aligns with
  Demographic's empty-Event return while retaining strict batch consistency.

## 5. Age mapping and validation

No authoritative bucket helper exists at the baseline, so the small private
`_age_to_bucket(age: int) -> int` helper will live only in
`assemble/assemble.py`. It will implement the locked mapping literally:

| Raw age | `age_bucket` |
|---|---:|
| 0-4 | 0 |
| 5-13 | 1 |
| 14-25 | 2 |
| 26-45 | 3 |
| 46-65 | 4 |
| 66 and above | 5 |

Validation occurs before bucketing. An age must be a built-in Python `int` and
must not be `bool`; strings, floats (including `12.0`), NumPy integers, and other
coercible values are rejected rather than converted. Negative integers are
rejected, not clamped. Because Demographic currently declares no output ceiling,
Assemble adds no arbitrary upper bound: every valid integer of 66 or greater maps
to bucket 5. The emitted value is a built-in `int`, and raw age is never copied
to a row.

## 6. Exact deterministic event ID

The exact format will be:

```text
event-{event_index}-{digest12}
```

`event_index` is the zero-based index in the original EventBatch. `digest12` is
the first 12 lowercase hexadecimal characters of a standard-library SHA-256 over
UTF-8 JSON with separators `(',', ':')` for this ordered array:

```python
[event["track_id"], int(event["event_type"]), float(event["timestamp"]).hex()]
```

The array representation needs no key sorting; JSON's normal Unicode handling
will be explicitly fixed (`ensure_ascii=True`) so the byte sequence is stable.
`float.hex()` provides an unambiguous stable representation after timestamp
validation/conversion. `_create_event_id(event, event_index)` will implement this
format with only `json` and `hashlib` from the standard library.

For the same ordered EventBatch, stable source fields, index, serialization, and
digest produce the same string. Within one OutputBatch the embedded decimal
index differs for every row, which proves uniqueness even if two Events have the
same track/type/timestamp or their truncated digests collide. The digest avoids
unnecessary disclosure of the track ID, while the index keeps the scheme compact
and deliberately batch-scoped. No clock, UUID4, database, model ID, global
counter, process address, or external state participates. This is not presented
as a permanent distributed identity scheme; later persistence may own that.

## 7. Validation and error model

The compact package will use `AssembleInputError(ValueError)` defined privately
in `assemble/assemble.py`; a separate exception module is not justified for one
public operation and one input-error category. Only `Assemble` is exported.

Private helpers will be `_require_mapping`, `_require_fields`,
`_validate_event_batch`, `_validate_demographics_batch`,
`_index_demographics`, `_age_to_bucket`, and `_create_event_id`. They will give
errors the full batch path and index. Mapping contracts follow current Event and
Demographic conventions: required keys are enforced while unrelated additional
input keys are tolerated; output keys remain exact.

Event validation will require an EventBatch mapping with list `events`, each
Event a mapping containing `track_id`, `timestamp`, `event_type`, and
`best_crop`; `track_id` must be a non-empty built-in string; timestamp must be a
finite built-in `int`/`float` but not bool; and `event_type` must be a built-in
integer exactly 0 or 1. Assemble will confirm `best_crop` is a mapping containing
`frame_id` and `bbox`, but will not repeat Demographic's crop geometry validation
because it neither consumes nor emits crop data and the immediately preceding
stages already enforce it.

Demographic validation will require a mapping with list `results`; every result
must be a mapping containing `track_id`, `age`, and `sex`; track ID is a non-empty
built-in string; age follows Section 5; and sex is a built-in integer exactly 0
or 1. Duplicate IDs fail while indexing. Errors identify
`DemographicsBatch.results[i]` and its invalid field. Missing, duplicate, and
unused join failures use the exact messages in Section 4. No broad exception is
caught; exception chaining is used only if a lower-level serialization error is
translated into the input error.

Rows explicitly construct built-in `str`, `int`, and `float` values. The return
contains no NumPy scalar, model, or borrowed mutable mapping, and Assemble retains
no input reference after return.

## 8. Package structure and exact file actions

### Files to create

| File | Exact future implementation action |
|---|---|
| `assemble/__init__.py` | Import `Assemble` from `.assemble` and set `__all__ = ["Assemble"]`; export nothing else. |
| `assemble/assemble.py` | Define the private validation/index/bucket/ID helpers, private `AssembleInputError`, and zero-state callable `Assemble`; perform the linear join and return the exact OutputBatch. |

`assemble/exceptions.py` will **not** be created; separating one private input
exception adds layout without clarifying the API.

### Files to modify

| File | Exact future implementation action |
|---|---|
| `test_tracking_v2_pipeline.py` | Import `Assemble`; update module/CLI wording to include Assemble; rename the JSON constant/argument variables as described below; delete `join_events_and_demographics` at current lines 171-184; replace its current call with `output_batch = Assemble()(event_batch, demographics_batch)`; pass that exact object to `write_json`; print the OutputBatch once as the sole structured console object; retain replay generation and concise scalar completion diagnostics. |
| `pyproject.toml` | Append `"assemble"` to the explicit `[tool.setuptools].packages` list, which is required because this project does not use automatic discovery. |

`.gitignore` will not change: it already ignores the entire `output/` directory.
No active documentation currently names the old CLI flag, so no other doc needs
an implementation edit. Runtime behavior under `detect/`, `track/`, `events/`,
`demographics/`, and `contracts/frame_batch.py` must not change.

## 9. Integration runner and final output decision

The final visible stage sequence in `main()` will be:

```python
event_batch = Event(tracking_state, LINE_CONFIG)
validate_event_best_crops(event_batch, frame_batch)
demographics_batch = Demographic()(event_batch, frame_batch)
output_batch = Assemble()(event_batch, demographics_batch)
draw_replay(frame_batch, detection_batch, tracking_state, replay_path, fps, frame_size)
write_json(output_batch, output_batch_path)
print(json.dumps(output_batch, sort_keys=True))
```

The existing `join_events_and_demographics` definition and its `final_output =
join_events_and_demographics(...)` call are deleted. There will be no inline
`demographics_by_track` replacement in the runner; joining belongs solely to
Assemble.

Rename `DEFAULT_EVENTS_OUTPUT` to `DEFAULT_OUTPUT_BATCH_PATH`, change its value
from `output/events_with_demographics.json` to the semantically accurate single
file `output/output_batch.json`, and replace `--events-output` with
`--output-batch`. Rename `events_output_path` accordingly and update help text.
This small breaking CLI correction prevents the locked `rows` object from being
mislabelled as Events and avoids creating both old and new files.

The JSON file will contain exactly the returned OutputBatch. No EventBatch,
DemographicsBatch, raw ages, tracks, detections, diagnostics, or enriched Events
will be separately saved or dumped. Scalar progress/completion lines may remain,
but the single `json.dumps(output_batch, ...)` line is the only structured data
printed.

Replay video remains a separate non-structured artifact. Line configuration,
drawing, annotations, codec, frame order, resolution, timestamps, and replay path
are unchanged. Assemble has no knowledge of FrameBatch, OpenCV, video,
LineConfig, drawing, or filesystem paths.

## 10. Efficiency design

For `D` demographic results and `E` Events, validation/indexing takes one
demographic pass, set consistency and row production take linear passes, and no
nested lookup or sorting occurs. Expected time is `O(D + E)` and additional
memory is `O(D + E output)`. The implementation builds the lookup once, emits
one row collection, and avoids deep copies, pandas, NumPy, database access,
per-row logging, input retention, process globals, mutable counters, background
work, and intermediate duplicate row collections.

## 11. Focused implementation validation

Use focused ephemeral Python invocations (not a committed test or temporary
script) to assert:

1. one matching Event/result yields the exact five-field row and only `rows` at
   batch level;
2. two Events for one track yield two rows with shared sex/bucket and different
   event IDs;
3. interleaved tracks and demographics in reverse order still join by track and
   preserve Event order;
4. ages map exactly as `0->0`, `4->0`, `5->1`, `13->1`, `14->2`, `25->2`,
   `26->3`, `45->3`, `46->4`, `65->4`, `66->5`, and `90->5`;
5. negative age, bool, float, string, and NumPy integer fail; 0 and a large
   built-in integer succeed under the documented rules;
6. only built-in integer 0/1 is accepted for each of sex and event type and is
   preserved under `sex`/`event`;
7. bool/non-finite timestamps and empty track IDs fail with indexed paths;
8. missing, duplicate, and unused demographics each raise
   `AssembleInputError` with the locked contextual message and no partial result;
9. empty/empty returns `{"rows": []}`, while empty/non-empty fails as unused;
10. repeated calls on the same input yield identical IDs; all IDs are strings and
    unique; identical Event records at different indexes get distinct IDs;
11. output values are plain Python primitives, inputs are unmodified, and an
    instance retains no input state.

Then run `python -m compileall assemble test_tracking_v2_pipeline.py` and the
project's available automated checks. Finally use `rg` to ensure the removed
helper, raw-age output, old CLI flag/path, and manual lookup no longer occur in
the runner.

## 12. Full pipeline validation

Run the real video path end to end:

```text
video -> FrameBatch -> Detect -> DetectionBatch -> Track -> TrackingState
-> EventBatch -> DemographicsBatch -> Assemble -> OutputBatch
```

Instrument only locally if necessary, removing instrumentation before commit,
to confirm Assemble is invoked exactly once. Confirm row count equals Event
count; every row's key set is exactly `event_id`, `event`, `timestamp`, `sex`,
and `age_bucket`; raw age, track ID, best crop, and model ID are absent; IDs are
unique; Event order is retained; `output/output_batch.json` exactly equals the
returned object; no other structured file is produced; and only that object is
printed as structured data. Open/probe the replay to confirm it is playable and
retains configured line/tracking annotations, codec, dimensions, frame order,
and timestamps. Search the committed diff to ensure Detect, Track, Event,
Demographic, and FrameBatch runtime files are untouched.

## 13. Future implementation order

1. Fetch and verify the latest pipeline branch and capture its starting SHA.
2. Capture current integration output and replay characteristics for comparison.
3. Reinspect and freeze current EventBatch and DemographicsBatch contracts.
4. Repeat the repository-wide age-bucket and ID convention searches.
5. Lock the exact ID serialization/format from Section 6.
6. Add the two-file Assemble package and explicit package metadata entry.
7. Run the boundary, primitive-type, and output-shape validations.
8. Validate strict track-ID joining, repeated Events, and ordering.
9. Validate missing, duplicate, unused, and empty consistency behavior.
10. Import and call Assemble once in the runner.
11. Delete the manual helper/call and switch to the sole OutputBatch JSON.
12. Run the real pipeline and compare row/event cardinality.
13. Confirm replay video behavior is unchanged.
14. Search for obsolete manual join, raw output, and old output-path logic.
15. Remove all temporary validation artifacts and inspect the final diff.
16. Commit only focused implementation changes.

## 14. Rollback points

Keep the future work separable into three small commits/revert points:

1. **Add Assemble package** — package files, metadata, and focused direct
   validation complete.
2. **Integrate Assemble into pipeline runner** — import/call and exact
   OutputBatch writing, with replay comparison complete.
3. **Remove manual join and old structured output** — delete helper and old
   path/flag/output wording after end-to-end verification.

If a stage fails, revert only its commit and retain the last verified pipeline;
do not compensate by modifying Detect, Track, Event, Demographic, or FrameBatch.

## 15. Final acceptance checklist

- [ ] Work is implementation-only when this plan is later executed; no second plan, standalone runner, permanent broad test suite, fixtures, or reports are added.
- [ ] `from assemble import Assemble` exports only the no-argument `Assemble` class.
- [ ] The call accepts EventBatch and DemographicsBatch and returns only `{"rows": [...]}`.
- [ ] Every row has exactly `event_id`, `event`, `timestamp`, `sex`, and `age_bucket` as plain Python primitives.
- [ ] `model_id` is absent and reserved for future Analyse.
- [ ] Join uses track ID, never position, and supports repeated Events per track.
- [ ] The locked six-bucket integer mapping and strict age validation are exact; raw age is absent.
- [ ] `event-{index}-{digest12}` follows the exact serialization and is deterministic and batch-unique.
- [ ] Event input order and one-row-per-Event cardinality are preserved without sorting or deduplication.
- [ ] Missing, duplicate, and unused demographics fail with `AssembleInputError` and contextual messages.
- [ ] Empty/empty returns empty rows; empty/non-empty fails as unused.
- [ ] `join_events_and_demographics` and its call are deleted; no manual/inline join remains.
- [ ] The runner calls Assemble exactly once and saves/prints only the returned OutputBatch as structured data.
- [ ] Only `output/output_batch.json` is created; old structured output and CLI flag are removed.
- [ ] Replay behavior and all Detect, Track, Event, Demographic, and FrameBatch runtime behavior remain unchanged.
- [ ] Complexity remains `O(D + E)` time and `O(D + E output)` memory.
- [ ] Focused and real-pipeline validation pass and temporary artifacts are removed.
- [ ] The implementation diff contains only the two Assemble files, `pyproject.toml`, and `test_tracking_v2_pipeline.py`.
