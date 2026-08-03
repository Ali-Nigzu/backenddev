# Full repository deletion and compression plan

## 1. Scope, verified base, and non-goals

This is an implementation plan only. The cleanup must preserve the current algorithms and primitive dictionary/list/NumPy representations while reducing the repository to the live Detect → Track → Event → Demographic → Assemble pipeline and its single video runner. Data-type redesign, memory/copy optimisation, `Analyse()`, input/output boundary validation, deployment packaging, and every GCP concern are explicitly deferred.

The planning checkout was inspected after attempting `git fetch origin --prune`. This checkout has **no configured Git remote**, so no fetch was possible. The latest locally available commit containing Assemble and the functioning call graph is therefore the supplied commit:

* verified source branch represented by the checkout: expected implementation lineage `codex/plan-new-assemble-production-module` (the checkout itself was initially named `work`);
* verified source commit: `f89550d9e8ae48cefbf7cd96f06778cc4b8d2997` (`Add final Event and demographic assembly`);
* planning branch: `codex/plan-full-repository-deletion`;
* tracked files at that commit: **26**.

Before implementing, repeat `git fetch origin --prune` in a clone with `origin`, inspect `origin/codex/plan-new-assemble-production-module`, and stop/rebase this plan if its tip is newer and contains the completed pipeline. Do not silently apply this inventory to a moved tree.

## 2. Full current tracked tree

```text
backenddev/
├── .gitignore
├── assemble/
│   ├── __init__.py
│   └── assemble.py
├── contracts/
│   ├── __init__.py
│   └── frame_batch.py
├── data/
│   └── samples/
│       └── 1000040807.jpg
├── demographics/
│   ├── __init__.py
│   ├── _mivolo/
│   │   ├── __init__.py
│   │   └── model/
│   │       ├── __init__.py
│   │       ├── cross_bottleneck_attn.py
│   │       └── mivolo_model.py
│   ├── demographic.py
│   ├── exceptions.py
│   ├── model.py
│   └── preprocessing.py
├── detect/
│   ├── __init__.py
│   ├── detect.py
│   └── yolov10n.pt
├── events/
│   ├── __init__.py
│   └── event.py
├── pyproject.toml
├── test_tracking_v2_pipeline.py
└── track/
    ├── __init__.py
    ├── config.py
    ├── matcher.py
    └── tracker.py
```

`demographics/demographicweights.pth` is a required but currently ignored runtime file, not one of the 26 tracked files. The old split checkpoint parts and assembly script were removed by the current commit. There is no tracked root `LICENSE`, `NOTICE`, README, test directory, notebook, generated output, cache, build artifact, or alternate runner.

## 3. Live call and import graph

### Current calls

The current runner decodes every frame once into RGB, then executes:

```text
build_frame_batch_from_video
  → Detect()(FrameBatch)
  → validate_detection_batch                         [delete]
  → Track(TrackingState, DetectionBatch)
  → Event(TrackingState, LINE_CONFIG)
  → validate_event_best_crops                        [delete]
  → Demographic()(EventBatch, FrameBatch)
  → Assemble()(EventBatch, DemographicsBatch)
  → draw_replay
  → write_json
  → stdout JSON
```

The implementation will retain this behavior but make the public stages consistently callable as `Detect()`, `Track()`, `Event()`, `Demographic()`, and `Assemble()`; `Track` and `Event` may remain functions only if changing them to zero-argument callable objects would alter more than the public shape. The required visible runner sequence is one call per stage, with `TrackingState = {"tracks": []}` created immediately before Track.

### Direct and transitive imports

```text
test_tracking_v2_pipeline.py
├── stdlib: argparse, json, pathlib.Path, typing.Any
├── third party: cv2, numpy
├── assemble.__init__ → assemble.assemble.Assemble
│   └── stdlib: collections.abc.Mapping, hashlib, json, math.isfinite, typing.Any
├── detect.__init__ → detect.detect.Detect
│   ├── stdlib: pathlib.Path
│   ├── third party: numpy, ultralytics.YOLO
│   └── contracts.validate_frame_batch                         [remove]
├── track.__init__ → track.tracker.Track
│   ├── tracker → collections.abc.Mapping, math.isfinite       [validation imports remove]
│   ├── tracker → track.config                                 [inline]
│   └── tracker → track.matcher → math, typing, track.config   [inline]
├── events.__init__ → events.event.Event
│   └── stdlib: collections.abc.Mapping, math.hypot/isfinite, typing.Any
└── demographics.__init__ → demographics.demographic.Demographic
    ├── contracts.FrameBatchError/build_frame_lookup           [remove]
    ├── demographics.exceptions                                [remove]
    └── demographics.model._MiVOLOModelRunner                  [absorb]
        ├── stdlib: importlib, math, threading, collections, dataclasses, pathlib, typing
        ├── dynamic: numpy, torch
        ├── demographics.preprocessing                         [absorb]
        └── demographics._mivolo.model.mivolo_model
            ├── torch, torch.nn, timm.layers, timm.models.volo.VOLO
            └── demographics._mivolo.model.cross_bottleneck_attn
                └── torch, torch.nn and timm layer helpers
```

After cleanup, unused imports disappear with validation (`Mapping`, `Any`, `isfinite`, and NumPy where no algorithmic use remains). Package `__init__.py` files export only the named production callable; remove every compatibility alias and all private/public exports other than `Detect`, `Track`, `Event`, `Demographic`, and `Assemble`.

### Dynamic and data loads

* `detect/detect.py` resolves `Path(__file__).with_name("yolov10n.pt")` and passes it to `ultralytics.YOLO`; retain tracked `detect/yolov10n.pt` (5.6 MB).
* `demographics/model.py` currently resolves `Path(__file__).with_name("demographicweights.pth")`, dynamically imports `torch`, `numpy`, and `demographics._mivolo.model.mivolo_model`, and calls `torch.load(..., map_location="cpu")`. Move this exact path and load behavior into `demographics/demographic.py`.
* `demographics/demographicweights.pth` is absent/ignored at the base even though runtime requires it. The cleanup must reconstruct it from the verified historical checkpoint or obtain the exact verified production checkpoint, verify its checksum privately, add it as a tracked runtime asset, and remove its ignore rule. Never substitute a different checkpoint.
* `data/samples/1000040807.jpg` has no live import or dynamic open and is deleted.
* The runner opens only its CLI video path and its replay/JSON output paths. No video is retained in Git.

## 4. Exact current-file classification

Every one of the 26 tracked paths is classified below. “Absorb” is still a `DELETE` classification for the source path.

### KEEP unchanged (5)

| Classification | Path | Kind and justification |
|---|---|---|
| KEEP | `detect/yolov10n.pt` | Runtime detector weights loaded by `Detect`. |
| KEEP | `demographics/_mivolo/__init__.py` | Runtime package boundary for isolated MiVOLO-derived architecture. |
| KEEP | `demographics/_mivolo/model/__init__.py` | Runtime package boundary for its relative architecture import. |
| KEEP | `demographics/_mivolo/model/cross_bottleneck_attn.py` | Substantial MiVOLO cross-attention architecture used by the forward path; keeping it isolated preserves attribution/readability. |
| KEEP | `demographics/_mivolo/model/mivolo_model.py` | Substantial D1-224 neural architecture used to instantiate the checkpoint; not sensible to bury in orchestration. Preserve its copyright header. |

### MODIFY or rename (12)

| Classification | Path | Exact action |
|---|---|---|
| MODIFY | `.gitignore` | Keep cache/output/video ignores, add coverage/build/dist/egg-info/notebook-checkpoint ignores, and remove the demographic checkpoint ignore because that required weight becomes tracked. |
| MODIFY | `assemble/__init__.py` | Export only `Assemble`. |
| MODIFY | `assemble/assemble.py` | Delete validation and retain lookup, ordered join, age bucket, deterministic ID, row construction, and batch return. |
| MODIFY | `demographics/__init__.py` | Export only `Demographic`. |
| MODIFY | `demographics/demographic.py` | Remove contract validation; absorb frame lookup, preprocessing, model runner/loading, metadata/output conversion, and constants. |
| MODIFY | `detect/__init__.py` | Export only `Detect`. |
| MODIFY | `detect/detect.py` | Remove FrameBatch validation/import and iterate `frame_batch["frames"]` directly; preserve inference/output behavior. |
| MODIFY | `events/__init__.py` | Export only `Event`. |
| MODIFY | `events/event.py` | Remove schema validation while preserving stable-side/segment-crossing behavior and event construction/order. |
| MODIFY | `test_tracking_v2_pipeline.py` | Reduce to decode, initialization, five calls, replay drawing, JSON write, and stdout. |
| MODIFY | `track/__init__.py` | Change the sole import to `track.track.Track`; export only `Track`. |
| MODIFY/RENAME | `track/tracker.py` → `track/track.py` | Retain lifecycle/reducer logic, absorb matching/constants, delete validation, and expose the sole Track implementation at the intended path. |

### DELETE (9)

| Classification | Path | Exact reason/action |
|---|---|---|
| DELETE | `contracts/__init__.py` | Delete contract package/export completely. |
| DELETE/ABSORB | `contracts/frame_batch.py` | Delete all validation; absorb only the plain frame-ID lookup comprehension into `demographic.py`. |
| DELETE | `data/samples/1000040807.jpg` | Unused development sample; no live load. |
| DELETE | `demographics/exceptions.py` | Validation-only input exception and unnecessary exception hierarchy; model-load failures may use standard runtime exceptions with operational context. |
| DELETE/ABSORB | `demographics/model.py` | Absorb the live loader, inference, checkpoint metadata needed by the algorithm, and output conversion into `demographic.py`; delete validation-only checkpoint constraints/error wrappers. |
| DELETE/ABSORB | `demographics/preprocessing.py` | Absorb crop, letterbox, normalisation, missing-face tensor, and batch stacking; delete contract/type/shape validators. |
| DELETE | `pyproject.toml` | Packaging/development metadata is outside this root-run repository. |
| DELETE/ABSORB | `track/config.py` | Inline all seven active tracking constants at the top of `track/track.py`; there is no reason for a config object/file. |
| DELETE/ABSORB | `track/matcher.py` | Absorb the active historical-anchor and one-to-one matching helpers into `track/track.py`; it is a small single-consumer boundary. |

Counts over the current tree: **5 unchanged + 12 modified/renamed + 9 deleted = 26**. Five deleted files contribute active logic that is absorbed (`contracts/frame_batch.py`, `demographics/model.py`, `demographics/preprocessing.py`, `track/config.py`, `track/matcher.py`); the rest is removed outright.

## 5. New required runtime/legal files and exact final tree

The implementation must add exactly four tracked files: the already-required checkpoint plus three legal records. The repository currently has no root project license, so the cleanup must not invent or assert an owner-selected repository license. Ask the owner to license original code separately; that is not a reason to discard required third-party notices.

* `demographics/demographicweights.pth`: runtime checkpoint, exact production bytes/checksum.
* `THIRD_PARTY_NOTICES.md`: identify modified MiVOLO-derived files and the distributed Ultralytics YOLO weight; record upstream project, copyright, license, modifications, and source URL/version without historical development prose.
* `LICENSES/Apache-2.0.txt`: full Apache-2.0 text required for the MiVOLO-derived source distribution.
* `LICENSES/AGPL-3.0.txt`: full AGPL-3.0 text accompanying the distributed Ultralytics model asset. Confirm the specific weight provenance/license with counsel/upstream metadata before release; do not remove the notice merely because Ultralytics is installed externally.

The planned final count is **21 tracked files**: 17 current files retained (five unchanged and twelve modified/renamed) plus four additions. Nine current files are deleted, five of those are absorbed, and no module-local config remains.

```text
backenddev/
├── .gitignore
├── LICENSES/
│   ├── AGPL-3.0.txt
│   └── Apache-2.0.txt
├── THIRD_PARTY_NOTICES.md
├── assemble/
│   ├── __init__.py
│   └── assemble.py
├── demographics/
│   ├── __init__.py
│   ├── _mivolo/
│   │   ├── __init__.py
│   │   └── model/
│   │       ├── __init__.py
│   │       ├── cross_bottleneck_attn.py
│   │       └── mivolo_model.py
│   ├── demographic.py
│   └── demographicweights.pth
├── detect/
│   ├── __init__.py
│   ├── detect.py
│   └── yolov10n.pt
├── events/
│   ├── __init__.py
│   └── event.py
├── test_tracking_v2_pipeline.py
└── track/
    ├── __init__.py
    └── track.py
```

The only files beyond each primary source and weight are four substantial/package-boundary MiVOLO files. `mivolo_model.py` (D1-224/VOLO architecture) and `cross_bottleneck_attn.py` (cross-person/face attention) are hundreds of lines of third-party-derived model code and merit isolation for readability and attribution. Their two empty `__init__.py` files provide explicit deterministic package boundaries for relative and dynamic imports. No other Demographic support file is retained.

## 6. Consolidation by stage

### Detect

Retain `detect/__init__.py`, `detect/detect.py`, and `detect/yolov10n.pt`; delete no other Detect files because none exist. Remove `contracts.validate_frame_batch`, directly iterate `frame_batch["frames"]`, and preserve the zero-box branch, bbox clipping/positive-area condition, person class `0`, confidence conversion, center calculation, ID format, frame order, and output dictionaries. Inline at the top of `detect.py`: weight path, person class `(0,)`, confidence `0.25`, IoU `0.70`, max detections `300`, device `"cpu"`, and `verbose=False`. There is no image-size preset in current behavior, so do not add one.

### Track

Rename `tracker.py` to `track.py`, inline all of `matcher.py`, and inline config values: history window `7`, anchor exponent `1.0`, maximum anchor distance `100.0`, tie distance `20.0`, confirmation hits `3`, active timeout `30`, and tentative timeout `15`. Keep `_track_sort_key`, `_historical_anchor`, `_beats`, `_best_for_detection`, `_match_tier`, `_classify_track`, `_append_detection`, `_create_track`, `_next_numeric_track_id`, and `_process_frame` because they directly implement matching/lifecycle. Preserve empty detections, distance gating, deterministic one-to-one association/ties, birth/update/closure, path and best-crop updates. (The current implementation is historical-anchor greedy matching, not SciPy Hungarian assignment; do not introduce Hungarian or motion prediction during a behavior-preserving deletion.) Remove `Mapping`/`isfinite` when validation is gone.

### Event

Keep only module constants and the signed-distance, side/deadband, path-point, event construction, stable crossing, ordering, and public call logic in `event.py`. Read trusted mapping fields directly. Delete validation/copy wrappers but retain copies needed to preserve crop ownership and output primitives. Preserve zero-length-line protection and epsilon/deadband checks as algorithmic division-by-zero/crossing control flow, empty tracks/events, minimum path/stable-side thresholds, direction assignment, timestamp selection, best-crop reference, and current ordering.

### Demographic

Trace remains `Demographic.__call__` → unique event descriptors → required-frame lookup → crop/preprocess → `_MiVOLOModelRunner.predict` → lazy model load → `_mivolo` factory → torch forward → age/sex conversion. Move descriptor representation, direct lookup comprehension, RGB crop clamping, letterbox resize, ImageNet normalisation, missing-face channels, batch stacking, thread-safe lazy load, device selection, checkpoint read/state load, chunked inference, metadata used for output conversion, and output conversion into `demographic.py`.

Inline module constants there: checkpoint path, input size `224`, ImageNet mean/std, missing-face tensor behavior, CPU chunk `16`, CUDA chunk `64`, ignored `fds.` state prefix, expected three outputs, and device selection. Retain only checkpoint checks required to safely construct/load the specified model (file load success and state-load compatibility); remove general input contracts, schema checks, artificial “realistic parameter count” validation, generic dynamic import wrappers, and validation-specific exception translation. Preserve crop-exists/index-clamping checks, missing referenced frame handling needed to avoid invalid access, empty EventBatch, one result per first-seen unique track, inference result cardinality needed to pair results safely, and model-load failure as operational/algorithmic control flow.

Do not absorb `_mivolo/model/mivolo_model.py` or `cross_bottleneck_attn.py`: each is substantial executable architecture, its separation makes third-party provenance clear, and combining it would make orchestration unmanageable. Delete no architecture methods reached by the D1-224 forward pass; separately remove only provably training-only branches inside retained files if a baseline-backed reachability review proves they cannot affect `eval()` inference (otherwise leave their code untouched in this deletion phase).

### Assemble

Keep a compact `Assemble` implementation with direct `event_batch["events"]` and `demographics_batch["results"]` access, demographic lookup by `track_id`, ordered event iteration, missing-demographic operational branch, `_age_to_bucket`, `_create_event_id`, row creation, and `{"output": rows}` return. Preserve event ID hashing inputs/format, JSON canonicalisation, age buckets, primitive coercions, and unused-result behavior only if it affects successful current inputs (it should be removed as validation). Delete every validation helper and `AssembleInputError`.

### Runner

Retain `parse_args`, `build_frame_batch_from_video`, `create_video_writer` (or absorb its few lines into `draw_replay`), `draw_replay`, rename `write_json` to the precise `write_output_batch`, and `main`. Preserve the current three CLI values: positional input (default `videoplayback.mp4`), `--output` (default `output/tracking_replay.mp4`), and `--output-batch` (default `output/output_batch.json`). Retain video-open/writer-open failures, decode-end condition, FPS fallback, no-frame guard, BGR→RGB decode, contiguous RGB, RGB→BGR replay, line/track drawing, directory creation, and `finally` releases as operational/algorithmic safeguards. Delete `validate_detection_batch`, `validate_event_best_crops`, their calls, `typing.Any` annotations if no longer useful, all internal output assertions, and no summaries/debug modes (none currently exist).

## 7. Validation deletion inventory

Delete the following exact validation symbols/classes/imports and their error formatting:

* contracts: `FrameBatchError`, `_validate_image`, `_validate_timestamp`, `validate_frame_batch`, required-field sets, validation portion of `build_frame_lookup`, `Mapping`, `isfinite`, `Any`, and NumPy validation import;
* Detect: `from contracts import validate_frame_batch` and its call;
* Track: `_require_fields`, `_require_finite_number`, `_validate_bbox`, `_validate_centre`, `_validate_tracking_state`, `_validate_frame_detections`, `_validate_detection_batch`, `Mapping`, `isfinite`, and their calls. Retain `_classify_track` timeout decisions; replace its impossible negative-time contract error only with direct trusted behavior, without redesigning timestamps;
* Event: `_require_mapping`, `_require_fields`, `_finite_number`, `_point_from_mapping`, `_validate_bbox`, `_validate_track_point`, `_validate_inputs`, and validation-only `Mapping`/`Any`/`isfinite`. Fold direct field reads and required crop copying into the algorithmic helpers rather than recreating validators;
* Demographic: `DemographicError`, `DemographicInputError`, `DemographicModelError`, `_require_mapping`, `_require_fields`, `_finite_number`, `_validate_bbox`, `_validate_event_batch`, all image dtype/shape/contiguity validators in `frame_image`, `crop_body`, and `mivolo_input_from_body_crop`, generic checkpoint required-field/schema/type/finite/parameter-count validators, and validation-only imports/error translations;
* Assemble: `AssembleInputError`, `_require_mapping`, `_require_fields`, `_event_values`, validation portion of `_index_demographics`, `_validate_event`, duplicate/unused-result enforcement, `Mapping`/`isfinite`/`Any` used only for validation, and all formatted validation paths;
* runner: `validate_detection_batch`, `validate_event_best_crops`, and both calls.

Classify remaining conditions carefully: model/weight load success, state-dict compatibility needed for PyTorch to execute, empty detections/events, match/timeout/crossing decisions, demographic match during assembly, crop bounds/existence, invalid-index/division guards, video decode termination, video/writer opening, and OpenCV release are **RETAIN AS ALGORITHMIC OR OPERATIONAL CONTROL FLOW**. Assertions internal to retained `timm`/PyTorch architecture are **THIRD-PARTY — DO NOT MODIFY** unless the architecture is deliberately relicensed/refactored. All listed validation matches elsewhere are **DELETE**; there is no replacement contract/schema/type package.

## 8. Configuration and packaging disposition

The exhaustive config search finds only:

| Path/value | Decision |
|---|---|
| `track/config.py` and all seven values | **INLINE INTO OWNER** (`track/track.py`), then delete. |
| `pyproject.toml` | **DELETE**; do not inline or replace. |
| Detect literals/path listed in §6 | **INLINE** as `detect.py` constants. |
| Event four constants | **RETAIN MODULE-LOCALLY** in `events/event.py`, not a config object/file. |
| Demographic constants listed in §6 | **INLINE** as `demographic.py` constants. |
| Runner paths and `LINE_CONFIG` | **RETAIN MODULE-LOCALLY** in the runner because they own this one test scene/CLI. |

No root `config.py`, `config/`, `configs/`, `settings/`, `configuration/`, shared config, or module-local config file will exist. No future Analyse API is designed here.

`pyproject.toml` currently supplies project name/version, Python `>=3.12`, five dependency ranges (`numpy`, `torch`, `timm`, headless OpenCV, Ultralytics), and explicit setuptools package discovery. It specifies no build-system/backend table, pytest, formatter, lint, or coverage configuration. Delete it because execution is directly from the repository root, local directories are importable via `__init__.py`, the environment is assumed provisioned, package metadata/version/build/discovery are unused, and dependency/deployment management is deferred. Add no `setup.py`, `setup.cfg`, requirements file, Poetry, Hatch, Flit, or PDM replacement.

## 9. Repository-debris deletion

Delete the whole `contracts/` and `data/` directories, `pyproject.toml`, Track helper/config sources after absorption, and Demographic helper/exception sources after absorption. No current legacy/compatibility/v1/v2/old/backup match identifies a separate implementation; the only matches are MiVOLO/timm compatibility comments and checkpoint compatibility logic on the live inference path, which remain. Do not preserve old API aliases.

At implementation completion delete the entire `docs/` directory, including this plan, in the same focused cleanup (so the final tree/count above contains no docs). Keep only `test_tracking_v2_pipeline.py`; there are currently no other tracked tests/fixtures/notebooks. Delete any untracked `__pycache__/`, `*.pyc`, `.pytest_cache/`, coverage files, build/dist, `*.egg-info`, notebook checkpoints, output JSON/video, profiling data, and temporary baseline files, and maintain corresponding ignores. Do not commit the representative source video or baseline artifacts.

## 10. Baseline and post-cleanup verification

### Fixed baseline

First restore/provision the exact `demographics/demographicweights.pth` and select one fixed representative crossing video outside Git. Record its absolute path and SHA-256 privately. The current CLI command is exactly:

```bash
python test_tracking_v2_pipeline.py /absolute/path/to/fixed-representative.mp4 \
  --output /tmp/backenddev-baseline/tracking_replay.mp4 \
  --output-batch /tmp/backenddev-baseline/output_batch.json
```

Capture stdout and copy the JSON privately. Use a short uncommitted Python/OpenCV command to parse JSON and record OutputRow count, ordered full rows, Event IDs, events, timestamps, sex, and age buckets, and open the replay to record frame count, width, height, FPS, and fourcc/codec when available. Record the input/checkpoint/detector SHA-256 values and visually inspect at least crossing frames for the white line and green tracked boxes/labels. Baseline commands/scripts and artifacts stay under `/tmp` and are never added to the repository.

### Same post-cleanup run and equality

```bash
python test_tracking_v2_pipeline.py /absolute/path/to/fixed-representative.mp4 \
  --output /tmp/backenddev-after/tracking_replay.mp4 \
  --output-batch /tmp/backenddev-after/output_batch.json
```

Require byte-for-structure equality of parsed OutputBatch dictionaries and list order, including OutputRow count, Event IDs, event values, timestamps, sex values, and age buckets. Require identical replay frame count, dimensions, and FPS; compare codec where the backend reports it but treat backend reporting absence as an environment limitation, not an algorithm difference. Visually confirm the line and tracking overlay. The source must be decoded once and each stage invoked exactly once. The direct command must work from the root with no installation step and produce only replay, `output/output_batch.json` (when defaults are used), and stdout.

### Mandatory final searches/checks

```bash
test ! -d contracts
! rg -n "from contracts|import contracts|contracts\." .

rg -n "validate|validation|validator|_require|required_fields|InputError|ValidationError" \
  detect track events demographics assemble test_tracking_v2_pipeline.py
# Expected: no internal validation; inspect/classify any third-party architecture match.

test ! -f pyproject.toml
test ! -f setup.py
test ! -f setup.cfg
test ! -f config.py
test ! -d config
test ! -d configs
test ! -d settings

find . \( -path './.git' -o -path './.venv' \) -prune -o \
  \( -name 'test_*.py' -o -name '*_test.py' -o -name tests -o -name pytest.ini \
     -o -name .pytest_cache -o -name '*.ipynb' \) -print
# Expected sole match: ./test_tracking_v2_pipeline.py

test ! -d docs
git ls-files | wc -l                 # exactly 21
git status --short                   # no generated output/caches

python - <<'PY'
from detect import Detect
from track import Track
from events import Event
from demographics import Demographic
from assemble import Assemble
print(Detect, Track, Event, Demographic, Assemble)
PY

python -m compileall detect track events demographics assemble test_tracking_v2_pipeline.py
```

Also inspect `git ls-files` against the exact §5 tree, verify both model checksums, verify only five package exports, and remove compile caches afterward.

## 11. Implementation sequence and rollback points

1. In a clone with a remote, fetch/prune, verify the latest implementation tip, and reconcile any moved files before touching code.
2. Provision exact weights/video, hash them, run the baseline command, capture private JSON/stdout/replay metadata, and visually inspect overlays.
3. Reconfirm the live static/dynamic graph and all file classifications against that tip.
4. **Rollback point 1 — Simplify core modules:** consolidate Detect, rename/consolidate Track, simplify Event and Assemble, and run import/compile plus focused baseline comparisons after each stage.
5. **Rollback point 2 — Collapse Demographic inference:** absorb `model.py` and `preprocessing.py`, retain the two substantial architecture files/package boundaries, add and checksum the exact checkpoint, run inference comparison, and add legal notices/licenses.
6. **Rollback point 3 — Remove validation/contracts:** remove every listed validator/exception/import, absorb plain frame lookup, delete `contracts/`, and rerun imports/compile/full pipeline comparison.
7. **Rollback point 4 — Simplify runner:** remove validator calls/helpers, preserve only decode/direct calls/replay/output, and rerun the same CLI and metadata/output comparison.
8. **Rollback point 5 — Delete debris:** delete sample data, packaging, all non-runner tests (none now), docs including this plan, helper/config remnants, compatibility/development debris, caches, and generated/baseline files; tighten `.gitignore`.
9. Run every final search, exact-tree/count check, import smoke test, compile, full pipeline equality check, and overlay inspection.
10. Commit the focused cleanup. Do not mix data types, memory, Analyse, GCP, APIs, or deployment into any rollback commit.

## 12. Acceptance checklist

* [ ] Latest remote implementation commit was verified; any divergence from this 26-file inventory was explicitly reconciled.
* [ ] A private fixed-video baseline and weight/input hashes exist; nothing generated is committed.
* [ ] Exactly the §5 21-file tree remains after implementation.
* [ ] All current files received an exact KEEP/MODIFY/DELETE classification.
* [ ] Each retained executable/legal file has the justification above.
* [ ] Only `Detect`, `Track`, `Event`, `Demographic`, and `Assemble` are package exports; TrackingState stays a plain runner-created structure.
* [ ] Video is decoded once; stage ordering/call counts and RGB/BGR behavior are unchanged.
* [ ] Contracts are deleted, no contract imports remain, and no replacement schema/type package exists.
* [ ] All listed internal validators, validation exceptions/imports/assertions, and runner checks are removed.
* [ ] Algorithmic/operational empty, matching, lifecycle, crossing, join, crop/index, model-load, decode, divide-by-zero, and cleanup conditions remain.
* [ ] Detect is only `__init__.py`, `detect.py`, and exact weights, with presets inline.
* [ ] Track is only `__init__.py` and `track.py`, with matching/constants absorbed and behavior unchanged.
* [ ] Event is only `__init__.py` and `event.py`, with line config owned by the runner.
* [ ] Demographic orchestration/preprocessing/loading is one file plus exact weights; only the two substantial attributed architecture files and their package boundaries remain beyond it.
* [ ] Assemble contains only join/transformation behavior, including unchanged age buckets and event IDs.
* [ ] Runner contains only minimal CLI, decode/state/line, direct calls, replay/output/stdout, and resource cleanup.
* [ ] No root/shared/module config file exists; all enumerated presets are inline in their owner.
* [ ] `pyproject.toml` is deleted and no packaging replacement exists.
* [ ] Only the integration runner matches test naming; samples, docs/plans, compatibility/development debris, caches, and generated files are deleted.
* [ ] Exact detector/demographic weights and required third-party notices/license texts remain; retained copyright headers are unchanged.
* [ ] Parsed OutputBatch and ordering/fields exactly equal baseline; replay frame count/dimensions/FPS equal baseline and overlays are visible.
* [ ] Root-run imports, compilation, full CLI, searches, exact count, and clean status pass without installation.
* [ ] Data types, precision, memory/copy ownership, Analyse, validation boundaries, GCP, and deployment remain deferred.
