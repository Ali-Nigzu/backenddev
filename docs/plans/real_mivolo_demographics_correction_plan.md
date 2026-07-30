# Real MiVOLO Demographics Correction Plan

This is an implementation-only cleanup plan. The follow-up implementation must make production demographic inference use only the real MiVOLO neural network, loaded from the manually supplied local checkpoint at `demographics/demographicweights.pth`. No implementation changes are included in this planning PR.

## Non-negotiable production behavior

Production demographic inference must have exactly two runtime paths:

```text
Real checkpoint available and compatible
    -> instantiate the genuine MiVOLO architecture, load checkpoint["state_dict"], run PyTorch inference

Real checkpoint absent, corrupt, incompatible, or import/load failure
    -> raise DemographicModelError
```

There must be no third path: no deterministic substitute, no random generator, no hard-coded default, no heuristic based on crop pixels, no fake backend reachable from production, and no silent fallback.

## Current inventory and discovered invalid paths

### Fake or synthetic demographic inference paths to remove

The implementation must remove or correct every discovered fake/synthetic path below:

| Path | Current issue | Required implementation action |
| --- | --- | --- |
| `demographics/model.py` | Defines `_DeterministicBodyModel`, which is not MiVOLO. It derives sex logits from `body_mean`/`body_std` and derives age from `body_mean / 4.0`. | Delete `_DeterministicBodyModel` entirely. Do not rename or retain it. Replace `MiVOLOBackend` with a real PyTorch MiVOLO backend. |
| `demographics/model.py` | `_load_metadata()` validates checkpoint metadata and selected tensor shapes but discards the trained weights instead of loading them into a network. | Load `checkpoint["state_dict"]` into the instantiated official MiVOLO architecture and fail on unexplained missing/unexpected keys. |
| `demographics/model.py` | Missing checkpoint error recommends `bash demographics/assemble_weights.sh`. | Replace with a direct message: `MiVOLO checkpoint not found at demographics/demographicweights.pth. Place the full checkpoint at that path before running demographic inference.` |
| `demographics/demographic.py` | Public constructor accepts `backend`, enabling test/fake backend injection through the production class. | Remove production-reachable backend injection. Keep only `checkpoint_path` and `device` in the public production constructor. If implementation tests need test doubles, keep them outside production execution paths. |
| `demographics/preprocessing.py` | Current body-only preprocessing assumes a zero-face + body six-channel tensor without proving it matches official MiVOLO inference. | Verify against the exact vendored official MiVOLO inference code and replace any incorrect channel order, face-missing representation, letterbox, normalization, or tensor-shape assumption. |
| `test_tracking_v2_harness_demographics.py` | Root-level harness test can exercise injected fake demographic behavior. | Delete this root-level test script. Do not replace it with another root `test*.py`. |
| `test_demographics.py` | Root-level tests include expectations around the assembly script and likely fake-compatible type checks. | Delete this root-level test script. Do not preserve fake-backend coverage in repository root. |
| `docs/plans/mivolo_demographics_and_tracking_harness_plan.md` | Existing plan documents checkpoint assembly, fake-backend tests, synthetic fixture crops, and CPU fallback language from the prior invalid approach. | Correct any future docs touched by implementation so they no longer instruct assembly, fake backend injection, or synthetic demographic validation. This planning-only PR leaves prior history intact except for this new plan. |

Search terms that must produce no production fake path after implementation: `_DeterministicBodyModel`, `body_mean`, `body_std`, `brightness`, `fake`, `synthetic`, `random age`, `random sex`, `mock demographic`, `fallback demographic`, `assemble_weights`, and `model_parts`.

## Checkpoint ownership and repository cleanup

The checkpoint is manually supplied by the user and must remain uncommitted. The package-relative default path must be exactly:

```text
demographics/demographicweights.pth
```

Implementation actions:

1. Keep the `.gitignore` rule for `demographics/demographicweights.pth`.
2. Delete `demographics/model_parts/` and all of its contents:
   - `demographics/model_parts/demographicweights.pth.part-00`
   - `demographics/model_parts/demographicweights.pth.part-01`
3. Delete `demographics/assemble_weights.sh`.
4. Delete any code, docs, or error messages that tell users to run `bash demographics/assemble_weights.sh`.
5. Do not add Git LFS configuration for this checkpoint.
6. Do not add runtime checkpoint download, release download, reconstruction, or source checkout logic.
7. Preserve explicit `checkpoint_path` support so callers can override the default with a local path.

Missing checkpoint behavior for non-empty `EventBatch` must raise `DemographicModelError` and include the expected path. Empty `EventBatch` must return `{"results": []}` before loading, hashing, importing, or requiring the model.

## MiVOLO source strategy

Use exactly this source strategy:

```text
Vendor the minimum official MiVOLO inference implementation required by the checkpoint
```

Vendored code must live under:

```text
demographics/_mivolo/
```

The implementation must record the exact official MiVOLO source revision used. Use the official repository `WildChlamydia/MiVOLO` and vendor only the inference/model files needed to instantiate the checkpoint-compatible network. Do not runtime-clone, import from `/tmp`, dynamically download source, include MiVOLO's detector, include training code, include datasets, or include unrelated CLI applications.

### Official source files planned for vendoring

Vendor these official files, adjusted only for package-relative imports, lint compatibility, and removal of unused detector/training dependencies:

| Official file | Vendored path | Purpose |
| --- | --- | --- |
| `mivolo/model/mi_volo.py` | `demographics/_mivolo/model/mi_volo.py` | Core MiVOLO/VOLO model architecture. |
| `mivolo/model/mivolo_model.py` | `demographics/_mivolo/model/mivolo_model.py` | MiVOLO model factory/wrapper used by official inference. |
| `mivolo/model/create_timm_model.py` | `demographics/_mivolo/model/create_timm_model.py` | Checkpoint-compatible timm model creation utilities. |
| `mivolo/model/cross_bottleneck_attn.py` | `demographics/_mivolo/model/cross_bottleneck_attn.py` | Cross-input attention layers used by the MiVOLO architecture. |
| `mivolo/model/__init__.py` | `demographics/_mivolo/model/__init__.py` | Narrow model package exports. |
| `mivolo/__init__.py` | `demographics/_mivolo/__init__.py` | Vendored package marker and source revision note. |
| `mivolo/data/misc.py` | `demographics/_mivolo/data/misc.py` | Only if required for official preprocessing/model configuration helpers. |
| `mivolo/predictor.py` | reference only; do not vendor wholesale unless needed | Use to verify official preprocessing and output conversion. If any tiny helper is required, copy only that helper into production preprocessing with attribution. |
| `demo.py` / `eval_pretrained.py` | reference only; do not vendor | Use only to confirm official invocation and output interpretation. |

The follow-up implementation must inspect the actual supplied checkpoint metadata before finalizing the exact model variant, input size, `with_persons_model`, output head shape, and age normalization constants. If the checkpoint requires a different subset of official files, document the verified reason in `demographics/_mivolo/README.md` and keep the vendored set minimal.

## Exact real checkpoint-loading sequence

`demographics/model.py` must be rewritten around this sequence:

1. Resolve `checkpoint_path` to the explicit local path, or to `Path(__file__).with_name("demographicweights.pth")` by default.
2. Defer loading until `predict()` is called with a non-empty batch.
3. If the resolved checkpoint path does not exist, raise `DemographicModelError` with the direct missing-file message and no assembly/download suggestion.
4. Select device: `cpu`, `cuda`, or `auto`; `cuda` must fail if unavailable, while `auto` may choose CUDA when available or CPU otherwise.
5. Import PyTorch and the vendored MiVOLO model. Any import failure raises `DemographicModelError`.
6. Load the checkpoint with `torch.load(checkpoint_path, map_location="cpu")`.
7. Validate the checkpoint is a dictionary containing `state_dict` and required MiVOLO metadata such as `min_age`, `max_age`, `avg_age`, `no_gender`, and `with_persons_model`.
8. Instantiate the checkpoint-compatible real MiVOLO `torch.nn.Module` from `demographics._mivolo` using the verified model name/input size/person-body configuration.
9. Load `checkpoint["state_dict"]` into the model. Use strict loading unless a specific upstream incompatibility is verified and documented. Missing required keys fail. Unexpected material keys fail.
10. Record and expose diagnostic load data: model class name, parameter count, missing keys, unexpected keys, checkpoint path, source revision, and selected device.
11. Verify the loaded model has a realistic trained parameter count for the selected MiVOLO variant; fail if the count is implausibly tiny or zero.
12. Move the model to the selected device and call `model.eval()`.
13. Cache the model once per `Demographic`/backend instance. Do not reload per track or per event.
14. Convert the preprocessed NumPy batch to a PyTorch tensor on the selected device.
15. Run `model(tensor)` under `torch.inference_mode()`.
16. Convert only the model's real outputs to public `age` and `sex` values. Validate output shape and finiteness. Raise `DemographicModelError` on invalid output rather than clamping bad shapes into plausible values.

## Proof that the checkpoint controls inference

The implementation is incomplete unless it adds programmatic proof that `checkpoint["state_dict"]` controls inference:

1. Inspect the backend model type and assert it is the real vendored MiVOLO `torch.nn.Module`, not `_DeterministicBodyModel` or a NumPy callable.
2. Count trainable and total parameters and assert a realistic trained MiVOLO parameter count for the selected checkpoint variant.
3. After `load_state_dict`, compare a named loaded parameter, preferably `head.weight` if present, to the corresponding `checkpoint["state_dict"]["head.weight"]` tensor with `torch.equal` or an exact CPU comparison after any documented key-prefix normalization.
4. Assert the loaded-state diagnostics report zero unexplained missing keys and zero unexplained unexpected keys.
5. Monkeypatching or local `/tmp` diagnostic scripts may be used during implementation, but no extra root-level test script may remain.
6. Add a negative validation path proving corrupt and incompatible checkpoints raise `DemographicModelError`.
7. Search the final repository for `_DeterministicBodyModel`, `body_mean`, `body_std`, and pixel-average demographic calculations; none may remain in production.
8. Verify inference increments/records a real forward-call path from the PyTorch model and never discards `state_dict` after metadata inspection.

## Public `Demographic` contract to preserve

Keep the public import:

```python
from demographics import Demographic
```

Keep expected usage:

```python
demographic = Demographic(checkpoint_path=None, device="auto")
demographics_batch = demographic(event_batch, frame_batch)
```

Keep expected result shape:

```python
{"results": [{"track_id": str, "age": int, "sex": int}]}
```

Implementation requirements:

- Produce one result per unique `track_id`.
- Multiple events for one track must reuse one demographic result joined by `track_id`.
- Return no race, age bucket, confidence, logits, checkpoint metadata, or model-internal fields.
- Encode `sex` as Python integer `1` for male and `0` for female.
- Return exact age as a Python integer.
- Return `{"results": []}` for empty `EventBatch` without loading/importing/requiring the checkpoint.
- Require the real checkpoint for non-empty `EventBatch`.
- Remove production-reachable fake backend injection from `Demographic.__init__`.

## Preprocessing verification and correction plan

Do not assume the current preprocessing is correct. Verify against the exact vendored official MiVOLO inference path and lock the following in implementation comments/tests:

- official input dimensions;
- body-only invocation method;
- face-missing representation;
- whether body-only inference uses six channels, paired face/body tensors, a missing-face mask, or a different official convention;
- body-channel placement;
- expected RGB/BGR convention;
- official letterbox/resize behavior;
- padding value;
- interpolation method;
- input scaling;
- normalization values;
- tensor channel order;
- batch dimensions;
- checkpoint metadata use.

Implementation rules:

1. Input pixels must originate from the real tracked-person crop.
2. Bounding boxes must come from `Event.best_crop.bbox`.
3. `frame_id` must come from `Event.best_crop.frame_id`.
4. Source pixels must come from the corresponding real source-video frame in `FrameBatch`.
5. RGB/BGR conversion must happen exactly once and be documented at the `Detect`/`Demographic` boundary.
6. Aspect-ratio handling, padding, interpolation, scaling, normalization, and face-missing handling must match official MiVOLO inference exactly.
7. Preprocessing must never generate or influence age/sex except by preparing the model input tensor.
8. Invalid crop/frame failures must raise `DemographicInputError` and include `track_id`, `frame_id`, and `bbox`.

## Tracking harness cleanup and truthful output

Only this root-level test/pipeline script may remain:

```text
test_tracking_v2_pipeline.py
```

The implementation must make it run the real sequence:

```text
real source video frame
    -> production Detect
    -> production Track
    -> production Event
    -> real Event.best_crop frame_id and bbox
    -> production Demographic
    -> real MiVOLO model
    -> enriched Event output
```

Harness requirements:

- Video frames come from the supplied video.
- Detection results come from production `Detect`.
- Tracks come from production `Track`.
- Events come from production `Event`.
- Event timestamps, `track_id`, and `event_type` come directly from Event output.
- Body crops use real `best_crop.frame_id` and `best_crop.bbox`.
- Frames are re-read from the real source video for demographic inference, and frame IDs resolve deterministically.
- `Demographic` is the production class with no fake backend injection.
- The harness must not generate synthetic frames, random crops, fake records, fake tracks, fake ages, fake sexes, hard-coded demographics, rebalanced distributions, forced sex variety, forced age spread, or default results after model failure.
- Missing checkpoint must fail for non-empty events.
- Empty events must still write/print `{"events": []}`.
- JSON output schema must be exactly:

```python
{
    "events": [
        {
            "track_id": str,
            "timestamp": float,
            "event_type": int,
            "age": int,
            "sex": int,
        }
    ]
}
```

Console sex display must be presentation-only. JSON contains only `sex`. Replace the current separate `sex` plus `sex_label` console wording with exactly one derived form, for example `sex=0(female)` or `sex=1(male)`. Invalid sex values must raise rather than defaulting to female.

## Root-level test script deletion inventory

Inventory command used for this plan:

```bash
find . -maxdepth 1 -type f -name 'test*.py' -print | sort
```

Discovered root-level test scripts at planning time:

```text
./test_demographics.py
./test_demographics_real_checkpoint.py
./test_event_demographic_integration.py
./test_events.py
./test_tracking_v2_harness_demographics.py
./test_tracking_v2_pipeline.py
```

Implementation deletion list:

```text
test_demographics.py
test_demographics_real_checkpoint.py
test_event_demographic_integration.py
test_events.py
test_tracking_v2_harness_demographics.py
```

Keep only:

```text
test_tracking_v2_pipeline.py
```

Do not replace deleted scripts with renamed root-level tests. Temporary validation scripts may exist only outside the repository or must be deleted before commit.

## File-by-file implementation plan

| Path | Planned action | Required outcome |
| --- | --- | --- |
| `demographics/model.py` | Rewrite. | Real MiVOLO `torch.nn.Module`; package-relative default checkpoint path; truthful missing-file error; strict state-dict loading; real PyTorch forward under `torch.inference_mode()`; no fake/fallback path. |
| `demographics/demographic.py` | Inspect and correct. | Preserve public `Demographic(checkpoint_path=None, device="auto")`; remove production fake backend injection; one result per unique track; empty batch returns without model load. |
| `demographics/preprocessing.py` | Verify against official source and correct. | Exact MiVOLO-compatible body preprocessing from real `Event.best_crop` crop; no pixel-derived demographic outputs. |
| `demographics/exceptions.py` | Retain or simplify. | Use `DemographicModelError` for model/checkpoint/import/output failures and `DemographicInputError` for invalid frame/crop details. |
| `demographics/__init__.py` | Retain narrow export. | Export the real `Demographic` public class only. |
| `demographics/_mivolo/` | Replace placeholder docs/source with minimal official vendored inference architecture. | Actual checkpoint-compatible MiVOLO implementation with recorded upstream revision; no detector/training/dataset/CLI bulk import. |
| `demographics/model_parts/` | Delete. | No split checkpoint storage remains. |
| `demographics/assemble_weights.sh` | Delete. | No reconstruction workflow remains. |
| `.gitignore` | Retain checkpoint ignore. | `demographics/demographicweights.pth` remains local and uncommitted. |
| `test_tracking_v2_pipeline.py` | Inspect and correct. | Real end-to-end video inference only; no fake backend injection; direct Event and MiVOLO outputs; console `sex=0(female)`/`sex=1(male)` representation. |
| `test_demographics.py` | Delete. | Remove root fake/assembly-script test path. |
| `test_demographics_real_checkpoint.py` | Delete. | Remove root checkpoint smoke script. |
| `test_event_demographic_integration.py` | Delete. | Remove root integration test script. |
| `test_events.py` | Delete. | Remove root Event test script per root-test cleanup requirement. |
| `test_tracking_v2_harness_demographics.py` | Delete. | Remove root harness demographic fake-backend test script. |
| `docs/plans/mivolo_demographics_and_tracking_harness_plan.md` | Do not edit in this planning PR; correct only if future implementation touches docs. | No future active instructions should recommend assembly, fake backend tests, or synthetic demographic validation. |

## Implementation acceptance checklist

- [ ] `_DeterministicBodyModel` no longer exists.
- [ ] No synthetic demographic generator exists.
- [ ] No crop mean, crop standard deviation, color, brightness, or random value is used as a demographic prediction.
- [ ] `model_parts/` no longer exists.
- [ ] `assemble_weights.sh` no longer exists.
- [ ] The full model is expected at `demographics/demographicweights.pth`.
- [ ] The full model remains ignored by Git.
- [ ] Real MiVOLO source is present under `demographics/_mivolo/`.
- [ ] Real MiVOLO architecture is instantiated.
- [ ] Real checkpoint `state_dict` is loaded into model parameters.
- [ ] Real PyTorch forward inference runs in eval mode under `torch.inference_mode()`.
- [ ] Missing checkpoint fails for non-empty events with a truthful path message.
- [ ] Empty events return an empty result without loading the model.
- [ ] `test_tracking_v2_pipeline.py` uses real source video data only.
- [ ] The harness does not inject a fake backend.
- [ ] The harness does not modify model predictions.
- [ ] Only `test_tracking_v2_pipeline.py` remains among root `test*.py` files.
- [ ] Enriched output retains the required schema.
- [ ] Repeated events for a track reuse one genuine prediction.
- [ ] No artificial age or sex distribution is enforced.
