# Track V2 Internal Architecture Redesign Plan

## Scope and non-negotiable constraints

This plan is intentionally implementation-only for `track/` plus the existing replay harness in `test_tracking_v2_pipeline.py`. It does not require any upstream or downstream pipeline changes.

Locked public contract:

- `Track(tracking_state, observation_batch, config=None)` remains the public reducer.
- Input remains dictionary-compatible `TrackingState` and `ObservationBatch`.
- Output remains the same `TrackingState` object shape.
- Track IDs, path schema, best crop schema, and observation schema remain unchanged.
- Candidate generation remains observation-first: each observation asks which existing tracks could realistically explain it; assignment arbitrates among those candidates; unmatched observations become births only after assignment.
- Motion remains primary. Appearance may only act as a deterministic tie-breaker and must never make physically impossible motion eligible.

## Current architectural problems

### 1. Configuration has multiple names for the same behavior

`TrackV2Config` exposes both current and legacy fields for confirmation, miss tolerance, tentative tolerance, reassociation, gates, speed limits, jitter, and classification thresholds. `track.normalize` then chooses the first non-null value or combines several legacy values into one private value. This means multiple public settings can influence the same internal decision, and some visible settings are ignored by the current implementation.

Examples:

- Confirmation can be controlled by `confirmation_hits`, `confirmation_min_path_points`, `active_confirmation_min_path_points`, and `tentative_confirmation_min_path_points`.
- Confirmed miss tolerance can be controlled by `detector_miss_tolerance_sec`, `confirmed_track_window_sec`, `confirmed_reassociation_window_sec`, and `max_reassociation_gap_sec`.
- Motion tolerance can be controlled by `motion_tolerance_px`, `prediction_gate_px`, `latest_position_gate_px`, `confirmed_prediction_gate_px`, and `confirmed_latest_position_gate_px`.
- Speed can be controlled by `max_physical_speed_px_per_sec`, `hard_speed_limit_px_per_sec`, `confirmed_max_speed_px_per_sec`, `max_speed_px_per_sec`, and legacy tentative speed fields.
- Jitter can be controlled by `localization_jitter_px`, `jitter_tolerance_px`, and `base_motion_gate_px`.

Several parameters are currently validation-only or effectively dead:

- `confirmed_prediction_gate_px`, `tentative_prediction_gate_px`, `confirmed_latest_position_gate_px`, `tentative_latest_position_gate_px`, `confirmed_max_speed_px_per_sec`, and `tentative_max_speed_px_per_sec` imply lifecycle-specific gates, but the normalized model consumes one shared motion tolerance and one shared speed limit.
- `strong_motion_threshold`, `normal_motion_threshold`, and `weak_motion_threshold` are validated but not used to classify candidates; candidate classification currently hard-codes `<= 1.0` and a weak limit derived from takeover margin.
- `max_believable_speed_px_per_sec` is validated but not normalized or used.
- `forced_continuity_break_normalized_motion` is validated but not used.

### 2. Lifecycle decisions are split across facts, assignment roles, and compatibility helpers

`derive_track_facts` determines confirmation, eligibility, and an ownership class. `candidate_builder` turns that ownership class into string roles such as `protected_continuation` and `reassociation_continuation`. `assignment` gives those roles priority and performs deterministic first-claim passes for protected and reassociation tracks. `tracker._partition_track_indices` independently exposes a compatibility view of active and tentative tracks for the replay script.

This creates hidden coupling:

- Confirmation is path-length based but configured through several aliases.
- Protection means both same-frame continuity and assignment priority.
- Reassociation means both eligible-after-miss and a second privileged assignment pass.
- Tentative means unconfirmed lifecycle and a lower assignment priority.
- Stale means no candidates, but stale tracks are never pruned because the public state is append-only.

### 3. Motion uses one score made from several overlapping concepts

The motion pipeline currently derives velocity from recent path segments, predicts a center, computes distance to the prediction, computes distance to the latest point, computes required speed from latest point to observation, and then creates a score using a tolerance made from base tolerance, localization jitter, and tolerance growth over missing time.

Important issues:

- `motion_tolerance_px` and `localization_jitter_px` are additive, so changing one can be masked by the other.
- `motion_tolerance_growth_px_per_sec` defaults from `max_speed_px_per_sec`, making the gate grow rapidly over missed time and making CCTV localization gating permissive.
- `speed_required` uses `max(age_seconds, epsilon)`, so same-timestamp duplicate updates can imply enormous speed unless distance is zero.
- The final motion score is `max(min(normalized_prediction, normalized_latest), normalized_speed)`, which lets either prediction distance or latest-position distance compensate for the other. That makes the physical meaning of the gate hard to reason about.
- Candidate weak eligibility is controlled by `takeover_margin`, an assignment/ownership concept, not by an explicit motion threshold.
- Velocity clamping and eligibility speed limit use the same normalized physical speed value, but the public config suggests separate believable and physical limits.

### 4. Assignment mixes ownership policy with global optimization

Assignment first lets protected tracks claim observations, then reassociation tracks claim observations, then runs exhaustive optimal assignment on the remainder. The first-claim passes use `_is_defensible_first_claim`, which compares a candidate to the best peer for the same observation using `takeover_margin` and `continuity_strength`.

This is deterministic, but it means assignment is compensating for lifecycle and candidate-generation permissiveness. The result is harder to tune because ownership takeover difficulty, incumbent continuity, and motion plausibility are all entangled.

### 5. Module boundaries expose private transitional concepts

Current module responsibilities are mostly present but not clean:

- `matching.py` is only a compatibility export and should disappear once no in-scope caller imports it.
- `normalize.py` exists primarily to reconcile old public names with current private behavior.
- `facts.py` imports motion prediction, so lifecycle facts also own prediction data.
- `candidate_builder.py` owns appearance similarity, motion classification, role conversion, sorting helpers, and candidate dataclass definition.
- `tracker.py` owns validation, reducer orchestration, ID generation, state ordering, and compatibility active/tentative partitioning.

### 6. Fail-loud behavior is incomplete

Validation catches malformed public dictionaries and non-finite config values. It does not yet assert internal invariants such as:

- every assignment references a candidate that was generated;
- no track or observation is assigned more than once;
- every unmatched observation index is valid;
- candidates are only produced for lifecycle-eligible tracks;
- appearance scores never decide eligibility;
- normalized config contains only positive, coherent values after alias resolution;
- the output state remains sorted and schema-valid after mutation.

## Desired architecture

The redesign should introduce a small, behavior-focused internal model with one owner per concept:

1. **Public contract layer**
   - Own dictionary validation, public reducer shape, deterministic track ID generation, and schema-preserving state mutation.
   - Does not own motion, lifecycle, or assignment policy.

2. **Behavior configuration layer**
   - Own the small supported configuration surface.
   - Converts public `TrackV2Config` into one private `TrackerPolicy` or `TrackerSettings` object.
   - During the redesign, compatibility aliases can be accepted at the edge, but internals must never read legacy names.
   - Later phases should remove in-scope references to unused public fields where safe, or mark them as ignored compatibility fields in one explicit place.

3. **Lifecycle layer**
   - Own confirmation, tentative eligibility, confirmed miss tolerance, reassociation eligibility, and ownership state.
   - Produces one `TrackStatus` per track for a timestamp.
   - Does not compute motion distances or candidate scores.

4. **Motion layer**
   - Own velocity estimation, prediction, distance calculations, speed calculations, and motion eligibility.
   - Produces one `MotionResult` with explicit components and an eligibility decision.
   - Does not know assignment roles or appearance.

5. **Candidate layer**
   - Observation-first generation only.
   - Combines `TrackStatus` plus `MotionResult` into `Candidate` records.
   - Applies appearance only after motion eligibility.
   - Does not perform assignment.

6. **Assignment layer**
   - Solves one deterministic one-track-to-one-observation selection problem from already-eligible candidates.
   - Owns continuity/takeover policy in one place.
   - Does not rescue impossible candidates.

7. **Mutation layer**
   - Owns appending observations, best crop updates, and track birth.
   - Does not decide eligibility or matching.

## Proposed module responsibilities

A concrete target layout can remain small and close to the existing files:

- `track/config.py`
  - Public dataclass retained for compatibility.
  - Public fields grouped into supported behavioral fields and deprecated compatibility fields.
  - Documentation explains which fields are authoritative.

- `track/policy.py` or replacement for `track/normalize.py`
  - New private dataclass, for example `TrackerPolicy`.
  - Contains only behavior-level values:
    - `confirmation_hits`
    - `tentative_max_age_sec`
    - `confirmed_max_missed_sec`
    - `prediction_uncertainty_px`
    - `miss_uncertainty_growth_px_per_sec`
    - `localization_jitter_px`
    - `max_speed_px_per_sec`
    - `weak_match_max_motion_score`
    - `continuity_bias`
    - `takeover_margin`
    - `appearance_tiebreak_enabled`
    - `epsilon`
  - All alias resolution and deprecation warnings-by-comment live here only.

- `track/lifecycle.py`
  - `TrackStatus` dataclass.
  - `classify_track(track, timestamp, policy)`.
  - Confirmation and eligibility logic only.
  - Mutation helpers can move to `track/state.py` or remain here under a clearly named mutation section.

- `track/motion.py`
  - `VelocityEstimate`, `Prediction`, and `MotionResult` dataclasses if useful.
  - One coherent motion model.
  - Explicit rejection reasons.
  - No assignment terminology.

- `track/candidates.py` or refactored `track/candidate_builder.py`
  - `Candidate` dataclass.
  - Observation-first candidate generation.
  - Candidate sort keys used for deterministic tie-breaking.
  - Appearance similarity helper remains private here or moves to `track/appearance.py` if it grows.

- `track/assignment.py`
  - Deterministic solver from `Candidate` list to selected matches.
  - Invariant checks for duplicate assignment.
  - A single continuity/takeover policy.

- `track/tracker.py`
  - Reducer orchestration only:
    1. validate inputs and config;
    2. normalize config to policy;
    3. sort state and observations deterministically;
    4. classify tracks;
    5. build candidates observation-first;
    6. assign;
    7. mutate matches;
    8. birth unmatched observations;
    9. validate output invariants.
  - `_partition_track_indices` may remain only if the replay script needs it; preferably replace the replay dependency with an in-scope public helper or documented private helper.

- `track/matching.py`
  - Remove if no in-scope code imports it.
  - If retained, make it a thin deprecated compatibility file with no production role.

## Configuration redesign

### Target behavioral configuration

The eventual internal configuration should be significantly smaller and should read like tracking behavior rather than implementation history:

| Proposed setting | Behavioral question answered | Notes |
| --- | --- | --- |
| `confirmation_hits` | How quickly does a track become confirmed? | Single source of truth for maturity. |
| `tentative_max_age_sec` | How long may an unconfirmed track wait for another hit? | Replaces tentative windows/frame aliases. |
| `confirmed_max_missed_sec` | How tolerant are confirmed tracks of detector misses? | Replaces reassociation/window aliases. |
| `prediction_uncertainty_px` | How much prediction/localization error is acceptable immediately? | One base spatial uncertainty. |
| `miss_uncertainty_growth_px_per_sec` | How much uncertainty grows during detector misses? | Must not silently default to max speed. |
| `localization_jitter_px` | How much detector jitter is acceptable independent of motion? | Keep only if distinct from prediction uncertainty; otherwise fold into base uncertainty. |
| `max_speed_px_per_sec` | What movement is physically believable? | One hard physical eligibility limit. |
| `weak_match_max_motion_score` | How far beyond normal can confirmed continuity stretch? | Replaces hard-coded weak limit from takeover margin. |
| `continuity_bias` | How much should incumbents be favored once candidates are plausible? | Assignment-only meaning. |
| `takeover_margin` | How much better must a challenger be to take ownership? | Assignment-only meaning. |
| `appearance_tiebreak_enabled` | May appearance break otherwise plausible ties? | Never affects eligibility. |
| `epsilon` | Numerical tolerance. | Internal/advanced only. |

### Parameters to remove or make internal

- Remove from internals: `confirmed_prediction_gate_px`, `tentative_prediction_gate_px`, `confirmed_latest_position_gate_px`, `tentative_latest_position_gate_px`, `confirmed_max_speed_px_per_sec`, `tentative_max_speed_px_per_sec`.
- Remove or deprecate: `active_confirmation_min_path_points`, `tentative_confirmation_min_path_points`, `tentative_recency_window_frames`, `max_reassociation_gap_sec`.
- Remove or deprecate: `prediction_gate_px`, `latest_position_gate_px`, `prediction_gate_growth_px_per_sec`, `latest_position_gate_growth_px_per_sec`.
- Remove or deprecate: `max_believable_speed_px_per_sec`, `hard_speed_limit_px_per_sec`, `max_physical_speed_px_per_sec` in favor of one `max_speed_px_per_sec`, unless a clear two-tier speed model is adopted.
- Remove or deprecate: `jitter_tolerance_px` and `base_motion_gate_px` if `localization_jitter_px` or `prediction_uncertainty_px` becomes authoritative.
- Remove: `forced_continuity_break_normalized_motion` because impossible candidates are excluded before assignment.
- Either use or remove: `strong_motion_threshold`, `normal_motion_threshold`, `weak_motion_threshold`. Prefer replacing them with one `weak_match_max_motion_score` if strong/normal labels are assignment decoration only.

### Compatibility strategy

Because external public IO is locked, the safest phased approach is:

1. Keep the `TrackV2Config` class importable and instantiable with existing fields.
2. Define a small authoritative subset in the class docstring.
3. Resolve all deprecated aliases in exactly one normalization/policy function.
4. Ensure no internal module imports or reads `TrackV2Config` directly except validation and policy construction.
5. After behavior is covered by tests, optionally remove unused fields only if no permitted in-repo caller relies on them. If external callers may instantiate them, leave fields as inert compatibility aliases but document that they map to one authoritative behavior or are ignored.

## Motion redesign

The motion model should become explicit and non-compensating:

1. Derive a deterministic velocity estimate from recent path points.
2. Predict the expected center at the observation timestamp.
3. Compute:
   - `dt_sec`
   - `prediction_distance_px`
   - `latest_distance_px`
   - `required_speed_px_per_sec`
   - `allowed_prediction_error_px`
4. Reject immediately if:
   - timestamp moves backwards;
   - lifecycle says track is ineligible;
   - required speed exceeds `max_speed_px_per_sec` for positive `dt`;
   - prediction distance exceeds the allowed prediction error;
   - same-timestamp distance is non-zero beyond jitter.
5. Compute one motion score from the primary constraint, preferably `prediction_distance_px / allowed_prediction_error_px`, with speed as an independent hard constraint rather than a compensating score.
6. If latest-position distance remains useful, use it as a secondary diagnostic or separate hard bound, not as a way to compensate for failed prediction.
7. Weak confirmed matching should be explicit: normal eligibility is score `<= 1.0`; confirmed weak eligibility is score `<= weak_match_max_motion_score` and speed-valid.

This prevents a permissive growth term or latest-position fallback from masking prediction failure.

## Lifecycle redesign

Lifecycle should be a small state machine derived from public track dictionaries:

- `tentative`: path length below `confirmation_hits`, eligible only until `tentative_max_age_sec` since last hit.
- `confirmed_live`: path length at or above `confirmation_hits`, last hit is at current timestamp or within same-frame epsilon.
- `confirmed_missing`: confirmed but missed for `0 < missing_sec <= confirmed_max_missed_sec`.
- `stale`: ineligible for matching but retained in output state for schema compatibility/history.
- `future`: invalid for current timestamp and should fail or be ineligible with a loud error depending on public-state expectations.

Ownership should be represented as assignment priority derived from status, not mixed into lifecycle itself:

- live confirmed continuation has highest incumbent priority;
- missing confirmed reassociation has second priority;
- tentative continuation is lower;
- stale/future generate no candidates.

## Assignment redesign

Assignment should operate only on realistic candidates and should not compensate for permissive motion gates.

Target behavior:

1. Input is a finite list of candidates with valid track and observation indices.
2. Group candidates by observation and by track.
3. Apply deterministic scoring tuple:
   - status priority;
   - motion score;
   - continuity/takeover adjustment;
   - appearance tie-break cost if enabled;
   - deterministic track ID and detection ID keys.
4. Select one-to-one matches with a deterministic solver.
5. Enforce invariants after selection:
   - no duplicate track index;
   - no duplicate observation index;
   - every selected pair exists in candidate set;
   - all unmatched observations are exactly observations not selected.

A simpler assignment option is to remove the separate protected/reassociation first-claim passes and encode continuity preference directly in the candidate cost tuple. If first-claim passes remain, they must be documented as the one assignment owner for incumbent protection and must not alter eligibility.

## Dead code and simplification plan

Remove or consolidate:

- `track/matching.py` compatibility exports if no permitted caller imports them.
- `_map_matches` if tracks are always sorted and indices are already state indices.
- `_rejected_observation_indices` return value from `assign_candidates`, currently always empty.
- Unused classification thresholds or wire them to real classification in one place.
- Deprecated config aliases from internal code paths.
- Lifecycle-specific gate names that are not actually lifecycle-specific.
- Stringly typed role names duplicated across modules; replace with constants or enum-like literals owned by lifecycle/assignment.
- Repeated numeric track ID logic if only one module needs it.
- Appearance helper from candidate builder if a separate appearance module clarifies boundaries.

## Internal consistency improvements

Add checks at module boundaries:

- Validate normalized policy values are positive where required and logically ordered.
- Assert candidate indices are in range.
- Assert candidate motion scores and distances are finite for eligible candidates.
- Assert no candidate has an impossible lifecycle status.
- Assert assignment result references generated candidates only.
- Assert selected matches are one-to-one.
- Revalidate tracking state after mutation in debug/test path or always if performance is acceptable.
- Fail if a track path contains a timestamp newer than the current observation batch timestamp beyond epsilon.

## Proposed implementation phases

### Phase 1: Characterization and tests

- Add focused tests in `test_tracking_v2_pipeline.py` or a permitted test harness section that exercises deterministic reducer behavior without running the full video replay.
- Cover births, same-track continuation, implausible motion birth, tentative expiration, confirmed reassociation, deterministic ordering, best crop update, duplicate ID validation, and appearance tie-breaking only after motion eligibility.
- Capture at least one regression case where current location gating is too permissive.

### Phase 2: Introduce policy object

- Replace `_NormalizedTrackConfig` with a clearly named private policy dataclass.
- Move all alias resolution to one function.
- Document authoritative settings and deprecated aliases.
- Ensure internals consume only the policy object.

### Phase 3: Separate lifecycle from motion

- Refactor `derive_track_facts` into lifecycle-only `TrackStatus` derivation.
- Move prediction and velocity responsibility fully into `motion.py`.
- Ensure candidates receive both status and motion result explicitly.

### Phase 4: Replace motion eligibility model

- Implement the explicit non-compensating motion model.
- Decouple uncertainty growth from max speed.
- Make weak confirmed eligibility controlled by a dedicated motion threshold, not takeover margin.
- Keep deterministic outputs and public schema unchanged.

### Phase 5: Simplify assignment

- Decide whether to remove first-claim passes or keep them as documented incumbent policy.
- Remove dead return values and duplicate role/classification logic.
- Add assignment invariants.

### Phase 6: Remove dead compatibility internals

- Remove unused helpers and modules.
- Remove validation of fields that are no longer behaviorally meaningful, or keep validation only for compatibility aliases that still map to policy.
- Update replay helper dependency on `_partition_track_indices` if possible inside the permitted files.

### Phase 7: Tune defaults for CCTV after architecture is clean

- Tune behavior-focused defaults only after the internal model is coherent.
- Prefer conservative default spatial uncertainty and modest miss-growth for static CCTV scenes.
- Validate with deterministic synthetic tests before replaying real video.

## Risks

- Public `TrackV2Config` fields may be used by external production callers even if they are not used in this repository. Removing dataclass fields can break construction. Prefer deprecating or aliasing before removal unless external compatibility is explicitly cleared.
- Tightening motion gates can increase track fragmentation. That is expected during tuning but should not be mixed with architecture refactors.
- Changing assignment policy can alter track IDs even when schema stays unchanged. Characterization tests should distinguish intentional architectural behavior from accidental nondeterminism.
- Validation that fails loudly may reveal malformed historical states. Decide whether those are invalid states or require edge compatibility before deploying.
- Exhaustive assignment can become expensive if candidate counts grow. Simplification should preserve deterministic pruning and consider bounded candidate counts per observation if necessary.

## Validation strategy

- Unit-style deterministic reducer scenarios in the permitted test file.
- Golden assertions on complete `TrackingState` dictionaries for small synthetic sequences.
- Invariant tests for validation failures and assignment consistency.
- Replay-level smoke check using `test_tracking_v2_pipeline.py` only as an integration harness, not as the main correctness oracle.
- Determinism check: run the same synthetic sequence twice and assert byte-for-byte identical states.
- Motion sanity checks:
  - same timestamp and same position matches;
  - same timestamp and different position does not match;
  - plausible velocity matches;
  - impossible jump births a new track;
  - miss gap grows uncertainty only by configured miss growth;
  - appearance cannot rescue impossible motion.

## Execution checklist

Before implementation begins, confirm the intended compatibility level for `TrackV2Config`:

1. **Strict external compatibility:** keep all existing dataclass fields, but make only the target subset authoritative through a single policy builder.
2. **Clean internal-only compatibility:** remove unused fields if all callers are in the permitted scope.

Given the hard rule that the wider production pipeline must not change, option 1 is the safer default for the first implementation pass.
