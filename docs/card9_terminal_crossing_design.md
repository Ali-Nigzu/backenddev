# Card 9 Terminal Crossing Engineering Design Proposal

## 1. Current detector review

Card 9 is a stateless geometric consumer of Card 8 runtime tracks. Its public contract is `detect_events(tracks, line_config)`, reading the line once, evaluating each track independently, and sorting emitted candidates deterministically. The current detector copies each `center_history` point, computes a side label for every point, drops `ON` points from the transition sequence while retaining original indices, and emits at most one event per track.

The locked specification defines the original stable-transition rule as follows: a non-`ON` side change is valid only when the new side persists for `MIN_STABLE_POINTS_AFTER_TRANSITION` consecutive non-`ON` observations including the transition point. With the current constant value of `2`, `A A B B` has two post-transition `B` observations and can emit, while the historical baseline `A A B` has only one post-transition `B` observation and is rejected as insufficient post-transition stability.

The implementation currently contains an additional terminal-transition path that is not fully reflected in the older problem statement: when a candidate transition occurs at the final non-`ON` point and there are not enough post-transition points, the detector can emit if the pre-transition side has at least `MIN_STABLE_POINTS_BEFORE_TERMINAL_TRANSITION` observations. That path makes `A A B` eligible, while still rejecting a two-point `A B` crossing because it lacks enough pre-transition evidence.

### Exact decision flow for `A A B B`

1. Normalize the line and track points.
2. Compute per-point side labels.
3. Build `compressed = [(0,A), (1,A), (2,B), (3,B)]`.
4. Skip same-side pair at compressed index `1`.
5. At compressed index `2`, discover `A -> B`.
6. Compute `stable_end = 2 + 2 - 1 = 3`, which is inside the compressed history.
7. Validate stable window `[B, B]`.
8. Emit `ENTRY / IN` with supporting positions from the last `A` through the second stable `B`.

### Exact decision flow for historical `A A B` without terminal logic

1. Build `compressed = [(0,A), (1,A), (2,B)]`.
2. At compressed index `2`, discover `A -> B`.
3. Compute `stable_end = 2 + 2 - 1 = 3`, which is outside the compressed history.
4. Reject because insufficient post-transition observations exist.

### Exact decision flow for current `A A B` implementation

1. Build `compressed = [(0,A), (1,A), (2,B)]`.
2. At compressed index `2`, discover `A -> B`.
3. Compute `stable_end = 3`, outside history.
4. Compute previous same-side run length as `2`.
5. Confirm the transition is terminal because it is the final non-`ON` point.
6. Emit because previous run length meets `MIN_STABLE_POINTS_BEFORE_TERMINAL_TRANSITION = 2`.

This is already close to the desired second decision path, but it should be formalized, hardened, and performance-reviewed before being treated as the approved architecture.

## 2. Runtime track lifecycle review

Card 8 creates a `RuntimeTrackV2` in `TENTATIVE` state with one center, a UUID runtime ID, timestamp metadata, hit/miss counters, detection history, and `center_history` seeded with the first observed center. Matched observations update the current center, velocity, timestamp, detection history, and append the new center to `center_history`; promotion to `ACTIVE` occurs when hit count reaches the configured activation threshold.

A track is not physically removed when it misses. Instead, stale tracks are marked `CLOSED` after their `miss_count` exceeds the configured threshold for their state and after the minimum lifetime has elapsed. `close_track` only changes `state` and records `closed_timestamp`; it does not append synthetic points, alter `center_history`, or preserve the reason for termination beyond the state/timestamp fields.

Therefore, at Card 9 evaluation time, the available deterministic terminal information is:

- the immutable observed `center_history` prefix accumulated by Card 8;
- `last_seen_timestamp`, which corresponds to the last matched observation, not necessarily the closure time;
- `state`, `closed_timestamp`, `miss_count`, and frame indices, although the current Card 9 specification says these are outside Phase B input semantics;
- no per-point timestamps, no explicit end-of-camera signal, no occlusion reason, and no tracker-fragment linkage.

The important architectural limitation is that Card 9 cannot know why a track ended unless the contract is expanded to admit lifecycle fields. A terminal crossing rule based only on the final observed point is deterministic and stateless, but it cannot distinguish a true camera exit from an arbitrary disappearance. Precision must therefore come from stronger geometric/history evidence, not from lifecycle inference.

## 3. Identified weaknesses

### Recall weakness

The stable-transition rule has an inherent length bias: it favors tracks that survive long enough to provide post-transition evidence. A true crossing followed by disappearance, occlusion, or fragmentation can be missed if the detector insists on two post-transition non-`ON` observations.

### Precision risks in the current terminal path

The existing terminal path uses pre-transition run length and terminal position in the compressed history. That is conservative for two-point crossings, but it still does not evaluate geometric margin, movement vector direction, segment-line intersection, distance from the line, or oscillation history before the terminal transition. A track like `A A B` can be emitted even if all points are extremely close to the line epsilon and could plausibly be jitter.

### Computational weakness

For every call and every track, the detector rescans the full history, recomputes all side labels, builds full debug structures, computes min/max x values, creates compressed side arrays, and prints debug output unconditionally. In repeated pipeline calls over unchanged historical tracks, this repeats work proportional to total history length even though only the tail changed. The code is still simple and small, but the asymptotic behavior is avoidable.

## 4. Candidate solution designs

### Option A — Global threshold reduction

Algorithm: set `MIN_STABLE_POINTS_AFTER_TRANSITION = 1` for every transition.

Complexity: `O(n)` per track with the current scan.

Recall: highest recall for short crossings.

Precision impact: unacceptable. Every single-point side flip becomes a valid event, including jitter and one-frame association noise.

Implementation difficulty: trivial.

Compatibility: technically compatible, but violates the requirement not to weaken all tracks.

Recommendation: reject.

### Option B — Terminal pre-history rule

Algorithm: keep the normal two-point post-transition rule. If and only if the side transition is at the final non-`ON` point, allow emission when the pre-transition side has at least `K` consecutive non-`ON` observations. Default `K = 2`, matching the current partial implementation.

Complexity: `O(n)` current implementation; can be `O(n)` single-pass with constant extra terminal bookkeeping.

Recall: recovers `A A B` and `B B A` terminal crossings.

Precision impact: moderate-to-low if `K >= 2`; still vulnerable to near-line jitter unless paired with geometric margins.

Implementation difficulty: low because the code already has the skeleton.

Compatibility: high; remains stateless, deterministic, and based on observed points only.

### Option C — Terminal segment-crossing rule with geometric margin

Algorithm: for a terminal transition candidate, evaluate the final segment from the last pre-transition non-`ON` point to the terminal post-transition point. Require:

1. terminal transition is the final non-`ON` point;
2. at least `K` stable pre-transition observations;
3. endpoints lie on opposite sides;
4. both endpoint signed distances exceed a configurable margin above `LINE_SIDE_EPSILON`;
5. the segment intersects the infinite line between its endpoints, not merely through a side-label artifact;
6. optional: displacement along the line-normal direction is at least a minimum normalized crossing distance.

Complexity: `O(1)` extra per transition after side/cross values are available.

Recall: recovers genuine terminal crossings while filtering weak edge cases near the line.

Precision impact: best of the terminal-only approaches because it adds deterministic geometric evidence instead of simply lowering stability.

Implementation difficulty: medium; requires retaining signed cross products and adding well-tested math helpers.

Compatibility: high; still observed-data-only and stateless.

### Option D — Movement-vector validation

Algorithm: for a terminal transition, compute the vector from an earlier stable pre-transition point or run centroid to the terminal point. Project it onto the oriented line normal and require the projection sign and magnitude to agree with the side transition.

Complexity: `O(K)` or `O(1)` if the pre-run summary is maintained.

Recall: good for direct crossings; weaker for curved trajectories that cross near the end.

Precision impact: improves robustness against jitter and duplicate detections; may reject valid slow or nearly tangential crossings.

Implementation difficulty: medium.

Compatibility: medium-high; uses only center history but adds a motion assumption the spec has historically avoided.

### Option E — Hybrid terminal confidence gate

Algorithm: assign deterministic evidence booleans rather than probabilistic confidence. Emit terminal event only when a required bundle is satisfied, for example:

- terminal side transition;
- pre-run length `>= 2`;
- previous and terminal endpoint distances from line exceed margin;
- final segment crosses the line;
- no earlier opposite-side oscillation in a short lookback window;
- terminal displacement from the pre-run centroid exceeds a minimum distance.

Complexity: `O(n)` with current scan; `O(1)` additional checks once summaries are available.

Recall: high for genuine `A A B` terminal crossings that are not borderline.

Precision impact: strong, because a single terminal point is accepted only if multiple deterministic evidence gates agree.

Implementation difficulty: medium.

Compatibility: high if the evidence fields are internal only and output schema remains unchanged.

### Option F — Delayed confirmation

Algorithm: do not emit on `A A B` immediately. Keep the track or pending event around until either another observation confirms `B`, the track closes, or a timeout expires.

Complexity: requires state outside a pure per-call detector.

Recall: high when delayed observations arrive.

Precision impact: strong if confirmation arrives, but terminal disappearances still need a terminal fallback.

Implementation difficulty: high because it changes Card 9 from stateless to stateful or requires pipeline-level pending event storage.

Compatibility: low with existing Card 9 constraints.

Recommendation: reject for Phase B; reconsider only if future architecture permits Card 9 state.

### Option G — Lifecycle-aware terminal evaluation

Algorithm: admit Card 8 lifecycle fields into the Card 9 contract and apply terminal rules only to `CLOSED` tracks, possibly using `closed_timestamp` and miss counts.

Complexity: low.

Recall: improves safety by avoiding terminal inference on still-active partial histories.

Precision impact: good for batch evaluation after closure; bad for immediate online event latency if closed tracks take several misses to close.

Implementation difficulty: medium because it changes the Card 9 contract and tests.

Compatibility: medium-low under the current spec, which explicitly ignores lifecycle state.

## 5. Comparison matrix

| Option | Recall gain | Precision risk | Complexity | Architecture fit | Recommendation |
|---|---:|---:|---:|---:|---|
| A. Global threshold = 1 | Very high | Very high | Low | Poor | Reject |
| B. Terminal pre-history | High for `A A B` | Medium | Low | Excellent | Keep as baseline |
| C. Terminal segment + margin | High | Low | Medium | Excellent | Adopt |
| D. Movement vector validation | Medium-high | Low-medium | Medium | Good | Optional gate |
| E. Hybrid terminal evidence | High | Low | Medium | Excellent | Recommended |
| F. Delayed confirmation | High | Low | High | Poor | Defer |
| G. Lifecycle-aware terminal only | Medium | Low | Medium | Medium | Future option |

## 6. Recommended design

Adopt a formal **terminal-only deterministic evidence path** in addition to the existing stable path.

The normal path remains unchanged:

- require `MIN_STABLE_POINTS_AFTER_TRANSITION = 2` consecutive non-`ON` points after transition;
- ignore `ON` points for side stability;
- emit only the first valid event per track.

The terminal path should be used only when the normal path cannot be evaluated because the transition is at the final non-`ON` point. It should emit only when all mandatory evidence gates pass:

1. **Terminal transition gate**: the candidate side change is the final non-`ON` point in the compressed history.
2. **Pre-history stability gate**: the previous side has at least `2` consecutive non-`ON` observations immediately before transition.
3. **No two-point crossing gate**: total non-`ON` history length must be at least `3`.
4. **Segment crossing gate**: the segment from the last previous-side non-`ON` point to the terminal point must geometrically cross the infinite line.
5. **Distance margin gate**: both endpoints should be farther from the line than a configurable terminal margin. The margin should be expressed in signed-cross units or normalized signed distance; normalized distance is easier to tune across line lengths.
6. **Oscillation guard**: reject if a recent lookback window before the pre-run contains the terminal side, preventing `B A A B`-style short oscillations from masquerading as clean terminal exits.

Optional gate after calibration:

7. **Minimum normal displacement gate**: require the final segment to move enough across the line normal to exceed a small pixel threshold. This is useful for cameras with jittery center estimates but should be tuned carefully to avoid rejecting slow genuine crossings.

This design preserves the strengths of Card 9: deterministic, stateless, simple, based only on observed centers, and resistant to online jitter. It avoids global threshold reduction because only terminal transitions receive the one-post-point exception, and only after additional evidence gates compensate for the missing post-transition stability.

## 7. Computational analysis and optimization plan

### Current behavior

Current work per `detect_events` call is `O(T log T + P)`, where `T` is emitted events/tracks for final sorting and `P` is total center-history points across input tracks. For each track the implementation does several avoidable things:

- copies every center point before knowing whether the track can emit;
- computes both `_signed_cross` and `compute_side`, duplicating cross-product work;
- builds `point_debug` for every point unconditionally;
- computes `min_x`, `max_x`, and vertical-line expected side only for debug;
- builds `compressed_side_sequence` only for debug;
- scans backward to compute pre-run length when the run length could be known while scanning;
- prints debug output on every track, which is expensive and noisy in production;
- rescans unchanged histories on every invocation.

### Recommended low-risk optimizations

1. Make debug collection opt-in via a parameter, environment flag, or logger level. Production calls should not allocate debug dictionaries or print.
2. Compute signed cross product once per point and derive both side and normalized distance from that value.
3. Store compressed entries as a small internal tuple containing original index, side, point, signed cross, normalized distance, and current run length.
4. Avoid building `side_sequence` and `compressed_side_sequence` unless debug is enabled.
5. Maintain previous-run length during the scan, eliminating `_run_length_before_transition` backward rescans.
6. Return early when a stable event is found, but only after enough data has been scanned to build required evidence for that transition. For terminal logic, full scan is inherently needed to know a transition is terminal.
7. Consider an optional upstream incremental summary only in a future phase. A truly stateless function cannot avoid rescanning if it receives only complete track objects and no prior summary cache.

These changes preserve behavior but reduce allocations and duplicate math. They also make the terminal margin checks essentially free because signed cross and normalized distances are already available.

## 8. Edge case analysis

| Edge case | Expected behavior under recommended design |
|---|---|
| Jitter around line | Rejected unless there is stable pre-history, terminal transition, real segment crossing, and endpoint distance margin. |
| `A B` two-point track | Rejected because pre-history length and total non-`ON` length gates fail. |
| `A A B` terminal crossing | Accepted if endpoint distances and segment crossing gates pass. |
| `A A B B` normal crossing | Accepted by unchanged stable path. |
| `A B A` oscillation | Rejected by unchanged stability rule and non-terminal condition. |
| `B A A B` terminal oscillation | Rejected by oscillation guard if configured lookback includes earlier `B`. |
| ON-line points between sides | Ignored for side compression; may be included in supporting positions if inside the transition window. |
| Track begins immediately after crossing | Not emitted, because no observed side transition exists. This is correct for an observed-data-only detector. |
| Duplicate detections on same side | Harmless; they lengthen pre-run or post-run stability but do not create transitions. |
| Fragmented tracks | First fragment ending after crossing can emit if terminal evidence is sufficient; second fragment beginning after crossing cannot emit without observed transition. |
| Repeated event generation across repeated calls | Event IDs remain deterministic; downstream de-duplication is still needed if `detect_events` is repeatedly called on the same complete track list. |
| Multiple crossings in one track | Existing Phase B emits only the first valid event. Terminal path should not change that contract. |
| False positives near epsilon | Mitigated by normalized distance margin and optional normal-displacement gate. |

## 9. Implementation roadmap

No production code should be written until this design is approved. After approval, implement in small reviewable steps:

1. Update `events/SPEC.md` to formalize terminal crossing semantics, constants, and evidence gates.
2. Refactor geometry computation so signed cross, side, and normalized distance are produced from one calculation.
3. Make debug instrumentation opt-in and remove unconditional printing from production detection.
4. Replace backward run-length scans with single-pass compressed-entry run lengths.
5. Implement terminal evidence gates behind named constants with conservative defaults.
6. Add synthetic and pipeline tests before changing runtime behavior.
7. Run full regression and performance tests.
8. Consider a later contract discussion for lifecycle-aware evaluation if product owners want terminal rules only on `CLOSED` tracks.

## 10. Risks and mitigations

- **Risk: terminal rule emits jitter as crossings.** Mitigate with distance margins, pre-history stability, segment crossing, and oscillation guard.
- **Risk: distance margin rejects valid near-line crossings.** Mitigate with conservative defaults and tests around slow/near-line paths.
- **Risk: lifecycle ambiguity remains.** Mitigate by documenting that terminal means final observed non-`ON` point in the supplied history, not confirmed tracker closure.
- **Risk: repeated calls produce repeated candidates.** Mitigate outside Card 9 with deterministic event ID de-duplication, or approve stateful event registry in a future phase.
- **Risk: optimizing debug changes observability.** Mitigate with opt-in debug output that preserves the current information when enabled.

## 11. Testing strategy

### Synthetic unit tests

- Stable crossing: `A A B B` and `B B A A` emit correct event type and direction.
- Historical short terminal crossing: `A A B` and `B B A` emit only when margins pass.
- Two-point crossing: `A B` and `B A` never emit.
- Near-line jitter: points whose normalized distances are below terminal margin do not emit.
- Oscillation: `A B A`, `A A B A`, `B A B`, and `B B A B` do not emit.
- ON handling: `A A ON B` terminal and `A ON B B` stable cases behave as specified.
- Supporting positions: terminal events include the last pre-transition point through terminal point, including intervening `ON` points in the original-index slice.
- Deterministic event ID: repeated calls over the same track object return identical IDs.

### Runtime/Card 8 integration tests

- Track disappears immediately after crossing and then closes: terminal event is emitted once.
- Tracker fragments after crossing: first fragment emits if it contains sufficient terminal evidence; second fragment does not emit without a transition.
- Active partial track with `A A B` during online processing: define expected behavior explicitly. Under current stateless semantics it may emit because terminal means end of supplied history; lifecycle-aware gating would defer this.
- Multiple simultaneous tracks: only tracks with valid evidence emit.

### False-positive tests

- Long same-side trajectories near the line never emit.
- Duplicate detections around one side of the line never emit.
- Single noisy point crossing the line after insufficient pre-history never emits.
- Oscillatory center estimates around the line never emit.

### Stress and performance tests

- Large number of tracks with long histories and no transitions.
- Large histories with one early stable crossing.
- Repeated `detect_events` calls over unchanged tracks to quantify the cost of stateless rescanning.
- Compare allocations and runtime before/after debug gating and single-pass summary refactor.

### Regression tests

- Existing Card 9 scenario tests must remain green.
- Existing Card 8 lifecycle tests must remain green.
- Public output schema remains unchanged.
- No changes to Card 5-8 behavior unless a future lifecycle-aware contract is explicitly approved.
