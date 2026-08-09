"""Human-understandable behaviour controls for Track."""

# ============================================================
# NEW ID CONFIRMATION
# ============================================================

BOOTSTRAP_WINDOW_FRAMES = 10
# Frames at the beginning of the batch where Tracks are easier to confirm
# because people may already be present.
# Practical range: 0–30 frames.
# Higher keeps startup rules active longer; lower starts normal confirmation sooner.

BOOTSTRAP_ACTIVATION_HITS = 2
# Real observations required to confirm a Track during the bootstrap period.
# Practical range: 1–3.
# Higher rejects more startup noise; lower confirms people already present faster.

NEW_TRACK_ACTIVATION_HITS = 4
# Real observations required before a brand-new mid-window identity is public.
# Practical range: 2–5.
# Higher strengthens continuity and suppresses replacement IDs; lower exposes new
# identities faster. The initial setting intentionally favours continuity.


# ============================================================
# TRACK MEMORY
# ============================================================

TENTATIVE_TIMEOUT_SECONDS = 1.0
# Time an unconfirmed Track may survive without another observation.
# Practical range: 0.3–1.5 seconds.
# Higher retains tentative evidence longer; lower removes tentative noise faster.

ACTIVE_TIMEOUT_SECONDS = 2.5
# Time an established Track stays alive after detections disappear.
# Practical range: 1.0–4.0 seconds.
# Higher strengthens continuity; lower closes Tracks and permits replacement IDs
# sooner. Too high lets stale IDs compete with different people.


# ============================================================
# DETECTION EVIDENCE
# ============================================================

HIGH_CONFIDENCE_THRESHOLD = 0.50
# Confidence required for normal matching and creation of a tentative Track.
# Practical range: 0.35–0.80.
# Higher requires stronger evidence; lower treats more detections as primary.

LOW_CONFIDENCE_THRESHOLD = 0.25
# Lowest confidence an established Track may use as recovery evidence.
# Detect currently floors detections at 0.25, so configuring Track lower has no
# effect without a Detect change. Higher ignores more weak evidence; lower
# preserves established IDs through more confidence dips.


# ============================================================
# NORMAL FRAME-TO-FRAME MATCHING
# ============================================================

NORMAL_MATCH_MAX_COST = 0.88
# Maximum normalized cost for normal Track-to-detection matching.
# Practical range: 0.65–0.95.
# Higher is more forgiving; lower is stricter. Continuity Rescue, rather than this
# value, provides the aggressive established-ID preservation policy.

LOW_CONFIDENCE_RECOVERY_MAX_COST = 0.80
# Permissiveness for established Tracks recovering weak detections.
# Practical range: 0.50–0.90.
# Higher tolerates weaker geometry; lower requires cleaner recovery evidence.


# ============================================================
# CONTINUITY RESCUE
# ============================================================

CONTINUITY_RESCUE_MAX_GAP_SECONDS = 1.75
# Maximum age of an established Track eligible for special Pass 1 rescue.
# Practical range: 0.5–2.5 seconds.
# Higher fights longer to preserve an ID; lower permits replacement IDs sooner.

CONTINUITY_RESCUE_MAX_COST = 0.94
# Maximum normalized cost for established-ID continuity rescue.
# Practical range: 0.70–0.98.
# Higher strengthens continuity; lower requires stronger rescue evidence. Hard
# physical gates apply regardless of this value.


# ============================================================
# WHOLE-WINDOW REFINEMENT
# ============================================================

REFINE_ENABLED = True
# True runs whole-window fragment refinement. False returns the v1.2 Pass 1 result.

REFINE_MAX_GAP_SECONDS = 1.75
# Maximum temporal gap Pass 2 may consider joining.
# Practical range: 0.5–2.5 seconds.
# Higher permits longer stitching; lower restricts repair to shorter breaks.

REFINE_MAX_COST = 0.48
# Maximum normalized final fragment-link cost.
# Practical range: 0.25–0.60.
# Higher merges more aggressively; lower is more conservative. Hard motion and
# geometry gates remain mandatory.

REFINE_MAX_ROUNDS = 2
# Maximum bounded refinement passes; processing stops early when no link is made.
# Practical range: 1–3. Two is the production default.
