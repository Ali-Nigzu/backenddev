"""Manually tuneable controls for the tracking stage."""

# Lifecycle / ID behaviour
BOOTSTRAP_WINDOW_FRAMES = 10
BOOTSTRAP_ACTIVATION_HITS = 2
ACTIVATION_HITS = 3
TENTATIVE_TIMEOUT_SECONDS = 1.0
ACTIVE_TIMEOUT_SECONDS = 2.0

# Detection-confidence behaviour
HIGH_CONFIDENCE_THRESHOLD = 0.50
LOW_CONFIDENCE_THRESHOLD = 0.25

# Association / motion behaviour
MAHALANOBIS_GATE = 13.28
MOTION_COST_WEIGHT = 0.70
IOU_COST_WEIGHT = 0.30
PRIMARY_MAX_COST = 0.85
RECOVERY_MAX_COST = 0.70

# Standard deviations are expressed in pixels (or pixels/second squared for
# process noise). Measurement values are relative to the observed box size.
CENTRE_PROCESS_NOISE = 25.0
SIZE_PROCESS_NOISE = 10.0
MEASUREMENT_CENTRE_NOISE = 0.05
MEASUREMENT_SIZE_NOISE = 0.10
INITIAL_VELOCITY_VARIANCE = 10_000.0

# Whole-window continuity refinement
REFINE_ENABLED = True
REFINE_MAX_GAP_SECONDS = 1.5
REFINE_MAX_ROUNDS = 2
REFINE_MAX_COST = 0.40
