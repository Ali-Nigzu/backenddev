# Vendored MiVOLO inference model

This package vendors the minimum MiVOLO model architecture needed for local
checkpoint inference in `demographics.model`.

Upstream repository: <https://github.com/WildChlamydia/MiVOLO>

Upstream commit: `37475e3f8818b5f22448003feec3e64b01bfb188`

Vendored files:

- `mivolo/model/mi_volo.py`
- `mivolo/model/mivolo_model.py`
- `mivolo/model/create_timm_model.py`
- `mivolo/model/cross_bottleneck_attn.py`
- `mivolo/model/__init__.py`
- `mivolo/__init__.py`

The detector, YOLO integration, training code, datasets, demo applications, and
evaluation applications are intentionally not vendored. Runtime code must not
clone or download MiVOLO source; this vendored package is the only model source
used by production demographic inference.
