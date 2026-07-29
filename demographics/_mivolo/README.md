# MiVOLO source attribution

The production demographic stage is designed for the MiVOLO v1 `mivolo_d1_224`
body+face checkpoint assembled at `demographics/demographicweights.pth`.

Source inspected for compatibility: <https://github.com/WildChlamydia/MiVOLO>.
Only the required public contract, metadata validation, preprocessing policy, and
output conversion are implemented locally. MiVOLO's YOLO detector, training code,
dataset tooling, and face detection flow are intentionally not vendored or used.
