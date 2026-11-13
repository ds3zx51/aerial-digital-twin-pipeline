# Runtime data

This directory is intentionally empty in Git. For the optional AdvTabDDPM
example, stage 11 copies its admitted adapter output into an
isolated run copy and materializes the NumPy/pickle inputs required by that
plug-in. The repository architecture does not require other AI plug-ins to use
this schema. Do not commit Colosseum datasets,
raw AODT tensors, or generated run data here.
