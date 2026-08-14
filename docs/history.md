# Project history

This document keeps decisions that explain the current repository shape. It is historical context, not an active task list.

## Refactoring plan (2026-07-31)

The repository was reorganized around model specifications, shared runtime utilities, generated reports, and explicit model contracts. The original assumption that every model could be compared at one common resolution was discarded because several models resize internally or require a different geometry convention.

## Repository cleanup (2026-07-31 to 2026-08-15)

- Export and build entry points were standardized.
- Model READMEs were normalized around input, output, preprocessing, postprocessing, checkpoint, and license information.
- HyDen and the unused `later/` backup were removed from the active model set.
- The optional `video/video2.mp4` demo input was removed and added to `.gitignore`.

## Directory rename (2026-08-13)

Model directories now use stable, lowercase names. References in scripts, reports, and documentation were updated together so that model IDs remain consistent.

## Historical measurements

Earlier native PyTorch timings and resolution sweeps are retained in reports and retired tools. They are not directly comparable with the current TensorRT tables because hardware, preprocessing, checkpoints, or input sizes differed. The most important lessons were:

- UniDepth V2 and UniK3D are sensitive to their native 672×896 input geometry.
- Many apparent accuracy differences are global scale differences rather than structural depth errors.
- A single build can be misleading even with a fixed GPU clock; repeated builds are required for builder tuning.
- Metric3D V2 produces canonical depth unless camera calibration is supplied.
