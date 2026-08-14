# Measurement findings

This is a concise record of the evidence behind the current defaults. Detailed raw measurements remain in `reports/` JSON and Markdown tables.

## P0 — measurement metadata

Every published row records model ID, checkpoint/variant, input shape, precision, TensorRT version, GPU, timing statistic, and output semantics. Missing measurements are represented explicitly rather than being treated as zero.

## P1 — shared preprocessing

Common runtime plumbing is shared, but model-specific resize, padding, normalization, coordinate reversal, and postprocessing remain declared in each `spec.json`. Unifying these operations blindly changes outputs.

## P2 — evaluation inputs

Evaluation uses an explicit manifest and preserves the original image aspect ratio where the model contract requires it. Local staging data under `reports/inputs/` is ignored and can be regenerated.

## P3 — ground-truth evaluation

The DIODE indoor subset is used for the main accuracy table. Relative and canonical outputs are evaluated with the appropriate scale-invariant or scale-only protocol; they are not silently reported as metric metres. Metric3D V2 clips after applying its scale conversion, matching the upstream order.

## P4 — precision comparison

FP16 is the production baseline. Some models cannot be built in FP32 with the current TensorRT graph. Any precision change must pass the accuracy check before it is accepted.

## P5 — deferred decisions

- uint8 preprocessing is retained as a documented variant, not the baseline.
- Resolution sweeps showed that larger input is not universally more accurate. The native geometry of UniDepth V2 and UniK3D remains the default.
- The scale-only report is separate from metric accuracy so that a good relative shape is not confused with calibrated distance.

## P6 — builder tuning

The default builder level remains 3. StreamVGGT level 4 is the only accepted exception after repeated builds and accuracy verification. MoGe-2 level 5 showed high run-to-run spread and a worse mean, so it was rejected. GPU clock locking reduces noise but does not make every high-level search reproducible.

## P7 — visualization

The default demo is a qualitative gallery with a per-panel 2–98% display range. `--view metric` is a separate audit view. Metric3D uses measured `--fx` when available; `--metric3d-fx-from-depth-pro` is explicitly labeled as an estimated focal length.

## Verification lessons

The regression suite, report consistency check, and engine/ONNX comparison are separate checks. A passing speed table does not prove accuracy, and a passing ONNX comparison does not prove the output is metric. Run all relevant checks before publishing a new engine.
