# reports/

Machine-readable benchmark results and the table generated from them.

```
reports/
  bench/                      one JSON per measured configuration
    <model>_<HxW>_<profile>_<variant>_<precision>.json
  comparison.md               generated - do not edit
```

## How a number gets here

1. Run a model's `onnx2trt.py`. It times through `core/bench.py` and writes one
   JSON into `bench/` at the end.
2. Run `python compare.py` at the repo root to regenerate `comparison.md`.

`python compare.py --check` exits non-zero when `comparison.md` no longer
matches the files in `bench/`, which is the useful thing to run in CI.

## Why the files are committed

They are the evidence behind the table. A result records the GPU, driver,
TensorRT version, input size, precision and every per-iteration sample, so a
figure that looks wrong six months from now can be traced to the conditions it
was taken under. That was the problem with the numbers previously typed into
each model's README: nothing recorded what produced them, so two of them could
not be compared even in principle.

## Reading the table

Rows are grouped by input size and **latency does not compare across groups**.
Three models do not run at 518: `depth_pro` is fixed at 1536x1536,
`metric3d_v2` at 616x1064, `moge_2` at 388x518. Attention cost grows faster
than pixel count -- measured on `depth_anything_ac`, 1.35x the pixels cost
1.70x the time -- so normalising by resolution does not rescue the comparison.

A result is only comparable to another taken on the same GPU. `compare.py`
prints a warning at the top of the table when it finds more than one device.

See `docs/model_contracts.md` for what each model's output actually means;
several are not directly comparable as *values* even when the latency is.
