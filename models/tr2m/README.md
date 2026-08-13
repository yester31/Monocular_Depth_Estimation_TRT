# TR2M
- **[TR2M: Transferring Monocular Relative Depth to Metric Depth with Language Descriptions and Dual-Level Scale-Oriented Contrast](https://arxiv.org/abs/2506.13387)** (CVPR 2026)
- **[TR2M official GitHub](https://github.com/BeileiCui/TR2M)**
- 2d image **+ text description** -> metric depth

## What makes this one different

Every other model here takes an image and returns depth. TR2M takes an image
**and a sentence describing the scene**, and uses the sentence to decide the
metric scale. It is not a depth network in the usual sense: it runs Depth
Anything for relative depth, then predicts a per-pixel scale `A` and shift `B`
from DINOv2 features fused with a CLIP text embedding, and returns

```
D_metric = 1 / (A * D_rel + B)
```

Three consequences worth knowing before reading its numbers:

| | |
| :--- | :--- |
| **Two inputs** | `image [1,3,434,560]` and `text_features [1,1,768]`. The only model here whose engine has more than one input. |
| **CLIP is outside the engine** | The text embedding changes only when the prompt does, so it stays an input and 900 MB of CLIP weights stay out of the graph. `onnx_export.py` writes the embedding to `onnx/<name>_text_features.npy`; `onnx2trt.py` reads it back and never loads CLIP. |
| **434x560, not 518x518** | Multiples of 14, as the architecture requires, but a different amount of work from most rows in the comparison table. Compare milliseconds with that in mind. |

The prompt used for the benchmark is fixed and recorded beside the embedding.
A measurement whose input depends on what someone typed is not a measurement;
changing the prompt means re-running the export, which records the new
sentence with the new vector.

**Its errors are two models' errors.** Metric depth here is Depth Anything's
relative depth times a predicted rescaling, so it inherits whatever the
relative model got wrong and adds whatever the rescaling gets wrong.

## TensorRT performance

<!-- BENCH:BEGIN -->
Not measured yet. Run `python run.py build tr2m` on the machine with the
engines and regenerate with `python compare.py`.
<!-- BENCH:END -->

## How to Run (Pytorch)

1. clone upstream and set up its environment.
    ```
    cd models/tr2m
    git clone https://github.com/BeileiCui/TR2M.git

    conda create -n tr2m -y python=3.10
    conda activate tr2m
    cd TR2M
    pip install -r requirements.txt
    pip install onnx onnxsim
    ```

2. download the weights.
    ```
    python download_weights.py
    ```
    Three separate downloads: the TR2M ScaleMap head (~75 MB, in the upstream
    release), Depth Anything ViT-S (~100 MB, from Hugging Face), and DINOv2
    ViT-L plus CLIP ViT-L/14 (~1.2 GB and ~900 MB, cached by torch.hub and the
    `clip` package on first use).

--------------------------------------------------------------------

## How to Run (TensorRT)

1. generate the onnx file and the prompt embedding.
    ```
    conda activate tr2m
    python onnx_export.py
    // onnx/tr2m_da_s_vitl_434x560.onnx  (~1.4 GB of fused weights)
    // onnx/tr2m_da_s_vitl_434x560_text_features.npy
    ```
    Tracing runs on the CPU on purpose -- the fused graph carries about 1.4 GB
    of weights and tracing it on a small GPU is the first thing that fails.

2. build the engine and benchmark.
    ```
    conda activate trte
    python onnx2trt.py
    // engine/tr2m_da_s_vitl_434x560_fp16.engine
    ```
    On a GPU with little memory, pass a lower `opt_level` to `get_engine`; the
    build is the memory-hungry part, not the inference.

## Licence

| Item | Licence |
| :--- | :--- |
| Upstream code ([BeileiCui/TR2M](https://github.com/BeileiCui/TR2M)) | see upstream; **`pos_embed.py` is CC BY-NC-SA 4.0 (non-commercial)** |
| Depth Anything ViT-S checkpoint | Apache-2.0 |
| DINOv2 | Apache-2.0 |
| CLIP (OpenAI) | MIT |
| Conversion scripts in this folder | MIT (this repository) |

The upstream clone lives at `models/tr2m/TR2M/` and is **not** committed --
`.gitignore` excludes it. Nothing from upstream is copied into this
repository, which is what keeps a non-commercial file in that tree from
reaching this repository's MIT licence. Do not vendor it.

Upstream licences checked 2026-08-14. This table is a convenience summary --
the upstream LICENSE files are authoritative.

**[Back](../../README.md)**
