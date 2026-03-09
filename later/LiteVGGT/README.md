# LiteVGGT
- **[LiteVGGT: Boosting Vanilla VGGT via Geometry-aware Cached Token Merging](https://arxiv.org/pdf/2512.04939)**
- **[LiteVGGT official GitHub](https://github.com/GarlicBa/LiteVGGT-repo)**

## How to Run (Pytorch)

1. set up a virtual environment.
    ```
    cd LiteVGGT
    git clone https://github.com/GarlicBa/LiteVGGT-repo.git
    mv LiteVGGT-repo LiteVGGT_repo
    cd LiteVGGT_repo

    conda create -n litevggt -y python=3.11
    conda activate litevggt 

    pip install -r requirements.txt
    pip install onnxsim
    pip install onnx
    pip install onnxscript

    pip install --no-build-isolation transformer_engine[pytorch]
    ```


2. download pretrained checkpoints.
    ```
    mkdir -p checkpoints
    wget https://huggingface.co/ZhijianShu/LiteVGGT/resolve/main/te_dict.pt -P checkpoints
    ```
    

3. run the original pytorch model on test images.
    ```
    python run_demo.py \
        --ckpt_path ../checkpoints/te_dict.pt \
        --img_dir kitchen/images \
        --output_dir results2
    ```


4. check pytorch model inference performance
    ```
    cd ..
    python infer.py
    ```
- 518 x 518 input
- 100 iterations time : 53.1731 [sec]
- Average FPS: 1.88 [fps]
- Average inference time: 531.73 [msec]
- max : 3.540 , min : 0.906

## How to Run (TensorRT)

1. generate onnx file
    ```
    python onnx_export.py
    ```

2. build tensorrt model and run
    ```
    conda activate trte
    pip install git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183
    pip install trimesh
    python onnx2trt.py
    ```
- 518 x 518 input
- 100 iterations time : 20.3727 [sec]
- Average FPS: 4.91 [fps]
- Average inference time: 203.73 [msec]
- max : 3.526 , min : 0.904

**[Back](../README.md)** 