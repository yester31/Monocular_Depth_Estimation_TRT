# DKT (Diffusion Knows Transparency)
- **[Diffusion Knows Transparency: Repurposing Video Diffusion for Transparent Object Depth and Normal Estimation
](https://arxiv.org/abs/2512.23705)**
- **[DKT official GitHub](https://github.com/Daniellli/DKT)**
- 2d image -> depth

## How to Run (Pytorch)

1. set up a virtual environment.
    ```
    cd DKT
    git clone https://github.com/Daniellli/DKT.git
    cd DKT
    conda create -n dkt -y python=3.11
    conda activate dkt
    pip install -r requirements.txt
    pip install onnxsim
    pip install onnx
    pip install onnxscript

    huggingface-cli login
    ```

2. download pretrained checkpoints.
    ```
    mkdir -p checkpoints
    wget https://huggingface.co/Daniellesry/DKT-Depth-1-3B-v1.1/resolve/main/TransPhy3D_cleargrasp_HISS_DREDS_1.3B_depth_70K_lora.safetensors -P checkpoints
    ```

3. run the original pytorch model on test images.
    ```
    python run.py --encoder vits --img-path assets/examples --outdir results/vits
    ```

4. check pytorch model inference performance
    ```
    cd ..
    python infer.py
    ```


## How to Run (TensorRT)

1. generate onnx file
    ```
    python onnx_export.py
    ```

2. build tensorrt model and run
    ```
    conda activate trte
    python onnx2trt.py
    ```


**[Back](../README.md)** 