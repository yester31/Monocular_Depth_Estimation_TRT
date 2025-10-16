# Map Anything
- **[MapAnything: Universal Feed-Forward Metric 3D Reconstruction](https://arxiv.org/pdf/2509.13414)**
- **[MapAnything official GitHub](https://github.com/facebookresearch/map-anything)**
- 2d image -> depth, (point cloud, extrinsic, intrinsic, Camera pose)

## How to Run (Pytorch)

1. set up a virtual environment.
    ```
    cd Map_Anything
    git clone https://github.com/facebookresearch/map-anything.git
    cd map-anything

    # Create a new conda environment with Python 3.11
    conda create -n mapanything  -y python=3.11

    # Activate the created environment
    conda activate mapanything 

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
    pip install -e .
    # pip install -e ".[all]"
    # pre-commit install

    pip install onnx
    pip install onnxsim
    ```

2. run the original pytorch model on test images.
    ```

    ```

3. check pytorch model inference performance
    ```
    cd ..
    python infer.py
    ```

--------------------------------------------------------------------
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