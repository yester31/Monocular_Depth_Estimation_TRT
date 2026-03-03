# Depth Anything V3
- **[Depth Anything V3](https://arxiv.org/pdf/2511.10647)**
- **[Depth Anything V3 official GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)**
- 2d image -> depth

## How to Run (Pytorch)

1. set up a virtual environment.
    ```
    cd Depth_Anything_V3
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3
    mv Depth-Anything-3 Depth_Anything_V3
    cd Depth_Anything_V3
    conda create -n dav3 -y python=3.11
    conda activate dav3

    pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
    pip install xformers --index-url https://download.pytorch.org/whl/cu128
    pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 # for gaussian head

    pip install -e ".[all]" # ALL
    pip install onnxsim
    pip install onnx
    pip install onnxscript
    ```

2. download pretrained checkpoints.
    ```
    mkdir -p depth-anything/DA3METRIC-LARGE
    wget https://huggingface.co/depth-anything/DA3METRIC-LARGE/resolve/main/model.safetensors -P depth-anything/DA3METRIC-LARGE
    wget https://huggingface.co/depth-anything/DA3METRIC-LARGE/resolve/main/config.json -P depth-anything/DA3METRIC-LARGE
    ```

3. check pytorch model inference performance
    ```
    cd ..
    python infer.py
    ```
    - 518 x 518 input
    - 100 iterations time : 11.1942 [sec]
    - Average FPS: 8.93 [fps]
    - Average inference time: 111.94 [msec]
    - max : 2.38608 , min : 0.34367

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
    - 518 x 518 input
    - 100 iterations time: 4.9711 [sec]
    - Average FPS: 20.12 [fps]
    - Average inference time: 49.71 [msec]
    - max : 2.37500 , min : 0.34187

## Reference
- [DA3-ROS2-CPP-TensorRT](https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt)


**[Back](../README.md)** 