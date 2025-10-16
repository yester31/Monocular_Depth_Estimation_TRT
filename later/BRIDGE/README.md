# BRIDGE
- **[BRIDGE - Building Reinforcement-Learning Depth-to-Image Data Generation Engine for Monocular Depth Estimation](https://arxiv.org/pdf/2509.25077)**
- **[BRIDGE official GitHub](https://github.com/lnbxldn/Bridge)**
- 2d image -> depth, 

## How to Run (Pytorch)

1. set up a virtual environment.
    ```
    cd BRIDGE
    git clone https://github.com/lnbxldn/Bridge.git
    cd Bridge

    # Create a new conda environment with Python 3.11
    conda create -n bridge -y python=3.11

    # Activate the created environment
    conda activate bridge

    # Install the required Python packages
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
    pip install matplotlib
    pip install opencv-python
    pip install onnx
    pip install onnxsim
    ```

2. download pretrained checkpoints.
    ```
    mkdir -p checkpoints
    wget https://huggingface.co/Dingning/BRIDGE/resolve/main/bridge.pth -P checkpoints
    ```

3. check pytorch model inference performance
    ```
    cd ..
    python infer.py
    ```
    - 518 x 518 input

--------------------------------------------------------------------

## How to Run (TensorRT)

1. generate onnx file

```
python onnx_export.py
// a file '.onnx' will be generated in onnx directory.
```

2. build tensorrt model and run

```
conda activate trte
python onnx2trt.py
// a file '.engine' will be generated in engine directory.
```
- 518 x 518 input


**[Back](../README.md)** 