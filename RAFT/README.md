# RAFT
- **[RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039)**
- **[RAFT official GitHub](https://github.com/princeton-vl/raft)**
- 2d image -> optical flow

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd RAFT
https://github.com/princeton-vl/RAFT.git
cd RAFT

# Create a new conda environment with Python 3.11
conda create -n raft -y python=3.11

# Activate the created environment
conda activate raft

# Install the required Python packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python
pip install matplotlib
pip install tensorboard
pip install scikit-image
```

2. download pretrained checkpoints.
```
./download_models.sh
```

3. run the original pytorch model on test images.
```
python demo.py --model=models/raft-things.pth --path=demo-frames
```

4. check pytorch model inference performance
```
cd ..
python ../gen_video2imgs.py
python infer.py
```

- input size: 288 x 512, iters: 20
- original
    - 249 iterations time: 26.1436 [sec]
    - Average FPS: 9.52 [fps]
    - Average inference time: 104.99 [msec]

- wrapper for onnx/trt
    - 249 iterations time: 41.2404 [sec]
    - Average FPS: 6.04 [fps]
    - Average inference time: 165.62 [msec]

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

- input size: 288 x 512, iters: 20
- 249 iterations time: 11.3397 [sec]
- Average FPS: 21.96 [fps]
- Average inference time: 45.54 [msec]

**[Back](../README.md)** 