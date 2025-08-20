# CoTracker3
- **[CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos](https://arxiv.org/abs/2410.11831)**
- **[CoTracker3 official GitHub](https://github.com/facebookresearch/co-tracker)**
- video -> video depth

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd CoTracker3
git clone https://github.com/facebookresearch/co-tracker.git
cd co-tracker/

# Create a new conda environment with Python 3.11
conda create -n cotracker3 -y python=3.11

# Activate the created environment
conda activate cotracker3 

# Install the required Python packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install opencv-python
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard
pip install onnxsim
pip install onnx
pip install onnxscript
```

2. download pretrained checkpoints.
```
mkdir -p checkpoints
cd checkpoints
# download the online (multi window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
# download the offline (single window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
cd ..
```

3. run the original pytorch model on test images.
```
python online_demo.py
python online_demo.py --video_path ../../video/video2.mp4 --grid_size 25

```

4. check pytorch model inference performance
```
cd ..
python infer.py
```
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

**[Back](../README.md)** 