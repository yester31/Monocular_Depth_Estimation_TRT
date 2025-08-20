# WAFT
- **[WAFT: Warping-Alone Field Transforms for Optical Flow](https://arxiv.org/abs/2506.21526)**
- **[WAFT official GitHub](https://github.com/princeton-vl/WAFT)**
- 2d image -> optical flow

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd WAFT
git clone https://github.com/princeton-vl/WAFT.git
cd WAFT

# Create a new conda environment with Python 3.12
conda create -n waft -y python=3.12

# Activate the created environment
conda activate waft

# Install the required Python packages
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -U xformers --index-url https://download.pytorch.org/whl/cu128
pip install gdown
```

2. download pretrained checkpoints.
```
# chairs-things.pth
gdown --fuzzy https://drive.google.com/file/d/1l-W5BSvhx2eV-lk8dMIPeI1yhxISNX0O/view?usp=drive_link -O ckpts/

# tar-c-t-kitti.pth
gdown --fuzzy https://drive.google.com/file/d/1hzpxBe80BmPCXjo9DvSszdCRsio5vLaB/view?usp=drive_link -O ckpts/

# chairs.pth
gdown --fuzzy https://drive.google.com/file/d/1YO5R2Ap7yijwGDIcHSmZfRTzy-IcTWEz/view?usp=drive_link -O ckpts/
```

3. run the original pytorch model on test images.
```
python evaluate --cfg config/chairs-things.json --ckpt ckpts/chairs-things.pth --dataset sintel
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