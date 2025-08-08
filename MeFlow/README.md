# MeFlow
- **[Memory-Efficient Optical Flow via Radius-Distribution Orthogonal Cost Volume](https://arxiv.org/pdf/2312.03790)**
- **[MeFlow official GitHub](https://github.com/gangweix/MeFlow)**
- 2d image -> optical flow

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd MeFlow
git clone https://github.com/gangweix/MeFlow.git
cd MeFlow

# Create a new conda environment with Python 3.11
conda create -n meflow -y python=3.11

# Activate the created environment
conda activate meflow

# Install the required Python packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install gdown
pip install tensorboard
pip install opencv-python
pip install scipy
pip install pillow==9.5.0
```

2. download pretrained checkpoints.
```
# things.pth
gdown --fuzzy https://drive.google.com/file/d/1_Ug6FZhK1V0spDPu578QG65uxTJ-ENQL/view?usp=drive_link -O pretrained_models/

# sintel.pth
gdown --fuzzy https://drive.google.com/file/d/1AcHbcWHRja-0IzxMVxZv9YbpCmI4L_EW/view?usp=drive_link -O pretrained_models/

# kitti.pth
gdown --fuzzy https://drive.google.com/file/d/1KuwYWOnJakECsPKOl6nj2E4V7bpJVH0j/view?usp=drive_link -O pretrained_models/

# chairs.pth
gdown --fuzzy https://drive.google.com/file/d/1NxBqOmpuxtVAA_R5GzNYdRXLLFoK2-9m/view?usp=drive_link -O pretrained_models/
```

3. run the original pytorch model on test images.
```
bash ./scripts/demo.sh
```

4. check pytorch model inference performance
```
cd ..
python ../gen_video2imgs.py
python infer.py
```
- input size: 288 x 512, iters: 20
- original
    - 249 iterations time: 60.6272 [sec]
    - Average FPS: 4.11 [fps]
    - Average inference time: 243.48 [msec]

- wrapper for onnx/trt
    - 249 iterations time: 192.1871 [sec]
    - Average FPS: 1.30 [fps]
    - Average inference time: 771.84 [msec]
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
- Engine build done! (197.0 [min] 7.06 [sec])
- 249 iterations time: 13.4447 [sec]
- Average FPS: 18.52 [fps]
- Average inference time: 53.99 [msec]

**[Back](../README.md)** 