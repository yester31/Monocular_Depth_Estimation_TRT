# Align3R
- **[Align3R: Aligned Monocular Depth Estimation for Dynamic Videos](https://arxiv.org/abs/2412.03079)**
- **[Align3R official GitHub](https://github.com/jiah-cloud/Align3R)**
- two 2d images -> depth, point cloud, Camera pose

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd Align3R
git clone https://github.com/jiah-cloud/Align3R.git
cd Align3R

# Create a new conda environment with Python 3.11
conda create -n align3r -y python=3.11 cmake=3.14.0

# Activate the created environment
conda activate align3r 

# Install the required Python packages
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_optional.txt
pip install onnx
pip install onnxsim
pip install evo
pip install gdown
pip install sam2

# Compile the cuda kernels for RoPE (as in CroCo v2)
# Change tokens.type() to tokens.scalar_type() at line 101 in croco/models/curope/kernels.cu. 
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../

# Install the monocular depth estimation model Depth Pro and Depth Anything V2
# Depth Pro
cd third_party/ml-depth-pro
pip install -e .
source get_pretrained_models.sh
# Depth Anything V2
pip install transformers==4.41.2
cd ../../
```

2. download pretrained checkpoints.
```
# Align3R_DepthAnythingV2_ViTLarge_BaseDecoder_512_dpt
gdown --fuzzy https://drive.google.com/file/d/1-qhRtgH7rcJMYZ5sWRdkrc2_9wsR1BBG/view?usp=sharing -O cyun9286/

# Align3R_DepthPro_ViTLarge_BaseDecoder_512_dpt
gdown --fuzzy https://drive.google.com/file/d/1PPmpbASVbFdjXnD3iea-MRIHGmKsS8Vh/view?usp=sharing -O cyun9286/

# Raft
cd third_party/RAFT
gdown --fuzzy https://drive.google.com/file/d/1KJxQ7KPuGHlSftsBCV1h2aYpeqQv3OI-/view?usp=drive_link -O models/
```

3. run the original pytorch model on test images.
```
export PYTHONPATH=$PYTHONPATH:Align3R
CUDA_VISIBLE_DEVICES=0 python tool/demo.py --input ../../../video_frames_50 --output_dir demo_dir --seq_name scene_name --interval=50
```

4. check pytorch model inference performance
```
cd ..
python infer.py
```
- 518 x 518 input
- 10 iterations time : 394.1773 [sec]
- Average FPS: 0.03 [fps]
- Average Latency: 39417.73 [msec]
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
pip install matplotlib
python onnx2trt.py
// a file '.engine' will be generated in engine directory.
```
- 518 x 518 input
- 100 iterations time: 17.3173 [sec]
- Average FPS: 5.77 [fps]
- Average Latency: 173.17 [msec]

**[Back](../README.md)** 