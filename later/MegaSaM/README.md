# MegaSaM
- **[MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos](https://arxiv.org/abs/2412.04463)**
- **[MegaSaM official GitHub](https://github.com/mega-sam/mega-sam)**
- 

## How to Run (Pytorch)

conda deactivate 
conda env remove -n wildgs -y 

1. set up a virtual environment.
```
cd MegaSaM
git clone --recursive git@github.com:mega-sam/mega-sam.git
cd mega-sam

# Create a new conda environment with Python 3.11
conda create -n megasam -y python=3.11

# Activate the created environment
conda activate megasam 

# Install the required Python packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

pip install ninja
pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

cd base; python setup.py install

pip install gdown
```

2. download pretrained checkpoints.
```
# Download DepthAnything checkpoint to mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth -P Depth-Anything/checkpoints/

# Download and include RAFT checkpoint at mega-sam/cvd_opt/raft-things.pth
gdown --fuzzy https://drive.google.com/file/d/1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM/view?usp=drive_link -O cvd_opt/

# Sintel data
gdown --fuzzy https://drive.google.com/file/d/1bSGX7JY73M3HzMS6xsJizRkPH-NQLPOf/view?usp=sharing -O sintel/
cd sintel
unzip Sintel.zip
cd .. 
# DAVIS data
gdown --fuzzy https://drive.google.com/file/d/1y5XItnTTgZJqRSOpG48v1FuHvPgaAvw8/view?usp=sharing -O DAVIS/
cd DAVIS
unzip DAVIS.zip
cd .. 
```

3. run the original pytorch model on test images.
```
bash scripts_downloading/download_demo_data.sh
python run.py  ./configs/Dynamic/Wild_SLAM_Mocap/crowd_demo.yaml

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