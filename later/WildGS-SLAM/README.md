# WildGS-SLAM
- **[WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments](https://arxiv.org/abs/2504.03886)**
- **[WildGS-SLAM official GitHub](https://github.com/GradientSpaces/WildGS-SLAM)**
- video -> video depth

## How to Run (Pytorch)

conda deactivate 
conda env remove -n wildgs -y 

1. set up a virtual environment.
```
cd WildGS-SLAM
git clone https://github.com/GradientSpaces/WildGS-SLAM.git
cd WildGS-SLAM
git submodule update --init --recursive

# Create a new conda environment with Python 3.11
conda create -n wildgs -y python=3.11

# Activate the created environment
conda activate wildgs 

# Install the required Python packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch-scatter
pip install -U xformers --index-url https://download.pytorch.org/whl/cu128

python -m pip install -e thirdparty/lietorch/
python -m pip install -e thirdparty/diff-gaussian-rasterization-w-pose/
python -m pip install -e thirdparty/simple-knn/

python -c "import torch; import lietorch; import simple_knn; import diff_gaussian_rasterization; print(torch.cuda.is_available())"

python -m pip install -e .
python -m pip install -r requirements.txt

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

pip install gdown
pip install munch
```

2. download pretrained checkpoints.
```
gdown --fuzzy https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing -O pretrained/
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