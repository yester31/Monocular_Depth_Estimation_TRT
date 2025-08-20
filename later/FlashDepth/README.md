# FlashDepth
- **[FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution](https://arxiv.org/abs/2504.07093)**
- **[FlashDepth official GitHub](https://github.com/Eyeline-Labs/FlashDepth)**
- video -> video depth

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd FlashDepth
git clone https://github.com/Eyeline-Labs/FlashDepth.git
cd FlashDepth

conda deactivate 
conda env remove -n flashdepth

# Create a new conda environment with Python 3.10
conda create -n flashdepth -y python=3.11

# Activate the created environment
conda activate flashdepth 

# Install the required Python packages
bash setup_env.sh

export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
export TORCH_CUDA_ARCH_LIST="8.6"  # for rtx3060
pip uninstall -y flash_attn
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attn
pip install -v --no-build-isolation .
cd ..
```

2. download pretrained checkpoints.
```
# FlashDepth (Full)
wget https://huggingface.co/Eyeline-Labs/FlashDepth/resolve/main/flashdepth/iter_43002.pth -P configs/flashdepth

# FlashDepth-L
wget https://huggingface.co/Eyeline-Labs/FlashDepth/resolve/main/flashdepth-l/iter_10001.pth -P configs/flashdepth-l

# FlashDepth-S
wget https://huggingface.co/Eyeline-Labs/FlashDepth/resolve/main/flashdepth-s/iter_14001.pth -P configs/flashdepth-s
```
- Generally, FlashDepth-L is most accurate and FlashDepth (Full) is fastest, but we recommend using FlashDepth-L when the input resolution is low (e.g. short side less than 518).

3. run the original pytorch model on test images.
```
torchrun train.py --config-path configs/flashdepth inference=true eval.random_input=examples/video1.mp4 eval.outfolder=output1
torchrun train.py --config-path configs/flashdepth inference=true eval.random_input=examples/video2.mp4 eval.outfolder=output2
```

4. check pytorch model inference performance
```
cd ..
python ../gen_video2imgs.py
python infer.py
```
- 518 x 518 
- 249 iterations time: 27.3043 [sec]
- Average FPS: 9.12 [fps]
- Average inference time: 109.66 [msec]

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