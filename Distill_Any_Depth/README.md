# Distill Any Depth
- **[Distill Any Depth: Distillation Creates a Stronger Monocular Depth Estimator](https://arxiv.org/abs/2502.19204)**
- **[Distill Any Depth official GitHub](https://github.com/Westlake-AGI-Lab/Distill-Any-Depth)**
- 2d image -> depth

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd Distill_Any_Depth
git clone https://github.com/Westlake-AGI-Lab/Distill-Any-Depth.git
cd Distill-Any-Depth

# Create a new conda environment with Python 3.10
conda create -n distill-any-depth -y python=3.10

# Activate the created environment
conda activate distill-any-depth

# Install the required Python packages
pip install -r requirements.txt

# Navigate to the Detectron2 directory and install it
cd detectron2
pip install -e .

cd ..
pip install -e .
```

2. download pretrained checkpoints.
```
mkdir -p checkpoint/small
wget https://huggingface.co/xingyang1/Distill-Any-Depth/resolve/main/small/model.safetensors -P checkpoint/small

mkdir -p checkpoint/base
wget https://huggingface.co/xingyang1/Distill-Any-Depth/resolve/main/base/model.safetensors -P checkpoint/base

mkdir -p checkpoint/large
wget https://huggingface.co/xingyang1/Distill-Any-Depth/resolve/main/large/model.safetensors -P checkpoint/large

mkdir -p checkpoint/Large-2w-iter
wget https://huggingface.co/xingyang1/Distill-Any-Depth/resolve/main/Distill-Any-Depth-Dav2-Teacher-Large-2w-iter/model.safetensors -P checkpoint/Large-2w-iter
```
3. run the original pytorch model on test images.
```
bash scripts/00_infer.sh
```

4. check pytorch model inference performance
```
cd ..
python infer.py
```
- 518 x 518 input
- see results/example_small_dad_torch.png    
- 100 iterations time: 1.9052 [sec]
- Average FPS: 52.49 [fps]
- Average inference time: 19.05 [msec]
--------------------------------------------------------------------

## How to Run (TensorRT)

1. generate onnx file

```
pip install onnx
pip install onnxsim
python onnx_export.py
// a file 'distill_any_depth_small.onnx' will be generated in onnx directory.
```

2. build tensorrt model and run

```
conda activate trte
pip install matplotlib
python onnx2trt.py
// a file 'distill_any_depth_small_fp16.engine' will be generated in engine directory.
```
- 518 x 518 input
- see results/example_dad_TRT.jpg 
- 100 iterations time: 0.8240 [sec]
- Average FPS: 121.36 [fps]
- Average inference time: 8.24 [msec]

**[Back](../README.md)** 