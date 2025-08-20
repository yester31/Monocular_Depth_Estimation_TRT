# SIDepth
- **[Scale-Invariant Monocular Depth Estimation via SSI Depth](https://yaksoy.github.io/papers/SIG24-SI-Depth.pdf)**
- **[SIDepth official GitHub](https://github.com/compphoto/SIDepth)**
- 2d image -> depth, 

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd SIDepth
git clone https://github.com/compphoto/SIDepth.git
cd SIDepth

# Create a new conda environment with Python 3.11
conda create -n sidepth -y python=3.11 

# Activate the created environment
conda activate sidepth

# Install the required Python packages
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install onnx
pip install onnxsim
pip install gdown
```

2. download pretrained checkpoints.
```
cd weights
gdown --fuzzy https://drive.google.com/file/d/1jbcgAkKNXxQO37iwjjWbEYCcVQVERwTc/view?usp=drive_link 
unzip weights.zip
```

3. run the original pytorch model on test images.
```
python eval.py -i ../../data/ -o results/ -m SI --colorize
python eval.py -i ../../data/ -o results/ -m SSI --colorize
python eval.py -i ../../data/ -o results/ -m SSIBase --colorize
```

4. check pytorch model inference performance
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
pip install matplotlib
python onnx2trt.py
// a file '.engine' will be generated in engine directory.
```
- 518 x 518 input


**[Back](../README.md)** 