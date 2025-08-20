# STream3R
- **[STream3R: Scalable Sequential 3D Reconstruction with Causal Transformer](https://arxiv.org/abs/2508.10893)**
- **[STream3R official GitHub](https://github.com/NIRVANALAN/STream3R)**
- 

## How to Run (Pytorch)

conda deactivate 
conda env remove -n wildgs -y 

1. set up a virtual environment.
```
cd STream3R
git clone  https://github.com/NIRVANALAN/STream3R.git
cd STream3R

# Create a new conda environment with Python 3.11
conda create -n stream3r python=3.11 cmake=3.14.0 -y

# Activate the created environment
conda activate stream3r 

# Install the required Python packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
pip install -e .
```

2. download pretrained checkpoints.
```
wget https://huggingface.co/yslan/STream3R/resolve/main/model.safetensors?download=true -P checkpoints/
```

3. run the original pytorch model on test images.
```
python app.py
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