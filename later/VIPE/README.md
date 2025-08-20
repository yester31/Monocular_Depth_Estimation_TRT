# VIPE
- **[ViPE: Video Pose Engine for 3D Geometric Perception](https://research.nvidia.com/labs/toronto-ai/vipe/assets/paper.pdf)**
- **[VIPE official GitHub](https://github.com/nv-tlabs/vipe)**
- video -> video depth

## How to Run (Pytorch)

1. set up a virtual environment.
```
cd VIPE
git clone https://github.com/nv-tlabs/vipe.git
cd vipe

# Create a new conda environment with Python 3.11
conda create -n vipe -y python=3.11

# Activate the created environment
conda activate vipe 

# Install the required Python packages
pip install -r envs/requirements.txt
pip install --no-build-isolation -e .
```

2. download pretrained checkpoints.
```

```

3. run the original pytorch model on test images.
```
# Using the ViPE CLI
vipe infer ../../video/video2.mp4

# visualize the results
vipe visualize vipe_results/ 

# Using the run.py script
python run.py pipeline=default streams=raw_mp4_stream streams.base_path=YOUR_VIDEO_OR_DIR_PATH
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