# Depth Pro

## How to Run

1. set up a virtual environment.
```
cd Depth_Pro
git clone https://github.com/apple/ml-depth-pro
cd ml-depth-pro
conda create -n depth-pro -y python=3.9
conda activate depth-pro
pip install -e .
```

2. download pretrained checkpoints.
```
source get_pretrained_models.sh   # Files will be downloaded to `checkpoints` directory.
```

3. run the original pytorch model on a single test image.
```
depth-pro-run -i ./data/example.jpg -o ./results
```

4. check pytorch model inference performance
```
cd ..
python infer.py
```
- 1536 x 1536 input
- see results/example_Torch.jpg    
- 20 iterations time: 124.4491 [sec]   
- Average FPS: 0.16 [fps]   
- Average inference time: 6222.46 [msec]   
--------------------------------------------------------------------

5. generate onnx file

```
pip install onnx
python onnx_export.py
// a file 'dinov2l16_384_cuda.onnx' will be generated in onnx directory.
```

6. build tensorrt model and run

```
conda activate trte
pip install matplotlib
python onnx2trt.py
// a file 'dinov2l16_384_fp16.engine' will be generated in engine directory.
```
- 1536 x 1536 input
- see results/example_TRT.jpg  
- 20 iterations time: 14.1636 [sec]
- Average FPS: 1.41 [fps]
- Average inference time: 708.18 [msec]

https://github.com/apple/ml-depth-pro
