# Depth Pro
- **[Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073)**
- **[Depth Pro official GitHub](https://github.com/apple/ml-depth-pro)**
- 2d image -> depth & focal length

## How to Run (Pytorch)

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


## How to Run (TensorRT)

1. generate onnx file

```
pip install onnx
python onnx_export.py
// a file 'depth_pro_dynamo.onnx' will be generated in onnx directory.
```

2-1. build tensorrt model and run single image process

```
conda activate trte
pip install matplotlib
```

```
python onnx2trt.py
// a file 'depth_pro_dynamo_fp16.engine' will be generated in engine directory.
```
- 1536 x 1536 input
- see results/example_TRT.jpg  
- 20 iterations time: 14.1636 [sec]
- Average FPS: 1.41 [fps]
- Average inference time: 708.18 [msec]

2-2. build tensorrt model and run video process

```
python onnx2trt_video.py
```

2-3. build tensorrt model and run wabcam process

```
python onnx2trt_webcam.py
```

**[Back](../README.md)** 