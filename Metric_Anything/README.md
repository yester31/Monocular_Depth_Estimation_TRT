# Metric Anything
- **[MetricAnything: Scaling Metric Depth Pretraining with Noisy Heterogeneous Sources](https://arxiv.org/pdf/2601.22054)**
- **[MetricAnything official GitHub](https://github.com/metric-anything/metric-anything)**

## How to Run (Pytorch)

1. set up a virtual environment.
    ```
    cd Metric_Anything
    git clone https://github.com/metric-anything/metric-anything.git
    mv metric-anything metric_anything

    conda create -n metric_anything -y python=3.11
    conda activate metric_anything

    pip install -r metric_anything/models/student_pointmap/requirements.txt
    pip install onnxsim
    pip install onnx
    pip install onnxscript
    ```


2. download pretrained checkpoints.
    ```
    mkdir -p checkpoints
    wget https://huggingface.co/yjh001/metricanything_student_depthmap/resolve/main/student_depthmap.pt -P checkpoints
    wget https://huggingface.co/yjh001/metricanything_student_pointmap/resolve/main/student_pointmap.pt -P checkpoints
    ```
    

3. run the original pytorch model on test images.
    ```
    python metric_anything/models/student_pointmap/infer.py \
        --input metric_anything/models/student_pointmap/example_images \
        --output metric_anything/models/student_pointmap/output_infer \
        --weights checkpoints/student_pointmap.pt \
        --save_glb \
        --save_ply
    ```


4. check pytorch model inference performance
    ```
    cd ..
    python infer.py
    ```


## How to Run (TensorRT)

1. generate onnx file
    ```
    python onnx_export.py
    ```

2. build tensorrt model and run
    ```
    conda activate trte
    python onnx2trt.py
    ```


**[Back](../README.md)** 