# model-shift

## Installation

```bash
conda create --name modelshift python=3.10 -y
conda activate modelshift

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install tensorrt-cu12 --use-pep517

pip install -r requirements.txt
```

## Model

https://huggingface.co/Ultralytics/YOLOv8/blob/main/yolov8n.pt

## Notes

- **mac**

  - **CoreML**: Apple’s machine learning framework for deploying models across iOS, macOS, watchOS, and tvOS platforms.

- **cross-platform (Windows, Linux, macOS)**
  - **TorchScript**: PyTorch’s method for exporting models, enabling them to be used in other languages and environments.
  - **ONNX**: An open format designed for exporting models from one framework to another, facilitating interoperability between different deep learning tools.
  - **TensorRT**: NVIDIA’s deep learning inference optimizer, designed to accelerate model inference on NVIDIA GPUs.
  - **TF Lite**: Google’s lightweight machine learning framework optimized for mobile and edge devices.
  - **TensorFlow Serving**: Google’s model server, designed for serving machine learning models in production environments.
  - **OpenVINO**: Intel’s deep learning inference optimizer, tailored for accelerating inference on Intel hardware.

## Usage

```
yolo predict model=yolov8n.mlmodel source=video.mp4 save=True show=True conf=0.75
```
