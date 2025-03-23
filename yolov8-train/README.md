# yolov8-train

## Installation

```
conda create --name yolov8-train python=3.10 -y
conda activate yolov8-train

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install ultralytics
pip install onnx
```

---

https://pytorch.org/get-started/locally/

conda info --envs

conda env remove --name yolov8-train -y

## Datasets

https://universe.roboflow.com/mcjeong-postech-ac-kr/soccer-214sc/dataset/3

## Model

https://huggingface.co/Ultralytics/YOLOv8/blob/main/yolov8n.pt

## Train

```
python train.py
```

## Test

```
python test.py
```
