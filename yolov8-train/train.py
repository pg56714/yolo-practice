from ultralytics import YOLO
import torch.multiprocessing as mp

if __name__ == '__main__':
    # Freeze support is needed for multiprocessing on Windows
    mp.freeze_support()

    # Initialize YOLO model
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(
        data="datasets/data.yaml",
        # epochs=10,
        epochs=100,
    )

    # Validate the model
    results = model.val()

    # Export the model
    success = model.export(format="ONNX")
