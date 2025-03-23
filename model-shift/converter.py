import os
import io
import torch
from ultralytics import YOLO

def convert_file(filename, file_bytes, field):
    dir_pathOutput = "models"

    if not os.path.exists(dir_pathOutput):
        os.makedirs(dir_pathOutput)
        print("Folder created.")

    file_nameOutput = filename
    file_pathOutput = os.path.join(dir_pathOutput, file_nameOutput)

    with open(file_pathOutput, "wb") as f:
        f.write(file_bytes.getbuffer())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(file_pathOutput)
    model.to(device)

    def export_model(model, format, device=0):
        try:
            success = model.export(format=format, device=device)
            if not success:
                raise ValueError(f"Export failed for format: {format}")
            
            with open(success, "rb") as f:
                file_bytesNew = io.BytesIO(f.read())
            file_data = io.BytesIO(file_bytesNew.getbuffer())
            file_data.seek(0)
            return os.path.basename(success), file_data
        except Exception as e:
            print(f"Error during {format} export: {e}")
            raise

    if field == "TorchScript":
        return export_model(model, "torchscript")
    elif field == "ONNX":
        return export_model(model, "onnx")
    elif field == "TensorRT":
        # TensorRT-specific handling
        try:
            return export_model(model, "engine")
        except ValueError as e:
            print(f"TensorRT export failed: {e}")
            return None, None
    elif field == "CoreML":
        return export_model(model, "coreml")
    elif field == "TF Lite":
        try:
            return export_model(model, "tflite")
        except ValueError as e:
            print(f"TF Lite export failed: {e}")
            return None, None
    elif field == "Tensorflow":
        try:
            success = model.export(format="saved_model", device=device, keras=True)
            if not success:
                raise ValueError("Tensorflow export failed.")
            
            successNew = os.path.join(success, "saved_model.pb")
            with open(successNew, "rb") as f:
                file_bytesNew = io.BytesIO(f.read())
            file_data = io.BytesIO(file_bytesNew.getbuffer())
            file_data.seek(0)
            return os.path.basename(successNew), file_data
        except Exception as e:
            print(f"Tensorflow export failed: {e}")
            return None, None
    elif field == "OpenVINO":
        try:
            success = model.export(format="openvino", device=device)
            filebase, _ = os.path.splitext(filename)
            successNew = os.path.join(success, filebase + ".xml")
            with open(successNew, "rb") as f:
                file_bytesNew = io.BytesIO(f.read())
            file_data = io.BytesIO(file_bytesNew.getbuffer())
            file_data.seek(0)
            return os.path.basename(successNew), file_data
        except Exception as e:
            print(f"OpenVINO export failed: {e}")
            return None, None
    else:
        file_data = io.BytesIO(file_bytes.getbuffer())
        file_data.seek(0)
        return filename, file_data
