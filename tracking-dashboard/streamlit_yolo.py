import streamlit as st
import asone
from asone import ASOne
from PIL import Image
import cv2
import tempfile
import os
import numpy as np
import time

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def main():
    # Streamlit app configuration
    st.set_page_config(page_title="YOLOv8 Tracking Dashboard", layout="wide")
    st.title("YOLOv8 Tracking Dashboard")

    # Side Panel
    st.sidebar.header("Setting")
    use_webcam = st.sidebar.checkbox("Use Webcam")
    conf_thres = st.sidebar.slider("Confidence Value", 0.0, 1.0, 0.25)
    tracker = st.sidebar.selectbox("Tracker", ["DeepSORT", "ByteTrack", "NoFair"])
    selected_classes = st.sidebar.multiselect("Classes to detect", COCO_CLASSES)

    if use_webcam:
        video_path = 0
    else:
        uploaded_file = st.sidebar.file_uploader("Upload video", type=["mp4", "avi"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        else:
            video_path = None

    if video_path is not None:
        if tracker == "DeepSORT":
            tracker = asone.DEEPSORT
        elif tracker == "ByteTrack":
            tracker = asone.BYTETRACK
        else:
            tracker = asone.NORFAIR

        dt_obj = ASOne(
            tracker=tracker,
            detector=asone.YOLOV8N_PYTORCH,
            weights=None,
            use_cuda=False,
        )

        track_fn = dt_obj.video_tracker(
            video_path,
            conf_thres=conf_thres,
            iou_thres=0.5,
            display=False,
            filter_classes=selected_classes if selected_classes else None,
            class_names=None,
        )

        video_cap = cv2.VideoCapture(video_path)
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = os.path.join("data/results/", "results.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_palceholder = st.empty()
        text_container = st.container()

        with text_container:
            st.markdown(
                """
                <style>
                    .info {
                        font-size: 20px;
                        font-weight: bold;
                        color:#4f8bf9;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )
            frame_num_placeholder = st.empty()
            num_objects_placeholder = st.empty()
            fps_placeholder = st.empty()

        while True:
            start_time = time.time()
            ret, frame = video_cap.read()
            if not ret:
                break

            model_output = next(track_fn)
            bbox_xyxy = model_output.dets.bbox
            ids = model_output.dets.ids
            class_ids = model_output.dets.class_ids
            frame_num = model_output.info.frame_no

            for i, bbox in enumerate(bbox_xyxy):
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                class_name = COCO_CLASSES[class_ids[i]]
                label = f"{class_name} {ids[i]}"
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                frame = cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame_palceholder.image(frame, width=800, output_format="PNG")
            end_time = time.time()
            processing_time = end_time - start_time
            actual_fps = 1 / processing_time
            actual_fps = int(actual_fps)

            frame_num_placeholder.markdown(
                f"<p class='info'>Frame number: {frame_num}</p>", unsafe_allow_html=True
            )
            num_objects_placeholder.markdown(
                f"<p class='info'>Number of objects being tracked: {len(ids)}</p>",
                unsafe_allow_html=True,
            )
            fps_placeholder.markdown(
                f"<p class='info'>FPS: {actual_fps}</p>", unsafe_allow_html=True
            )

            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame)

        video_cap.release()
        out.release()


if __name__ == "__main__":
    main()
