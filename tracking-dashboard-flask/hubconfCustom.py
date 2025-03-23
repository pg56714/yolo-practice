from asone.utils.draw import draw_ui_box
import asone
from asone import ASOne
import cv2
import time
import random


def video_detection(path_x="", conf_=0.25):
    total_detections = 0
    names = [
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
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    filter_classess = None

    if filter_classess:
        filter_classess = [filter_classess]

    dt_obj = ASOne(
        tracker=asone.BYTETRACK,
        detector=asone.YOLOV8N_PYTORCH,
        weights=None,
        use_cuda=False,
    )

    start_time = time.time()
    video_path = path_x

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for j in range(nframes):
        total_detections = 0
        ret, img0 = video.read()
        fps_x = int((j + 1) / (time.time() - start_time))
        if ret:
            detected = dt_obj.detecter(source=img0, conf_thres=conf_, iou_thres=0.45)

            bboxes_xyxy = detected.dets.bbox
            scores = detected.dets.score
            class_ids = detected.dets.class_ids

            for i in range(len(bboxes_xyxy)):
                total_detections += 1
                box = bboxes_xyxy[i]
                label = f"{names[int(class_ids[i])]} {scores[i]:.2f}"
                draw_ui_box(
                    box,
                    img0,
                    label=label,
                    color=colors[int(class_ids[i])],
                    line_thickness=3,
                )

            yield img0, fps_x, img0.shape, total_detections

        else:
            break
    video.release()
