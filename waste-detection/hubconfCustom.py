from asone.utils.draw import draw_ui_box
import asone
from asone import ASOne
import cv2
import random
import numpy as np
from tracker import *


def video_detection(path_x="", conf_=0.25):
    names = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    filter_classes = None

    if filter_classes:
        filter_classes = [filter_classes]

    dt_obj = ASOne(
        tracker=asone.BYTETRACK,
        detector=asone.YOLOV8N_PYTORCH,
        weights="best.pt",
        use_cuda=False,
    )
    video_path = path_x

    video = cv2.VideoCapture(video_path)
    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = EuclideanDistTracker()

    for j in range(nframes):
        degradable = []
        glass = []
        metal = []

        ret, img0 = video.read()

        if ret:

            detectionTracker = []
            detected = dt_obj.detecter(
                source=img0, conf_thres=0.25, iou_thres=0.45, filter_classes=None
            )
            output_detected = detected.dets.bbox
            scores = detected.dets.score
            class_ids = detected.dets.class_ids

            for idx, box in enumerate(output_detected):
                label = f"{names[int(class_ids[idx])]} {scores[idx]:.2f}"
                detectionTracker.append([box[0], box[1], box[2], box[3], label])

            boxes_ids = tracker.update(detectionTracker)
            for box_id in boxes_ids:
                x, y, w, h, id, label_ = box_id
                if label_.split(" ")[0] == "BIODEGRADABLE":
                    degradable.append(id)
                    draw_ui_box(
                        [x, y, w, h],
                        img0,
                        label=str(id) + " " + label_,
                        color=colors[int(0)],
                        line_thickness=3,
                    )
                if label_.split(" ")[0] == "GLASS":
                    glass.append(id)
                    draw_ui_box(
                        [x, y, w, h],
                        img0,
                        label=str(id) + " " + label_,
                        color=colors[int(2)],
                        line_thickness=3,
                    )
                if label_.split(" ")[0] == "METAL":
                    metal.append(id)
                    draw_ui_box(
                        [x, y, w, h],
                        img0,
                        label=str(id) + " " + label_,
                        color=colors[int(3)],
                        line_thickness=3,
                    )

            yield img0, len(list(set(degradable))), len(list(set(glass))), len(
                list(set(metal))
            )

        else:
            break

    video.release()
