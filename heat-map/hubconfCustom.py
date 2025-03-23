import asone
from asone import ASOne
import cv2
import time
import random as random
import numpy as np


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
global_img_np_array = None


def compute_color_for_labels(label):
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return img


def video_detection(path_x='', conf_=0.25,dt_obj=None):
    total_detections = 0
    filter_classes = ['person']
    # dt_obj = ASOne(
    #     tracker=asone.DEEPSORT,
    #     detector=asone.YOLOV8N_PYTORCH,
    #     weights=None,
    #     use_cuda=False
    # )

    video_path = path_x

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    global global_img_np_array

    super_imposed_img = None
    global_img_np_array = np.ones([height, width], dtype=np.uint32)
    track_fn = dt_obj.video_tracker(path_x, conf_thres=conf_, iou_thres=0.45,
                                  display=False, filter_classes=filter_classes)
                                  
    for model_output in track_fn:
        total_detections = 0
        bbox_xyxy = model_output.dets.bbox
        ids = model_output.dets.ids
        scores = model_output.dets.score
        class_ids = model_output.dets.class_ids
        frame = model_output.info.image
        frame_num = model_output.info.frame_no
        fps = model_output.info.fps
        total_detections = len(bbox_xyxy)

        if len(bbox_xyxy) > 0:

            draw_boxes(frame, bbox_xyxy, ids)

            for i, box in enumerate(bbox_xyxy):
                x1, y1, x2, y2 = [int(i) for i in box]

                global_img_np_array[y1:y2, x1:x2] += 1

            global_img_np_array_norm = ((global_img_np_array - global_img_np_array.min()) / (
                global_img_np_array.max() - global_img_np_array.min()))*255

            global_img_np_array_norm = global_img_np_array_norm.astype('uint8')

            global_img_np_array_norm = cv2.GaussianBlur(
                global_img_np_array_norm, (9, 9), 0)
            heatmap_img = cv2.applyColorMap(
                global_img_np_array_norm, cv2.COLORMAP_JET)

            super_imposed_img = cv2.addWeighted(
                heatmap_img, 0.5, frame, 0.5, 0)

            yield super_imposed_img, total_detections
