import sys
import argparse
import asone
import torch
import os


def main(args):
    filter_classes = args.filter_classes

    if filter_classes:
        filter_classes = ["person"]

    if args.use_cuda and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    if sys.platform.startswith("darwin"):
        detector = asone.YOLOV8N_MLMODEL
    else:
        detector = asone.YOLOV8N_PYTORCH

    detect = asone.ASOne(
        tracker=asone.NORFAIR,
        detector=detector,
        weights=args.weights,
        # sam_weights=args.sam_weights,
        use_cuda=args.use_cuda,
    )

    track = detect.video_tracker(
        args.video_path,
        output_dir=args.output_dir,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        display=args.display,
        draw_trails=args.draw_trails,
        filter_classes=filter_classes,
        class_names=None,
    )

    # for model_output in track:
    #     bbox_xyxy = model_output.dets.bbox
    #     ids = model_output.dets.ids
    #     scores = model_output.dets.score
    #     class_ids = model_output.dets.class_ids
    #     frame = model_output.info.image
    #     frame_num = model_output.info.frame_no
    #     fps = model_output.info.fps
    #     print(frame_num)

    output_file_path = os.path.join("tracking_results.txt")
    output_file = open(output_file_path, "w")

    for model_output in track:
        bbox_xyxy = model_output.dets.bbox
        ids = model_output.dets.ids
        scores = model_output.dets.score
        class_ids = model_output.dets.class_ids
        frame = model_output.info.image
        frame_num = model_output.info.frame_no
        fps = model_output.info.fps
        print(frame_num)

        num_detections = len(bbox_xyxy)
        for i in range(num_detections):
            bb = bbox_xyxy[i]
            obj_id = ids[i] if ids is not None else -1
            class_id = class_ids[i]

            x1, y1, x2, y2 = bb
            width = x2 - x1
            height = y2 - y1
            output_line = f"{frame_num}, {obj_id}, {x1}, {y1}, {width}, {height}, -1, -1, -1, -1\n"
            output_file.write(output_line)
    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument(
        "--cpu",
        default=True,
        action="store_false",
        dest="use_cuda",
        help="run on cpu if not provided the program will run on gpu.",
    )
    parser.add_argument(
        "--no_save",
        default=True,
        action="store_false",
        dest="save_result",
        help="whether or not save results",
    )
    parser.add_argument(
        "--no_display",
        default=True,
        action="store_false",
        dest="display",
        help="whether or not display results on screen",
    )
    parser.add_argument(
        "--output_dir", default="data/results", help="Path to output directory"
    )
    parser.add_argument(
        "--draw_trails",
        action="store_true",
        default=False,
        help="if provided object motion trails will be drawn.",
    )
    parser.add_argument("--filter_classes", default=None, help="Filter class name")
    parser.add_argument("-w", "--weights", default=None, help="Path of trained weights")
    parser.add_argument("--sam_weights", default=None, help="Path of sam weights")
    parser.add_argument(
        "-ct",
        "--conf_thres",
        default=0.25,
        type=float,
        help="confidence score threshold",
    )
    parser.add_argument(
        "-it", "--iou_thres", default=0.45, type=float, help="iou score threshold"
    )

    args = parser.parse_args()

    main(args)
