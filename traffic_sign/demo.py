import os
import cv2
import random

from utils import get_image_list, get_video_list, mkdir, imread

from detectron2d.detetron import OnnxDetectron
from detectron2d import det2d_class_names
from traffic_sign.traffic_sign_predictor import TrafficSignPredictorV2, OcrOnnxPredictor, TrafficSignOnnxClassifier
from traffic_sign import tsr_class_names
from tracker.kalman_filter import KalmanFilter
from traffic_sign.load_model import init_cls_model, init_ocr_model


def init_fcos_det():
    onnx_name = "fcos-det-roadEdge.onnx"
    onnx_path = os.path.join(r"D:\open_infer\onnx", onnx_name)
    input_size = (512, 288)
    show_size = (1280, 720)

    predictor = OnnxDetectron(
        model_path=onnx_path,
        input_node_name="input",
        output_node_name=["cls_id", "cls_score", "box_reg"],
        input_size=input_size,
        show_size=show_size,
        color_mode="rgb",
        class_names=det2d_class_names,
        normalize=[[127.5, 127.5, 127.5], [127.5, 127.5, 127.5]]
    )

    return predictor


def infer_crop_images(image_path_dir, cls_predictor, ocr_predictor, save_dir=None):
    image_list = get_image_list(image_path_dir)
    random.shuffle(image_list)
    for image_name in image_list:
        image_np = imread(image_name)
        label, score = cls_predictor.inference(image_np)
        num_str = ""
        if label in [1, 2, 3, 4]:
            num_str = ocr_predictor.inference(image_np)
        print(tsr_class_names[label], num_str, image_name)
        cv2.imshow("image", image_np)
        cv2.waitKey(0)


def infer_videos(det_predictor, cls_predictor, ocr_predictor, show_size):
    tsr_predictor = TrafficSignPredictorV2(det_predictor, cls_predictor, ocr_predictor, show_size)
    kalman_filter = KalmanFilter()
    videos_dir = r"D:\dataset\terrain_data\videos_20230711"
    video_list = get_video_list(os.path.join(videos_dir, "videos"))
    random.shuffle(video_list)
    save_dir = os.path.join(videos_dir, "visual_results")
    mkdir(save_dir)
    for video_name in video_list:
        tsr_predictor.inference_video(os.path.join(videos_dir, "videos", video_name), kalman_filter, save_dir=save_dir)


if __name__ == "__main__":
    det_predictor = init_fcos_det()
    cls_predictor = init_cls_model()
    ocr_predictor = init_ocr_model()
    show_size = (1280, 720)
    # infer_crop_images(r"F:\traffic_sign_cls\highest_speed_limit__最高限速", cls_predictor, ocr_predictor)
    infer_videos(det_predictor, cls_predictor, ocr_predictor, show_size)
