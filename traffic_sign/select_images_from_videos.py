import os
from utils import get_video_list, mkdir
import random

from traffic_sign.traffic_sign_predictor import TrafficSignPredictorV2
from traffic_sign.load_model import init_cls_model, init_ocr_model
from detectron2d.load_model import init_fcos_det

if __name__ == "__main__":
    det_predictor = init_fcos_det()
    cls_predictor = init_cls_model()
    ocr_predictor = init_ocr_model()
    show_size = (1280, 720)

    tsr_predictor = TrafficSignPredictorV2(det_predictor, cls_predictor, ocr_predictor, show_size)
    videos_dir = r"D:\dataset\terrain_data\videos_0711"
    video_list = get_video_list(os.path.join(videos_dir, "videos"))
    random.shuffle(video_list)

    save_dir = os.path.join(videos_dir, "sample_images")
    mkdir(save_dir)
    for video_name in video_list:
        tsr_predictor.inference_video_for_images(video_name, save_dir=save_dir)
