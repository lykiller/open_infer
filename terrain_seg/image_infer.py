import os
from terrain_predictor import TerrainOnnxPredictor
from utils import get_image_list, get_video_list, mkdir
from terrain_seg import *
from terrain_seg.load_model import load_model

def infer_images(terrain_predictor, onnx_name_t):
    src_path = r"D:\dataset\terrain_data\images_error_0712"
    src_image_path = os.path.join(src_path, "src_images")
    image_list = get_image_list(src_image_path)
    # save_dir = os.path.join(r"E:\比亚迪全地形分割\ORFD_Dataset_ICRA2022_ZIP", onnx_name)
    save_dir = os.path.join(src_path, onnx_name_t)
    mkdir(save_dir)

    terrain_predictor.inference_images(image_list, None, None, save_dir=save_dir)


def infer_videos(terrain_predictor, onnx_name_t):
    src_path = r"F:\videos_202305"
    src_video_path = os.path.join(src_path, "videos")
    video_list = get_video_list(src_video_path)
    # save_dir = os.path.join(r"E:\比亚迪全地形分割\ORFD_Dataset_ICRA2022_ZIP", onnx_name)
    save_dir = os.path.join(src_path, onnx_name_t)
    mkdir(save_dir)
    for video_path in video_list:
        terrain_predictor.inference_video(video_path, None, None, save_dir=save_dir)


def infer_videos_for_images(terrain_predictor):
    src_path = r"E:\7-8avi"
    src_video_path = os.path.join(src_path, "videos")
    video_list = get_video_list(src_video_path)
    # save_dir = os.path.join(r"E:\比亚迪全地形分割\ORFD_Dataset_ICRA2022_ZIP", onnx_name)

    save_dir = os.path.join(src_path, "sample_images")

    mkdir(save_dir)
    for video_path in video_list:
        terrain_predictor.inference_video_save_images(video_path, None, None, save_dir=save_dir)


if __name__ == "__main__":
    predictor = load_model()
    # infer_videos(predictor, onnx_name)
    infer_videos_for_images(predictor)
    # infer_images(predictor, onnx_name)
