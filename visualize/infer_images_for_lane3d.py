import os

import cv2

import random

from mono_task.model.load_model import init_mono_lane3d_model, imread

if __name__ == "__main__":
    image_path_dir = r"G:\BEVDATA\extracted_bag_data\2023-12-21\samples"
    anno_path_dir = r"G:\公开数据集\BEV_车道线数据集\OpenlaneV1\lane3d_1000_v1.2\lane3d_1000\validation"

    show_size = (1280, 720)
    down_sample = 1
    lane3d_predictor = init_mono_lane3d_model()

    scene_name_list = os.listdir(image_path_dir)
    random.shuffle(scene_name_list)

    save_path = r"G:\BEVDATA\mono_lane3d_test_visual_video"
    os.makedirs(save_path, exist_ok=True)

    for scene_name in scene_name_list:
        sample_scene_image_name_list = os.listdir(os.path.join(
            image_path_dir, scene_name, "CAM_FRONT"))
        sweep_scene_image_name_list = os.listdir(os.path.join(
            image_path_dir.replace("samples", "sweeps"), scene_name, "CAM_FRONT"))
        scene_image_name_list = sample_scene_image_name_list + sweep_scene_image_name_list
        scene_image_name_list = sorted(scene_image_name_list)
        print(scene_name, " begin")
        per_save_path = os.path.join(save_path, scene_name + ".mp4")
        writer = cv2.VideoWriter(
            per_save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            (1680, 720),
            True
        )
        for idx, image_name in enumerate(scene_image_name_list):
            if idx % 5 == 0:
                image_src = imread(os.path.join(
                    image_path_dir, scene_name, "CAM_FRONT", image_name))
            else:
                image_src = imread(os.path.join(
                    image_path_dir.replace("samples", "sweeps"), scene_name, "CAM_FRONT", image_name))
            if image_src is not None:
                # json_path = os.path.join(anno_path_dir, scene_name, image_name.replace(".jpg", ".json"))
                # if os.path.exists(json_path):
                # data = json.load(open(json_path))
                save_frame = lane3d_predictor.visual_result_using_metoak(image_src, True)
                writer.write(save_frame)
        writer.release()
