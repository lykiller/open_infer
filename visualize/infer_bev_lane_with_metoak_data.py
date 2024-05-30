import os
import cv2

from core.camera import MultiCamera
from data.image.scene_image_manager import SceneImageLoader
from mono_task.model.load_model import init_bev_lane3d_model

if __name__ == "__main__":

    date_time = "2024-01-17"

    root_samples_path = rf"G:\BEVDATA\extracted_bag_data\{date_time}\samples"
    root_sweeps_path = rf"G:\BEVDATA\extracted_bag_data\{date_time}\sweeps"

    multi_camera = MultiCamera(r"D:\文档\标注\camera_params_without_discoffs\camera_params_without_discoffs")

    sensor_name_list = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT"
    ]

    bev_lane3d_model = init_bev_lane3d_model()
    for scene_name in os.listdir(root_samples_path):
        scene_image_manager = SceneImageLoader(
            sample_path=os.path.join(root_samples_path, scene_name),
            sweep_path=os.path.join(root_sweeps_path, scene_name),
            scene_name=scene_name,
            sensor_name_list=sensor_name_list
        )
        for frame_idx in range(scene_image_manager.max_frame):
            visual_image = bev_lane3d_model.visual_result(next(scene_image_manager))

            cv2.imshow("frame_idx", visual_image)
            cv2.waitKey(0)
