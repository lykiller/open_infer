import numpy as np
import os
import cv2
from core.utils import get_video_list
from mono_task.model import TerrainOnnxPredictor

if __name__ == "__main__":
    video_dir = r"D:\dataset\terrain_data\videos"
    video_list = get_video_list(video_dir)
    show_size = (1280, 720)
    save_dir = None

    ipm_x = 960
    ipm_y = 320

    pt_img = np.float32(
        [
            [290, 0],
            [640, 0],
            [0, 320],
            [960, 320]
        ]
    )

    pt_world = np.float32(
        [[0, 0], [ipm_x, 0], [0, ipm_y], [ipm_x, ipm_y]]
    )
    matrix = cv2.getPerspectiveTransform(pt_img, pt_world)

    onnx_name = "lcnet_2.0_clean_data_v5.onnx"
    onnx_path = os.path.join(r"D:\自动驾驶\比亚迪全地形分割\onnx", onnx_name)

    roi_shape = (960, 320)
    top = (0.04, 0.24)
    bottom = (0.06, 0.36)

    crop = [[160, 400], [1120, 720]]
    show_size = (1280, 720)
    input_size = (768, 256)
    class_names = ["background", "asphalt", "gravel", "earth", "grassland", "mud", "wading", "rock", "sand",
                   "light_snow", "deep_snow"]
    predictor = TerrainOnnxPredictor(
        model_path=onnx_path,
        input_node_name="data",
        input_size=input_size,
        show_size=show_size,
        crop=crop,
        normalize=[[128., 128., 128.], [128., 128., 128.]],
        color_mode="rgb",
        class_names=class_names
    )
    print(matrix)
    for video_path in video_list:
        cap = cv2.VideoCapture(video_path)
        while True:
            ret_val, frame = cap.read()
            print(ret_val)
            if ret_val:
                frame = cv2.resize(frame, show_size)
                ipm_output = cv2.warpPerspective(frame[400:720, 160:1120], matrix, (ipm_x, ipm_y))
                cv2.imshow("src_images", frame[400:720, 160:1120])
                cv2.imshow("ipm_result", ipm_output)
                predictor.inference_to_bev(frame, matrix)
                cv2.waitKey(0)
            else:
                cap.release()
                break
