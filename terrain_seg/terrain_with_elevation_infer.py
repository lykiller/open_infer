import os

from utils import get_image_list, get_video_list, mkdir
from terrain_seg.terrain_predictor import TerrainWithElevationOnnxPredictor


def ddr_infer(image_path, save_dir, mode="video", infer_mode=1):
    onnx_name = "ddrnet_shared_v2.1.21_10cls_288x1024.onnx"
    onnx_path = os.path.join(r"E:\比亚迪全地形分割\onnx", onnx_name)

    roi_shape = (1024, 288)
    top = (0.05, 0.10)
    bottom = (0.15, 0.30)

    crop = [[128, 432], [1152, 720]]
    show_size = (1280, 720)
    input_size = (1024, 288)
    class_names = ["background", "asphalt", "gravel", "earth", "grassland", "mud", "wading", "rock", "sand",
                   "light_snow", "deep_snow"]
    predictor = TerrainWithElevationOnnxPredictor(
        model_path=onnx_path,
        input_node_name="input",
        output_node_name=["output0", "output1"],
        input_size=input_size,
        show_size=show_size,
        crop=crop,
        color_mode="rgb",
        class_names=class_names
    )

    from driving_area import emulate_straight_mask

    driving_area_mask, line_xy = emulate_straight_mask(roi_shape, top, bottom)
    save_path = None
    if save_dir is not None:
        save_path = os.path.join(save_dir, onnx_name)
        mkdir(save_path)
    if mode == "image":
        image_list = get_image_list(image_path)
        predictor.inference_images(image_list, driving_area_mask, line_xy, save_dir=save_path)
    elif mode == "video":
        video_list = get_video_list(src_path)
        for video_name in video_list:
            predictor.inference_video(
                os.path.join(src_path, video_name),
                driving_area_mask,
                line_xy,
                save_dir=save_path,
                infer_mode=infer_mode
            )


if __name__ == "__main__":
    scene = "terrain"
    # src_path = rf"F:\{scene}_test"
    # save_dir = rf"F:\{scene}_test_result"
    src_path = r"F:\terrain_test_result\src_images"
    save_dir = r"F:\terrain_test_result\ddr_sample"
    if scene == "terrain":
        infer_mode = 0
    else:
        infer_mode = 1
    ddr_infer(src_path, save_dir, mode="image", infer_mode=infer_mode)
