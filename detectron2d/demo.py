import os
from detetron import OnnxDetectron

from utils import get_video_list, mkdir
from detectron2d import det2d_class_names


def run_fcos_det():
    onnx_name = "fcos-det-roadEdge.onnx"
    onnx_path = os.path.join("onnx", onnx_name)
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
    videos_dir = r"E:\视频标识牌"
    video_list = get_video_list(os.path.join(videos_dir, "videos"))
    save_dir = os.path.join(videos_dir, onnx_name)
    mkdir(save_dir)
    for video_name in video_list:
        predictor.inference_video(os.path.join(videos_dir, "videos", video_name), save_dir=None)


def run_byd_det_seg():
    onnx_name = "byd_halfcrop_last_0610.onnx"
    onnx_path = os.path.join(r"D:\自动驾驶\比亚迪全地形分割\onnx", onnx_name)

    roi_shape = (1280, 360)
    top = (0.05, 0.10)
    bottom = (0.15, 0.30)

    crop = [[0, 360], [1280, 720]]
    show_size = (1280, 720)
    input_size = (768, 320)

    predictor = OnnxDetectron(
        model_path=onnx_path,
        input_node_name="image",
        output_node_name=["cls_id", "cls_score", "reg_pred"],
        input_size=input_size,
        show_size=show_size,
        crop=crop,
        color_mode="rgb",
        class_names=['manhole cover', 'iron plate', 'storm drain', 'road damage', 'speed bump']
    )

    videos_dir = r"E:\视频标识牌"
    video_list = get_video_list(os.path.join(videos_dir, "videos"))
    save_dir = os.path.join(videos_dir, onnx_name)
    mkdir(save_dir)
    for video_name in video_list:
        predictor.inference_video(os.path.join(videos_dir, "videos", video_name), save_dir=None)


if __name__ == "__main__":
    run_fcos_det()
