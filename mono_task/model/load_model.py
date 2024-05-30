import os

from fisheye_task.apa_predictoer import ApaOnnxPredictor
from mono_task.model.detetron import OnnxDetectron, det2d_class_names
from mono_task.model.traffic_sign_predictor import TrafficSignOnnxClassifier, OcrOnnxPredictor, TrafficSignPredictor
from mono_task.model.terrain_predictor import TerrainOnnxPredictor, terrain_class_names
from mono_task.model.orientation_classifier import OrientationOnnxClassifier
from mono_task.model.mono_lane3d_predictor import MonoLane3DPredictor
from bev_task.bev_lane3d_predictor import BevLane3DOnnxPredictor

onnx_dir = r"D:\open_infer\onnx"


def init_orientation_classifier():
    input_size = (40, 40)
    onnx_name = "orientation_cls_40x40_20230915.onnx"
    onnx_path = os.path.join(onnx_dir, onnx_name)
    predictor = OrientationOnnxClassifier(
        model_path=onnx_path,
        input_node_name="data",
        output_node_name=["label", "score"],
        input_size=input_size,
        color_mode="rgb",
        normalize=[[0., 0., 0.], [255., 255., 255.]]
    )
    return predictor


def init_terrain_seg_model():
    onnx_name = "V0.5.2.20240307_Terrain_cls12_256x768.onnx"
    onnx_path = os.path.join(onnx_dir, onnx_name)

    input_size = (768, 256)
    show_size = (1280, 720)
    predictor = TerrainOnnxPredictor(
        model_path=onnx_path,
        input_node_name="data",
        output_node_name=["terrain"],
        input_size=input_size,
        show_size=show_size,
        crop=[[160, 400], [1120, 720]],
        normalize=[],
        color_mode="rgb",
        class_names=terrain_class_names,
    )
    return predictor


def init_det_model():
    onnx_name = "fcos-det-roadEdge.onnx"
    onnx_path = os.path.join(onnx_dir, onnx_name)
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


def init_apa_model():
    onnx_name = "center-det.onnx"
    onnx_path = os.path.join(onnx_dir, onnx_name)
    predictor = ApaOnnxPredictor(
        input_size=(512, 512),
        input_node_name="input",
        output_node_name=["points_offsets", "cls_score", "box_reg"],
        normalize=[[127.5, 127.5, 127.5], [127.5, 127.5, 127.5]],
        strides=[4],
        show_size=[1024, 1024],
        model_path=onnx_path,
        if_transpose=False
    )
    return predictor


def init_tsr_cls_model():
    onnx_name = "tsr_cls_32x32_20231114.onnx"
    onnx_path = os.path.join(onnx_dir, onnx_name)
    input_size = (32, 32)

    predictor = TrafficSignOnnxClassifier(
        model_path=onnx_path,
        input_node_name="data",
        output_node_name=["label", "score"],
        input_size=input_size,
        color_mode="rgb",
        normalize=[[0., 0., 0.], [255., 255., 255.]]
    )

    return predictor


def init_tsr_ocr_model():
    onnx_name = "tsr_ocr_32x32_20240112.onnx"
    onnx_path = os.path.join(onnx_dir, onnx_name)
    input_size = (32, 32)

    predictor = OcrOnnxPredictor(
        model_path=onnx_path,
        input_node_name="data",
        output_node_name=["label", "score", "reg"],
        input_size=input_size,
        color_mode="rgb",
        normalize=[[0., 0., 0.], [255., 255., 255.]]
    )

    return predictor


def init_tsr_model():
    return TrafficSignPredictor(
        cls_predictor=init_tsr_cls_model(),
        ocr_predictor=init_tsr_ocr_model()
    )


def init_mono_lane3d_model():
    onnx_name = "V0.1.0.20240319_MonoLane3D_320x640.onnx"
    onnx_path = os.path.join(onnx_dir, onnx_name)
    input_size = (640, 320)

    return MonoLane3DPredictor(
        model_path=onnx_path,
        input_node_name="data",
        output_node_name=["heatmap", "u_offset", "depth", "relative_id", "category_id"],
        normalize=[[128., 128., 128.], [128., 128., 128.]],
        input_size=input_size,
        show_size=(1280, 720),
        crop=[[0, 240], [3840, 2160]]

    )


def init_bev_lane3d_model():
    onnx_name = "V0.1.1.20240401_MonoLane3D_320x640.onnx"
    onnx_path = os.path.join(onnx_dir, onnx_name)
    crop_list = [
        [[0, 240], [3840, 2160]],
        [[0, 120], [1920, 1080]],
        [[0, 120], [1920, 1080]],
        [[0, 120], [1920, 1080]],
        [[0, 120], [1920, 1080]],
        [[0, 120], [1920, 1080]],

    ]
    return BevLane3DOnnxPredictor(
        model_path=onnx_path,
        input_node_name="data",
        output_node_name=["heatmap", "u_offset", "depth", "height", "relative_id", "category_id"],
        normalize=[[128., 128., 128.], [128., 128., 128.]],
        input_size=(640, 320),
        feat_size=(160, 80),
        sensor_name_list=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                          "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"],

        crop_list=crop_list,
        scale_u_list=[24, 12, 12, 12, 12, 12, 12],
        scale_v_list=[24, 12, 12, 12, 12, 12, 12],
        camera_config_path=r"D:\文档\标注\camera_params_without_discoffs\camera_params_without_discoffs",
        vis_threshold=0.1,


    )
