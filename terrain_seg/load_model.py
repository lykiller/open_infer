import os

from terrain_seg import *
from terrain_seg.terrain_predictor import TerrainOnnxPredictor


def load_model():
    onnx_name = "teacher_0821.onnx"
    onnx_path = os.path.join(r"D:\open_infer\onnx", onnx_name)

    predictor = TerrainOnnxPredictor(
        model_path=onnx_path,
        input_node_name=input_node_name,
        input_size=input_size,
        show_size=show_size,
        crop=[],
        normalize=normalize,
        color_mode=color_mode,
        class_names=class_names
    )
    return predictor
