import os

from traffic_sign import *
from traffic_sign.traffic_sign_predictor import TrafficSignOnnxClassifier, OcrOnnxPredictor


def init_cls_model():
    onnx_name = "tsr_cls_32x32_20230823.onnx"
    onnx_path = os.path.join(r"D:\open_infer\onnx", onnx_name)
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


def init_ocr_model():
    onnx_name = "tsr_ocr_32x32_20230903.onnx"
    onnx_path = os.path.join(r"D:\open_infer\onnx", onnx_name)
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
