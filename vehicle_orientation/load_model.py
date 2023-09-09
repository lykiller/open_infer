import os

from vehicle_orientation import onnx_dir
from vehicle_orientation.orientation_classifier import OrientationOnnxClassifier


def init_orientation_classifier():
    input_size = (56, 56)
    onnx_name = "orientation_cls_56x56_20230831.onnx"
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
