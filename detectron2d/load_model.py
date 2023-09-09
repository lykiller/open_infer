import os
from detectron2d import *
from detectron2d.detetron import OnnxDetectron


def init_fcos_det():
    onnx_name = "fcos-det-roadEdge.onnx"
    onnx_path = os.path.join(r"D:\open_infer\onnx", onnx_name)
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
