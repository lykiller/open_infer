import os
import cv2


class BaseMnnPredictor:
    def __init__(
            self,
            model_path,
            input_node_name,
            output_node_name=None,
            color_mode="rgb",
            crop=[],
            input_size=[],
            show_size=[],
            normalize=[],
    ):
        pass
