import os
import cv2
import onnxruntime

import numpy as np


class BaseOnnxPredictor:
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
            class_names=[],
            dtype=np.float32,
            strides=[4],
            if_transpose=True,

    ):
        self.model_path = model_path
        self.input_node_name = input_node_name
        self.output_node_name = output_node_name
        self.color_mode = color_mode
        self.crop = crop
        self.input_size = tuple(input_size)
        self.show_size = tuple(show_size)
        self.normalize = normalize
        self.class_names = class_names
        self.dtype = dtype
        self.strides = strides
        self.if_transpose = if_transpose
        self.sess = onnxruntime.InferenceSession(
            model_path,
            # providers=['CPUExecutionProvider']
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        if len(normalize):
            mean, std = self.normalize
            self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
            self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def pre_process(self, image):
        if self.color_mode == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(self.crop) == 2:
            lt, rb = self.crop
            x_min, y_min = lt
            x_max, y_max = rb
            image = image[y_min:y_max, x_min:x_max]

        if len(self.input_size) == 2:
            image = cv2.resize(image, self.input_size)

        if len(self.normalize) == 2:
            image = (image.astype(float) - self.mean) / self.std
        if self.if_transpose:
            image = image.transpose(2, 0, 1)
        return np.expand_dims(image.astype(self.dtype), axis=0)
