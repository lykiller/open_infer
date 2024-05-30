from mono_task.model.base_onnx_predictor import BaseOnnxPredictor

chinese_class_names = [
    "车尾",
    "车尾右侧",
    "单右侧",
    "车头右侧",
    "车头",
    "车头左侧",
    "单左侧",
    "车尾左侧",
]

english_class_names = [
    "rear",
    "rear_right",
    "right",
    "front_right",
    "front",
    "front_left",
    "left",
    "rear_left"
]

name_to_id = {
    "rear": 0,
    "rear_right": 1,
    "right": 2,
    "front_right": 3,
    "front": 4,
    "front_left": 5,
    "left": 6,
    "rear_left": 7,
}

orientation_names = [
    "forward",
    "right_forward",
    "right",
    "right_backward",
    "backward",
    "left_backward",
    "left",
    "left_forward",

]


class OrientationOnnxClassifier(BaseOnnxPredictor):
    def inference(self, src_image):
        image = self.pre_process(src_image)
        label, score = self.sess.run(self.output_node_name, {self.input_node_name: image})
        return label[0], score[0]
