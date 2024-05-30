from mono_task.model.base_onnx_predictor import BaseOnnxPredictor

terrain_class_names = [
    "background",  # 0 背景
    "asphalt",  # 1 人工
    "gravel",  # 2 砾石
    "earth",  # 3 土
    "grass",  # 4 草
    "mud",  # 5 泥
    "water",  # 6 水
    "rock",  # 7 岩石
    "sand",  # 8 沙
    "light_snow",  # 9 冰
    "deep_snow",  # 10 雪
    "others"  # 11 其他
]


class TerrainOnnxPredictor(BaseOnnxPredictor):

    def inference(self, image_src):
        image = self.pre_process(image_src)
        model_out = self.sess.run(self.output_node_name, {self.input_node_name: image})
        model_out = model_out[0].squeeze()

        return model_out

    def view_transform(self, image_src, model_out):
        return