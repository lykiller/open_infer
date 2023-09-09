from core.base_onnx_predictor import BaseOnnxPredictor


class OrientationOnnxClassifier(BaseOnnxPredictor):
    def inference(self, src_image):
        image = self.pre_process(src_image)
        label, score = self.sess.run(self.output_node_name, {self.input_node_name: image})
        return label[0], score[0]
