import numpy as np
import math
import operator

from mono_task.model.base_onnx_predictor import BaseOnnxPredictor
from traffic_object.traffic_sign import ocr_class_names, tsr_class_names, need_ocr_traffic_sign
from core.bbox import distance2bbox
from core.nms import nms


class TrafficSignOnnxClassifier(BaseOnnxPredictor):
    def inference(self, src_image):
        image = self.pre_process(src_image)
        label, score = self.sess.run(self.output_node_name, {self.input_node_name: image})
        return label[0], score[0]


class OcrOnnxPredictor(BaseOnnxPredictor):
    def inference(self, src_image):
        '''
        根据检测出的字符和x_min坐标进行拼装，得到最终识别的速度

        Args:
            src_image:
            strides:

        Returns:

        '''
        image = self.pre_process(src_image)
        labels, scores, regs = self.sess.run(self.output_node_name, {self.input_node_name: image})
        labels, scores, regs = self.get_bboxes(labels, scores, regs)
        if len(labels):
            number_result = {}
            for label, point_x in zip(labels, regs[:, 0]):
                number_result[point_x] = ocr_class_names[label]

            sorted_number = sorted(number_result.items(), key=operator.itemgetter(0))

            return "".join([x[1] for x in sorted_number])
        else:
            return "unknown"

    def get_single_level_center_priors(self, feat_map_size, stride, dtype):
        """Generate centers of a single stage feature map.
        Args:
            batch_size (int): Number of images in one batch.
            feat_map_size (tuple[int]): height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype (obj:`torch.dtype`): data type of the tensors
            device (obj:`torch.device`): device of the tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        h, w = feat_map_size
        x_range = (np.arange(w, dtype=dtype)) * stride
        y_range = (np.arange(h, dtype=dtype)) * stride
        x, y = np.meshgrid(x_range, y_range)
        y = y.flatten()
        x = x.flatten()
        strides = np.array(stride).repeat(len(x))
        proiors = np.stack([x, y, strides, strides], axis=-1)
        return proiors

    def get_bboxes(self, labels, scores, reg_preds, strides=[4, 8]):
        '''
        Args:
            labels: argmax的索引
            scores: 置信度
            reg_preds:  (left, top, right, bottom)， 需要乘以stride
            strides: 特征图strides

        Returns:

        '''
        input_width, input_height = self.input_size

        feat_map_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in strides
        ]
        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                feat_map_sizes[i],
                stride,
                dtype=np.float32
            )
            for i, stride in enumerate(strides)
        ]
        center_priors = np.concatenate(mlvl_center_priors, axis=0)
        dis_preds = reg_preds * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=(input_height, input_width))

        results = nms(labels, scores, bboxes, iou_thresh=0.1, score_thresh=0.5)
        return results


class TrafficSignPredictor:
    def __init__(
            self,
            cls_predictor,
            ocr_predictor,
    ):
        '''
        没有跟踪，应参考V2逻辑
        Args:
            det_predictor:
            cls_predictor:
            ocr_predictor:
            show_size:
        '''
        self.cls_predictor = cls_predictor
        self.ocr_predictor = ocr_predictor

    def inference(self, src_image, det_bboxes):
        result = list()
        if len(det_bboxes):
            for bbox in det_bboxes:
                bbox_t = [int(x) for x in bbox[:-1]]
                x_min, y_min, x_max, y_max = bbox_t
                det_score = bbox[-1]
                w = x_max - x_min
                h = y_max - y_min
                area = w * h
                ratio = w / h
                if 0.5 < ratio < 2 and 900 < area < 10000:
                    label, score = self.cls_predictor.inference(src_image[y_min:y_max, x_min:x_max])
                    show_name = tsr_class_names[label]
                    if show_name in need_ocr_traffic_sign:
                        show_name = show_name + "-" + self.ocr_predictor.inference(src_image[y_min:y_max, x_min:x_max])
                    result.append([x_min, y_min, x_max, y_max, det_score, show_name])

        return result
