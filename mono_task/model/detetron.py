import numpy as np
import math

from mono_task.model.base_onnx_predictor import BaseOnnxPredictor
from core.bbox import distance2bbox
from core.nms import nms
from core.utils import fast_sigmoid

from core.bbox import warp_boxes, get_resize_matrix

det2d_class_names = [
    'car',  # 0
    'bus',  # 1
    'truck',  # 2
    'tricycle',  # 3
    'person',  # 4
    'bicycle',  # 5
    'front_rear',  # 6
    'vehicle_side',  # 7
    'wheel',  # 8
    'tunnel',  # 9
    'traffic_lights',  # 10
    'traffic_sign',  # 11
    'zebra_crossing',  # 12
    'pavement_signs',  # 13
    'human_like'  # 14
]


class OnnxDetectron(BaseOnnxPredictor):
    def inference(self, image_src):
        image = self.pre_process(image_src)
        labels, scores, reg_preds = self.sess.run(self.output_node_name, {self.input_node_name: image})
        labels, scores, bboxes = self.get_bboxes_center_net(labels[0], scores[0], reg_preds[0], 4)
        # labels, scores, bboxes = self.get_bboxes(labels[0], scores[0], reg_preds[0])
        det_result = self.post_process(
            labels, scores, bboxes, image_src)
        return det_result

    def post_process(self, labels, scores, det_bboxes, image_src):
        det_result = {}
        if len(self.crop):
            lt, rb = self.crop
            x_min, y_min = lt
            x_max, y_max = rb
            img_width = x_max - x_min
            img_height = y_max - y_min
        else:
            img_height, img_width, _ = image_src.shape
            x_min = 0
            y_min = 0
            x_max = img_width
            y_max = img_height

        det_bboxes = warp_boxes(
            det_bboxes,
            get_resize_matrix(self.input_size, (img_width, img_height)),
            img_width,
            img_height
        )

        for idx, class_name in enumerate(self.class_names):
            inds = labels == idx
            det_result[class_name] = np.concatenate(
                [
                    det_bboxes[inds, :].astype(np.float32),
                    scores[inds, np.newaxis].astype(np.float32),
                ],
                axis=1,
            ).tolist()

        return det_result

    def get_bboxes(self, labels, scores, reg_preds, strides=[8, 16, 32, 64]):
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
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=(input_height, input_width))

        results = nms(labels, scores, bboxes)
        return results

    def get_bboxes_center_net(self, labels, scores, reg_preds, stride):
        input_width, input_height = self.input_size

        feat_map_size = (math.ceil(input_height / stride), math.ceil(input_width / stride))

        # get grid cells of one image
        center_priors = self.get_single_level_center_priors(
            feat_map_size,
            stride,
            dtype=np.float32
        )

        dis_preds = reg_preds.reshape(len(center_priors), 4)
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=(input_height, input_width))

        results = nms(labels.flatten(), scores.flatten(), bboxes)
        return results

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

    def distribution_project(self, reg_preds, reg_max=7):
        reg_preds = fast_sigmoid(reg_preds)
        weights = np.array(range(reg_max + 1))
        dis_preds = np.split(reg_preds, np.array(range(1, 4)) * (reg_max + 1), axis=-1)
        results = []
        for distance in dis_preds:
            results.append((np.sum(distance * weights, axis=-1) / np.sum(distance, axis=1))[:, np.newaxis])
        return np.concatenate(results, axis=1)
