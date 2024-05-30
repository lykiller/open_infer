import math
import numpy as np

from core.key_points import warp_key_points
from core.bbox import distance2bbox, warp_boxes, get_resize_matrix
from core.nms import parking_slots_nms
from core.utils import fast_sigmoid
from mono_task.model.base_onnx_predictor import BaseOnnxPredictor

from traffic_object.parking_slot import ParkingSlot


class ApaOnnxPredictor(BaseOnnxPredictor):
    def inference(self, image_src):
        image = self.pre_process(image_src)
        points_offsets, cls_score, box_reg = self.sess.run(self.output_node_name, {self.input_node_name: image})
        labels = np.argmax(cls_score, axis=-1)
        scores = np.max(cls_score, axis=-1)
        labels, scores, det_bboxes, key_points = self.get_bboxes(labels, scores, box_reg, points_offsets)
        return self.post_process(labels, scores, det_bboxes, key_points, image_src)

    def post_process(self, labels, scores, det_bboxes, key_points, image_src):
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

        M = get_resize_matrix(self.input_size, (img_width, img_height))
        det_bboxes = warp_boxes(
            det_bboxes,
            M,
            img_width,
            img_height
        )

        key_points = warp_key_points(
            key_points,
            M,
            img_width,
            img_height
        )

        det_result = list()
        labels = labels.tolist()
        scores = scores.tolist()
        bboxes = det_bboxes.tolist()
        key_points = key_points.tolist()

        for label, score, bbox, key_point in zip(labels, scores, bboxes, key_points):
            parking_slot = ParkingSlot(
                occupy=label,
                score=score,
                bbox=bbox,
                key_point=key_point,
            )
            det_result.append(parking_slot)

        return det_result

    def get_bboxes(self, labels, scores, reg_preds, points_offsets):
        input_width, input_height = self.input_size
        if len(self.strides) > 1:
            feat_map_sizes = [
                (math.ceil(input_height / stride), math.ceil(input_width) / stride)
                for stride in self.strides
            ]
            # get grid cells of one image
            mlvl_center_priors = [
                self.get_single_level_center_priors(
                    feat_map_sizes[i],
                    stride,
                    dtype=np.float32
                )
                for i, stride in enumerate(self.strides)
            ]
            center_priors = np.concatenate(mlvl_center_priors, axis=0)
        else:
            stride = self.strides[0]
            feat_map_size = (math.ceil(input_height / stride), math.ceil(input_width / stride))

            # get grid cells of one image
            center_priors = self.get_single_level_center_priors(
                feat_map_size,
                stride,
                dtype=np.float32
            )
        dis_preds = reg_preds.reshape(len(center_priors), 4)
        dis_preds = np.exp(dis_preds) * center_priors[..., 2, None]
        labels = labels.reshape(len(center_priors))
        scores = scores.reshape(len(center_priors))
        points_offsets = points_offsets.reshape(len(center_priors), 8)

        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=(input_height, input_width))
        key_points = self.offsets_to_key_points(points_offsets, center_priors[..., :2])
        results = parking_slots_nms(labels, scores, bboxes, key_points)
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

    def offsets_to_key_points(self, offsets, center_points):
        key_points = offsets.reshape(len(center_points), 4, 2) * 512
        for i in range(4):
            key_points[:, i, 0] += center_points[:, 0]
            key_points[:, i, 1] += center_points[:, 1]

        key_points[:, :, 0] = key_points[:, :, 0].clip(min=0, max=self.input_size[0])
        key_points[:, :, 1] = key_points[:, :, 1].clip(min=0, max=self.input_size[1])
        return key_points
