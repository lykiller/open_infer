import os
import cv2
import numpy as np
import math

from core.base_onnx_predictor import BaseOnnxPredictor
from detectron2d.bbox import distance2bbox
from detectron2d.nms import nms
from utils import fast_sigmoid
from utils import get_image_list
from utils import imwrite
from detectron2d.visual_bboxes import overlay_bbox_cv
from detectron2d.bbox import warp_boxes, get_resize_matrix

FPS = 15
down_sample = 2


class OnnxDetectron(BaseOnnxPredictor):
    def inference(self, image_src, need_show_class_list=[11], if_draw=True):
        image = self.pre_process(image_src)
        labels, scores, reg_preds = self.sess.run(self.output_node_name, {self.input_node_name: image})
        labels, scores, bboxes = self.get_bboxes_center_net(labels[0], scores[0], reg_preds[0], 4)
        # labels, scores, bboxes = self.get_bboxes(labels[0], scores[0], reg_preds[0])
        det_result, image_with_bboxes = self.post_process(
            labels, scores, bboxes, image_src, need_show_class_list, if_draw)
        return det_result, image_with_bboxes

    def post_process(self, labels, scores, det_bboxes, image_src, need_show_class_list, if_draw=True):
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
            det_bboxes, get_resize_matrix(self.input_size, (img_width, img_height)), img_width, img_height
        )

        for i in range(len(self.class_names)):
            if i in need_show_class_list:
                inds = labels == i
                det_result[i] = np.concatenate(
                    [
                        det_bboxes[inds, :].astype(np.float32),
                        scores[inds, np.newaxis].astype(np.float32),
                    ],
                    axis=1,
                ).tolist()
            else:
                det_result[i] = []
        if if_draw:
            image_src[y_min:y_max, x_min:x_max] = overlay_bbox_cv(
                image_src[y_min:y_max, x_min:x_max],
                det_result,
                self.class_names
            )
        return det_result, image_src

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

    def inference_images(self, image_path, save_dir=None):
        image_list = []

        if isinstance(image_path, list):
            image_list = image_path
        elif os.path.exists(image_path):
            image_list = get_image_list(image_path)

        for image_name in image_list:
            image_src = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), 1)
            image_src = cv2.resize(image_src, self.show_size)

            det_result, image_with_bboxes = self.inference(image_src)

            if save_dir is None:
                cv2.imshow("result", image_with_bboxes)
                cv2.waitKey(0)
            else:

                imwrite(os.path.join(save_dir, os.path.basename(image_name)).replace("png", "jpg"), image_with_bboxes)

    def inference_video(self, video_path, save_dir=None):
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)

        if save_dir is not None:
            save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(video_path))[0] + ".mp4")
            vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, self.show_size, True
            )
        print(f"{video_path} begin")
        count = 0
        while True:
            ret_val, frame = cap.read()
            count += 1
            if ret_val:
                if count % down_sample == 0:
                    frame = cv2.resize(frame, self.show_size)

                    image_src = cv2.resize(frame, self.show_size)

                    det_result, image_with_bboxes = self.inference(image_src)

                    if save_dir is None:
                        cv2.imshow("result", image_with_bboxes)
                        cv2.waitKey(0)
                    else:
                        vid_writer.write(image_with_bboxes)
            else:
                if save_dir is not None:
                    vid_writer.release()
                cap.release()
                print(f"{video_path} end")
                break
