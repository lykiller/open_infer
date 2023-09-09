import cv2
import numpy as np
import math
import operator

from core.base_onnx_predictor import BaseOnnxPredictor
from traffic_sign import *
from traffic_sign.traffic_sign_tracker import TrackObject
from utils import _COLORS, imwrite, mkdir
from detectron2d.bbox import distance2bbox
from detectron2d.nms import nms


class TrafficSignOnnxClassifier(BaseOnnxPredictor):
    def inference(self, src_image):
        image = self.pre_process(src_image)
        label, score = self.sess.run(self.output_node_name, {self.input_node_name: image})
        return label[0], score[0]


class OcrOnnxPredictor(BaseOnnxPredictor):
    def inference(self, src_image, strides=[4, 8]):
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
            return ""

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
            det_predictor,
            cls_predictor,
            ocr_predictor,
            show_size,
    ):
        '''
        没有跟踪，应参考V2逻辑
        Args:
            det_predictor:
            cls_predictor:
            ocr_predictor:
            show_size:
        '''
        self.det_predictor = det_predictor
        self.cls_predictor = cls_predictor
        self.ocr_predictor = ocr_predictor
        self.show_size = show_size

    def inference(self, src_image, need_show_class_list=[11]):
        det_results, _ = self.det_predictor.inference(
            src_image,
            need_show_class_list,
            if_draw=False
        )
        bboxes = det_results[11]
        if len(bboxes):
            for bbox in bboxes:
                bbox = [int(x) for x in bbox]
                x_min, y_min, x_max, y_max, _ = bbox
                label, score = self.cls_predictor.inference(src_image[y_min:y_max, x_min:x_max])

                show_name = [tsr_class_names[label]]
                if label in need_ocr_index:
                    ocr_result = self.ocr_predictor.inference(src_image[y_min:y_max, x_min:x_max])
                    show_name.append(ocr_result)

                color = (_COLORS[label] * 255).astype(np.uint8).tolist()

                text = "--".join(show_name)
                txt_color = (0, 0, 0) if np.mean(_COLORS[label]) > 0.5 else (255, 255, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
                cv2.rectangle(src_image, (x_min, y_min), (x_max, y_max), color, 2)

                cv2.rectangle(
                    src_image,
                    (x_min, y_min - txt_size[1] - 1),
                    (x_min + txt_size[0] + txt_size[1], y_min - 1),
                    color,
                    -1,
                )
                cv2.putText(src_image, text, (x_min, y_min - 1), font, 0.5, txt_color, thickness=1)

        return src_image

    def inference_video(self, video_path, save_dir=None):
        FPS = 10
        down_sample = 1
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

                    image_with_bboxes = self.inference(image_src)

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


class TrafficSignPredictorV2:
    def __init__(
            self,
            det_predictor,
            cls_predictor,
            ocr_predictor,
            show_size,
    ):
        self.det_predictor = det_predictor
        self.cls_predictor = cls_predictor
        self.ocr_predictor = ocr_predictor
        self.show_size = show_size

    def inference(self, src_image, track_sign_dict, kalman_filter, need_show_class_list=[11]):
        det_results, _ = self.det_predictor.inference(
            src_image,
            need_show_class_list,
            if_draw=False
        )
        det_results = det_results[11]
        delete_id_list = []
        # print(track_sign_dict)
        for track_id, track_sign in track_sign_dict.items():
            flag = track_sign.update(det_results, kalman_filter)
            # print(flag)
            if flag:
                bbox = [int(x) for x in track_sign.bbox]
                x_min, y_min, x_max, y_max = bbox[:4]
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(self.show_size[0], x_max)
                y_max = min(self.show_size[1], y_max)
                # print(x_min, y_min, x_max, y_max)
                label, score = self.cls_predictor.inference(src_image[y_min:y_max, x_min:x_max])

                show_name = [tsr_class_names[label]]
                if label in need_ocr_index:
                    ocr_result = self.ocr_predictor.inference(src_image[y_min:y_max, x_min:x_max])
                    show_name.append(ocr_result)

                color = (_COLORS[label] * 255).astype(np.uint8).tolist()

                text = "--".join(show_name)
                txt_color = (0, 0, 0) if np.mean(_COLORS[label]) > 0.5 else (255, 255, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
                cv2.rectangle(src_image, (x_min, y_min), (x_max, y_max), color, 2)

                cv2.rectangle(
                    src_image,
                    (x_min, y_min - txt_size[1] - 1),
                    (x_min + txt_size[0] + txt_size[1], y_min - 1),
                    color,
                    -1,
                )
                cv2.putText(src_image, text, (x_min, y_min - 1), font, 0.5, txt_color, thickness=1)
            else:
                delete_id_list.append(track_id)
        for track_id in delete_id_list:
            del track_sign_dict[track_id]

        for bbox in det_results:
            new_track_id = 0
            while track_sign_dict.__contains__(new_track_id):
                new_track_id += 1
            track_sign_dict[new_track_id] = TrackObject(new_track_id, bbox, kalman_filter)

        return src_image

    def inference_video(self, video_path, kalman_filter, save_dir=None):
        FPS = 10
        down_sample = 1
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
        track_sign_dict = {}
        while True:
            ret_val, frame = cap.read()
            count += 1
            if ret_val:
                if count % down_sample == 0:
                    frame = cv2.resize(frame, self.show_size)

                    image_src = cv2.resize(frame, self.show_size)

                    image_with_bboxes = self.inference(image_src, track_sign_dict, kalman_filter)

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

    def inference_video_for_images(self, video_path, save_dir=None):
        down_sample = 10
        cap = cv2.VideoCapture(video_path)

        print(f"{video_path} begin")
        frame_id = 0
        src_image_save_path = os.path.join(save_dir, "src_images")
        bbox_image_save_path = os.path.join(save_dir, "image_with_bboxes")
        crop_image_save_path = os.path.join(save_dir, "crop_images")
        mkdir(src_image_save_path)
        mkdir(bbox_image_save_path)
        mkdir(crop_image_save_path)
        for tsr_name in tsr_class_names:
            mkdir(os.path.join(crop_image_save_path, tsr_name))
        base_name = os.path.basename(video_path).split(".")[0]
        while True:
            ret_val, frame = cap.read()
            frame_id += 1
            if ret_val:
                if frame_id % down_sample == 7:
                    frame_name = base_name + "_" + str(frame_id) + ".jpg"
                    frame = cv2.resize(frame, self.show_size)

                    src_image = cv2.resize(frame, self.show_size)
                    # source = copy.copy(src_image)
                    # imwrite(os.path.join(src_image_save_path, frame_name), src_image)
                    det_results, _ = self.det_predictor.inference(
                        src_image,
                        [11],
                        if_draw=False
                    )
                    # imwrite(os.path.join(bbox_image_save_path, frame_name), image_with_bboxes)
                    det_results = det_results[11]

                    for bbox_id, bbox in enumerate(det_results):
                        bbox = [int(x) for x in bbox]
                        x_min, y_min, x_max, y_max = bbox[:4]
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(self.show_size[0], x_max)
                        y_max = min(self.show_size[1], y_max)
                        # print(x_min, y_min, x_max, y_max)
                        label, score = self.cls_predictor.inference(src_image[y_min:y_max, x_min:x_max])
                        show_name = tsr_class_names[label]
                        imwrite(
                            os.path.join(crop_image_save_path, show_name, str(bbox_id) + "_" + frame_name),
                            src_image[y_min:y_max, x_min:x_max]
                        )
            else:

                cap.release()
                print(f"{video_path} end")
                break
