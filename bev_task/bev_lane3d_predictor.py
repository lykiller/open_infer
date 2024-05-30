import os
import cv2
import onnxruntime

import numpy as np

from core.utils import imread
from core.camera import MultiCamera

LIDAR_TOP_HEIGHT = 1.9


class BevLane3DOnnxPredictor:
    color_list = [
        [255, 255, 0],  # not-interest

        [255, 0, 0],  # left-left
        [0, 255, 0],  # left
        [0, 0, 255],  # right
        [0, 255, 255],  # right-right

        [255, 0, 255],  # left_curbside
        [128, 0, 100],  # right_curbside
    ]

    def __init__(
            self,
            model_path,
            input_node_name,
            input_size,
            feat_size,
            sensor_name_list,
            crop_list,
            scale_u_list,
            scale_v_list,
            camera_config_path,
            output_node_name=None,
            color_mode="rgb",
            normalize=[],
            dtype=np.float32,
            if_transpose=True,
            vis_threshold=0.3

    ):
        self.model_path = model_path
        self.input_node_name = input_node_name
        self.output_node_name = output_node_name
        self.color_mode = color_mode
        self.input_size = tuple(input_size)
        self.normalize = normalize
        self.vis_threshold = vis_threshold

        self.sensor_name_list = sensor_name_list
        self.crop_list = crop_list
        self.scale_u_list = scale_u_list
        self.scale_v_list = scale_v_list

        self.camera = MultiCamera(camera_config_path)

        self.dtype = dtype
        self.if_transpose = if_transpose
        self.sess = onnxruntime.InferenceSession(
            model_path,
            # providers=['CPUExecutionProvider']
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        self.feat_size = feat_size
        feat_w, feat_h = feat_size

        u_step = np.array([x for x in range(feat_w)])
        self.u_step = np.repeat(u_step, 5)
        self.v_step = np.array([[y + 0.1, y + 0.3, y + 0.5, y + 0.7, y + 0.9] for y in range(feat_h)])

        if len(normalize):
            mean, std = self.normalize
            self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
            self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def pre_process_per_image(self, image, crop):
        if self.color_mode == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(crop) == 2:
            lt, rb = crop
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

    def pre_process(self, image_np_list):

        result = []
        for image, crop in zip(image_np_list, self.crop_list):
            result.append(self.pre_process_per_image(image, crop))
        return np.concatenate(result, axis=0)

    def inference(self, image_np_list):

        multi_image = self.pre_process(image_np_list)
        predict = self.sess.run(
            self.output_node_name,
            {self.input_node_name: multi_image}
        )

        return self.post_process(predict)

    def post_process(self, predict):
        result = dict()
        predict_vis, predict_u, predict_depth, predict_height, predict_id, predict_category = predict
        for sensor_name, per_view_vis, per_view_u, per_view_d, per_view_h, per_view_id, per_view_cate, crop, scale_u, scale_v in zip(
                self.sensor_name_list, predict_vis, predict_u, predict_depth, predict_height, predict_id,
                predict_category,
                self.crop_list, self.scale_u_list, self.scale_v_list
        ):
            result[sensor_name] = self.post_process_per_view(
                per_view_vis, per_view_u, per_view_d, per_view_h, per_view_id, per_view_cate,
                scale_u, scale_v, crop[0][0], crop[0][1])

        return result

    def post_process_per_view(
            self,
            predict_vis,
            predict_u,
            predict_depth,
            predict_height,
            predict_id,
            predict_category,
            scale_u,
            scale_v,
            crop_u,
            crop_v,

    ):

        valid_mask = np.where(predict_vis[0] > self.vis_threshold)

        u_step = self.u_step[np.repeat(valid_mask[1], 5) * 5]
        v_step = self.v_step[valid_mask[0]].flatten() * scale_v + crop_v

        predict_u = predict_u[:, valid_mask[0], valid_mask[1]].T.flatten()
        result_u = (u_step + predict_u) * scale_u + crop_u

        predict_depth = predict_depth[:, valid_mask[0], valid_mask[1]].T.flatten()
        predict_height = predict_height[:, valid_mask[0], valid_mask[1]].T.flatten()

        predict_id = predict_id[:, valid_mask[0], valid_mask[1]]
        predict_id = np.repeat(predict_id, 5)

        predict_category = predict_category[:, valid_mask[0], valid_mask[1]]
        predict_category = np.repeat(predict_category, 5)

        return result_u, v_step, predict_depth, predict_height, predict_id, predict_category

    def visual_result(self, image_path_list):
        image_np_list = []
        for image_path in image_path_list:
            image_np_list.append(imread(image_path))

        predict_result = self.inference(image_np_list)

        per_view_w = 768
        per_view_h = 448
        bev_image = np.zeros((per_view_h * 2, 640, 3), dtype=np.uint8)
        bev_image[::20, ::20, :] = 255

        visual_image = np.zeros((per_view_h * 2, per_view_w * 3, 3), dtype=np.uint8)
        for idx, (sensor_name, per_view_predict_result) in enumerate(predict_result.items()):
            u_np, v_np, depth_np, height_np, id_np, category_np = per_view_predict_result
            camera = getattr(self.camera, sensor_name)
            # height_np = height_np + LIDAR_TOP_HEIGHT + camera.translation[-1]
            # if idx == 0:
            depth_bias = (camera.translation[-1] + LIDAR_TOP_HEIGHT) / (
                    image_np_list[idx].shape[0] - camera.intrinsic[1, 2]) * 10 * self.scale_v_list[idx] / 12 - 7/208.4
            print(sensor_name, camera.translation[-1] + LIDAR_TOP_HEIGHT, depth_bias, image_np_list[idx].shape[0] - camera.intrinsic[1, 2])
            depth_np = (depth_np + depth_bias) * camera.intrinsic[0][0] / (10 * self.scale_v_list[idx] / 12)
            # else:
            #     depth_np = camera.intrinsic[1, 1] * height_np / (v_np - camera.intrinsic[1, 2])
            ego_xyz = camera.uv_to_ego_xyz(u_np, v_np, depth_np)
            # print(sensor_name, camera.intrinsic)
            for idp, (relative_id, xyz) in enumerate(zip(id_np, ego_xyz.T)):
                u_idx = int(320 - int(xyz[1] / 0.05))
                v_idx = 448 - int(xyz[0] / 0.08)
                height = xyz[2] + LIDAR_TOP_HEIGHT
                cv2.circle(
                    bev_image,
                    (u_idx, v_idx),
                    1,
                    self.color_list[relative_id],
                    2
                )
                if idp % 80 == 1:
                    cv2.putText(
                        bev_image,
                        format(height, "0.2f"),
                        (u_idx, v_idx),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        thickness=2
                    )

            for idp, (u, v, depth, relative_id, category_id) in enumerate(
                    zip(u_np, v_np, depth_np, id_np, category_np)):
                # print(u, v, depth, relative_id, self.category_names[relative_id])
                cv2.circle(
                    image_np_list[idx],
                    (int(u), int(v)),
                    2,
                    self.color_list[relative_id],
                    2
                )
                if idp % 20 == 1:
                    cv2.putText(
                        image_np_list[idx],
                        format(depth, "0.2f"),
                        (int(u), int(v)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        thickness=2
                    )

            if sensor_name == "CAM_FRONT":
                visual_image[:per_view_h, per_view_w:per_view_w * 2, :] = cv2.resize(
                    image_np_list[idx], (per_view_w, per_view_h))
            elif sensor_name == "CAM_FRONT_LEFT":
                visual_image[:per_view_h, :per_view_w, :] = cv2.resize(
                    image_np_list[idx], (per_view_w, per_view_h))
            elif sensor_name == "CAM_FRONT_RIGHT":
                visual_image[:per_view_h, per_view_w * 2:, :] = cv2.resize(
                    image_np_list[idx], (per_view_w, per_view_h))
            elif sensor_name == "CAM_BACK":
                visual_image[per_view_h:, per_view_w:per_view_w * 2, :] = cv2.resize(
                    image_np_list[idx], (per_view_w, per_view_h))
            elif sensor_name == "CAM_BACK_LEFT":
                visual_image[per_view_h:, :per_view_w, :] = cv2.resize(
                    image_np_list[idx], (per_view_w, per_view_h))
            elif sensor_name == "CAM_BACK_RIGHT":
                visual_image[per_view_h:, per_view_w * 2:, :] = cv2.resize(
                    image_np_list[idx], (per_view_w, per_view_h))
            else:
                raise EOFError

        return np.concatenate([visual_image, bev_image], axis=1)
