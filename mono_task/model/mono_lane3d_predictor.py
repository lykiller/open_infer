import copy

import cv2
import numpy as np

from mono_task.model.base_onnx_predictor import BaseOnnxPredictor


class MonoLane3DPredictor(BaseOnnxPredictor):
    scale_x = 24
    scale_y = 24
    crop_y = 240
    vis_threshold = 0.3
    feat_w = 160
    feat_h = 80
    u_step = np.array([x for x in range(feat_w)])
    u_step = np.repeat(u_step, 5)
    v_step = np.array([[y + 0.1, y + 0.3, y + 0.5, y + 0.7, y + 0.9] for y in range(feat_h)])

    intrinsic = np.array(
        [
            [1912.6069702676, 0, 1895.7022972943],
            [0, 1912.2738506415001, 1038.9911795727],
            [0, 0, 1]
        ]
    )

    rotation = np.array([
        [-0.0050846031, 0.0016860227, 0.99998558],
        [-0.9999426, -0.0094001098, -0.0050685355],
        [0.0093914289, -0.99995399, 0.0017337218]
    ])
    translation = np.array([0.054415837, 0.023906555, -0.12925074])

    color_list = [
        [255, 255, 0],  # not-interest

        [255, 0, 0],  # left-left
        [0, 255, 0],  # left
        [0, 0, 255],  # right
        [0, 255, 255],  # right-right

        [255, 0, 255],  # left_curbside
        [128, 0, 100],  # right_curbside
    ]

    category_names = [
        'unknown',
        'white-dash',
        'white-solid',
        'double-white-dash',
        'double-white-solid',
        'white-ldash-rsolid',
        'white-lsolid-rdash',
        'yellow-dash',
        'yellow-solid',
        'double-yellow-dash',
        'double-yellow-solid',
        'yellow-ldash-rsolid',
        'yellow-lsolid-rdash',
        'left-curbside',
        'right-curbside',
    ]

    def inference(self, image_src):
        image = self.pre_process(image_src)
        predict = self.sess.run(
            self.output_node_name,
            {self.input_node_name: image}
        )

        return self.post_process(predict)

    def post_process(
            self,
            predict,
    ):
        predict_vis, predict_u, predict_depth, predict_id, predict_category = predict

        valid_mask = np.where(predict_vis[0][0] > self.vis_threshold)

        u_step = self.u_step[np.repeat(valid_mask[1], 5) * 5]
        v_step = self.v_step[valid_mask[0]].flatten() * self.scale_y + self.crop_y

        predict_u = predict_u[:, :, valid_mask[0], valid_mask[1]][0].T.flatten()
        result_u = (u_step + predict_u) * self.scale_x

        predict_depth = predict_depth[:, :, valid_mask[0], valid_mask[1]][0].T.flatten()
        predict_depth = np.exp(predict_depth)

        predict_id = predict_id[:, :, valid_mask[0], valid_mask[1]]
        predict_id = np.repeat(predict_id, 5)

        predict_category = predict_category[:, :, valid_mask[0], valid_mask[1]]
        predict_category = np.repeat(predict_category, 5)

        return result_u, v_step, predict_depth, predict_id, predict_category

    def visual_result(self, image_src, annotation, if_save):
        image_copy = copy.deepcopy(image_src)
        u_list, v_list, depth_list, id_list, category_list = self.inference(image_src)

        intrinsic = np.array(annotation["intrinsic"])
        camera_xyz = np.matmul(
            np.linalg.inv(intrinsic), np.stack([u_list * depth_list, v_list * depth_list, depth_list]))

        extrinsic = np.array(annotation["extrinsic"])
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]

        camera_xyz = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ]) @ camera_xyz
        ego_xyz = rotation @ camera_xyz + translation.repeat(camera_xyz.shape[1]).reshape(3, -1)

        bev_image = np.zeros((720, 400, 3), dtype=np.uint8)
        bev_image[672:, 190:211, :] = 255
        bev_image[::50, ::40, :] = 255

        gt_bev_image = np.zeros((720, 400, 3), dtype=np.uint8)
        gt_bev_image[672:, 190:211, :] = (0, 255, 0)
        gt_bev_image[::50, ::40, :] = (0, 255, 0)
        for per_lane_line in annotation["lane_lines"]:
            src_xyz = np.array(per_lane_line["xyz"])
            category_id = per_lane_line["category"]
            relative_id = per_lane_line["attribute"]
            if category_id == 20:
                relative_id = 5
            elif category_id == 21:
                relative_id = 6

            gt_ego_xyz = rotation @ src_xyz + translation.repeat(src_xyz.shape[1]).reshape(3, -1)
            for idx, xyz in enumerate(gt_ego_xyz.T):
                u_idx = int(200 - int(xyz[1] / 0.08))
                v_idx = 720 - int(xyz[0] / 0.2)
                height = xyz[2]
                cv2.circle(
                    gt_bev_image,
                    (u_idx, v_idx),
                    1,
                    self.color_list[relative_id],
                    2
                )
                if idx % 80 == 1:
                    cv2.putText(
                        gt_bev_image,
                        format(height, "0.2f"),
                        (u_idx, v_idx),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        thickness=1
                    )

        for idx, (relative_id, xyz) in enumerate(zip(id_list, ego_xyz.T)):
            u_idx = int(200 - int(xyz[1] / 0.08))
            v_idx = 720 - int(xyz[0] / 0.2)
            height = xyz[2]
            cv2.circle(
                bev_image,
                (u_idx, v_idx),
                1,
                self.color_list[relative_id],
                2
            )
            if idx % 80 == 1:
                cv2.putText(
                    bev_image,
                    format(height, "0.2f"),
                    (u_idx, v_idx),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    thickness=1
                )

        for idx, (u, v, depth, relative_id, category_id) in enumerate(
                zip(u_list, v_list, depth_list, id_list, category_list)):
            # print(u, v, depth, relative_id, self.category_names[relative_id])
            cv2.circle(
                image_copy,
                (int(u), int(v)),
                1,
                self.color_list[relative_id],
                2
            )
            if idx % 20 == 1:
                cv2.putText(
                    image_copy,
                    format(depth, "0.2f"),
                    (int(u), int(v)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    thickness=1
                )
        image_copy = cv2.resize(image_copy, self.show_size)
        image_copy = np.concatenate([image_copy, bev_image, gt_bev_image], axis=1)
        if if_save:
            return image_copy
        else:
            cv2.imshow("result", image_copy)

            cv2.waitKey(0)
            return image_copy

    def visual_result_using_metoak(self, image_src, if_save):
        image_copy = copy.deepcopy(image_src)
        u_list, v_list, depth_list, id_list, category_list = self.inference(image_src)
        depth_list = depth_list * 0.4587332

        camera_xyz = np.matmul(
            np.linalg.inv(self.intrinsic), np.stack([u_list * depth_list, v_list * depth_list, depth_list]))
        ego_xyz = self.rotation @ camera_xyz + self.translation.repeat(camera_xyz.shape[1]).reshape(3, -1)

        bev_image = np.zeros((720, 400, 3), dtype=np.uint8)
        bev_image[672:, 190:211, :] = 255
        bev_image[::50, ::40, :] = 255
        for idx, (relative_id, xyz) in enumerate(zip(id_list, ego_xyz.T)):
            u_idx = int(200 - int(xyz[1] / 0.08))
            v_idx = 720 - int(xyz[0] / 0.15)
            height = xyz[2] + 1.8
            cv2.circle(
                bev_image,
                (u_idx, v_idx),
                1,
                self.color_list[relative_id],
                2
            )
            if idx % 80 == 1:
                cv2.putText(
                    bev_image,
                    format(height, "0.2f"),
                    (u_idx, v_idx),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    thickness=1
                )

        for idx, (u, v, depth, relative_id, category_id) in enumerate(
                zip(u_list, v_list, depth_list, id_list, category_list)):
            # print(u, v, depth, relative_id, self.category_names[relative_id])
            cv2.circle(
                image_copy,
                (int(u), int(v)),
                1,
                self.color_list[relative_id],
                2
            )
            if idx % 20 == 1:
                cv2.putText(
                    image_copy,
                    format(depth, "0.2f"),
                    (int(u), int(v)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    thickness=2
                )
        image_copy = cv2.resize(image_copy, self.show_size)
        image_copy = np.concatenate([image_copy, bev_image], axis=1)
        if if_save:
            return image_copy
        else:
            cv2.imshow("result", image_copy)

            cv2.waitKey(0)
            return image_copy
