import cv2
import numpy as np
import os
import copy

from core.base_onnx_predictor import BaseOnnxPredictor
from terrain_seg.visualize import get_image_with_mask, add_mask_to_roi
from utils import get_image_list, imwrite, mkdir
from detectron2d.visual_bboxes import overlay_bbox_cv
from terrain_seg import *


def pt_add(pt1, pt2):
    x = pt1[0] + pt2[0]
    y = pt1[1] + pt2[1]
    return x, y


class TerrainOnnxPredictor(BaseOnnxPredictor):
    weights_of_state = np.array(
        #     "asphalt", "gravel", "earth", "grassland", "mud",
        #     "water", "rock", "sand", "light_snow", "deep_snow",
        #     "others"
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0],  # rock

            [0.0, 0.0, 0.1, 0.0, 1.2, 1.2, 0.0, 0.0, 0.2, 0.2, 0.0],  # mud

            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 1.0, 0.0],  # deep_snow

            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # sand

            [0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # wading

            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],  # light_snow

            [0.0, 1.0, 0.4, 0.6, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0],  # gravel

            [0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # grassland

            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # earth

            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],  # asphalt

            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # unknown
        ]
    )
    state_name_list = ["rock", "mud", "deep_snow", "sand", "wading",
                       "light_snow", "gravel", "grassland", "earth", "asphalt", "unknown"]
    valid_threshold = 0.5

    history_mask_ratio = None
    last_mask_ratio = None
    history_machine_state = []

    frame_count = 0
    current_state = -1
    confirm_state = -1
    out_state = -1

    num_state_update_frames = {
        0: 5,
        1: 5,
        2: 10,
        3: 10,
        4: 10,
        5: 10,
        6: 15,
        7: 15,
        8: 15,
        9: 15,
        10: 15,
        11: 15
    }

    def inference(self, image_src, driving_area_mask=None, line_xy=None, keep=False):
        image = self.pre_process(image_src)
        model_out = self.sess.run(self.output_node_name, {self.input_node_name: image})
        model_out = model_out[0].squeeze()
        if driving_area_mask is not None:
            model_out = model_out * driving_area_mask
        result = self.post_process(model_out, image_src, keep)
        if line_xy is not None:
            self.draw_line(result, line_xy)

        return result

    def inference_to_bev(self, image_src, matrix):
        image = self.pre_process(image_src)
        model_out = self.sess.run(self.output_node_name, {self.input_node_name: image})
        model_out = model_out[0].squeeze()
        ipm_y, ipm_x = (320, 960)
        bev_out = cv2.warpPerspective(model_out[:, :, np.newaxis].astype(np.uint8), matrix, (ipm_x, ipm_y),
                                      cv2.INTER_NEAREST)
        show_bev = np.zeros((ipm_y, ipm_x, 3))
        for index_x in range(12):
            for index_y in range(4):
                mask_count = np.bincount(
                    bev_out[index_y * 80:(index_y + 1) * 80, index_x * 80:(index_x + 1) * 80].astype(
                        np.uint8).flatten(), minlength=len(self.class_names))
                confirm_label = np.argmax(mask_count)
                show_bev[index_y * 80:(index_y + 1) * 80, index_x * 80:(index_x + 1) * 80] = mask_color[
                    confirm_label]

        cv2.imshow("bev", show_bev)

    def post_process(self, model_out, frame, keep=False):
        distance = 0

        mask_count = np.bincount(model_out.astype(np.uint8).flatten(), minlength=len(self.class_names))
        mask_ratio = mask_count / np.sum(mask_count)
        mask_ratio_without0 = mask_count[1:] / (np.sum(mask_count[1:]) + 1e-5)

        if mask_ratio[0] > 0.8:
            class_index = [0]
        else:
            distance = self.update_mask_ratio_state(mask_ratio_without0)
            labels = mask_ratio_without0.argsort()[-4:][::-1]
            label_keep = np.where(mask_ratio_without0[labels] > 0.15)
            class_index = (labels[label_keep] + 1).tolist()

        self.current_state = self.get_current_state_machine(class_index)
        if self.current_state:
            if len(self.history_machine_state) == 0:
                self.history_machine_state.append(self.current_state)
            elif self.history_machine_state[-1] == self.current_state:
                self.history_machine_state.append(self.current_state)
            else:
                self.history_machine_state = [self.current_state]

        if len(self.history_machine_state) > self.num_state_update_frames[self.current_state]:
            self.confirm_state = self.current_state

        if self.confirm_state:
            self.out_state = self.confirm_state

        self.put_text(frame, mask_ratio_without0, class_index, distance)
        self.put_state(frame)

        h, w, _ = color_mask_map.shape
        if len(self.crop) == 2:
            image_with_mask = add_mask_to_roi(frame, model_out, self.crop, color_list=mask_color)
        else:
            result = cv2.resize(model_out, self.show_size, interpolation=cv2.INTER_NEAREST)
            image_with_mask = get_image_with_mask(frame, result, color_list=mask_color)

        image_with_mask[:h, :w] = color_mask_map

        if keep:
            self.frame_count += 1
        return image_with_mask

    def draw_line(self, image, line_xy):
        lr = self.crop[0]
        for xy in line_xy:
            pt_start, pt_end = xy
            cv2.line(image, pt_add(pt_start, lr), pt_add(pt_end, lr), (255, 0, 0))

    def update_mask_ratio_state(self, current_mask_ratio):
        if self.history_mask_ratio is None:
            self.history_mask_ratio = current_mask_ratio
            self.last_mask_ratio = current_mask_ratio
            return 0
        else:
            distance = 1 - np.dot(current_mask_ratio, self.last_mask_ratio)
            self.last_mask_ratio = current_mask_ratio
            weight_decay = 0.04 * (1.15 - distance)

        self.history_mask_ratio = self.history_mask_ratio * (1 - weight_decay) + current_mask_ratio * weight_decay
        return distance

    def get_current_state_machine(self, class_index):
        if self.history_mask_ratio is None:
            return 0
        current_machine_state = np.sum(self.history_mask_ratio * self.weights_of_state, axis=1)
        for idx, score in enumerate(current_machine_state):
            if score > self.valid_threshold:
                if idx == 1 and 5 not in class_index:
                    continue
                else:
                    return idx

    def inference_images(self, image_path, driving_area_mask, line_xy, save_dir=None):
        image_list = []

        if isinstance(image_path, list):
            image_list = image_path
        elif os.path.exists(image_path):
            image_list = get_image_list(image_path)

        for image_name in image_list:
            image_src = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), 1)
            image_src = cv2.resize(image_src, self.show_size)

            result = self.inference(image_src, driving_area_mask, line_xy, keep=False)
            # model_out = model_out * driving_area_mask
            # status = np.bincount(model_out.astype(np.uint8).flatten(), minlength=len(self.mask_color))[1:]
            #
            # class_index = np.argmax(status) + 1
            # self.put_text(result, status, class_index, 0)
            if save_dir is None:
                cv2.imshow("result", result)
                cv2.waitKey(0)
            else:
                cv2.imencode(".jpg", result)[1].tofile(
                    os.path.join(save_dir, os.path.basename(image_name)).replace(".png", ".jpg"))

    def inference_video(self, video_path, driving_area_mask, line_xy, save_dir=None):
        cap = cv2.VideoCapture(video_path)
        FPS = 10
        down_sample = 1

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

                    image_with_mask = self.inference(frame, driving_area_mask, line_xy, keep=True)
                    # model_out = model_out * driving_area_mask

                    # cosine_distance = self.analysis_status(status)
                    # count += 1
                    # print(count)
                    if save_dir is not None:
                        vid_writer.write(image_with_mask)
                    else:
                        cv2.imshow("image_with_mask", image_with_mask)
                        cv2.waitKey(0)
            else:
                if save_dir is not None:
                    vid_writer.release()
                cap.release()
                self.history_mask_ratio = None
                self.frame_count = 0
                self.confirm_state = -1
                self.out_state = -1
                self.current_state = -1
                print(f"{video_path} end")
                break

    def inference_video_save_images(self, video_path, driving_area_mask, line_xy, save_dir=None):
        cap = cv2.VideoCapture(video_path)
        down_sample = 10

        print(f"{video_path} begin")
        if save_dir is not None:
            save_src_dir = os.path.join(save_dir, "src_images")
            save_visual_mask = os.path.join(save_dir, "visual_masks")
            mkdir(save_src_dir)
            mkdir(save_visual_mask)
        count = 0
        while True:
            ret_val, frame = cap.read()
            count += 1
            if ret_val:
                if count % down_sample == 7:
                    frame = cv2.resize(frame, self.show_size)
                    if save_dir is not None:
                        imwrite(
                            os.path.join(
                                save_src_dir, os.path.splitext(os.path.basename(video_path))[0] + f"_{count}.jpg"
                            ),
                            frame
                        )
                    image_with_mask = self.inference(frame, driving_area_mask, line_xy, keep=True)
                    # model_out = model_out * driving_area_mask

                    # cosine_distance = self.analysis_status(status)
                    # count += 1
                    # print(count)
                    if save_dir is not None:

                        imwrite(
                            os.path.join(
                                save_visual_mask, os.path.splitext(os.path.basename(video_path))[0] + f"_{count}.jpg"
                            ),
                            image_with_mask
                        )
                    else:
                        cv2.imshow("image_with_mask", image_with_mask)
                        cv2.waitKey(0)
            else:

                cap.release()
                self.history_mask_ratio = None
                self.frame_count = 0
                self.confirm_state = 0
                self.out_state = 0
                print(f"{video_path} end")
                break

    def put_text(self, image, ratio, class_index, distance):
        h, w, _ = color_mask_map.shape
        if ratio is not None:
            for idx, r in enumerate(ratio):
                cv2.putText(
                    image,
                    "%.1f" % (r * 100),
                    (w + 10, h * (idx + 2) // len(self.class_names)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255)
                )
        if isinstance(class_index, int):
            cv2.putText(
                image,
                self.class_names[class_index],
                (400, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3,
                (0, 255, 0),
                2
            )
        else:
            class_names_list = [self.class_names[index] for index in class_index]
            cv2.putText(
                image,
                ",".join(class_names_list),
                (400, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 255, 0),
                2
            )

        if distance > 0:
            cv2.putText(
                image,
                f"distance:{float('%.2g' % distance)}",
                (400, 380),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

    def put_state(self, image):
        cv2.putText(
            image,
            f"current_state:{self.state_name_list[self.current_state]}",
            (400, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        cv2.putText(
            image,
            f"confirm_state:{self.state_name_list[self.confirm_state]}",
            (400, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        cv2.putText(
            image,
            f"output_state:{self.state_name_list[self.out_state]}",
            (400, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )


class TerrainWithElevationOnnxPredictor(BaseOnnxPredictor):
    color_mask_map = cv2.imdecode(np.fromfile(mask_color_path, dtype=np.uint8), 1)

    elevation_class_names = ["RoadDamaged", "SpeedDump", "IronPlate", "ManholeCover", "StormDrain"]
    history_status = None

    def inference(self, image_src):
        image = self.pre_process(image_src, dtype=np.uint8)
        elevation, terrain = self.sess.run(self.output_node_name, {self.input_node_name: image})
        copy_image_src = copy.copy(image_src)
        terrain_result = self.terrain_post_process(terrain[0], image_src)
        elevation_result = self.elevation_post_process(elevation[0], copy_image_src)
        return terrain_result, elevation_result

    def terrain_post_process(self, model_out, frame):
        status = np.bincount(model_out.astype(np.uint8).flatten(), minlength=len(mask_color))[1:]
        status = status / np.sum(status)
        class_index = np.argmax(status) + 1
        self.put_text(frame, status, class_index, 0)

        h, w, _ = self.color_mask_map.shape
        if len(self.crop) == 2:
            image_with_mask = add_mask_to_roi(frame, model_out, self.crop, color_list=mask_color)
        else:
            result = cv2.resize(model_out, self.show_size)
            image_with_mask = get_image_with_mask(frame, result, color_list=mask_color)

        image_with_mask[:h, :w] = self.color_mask_map

        return image_with_mask

    def elevation_post_process(self, model_out, frame):
        results = {}
        for cls in [5, 6, 7, 8, 9]:
            bboxes = []
            label_cls = (model_out == cls).astype(np.uint8)
            if label_cls.sum() == 0:
                continue
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(label_cls, connectivity=8)
            for stat in stats[1:]:
                x, y, w, h = stat[:-1]
                score = w * h / 200
                if score > 1:
                    bboxes.append([x, y, x + w, y + h, 1.0])
                elif score > 0.3:
                    bboxes.append([x, y, x + w, y + h, score])
            results[cls - 5] = bboxes

        lt, rb = self.crop
        x_min, y_min = lt
        x_max, y_max = rb
        img_width = x_max - x_min
        img_height = y_max - y_min
        frame[y_min:y_max, x_min:x_max] = overlay_bbox_cv(
            frame[y_min:y_max, x_min:x_max],
            results,
            self.elevation_class_names
        )
        return frame

    def draw_line(self, image, line_xy):
        lr = self.crop[0]
        for xy in line_xy:
            pt_start, pt_end = xy
            cv2.line(image, pt_add(pt_start, lr), pt_add(pt_end, lr), (255, 0, 0))

    def analysis_status(self, status):
        sum_status = np.sum(status)
        if sum_status == 0:
            return 0
        current_ratio = status / sum_status
        if self.history_status is None:
            self.history_status = current_ratio
            cosine_distance = 0
        else:
            cosine_distance = 1 - np.dot(current_ratio, self.history_status)
            weight_decay = 0.04 * (1.2 - cosine_distance) * (1.2 - cosine_distance)
            self.history_status = self.history_status * (1 - weight_decay) + current_ratio * weight_decay
        return cosine_distance

    def inference_images(self, image_path, driving_area_mask, line_xy, save_dir=None):
        image_list = []

        if isinstance(image_path, list):
            image_list = image_path
        elif os.path.exists(image_path):
            image_list = get_image_list(image_path)

        for image_name in image_list:
            image_src = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), 1)
            image_src = cv2.resize(image_src, self.show_size)

            terrain_result, elevation_result = self.inference(image_src)

            if save_dir is None:
                cv2.imshow("terrain_result", terrain_result)
                cv2.imshow("elevation_result", elevation_result)
                cv2.waitKey(0)
            else:
                imwrite(os.path.join(save_dir, os.path.basename(image_name)), terrain_result)

    def inference_video(self, video_path, driving_area_mask, line_xy, save_dir=None, infer_mode=0):
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        color_mask_map = cv2.imread(self.mask_color_path)
        FPS = 10
        down_sample = 1

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
                    terrain_result, elevation_result = self.inference(frame)
                    # count += 1
                    # print(count)
                    if save_dir is not None:
                        if infer_mode:
                            vid_writer.write(elevation_result)
                        else:
                            vid_writer.write(terrain_result)
                    else:
                        cv2.imshow("terrain_result", terrain_result)
                        cv2.imshow("elevation_result", elevation_result)
                        cv2.waitKey(0)
            else:
                if save_dir is not None:
                    vid_writer.release()
                cap.release()
                self.history_status = None
                print(f"{video_path} end")
                break

    def put_text(self, image, ratio, class_index, distance=0):
        h, w, _ = self.color_mask_map.shape
        for idx, r in enumerate(ratio):
            cv2.putText(
                image,
                "%.1f" % (r * 100),
                (w + 10, h * (idx + 2) // len(self.class_names)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255)
            )

        cv2.putText(
            image,
            self.class_names[class_index],
            (500, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 3,
            (0, 255, 0),
            3
        )
        if distance > 0:
            cv2.putText(
                image,
                f"distance:{distance}",
                (320, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )
