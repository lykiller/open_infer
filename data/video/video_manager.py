import os

import cv2

from data.visual.visual_frame import VisualFrame
from core.utils import imwrite
from mono_task.model.detetron import det2d_class_names


class VideoManager:
    def __init__(
            self,
            video_path,
            show_size,
            down_sample,
            save_path_dict,
            FPS=10,
            show_type="cv2",
    ):
        self.video_path = video_path
        self.frame_id = -1
        self.last_result = dict(
            det_bboxes=dict(),
            mot_bboxes=dict(),
            terrain_seg=dict(),
        )
        for name in det2d_class_names:
            self.last_result["det_bboxes"][name] = list()
            self.last_result["mot_bboxes"][name] = list()
        self.last_result["terrain_seg"]["state"] = dict()

        self.show_size = show_size
        self.down_sample = down_sample
        self.save_path_dict = save_path_dict
        self.FPS = FPS
        self.show_type = show_type

    def infer_video(
            self,
            model_manager,
            if_show_src=False,
            if_show_det=False,
            if_show_mot=False,
            if_show_terrain=False
    ):
        cap = cv2.VideoCapture(self.video_path)
        vid_writer_dict = dict()
        if len(self.save_path_dict) and self.show_type == "video":
            for key, save_path in self.save_path_dict.items():
                vid_writer_dict[key] = (
                    cv2.VideoWriter(
                        os.path.join(save_path, self.video_path.split("\\")[-1]),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.FPS,
                        self.show_size,
                        True
                    )
                )
        print(f"{self.video_path} begin")
        while True:
            ret_val, frame = cap.read()
            self.frame_id += 1
            if ret_val:
                if self.frame_id % self.down_sample == 0:
                    frame = cv2.resize(frame, self.show_size)
                    result = model_manager.inference(frame, self.last_result)
                    frame_data = VisualFrame(
                        src_image=frame,
                        result=result
                    )
                    visual_images = frame_data.visual(
                        if_show_src=if_show_src,
                        if_show_det=if_show_det,
                        if_show_mot=if_show_mot,
                        if_show_terrain=if_show_terrain,
                    )

                    for visual_name, visual_image in visual_images.items():
                        if self.show_type == "image":
                            imwrite(
                                os.path.join(self.save_path_dict[visual_name], os.path.basename(self.video_path)[:-4] + f"__{self.frame_id}.jpg"),
                                visual_image
                                    )
                        elif self.show_type == "video":
                            vid_writer_dict[visual_name].write(visual_image)
                        else:
                            cv2.imshow(visual_name, visual_image)
                    cv2.waitKey(0)
                    self.last_result = result

            else:
                if len(vid_writer_dict):
                    for _, vid_writer in vid_writer_dict.items():
                        vid_writer.release()
                cap.release()
                break

