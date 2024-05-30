import os
import cv2

from data.visual.visual_frame import VisualFrame
from core.utils import imwrite, get_image_list, imread


class ImageManager:
    def __init__(
            self,
            image_path,
            save_path_dict,
            show_type,
            show_size,
    ):
        self.image_list = get_image_list(image_path)
        self.save_path_dict = save_path_dict
        self.show_type = show_type
        self.show_size = show_size

    def infer_images(
            self,
            model_manager,
            if_show_src=False,
            if_show_det=False,
            if_show_mot=False,
            if_show_terrain=False
    ):
        for image_path in self.image_list:
            frame = imread(image_path)
            frame = cv2.resize(frame, self.show_size)
            result = model_manager.inference(frame, dict())
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
                        os.path.join(
                            self.save_path_dict[visual_name], os.path.basename(image_path)),
                                     visual_image
                        )
                else:
                    cv2.imshow(visual_name, visual_image)
            cv2.waitKey(0)

    def rename(self):
        pass

    def down_sample(self):
        pass

    def move(self):
        pass

    def inference(self):
        pass
