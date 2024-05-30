import copy
import cv2

from data.visual.visual_bboxes import overlay_bbox_cv
from data.visual.visualize import add_mask_to_roi, get_image_with_mask
from traffic_object.terrain import terrain_mask_color_map, terrain_mask_color_list


class VisualFrame:
    def __init__(
            self,
            src_image,
            result,
    ):
        self.src_image = src_image
        self.result = result

    def visual(
            self,
            if_show_src,
            if_show_det,
            if_show_mot,
            if_show_terrain,
    ):
        visual_images = dict()
        if if_show_src:
            visual_images["src_images"] = self.src_image

        if self.result.__contains__("det_bboxes") and if_show_det:
            visual_images["det_images"] = self.visual_det_result(
                self.result["det_bboxes"]
            )
        if self.result.__contains__("mot_bboxes") and if_show_mot:
            visual_images["mot_images"] = self.visual_mot_result(
                self.result["mot_bboxes"]
            )
        if self.result.__contains__("terrain_seg") and if_show_terrain:
            visual_images["terrain_seg_images"] = self.visual_terrain_result(
                self.result["terrain_seg"]["mask"],
                self.result["terrain_seg"]["crop"],
                self.result["terrain_seg"]["state"],
            )
        if self.result.__contains__("parking_slots"):
            visual_images["parking_slots"] = self.visual_apa_result(self.result["parking_slots"])

        return visual_images

    def visual_det_result(self, det_bboxes):
        image_src = copy.deepcopy(self.src_image)
        return overlay_bbox_cv(image_src, det_bboxes)

    def visual_mot_result(self, mot_bboxes):
        return 0

    def visual_terrain_result(self, mask, crop, state):
        image_src = copy.deepcopy(self.src_image)
        h, w, _ = terrain_mask_color_map.shape
        if len(crop) == 2:
            image_with_mask = add_mask_to_roi(image_src, mask, crop, color_list=terrain_mask_color_list)
        else:
            result = cv2.resize(mask, image_src.shape, interpolation=cv2.INTER_NEAREST)
            image_with_mask = get_image_with_mask(image_src, result, color_list=terrain_mask_color_list)

        image_with_mask[:h, :w] = terrain_mask_color_map

        if len(state):
            for idx, (key, value) in enumerate(state.items()):
                text = f"{key}: {str(value)}"
                cv2.putText(
                    image_src,
                    text,
                    org=(250, 25 + 50 * idx),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 0, 255),
                    thickness=1
                )

        return image_with_mask

    def visual_apa_result(self, apa_result):
        image_src = copy.deepcopy(self.src_image)
        for parking_slot in apa_result:

            if parking_slot.occupy:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(image_src, parking_slot.pmin, parking_slot.pmax, color, 2)
            cv2.line(image_src, parking_slot.entry_center, parking_slot.end_center, (255, 0, 0), 3)
            if parking_slot.valid:
                cv2.line(image_src, parking_slot.pa, parking_slot.pb, (255, 255, 255), 3)
                cv2.line(image_src, parking_slot.pa, parking_slot.pd, (255, 0, 0), 3)
                cv2.line(image_src, parking_slot.pb, parking_slot.pc, (255, 0, 0), 3)
        return image_src
