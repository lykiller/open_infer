import os
import numpy as np
import cv2

from data.visual.visualize import get_image_with_mask
from core.utils import imwrite, imread, mkdir
from terrain_seg import class_names, terrain_mask_color_list, terrain_mask_color_map


if __name__ == "__main__":
    src_dir = r"D:\dataset\terrain_data\images_in_server"

    mask_path = os.path.join(src_dir, "masks")
    image_path = os.path.join(src_dir, "src_images")
    visual_mask_path = os.path.join(src_dir, "visual_masks")
    mkdir(visual_mask_path)
    if_show = False
    for mask_name in os.listdir(mask_path):
        mask = imread(os.path.join(mask_path, mask_name), 0)
        per_image_path = os.path.join(image_path, mask_name.replace(".png", ".jpg"))
        if os.path.exists(per_image_path):
            image = imread(per_image_path)
            print(mask_name)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            image_with_mask = get_image_with_mask(image, mask, terrain_mask_color_list)
            height, width, _ = image.shape
            h, w, _ = terrain_mask_color_map.shape
            image_with_mask[:h, :w, ] = terrain_mask_color_map
            percent = np.bincount(mask.flatten(), minlength=len(class_names)) / (height * width)
            for index, per in enumerate(percent):
                cv2.putText(
                    image_with_mask,
                    "%.1f" % (per * 100),
                    (w + 10, h * (index + 1) // len(class_names)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255)
                )
            if if_show:
                cv2.imshow("image", image_with_mask)
                cv2.waitKey(0)

            imwrite(os.path.join(visual_mask_path, mask_name.replace(".png", ".jpg")), image_with_mask)
