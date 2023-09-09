import os
import numpy as np
import cv2

from terrain_seg.visualize import get_image_with_mask
from utils import imwrite, imread, mkdir
from terrain_seg import class_names, mask_color, color_mask_map


if __name__ == "__main__":
    src_dir = r"D:\dataset\terrain_data\need_label_dianwo_20230822"

    mask_path = os.path.join(src_dir, "masks")
    image_path = os.path.join(src_dir, "src_images")
    visual_mask_path = os.path.join(src_dir, "visual_masks")
    mkdir(visual_mask_path)
    if_show = False
    for mask_name in os.listdir(mask_path):
        mask = imread(os.path.join(mask_path, mask_name), 0)
        image = imread(os.path.join(image_path, mask_name.replace(".png", ".jpg")))
        image_with_mask = get_image_with_mask(image, mask, mask_color)
        height, width, _ = image.shape
        h, w, _ = color_mask_map.shape
        image_with_mask[:h, :w, ] = color_mask_map
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
