import os
import numpy as np

from core.utils import imread, imwrite
from terrain_seg import terrain_mask_color_list
from data.visual.visualize import get_image_with_mask

if __name__ == "__main__":
    elevation_label_dir = r"D:\dataset\elevation_data\train\labels"
    image_dir = r"D:\dataset\elevation_data\train\images"

    exist_terrain_mask_list = os.listdir(r"D:\dataset\terrain_data\images_in_server\new_masks")
    for per_elevation_label_name in os.listdir(elevation_label_dir):
        if per_elevation_label_name not in exist_terrain_mask_list:
            mask = imread(os.path.join(elevation_label_dir, per_elevation_label_name))[:, :, 0]
            new_mask = np.zeros(mask.shape)
            new_mask[np.where(mask > 0)] = 1
            new_mask[np.where(mask > 5)] = 11
            new_mask[np.where(mask == 3)] = 0

            image = imread(os.path.join(image_dir, per_elevation_label_name.replace(".png", ".jpg")))
            image_with_mask = get_image_with_mask(image, new_mask, terrain_mask_color_list)

            imwrite(os.path.join(r"D:\dataset\elevation_data\train\new_masks", per_elevation_label_name), new_mask)
            imwrite(
                os.path.join(r"D:\dataset\elevation_data\train\visual_masks",
                             per_elevation_label_name.replace(".png", ".jpg")),
                image_with_mask
            )
