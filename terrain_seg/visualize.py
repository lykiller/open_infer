import cv2
import numpy as np


def get_color_mask(mask, color_list):
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3))
    for idx, color in enumerate(color_list):
        if idx:
            color_mask[mask == idx] = tuple(color)
    return color_mask


def get_image_with_mask(image, mask, color_list):
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3))
    for idx, color in enumerate(color_list):
        color_mask[mask == idx] = tuple(color)

    return cv2.addWeighted(image, 0.7, color_mask.astype(np.uint8), 0.3, 0)


def add_mask_to_roi(image, mask, crop, color_list):
    lt, rb = crop
    x_min, y_min = lt
    x_max, y_max = rb
    image[y_min:y_max, x_min:x_max] = get_image_with_mask(image[y_min:y_max, x_min:x_max], mask, color_list)

    return image
