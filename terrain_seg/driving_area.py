import os
import cv2
import numpy as np
from utils import get_image_list


def emulate_straight_mask(mask_shape, top, bottom):
    width, height = mask_shape
    top_min, top_max = top
    bottom_min, bottom_max = bottom
    mask = np.zeros((height, width))
    left_area = [
        [int(width * (0.5 - top_max)), 0],
        [int(width * (0.5 - top_min)), 0],
        [int(width * (0.5 - bottom_min)), height],
        [int(width * (0.5 - bottom_max)), height],
    ]

    right_area = [
        [int(width * (0.5 + top_max)), 0],
        [int(width * (0.5 + top_min)), 0],
        [int(width * (0.5 + bottom_min)), height],
        [int(width * (0.5 + bottom_max)), height],
    ]
    poly = [np.array(left_area), np.array(right_area)]
    cv2.fillPoly(mask, poly, 1)
    line_xy = [
        [[int(width * (0.5 - top_max)), 0], [int(width * (0.5 - bottom_max)), height]],
        [[int(width * (0.5 - top_min)), 0], [int(width * (0.5 - bottom_min)), height]],
        [[int(width * (0.5 + top_min)), 0], [int(width * (0.5 + bottom_min)), height]],
        [[int(width * (0.5 + top_max)), 0], [int(width * (0.5 + bottom_max)), height]],
    ]
    return mask, line_xy


if __name__ == "__main__":
    roi_shape = (960, 320)
    top = (0.04, 0.28)
    bottom = (0.06, 0.42)

    crop = [[160, 400], [1120, 720]]
    lt, rb = crop
    x_min, y_min = lt
    x_max, y_max = rb

    show_size = (1280, 720)

    mask, _ = emulate_straight_mask(roi_shape, top, bottom)
    # mask = cv2.blur(mask, ksize=(31, 31))
    mask = (mask * 255).astype(np.uint8)

    image_path = r"E:\dataset\images"

    image_list = get_image_list(image_path)

    for per_image_path in image_list:
        image = cv2.imdecode(np.fromfile(per_image_path, dtype=np.uint8), -1)
        image = cv2.resize(image, show_size)
        image[y_min:y_max, x_min:x_max] = cv2.addWeighted(image[y_min:y_max, x_min:x_max], 0.5,
                                                          np.repeat(mask[:, :, np.newaxis], 3, axis=-1), 0.5, 0)

        cv2.imshow("mask", image)
        cv2.waitKey(0)
