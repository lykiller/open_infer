import cv2
import os
import numpy as np

from utils import get_image_list

image_list = get_image_list(r"D:\dataset\terrain_data\images_no_label\need_label_images_0620_lh\images")
for image_path in image_list:
    image_path = image_path.replace(".jpg", ".png")
    if os.path.exists(image_path):
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        cv2.imwrite(image_path.replace(".png", ".jpg"), image)
