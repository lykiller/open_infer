import os
import shutil
import cv2

from core.utils import imread, imwrite


def down_sample():
    src_data_path = r"D:\文档\自动泊车\src_images"
    dst_data_path = r"D:\文档\自动泊车\need_label_avm_images"

    for scene_name in os.listdir(src_data_path):
        src_scene_path = os.path.join(src_data_path, scene_name)
        dst_scene_path = os.path.join(dst_data_path, scene_name)
        os.makedirs(dst_scene_path, exist_ok=True)
        image_list = os.listdir(src_scene_path)
        image_list.sort()
        for idx, image_name in enumerate(image_list):
            if idx % 5 == 0:
                shutil.move(os.path.join(src_scene_path, image_name), os.path.join(dst_scene_path, image_name))


if __name__ == "__main__":
    src_data_path = r"E:\need_label_avm_images"
    for scene_name in os.listdir(src_data_path):
        for image_name in os.listdir(os.path.join(src_data_path, scene_name)):
            image = imread(os.path.join(src_data_path, scene_name, image_name))
            imwrite(os.path.join(src_data_path, scene_name, image_name), cv2.resize(image, (512, 512)))
