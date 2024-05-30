# encoding:utf-8

import os
import shutil

from core.utils import mkdir

labeling_and_labeled_path = [
    r"D:\dataset\terrain_data\need_label_dianwo_0714",
    r"D:\dataset\terrain_data\images_in_server\images",
    r"D:\dataset\terrain_data\images_no_label\labeling",
    r"D:\dataset\terrain_data\need_label_dianwo_0718_and_0721",
]


def move_source_images(src_path, dst_path, if_move_list):
    mkdir(dst_path)
    # json_path = r"E:\比亚迪全地形分割\new_json_0608"
    # json_dst_path = r"E:\need_label6_json"
    for image_name in os.listdir(src_path):
        if image_name.replace(".png", ".jpg") in if_move_list:
            shutil.move(os.path.join(src_path, image_name), os.path.join(dst_path, image_name))
            print(image_name)


def copy_source_images(src_path, dst_path, if_move_list):
    mkdir(dst_path)
    # json_path = r"E:\比亚迪全地形分割\new_json_0608"
    # json_dst_path = r"E:\need_label6_json"
    for image_name in os.listdir(src_path):
        if image_name.replace(".png", ".jpg") in if_move_list:
            shutil.copy(os.path.join(src_path, image_name), os.path.join(dst_path, image_name))
            print(image_name)


def delete_source_images(src_path, if_delete_list):
    for image_name in os.listdir(src_path):
        if image_name.replace(".png", ".jpg") in if_delete_list:
            os.remove(os.path.join(src_path, image_name))
            print(image_name)


def move_need_label_images():
    if_move_dir_list = [
        r"G:\自测结果\全地形自测\20231120\error_masks",
    ]

    dst_path = r"G:\自测结果\全地形自测\20231120\need_labeled_image"
    mkdir(dst_path)

    for if_move_dir in if_move_dir_list:
        src_path = if_move_dir.replace("error_masks", "src_images")
        move_source_images(src_path, dst_path, os.listdir(if_move_dir))

    # label_list = []
    # for label_path in labeling_and_labeled_path:
    #     label_list += os.listdir(label_path)

    # delete_source_images(dst_path, label_list)


def if_repeat():
    src_path = r"D:\dataset\terrain_data\need_label_dianwo_20230822"
    label_list = []
    for label_path in labeling_and_labeled_path:
        label_list += os.listdir(label_path)
    for image_name in os.listdir(src_path):
        if image_name in label_list:
            print(image_name)


if __name__ == "__main__":
    move_need_label_images()

