import csv
import operator
import os
import shutil
import random

from traffic_sign import chinese_to_english, english_to_chinese

dataset_path = r"D:\dataset\traffic_sign_data\traffic_sign_cls\traffic_sign_cls"
dst_image_path = os.path.join(dataset_path, "images")
dst_annotation_path = os.path.join(dst_image_path, "annotations")


def statistics_per_class_num():
    per_class_num_dict = {}
    for per_class_name in os.listdir(dst_image_path):
        per_class_num_dict[per_class_name] = len(os.listdir(os.path.join(dst_image_path, per_class_name)))

    per_class_num_dict = sorted(per_class_num_dict.items(), key=operator.itemgetter(1))
    for key, value in per_class_num_dict:
        print(key, value)


def build_cls_dataset():
    train_data_list = []
    val_data_list = []

    for idx, chinese_name, english_name in chinese_to_english.items():
        class_id = idx
        os.rename(os.path.join(dst_image_path, chinese_name), os.path.join(dst_image_path, english_name))
        per_class_path = os.path.join(dst_image_path, english_name)
        per_class_num = len(os.listdir(per_class_path))
        ratio = 2000 // per_class_num + 1
        count = 0
        for image_name in os.listdir(per_class_path):
            image_path = os.path.join(per_class_path, image_name)
            dst_path = os.path.join(dst_image_path, image_name)
            if not os.path.exists(dst_path):
                shutil.copy(image_path, os.path.join(dst_image_path, image_name))
            if count % 10 == 0:
                for _ in range(ratio):
                    val_data_list.append((image_name, class_id, 1.0))
            else:
                for _ in range(ratio):
                    train_data_list.append((image_name, class_id, 1.0))
            count += 1
        print(chinese_name, "expand ratio:", ratio, "image num:", count)

    random.shuffle(train_data_list)
    random.shuffle(val_data_list)
    train_data_list.insert(0, ("file_name", "category_id", "label_scores"))
    val_data_list.insert(0, ("file_name", "category_id", "label_scores"))

    with open(os.path.join(dst_annotation_path, "train.csv"), "wt") as f:
        cw = csv.writer(f, lineterminator='\n')
        for item in train_data_list:
            cw.writerow(item)

    with open(os.path.join(dst_annotation_path, "val.csv"), "wt") as f:
        cw = csv.writer(f, lineterminator='\n')
        for item in val_data_list:
            cw.writerow(item)


if __name__ == "__main__":
    statistics_per_class_num()
