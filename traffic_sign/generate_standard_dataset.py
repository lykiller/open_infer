import csv
import os
import shutil
import random

from traffic_sign import tsr_class_names, name2id_map


def build_cls_dataset():
    src_path = r"F:\traffic_sign_cls"
    dst_image_path = r"F:\cls\images"
    dst_annotation_path = r"F:\cls\annotations"

    train_data_list = []
    val_data_list = []

    for label_name in os.listdir(src_path):
        class_name = label_name.split("__")[0]
        if class_name in tsr_class_names:
            count = 0
            class_id = name2id_map[class_name]
            per_class_path = os.path.join(src_path, label_name)
            per_class_num = len(os.listdir(per_class_path))
            ratio = 20000 // per_class_num + 1

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
            print(class_name, "expand ratio:", ratio, "image num:", count)

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
    build_cls_dataset()
