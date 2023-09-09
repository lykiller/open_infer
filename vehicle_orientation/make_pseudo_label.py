import os
import shutil

from utils import mkdir, get_video_list, get_image_list, imread
from vehicle_orientation import class_names
from load_model import init_orientation_classifier

if __name__ == "__main__":
    predictor = init_orientation_classifier()
    src_dir = r"D:\dataset\vehicle_orientation"
    dst_dir = os.path.join(src_dir, "pseudo_label")

    for name in class_names:
        dst_path_dir = os.path.join(dst_dir, name)
        mkdir(dst_path_dir)
        print(dst_path_dir)

    type_list = ["bus", "truck", "car"]
    for vehicle_type in type_list:
        image_path_dir = os.path.join(src_dir, vehicle_type)
        for image_idx, image_name in enumerate(os.listdir(image_path_dir)):
            src_image_path = os.path.join(image_path_dir, image_name)
            image_np = imread(src_image_path)
            label, score = predictor.inference(image_np)
            pseudo_label_name = class_names[int(label)]
            shutil.move(src_image_path, os.path.join(dst_dir, pseudo_label_name, image_name))
