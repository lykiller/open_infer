import os
import shutil

if __name__ == "__main__":
    type_list = ["car", "bus", "truck"]
    src_dir = r"D:\dataset\vehicle_orientation"

    for vehicle_type in type_list:
        image_path_dir = os.path.join(src_dir, vehicle_type)
        for image_idx, image_name in enumerate(os.listdir(image_path_dir)):
            src_image_path = os.path.join(image_path_dir, image_name)
            new_image_name = vehicle_type + "-" + str(image_idx) + "-20230901.jpg"
            dst_image_path = os.path.join(image_path_dir, new_image_name)
            os.rename(src_image_path, dst_image_path)
