import os
import shutil

if __name__ == "__main__":
    src_path = r"D:\dataset\elevation_data\train\visual_masks"
    dst_path = r"D:\dataset\elevation_data\train\mini_visual_masks"

    for idx, image_name in enumerate(os.listdir(src_path)):
        if idx % 3 == 1:
            shutil.move(os.path.join(src_path, image_name), os.path.join(dst_path, image_name))
