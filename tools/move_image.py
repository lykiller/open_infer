import os
import shutil

if __name__ == "__main__":
    image_path = r"D:\自测结果\visual_masks12.29"

    error_path = r"D:\自测结果\mask"

    dst_path = r"D:\自测结果\新建文件夹"
    for image_name in os.listdir(error_path):
        if os.path.exists(
                os.path.join(image_path, image_name.replace(".png", ".jpg"))):
            shutil.move(os.path.join(error_path, image_name), os.path.join(dst_path, image_name))
