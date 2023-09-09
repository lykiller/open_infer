import os
import shutil
import pandas as pd

if __name__ == "__main__":
    image_dir = r"F:\cls\images"
    dst_dir = r"F:\cls\new_images"
    exist_images = pd.read_csv(r"F:\cls\exist_images.csv")["file_name"].tolist()
    for image_name in os.listdir(image_dir):
        if image_name not in exist_images:
            print(image_name)
            shutil.move(os.path.join(image_dir, image_name), os.path.join(dst_dir, image_name))
