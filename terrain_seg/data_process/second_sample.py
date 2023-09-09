import os
from utils import mkdir
import shutil

if __name__ == "__main__":
    scene_sample_frequency = {
        "冰雪地": 48,
        "草地": 6,
        "测试场地": 6,
        "简单道路": 6,
        "泥泞地": 6,
        "沙地": 24,
        "山路": 6,
        "涉水": 6,
    }

    src_dir = r"E:\比亚迪全地形分割\images"
    save_dir = r"E:\比亚迪全地形分割\标注样本"
    for scene_name in os.listdir(src_dir):
        per_scene_src_dir = os.path.join(src_dir, scene_name)
        per_scene_save_dir = os.path.join(save_dir, scene_name)
        mkdir(per_scene_save_dir)
        per_scene_image_list = os.listdir(os.path.join(src_dir, scene_name))
        print(scene_name, ":", len(per_scene_image_list), ",二次抽帧:",
              len(per_scene_image_list) // scene_sample_frequency[scene_name])
        for idx, image_name in enumerate(per_scene_image_list):
            if idx % scene_sample_frequency[scene_name] == 0:
                shutil.move(os.path.join(per_scene_src_dir, image_name), os.path.join(per_scene_save_dir, image_name))
