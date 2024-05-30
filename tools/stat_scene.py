import os
import numpy as np

from core.utils import imread
from terrain_seg import class_names

if __name__ == "__main__":
    mask_dir = r"D:\dataset\terrain_data\images_in_server\new_masks"
    sum_count = None

    exist_scene_dict = {}

    for per_mask in os.listdir(mask_dir):

        mask = imread(os.path.join(mask_dir, per_mask), flag=0)
        h, w = mask.shape
        per_mask_count = np.bincount(mask.flatten(), minlength=len(class_names))
        if sum_count is None:
            sum_count = per_mask_count
        else:
            sum_count += per_mask_count

        ratio = per_mask_count/np.sum(per_mask_count)
        if ratio[0] > 0.85:
            scene_name = "background"
        else:
            keep = np.where(ratio > 0.05)
            keep_class_name = []
            for k in keep[0].tolist():
                keep_class_name.append(class_names[k])
            scene_name = ",".join(keep_class_name[1:])

        if exist_scene_dict.__contains__(scene_name):
            exist_scene_dict[scene_name] += 1
        else:
            exist_scene_dict[scene_name] = 1

    stat_result = sorted(exist_scene_dict.items(), key=lambda d: d[1], reverse=True)
    print(stat_result)
    total_ratio = sum_count/np.sum(sum_count)
    for name, r in zip(class_names, total_ratio):
        print(name, r*100, "%")
