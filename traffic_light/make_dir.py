import os

from utils import mkdir
from traffic_light import direction_class_names, color_class_names

if __name__ == "__main__":
    dst_dir = r"E:\traffic_light\label"
    for color in color_class_names:
        for direction in direction_class_names:
            mkdir(os.path.join(dst_dir, color + "-" + direction))
