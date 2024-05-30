import os
import cv2
import json

import numpy as np

image_ext = [".jpg", ".JPG", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = [".mov", ".avi", ".mkv", ".h265", ".mp4"]

_COLORS = (
    np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.314, 0.717, 0.741,
            0.500, 0.500, 0.000,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


def fast_sigmoid(x):
    return 1/(1 + np.exp(-x))


def get_image_list(path):
    image_names = []
    for main_dir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(main_dir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names


def get_video_list(path):
    video_names = []
    for main_dir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(main_dir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in video_ext:
                video_names.append(apath)
    return video_names


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def imread(path, flag=1):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)


def imwrite(path, image):
    cv2.imencode(os.path.splitext(path)[1], image)[1].tofile(path)

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)