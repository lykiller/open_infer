from utils import imread
from terrain_seg.driving_area import emulate_straight_mask

mask_color = [
    [0, 0, 0],  # background
    [255, 255, 0],  # asphalt
    [255, 0, 0],  # gravel
    [192, 192, 192],  # earth
    [0, 255, 0],  # grassland
    [42, 42, 128],  # mud
    [0, 91, 255],  # water
    [34, 139, 34],  # rock
    [33, 145, 237],  # sand
    [205, 235, 255],  # light_snow
    [240, 32, 160],  # deep_snow
    [0, 255, 255],  # others
]

class_names = [
    "background",  # 0 背景
    "asphalt",  # 1 人工
    "gravel",  # 2 砾石
    "earth",  # 3 土
    "grass",  # 4 草
    "mud",  # 5 泥
    "water",  # 6 水
    "rock",  # 7 岩石
    "sand",  # 8 沙
    "light_snow",  # 9 冰
    "deep_snow",  # 10 雪
    "others"  # 11 其他
]

onnx_dir = r"D:\自动驾驶\比亚迪全地形分割\onnx"
mask_color_path = r"D:\open_infer\terrain_seg\doc\color_mask.png"
color_mask_map = imread(mask_color_path)

roi_shape = (960, 320)
top = (0.04, 0.24)
bottom = (0.06, 0.36)

crop = []
show_size = (1280, 720)
input_size = (576, 320)

input_node_name = "data"
color_mode = "rgb"
normalize = [[128., 128., 128.], [128., 128., 128.]]

driving_area_mask, line_xy = emulate_straight_mask(roi_shape, top, bottom)
