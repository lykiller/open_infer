from core.utils import imread
from core.driving_area import emulate_straight_mask

# BGR
terrain_mask_color_list = [
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

mask_color_path = r"D:\open_infer\doc\color_mask.png"
terrain_mask_color_map = imread(mask_color_path)

roi_shape = (960, 320)
top = (0.04, 0.24)
bottom = (0.06, 0.36)

crop = []
show_size = (1280, 720)
input_size = (576, 320)

input_node_name = "data"
color_mode = "rgb"

driving_area_mask, line_xy = emulate_straight_mask(roi_shape, top, bottom)
