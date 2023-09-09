import os

tsr_class_names = [
    'others',
    'highest_speed_limit',
    'lowest_speed_limit',
    'height_limit',
    'weight_limit',
    'no_overtaking',
    'no_entry',
    'no_parking',
    'release_speed_limit',
    'release_no_overtaking',
]

name2id_map = {
    'others': 0,
    'highest_speed_limit': 1,
    'lowest_speed_limit': 2,
    'height_limit': 3,
    'weight_limit': 4,
    'no_overtaking': 5,
    'no_entry': 6,
    'no_parking': 7,
    'release_speed_limit': 8,
    'release_no_overtaking': 9,
    'direction_sign': 0,
}


need_ocr_index = [1, 2, 3, 4]

ocr_class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'm', 't']
