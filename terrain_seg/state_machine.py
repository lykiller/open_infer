import cv2
import os

import numpy as np


FPS = 10

invalid_threshold = 0.7

label_map_dict = {
    "background": -1,

    "asphalt": 0,
    "earth": 0,

    "light_snow": 1,
    "gravel": 1,
    "grassland": 1,

    "wading": 2,

    "deep_snow": 3,
    "sand": 3,

    "mud": 4,

    "rock": 5,
}

special_label_map_dict = {
    "water, earth": 4,

    "grassland,asphalt": 0,
    "grassland,earth": 0,
    "earth, grassland": 0,
    "asphalt, grassland": 0,

}


def argmax_state_machine(mask, class_names):
    state = np.bincount(mask.astype(np.uint8).flatten(), minlength=len(class_names))
    state_ratio = state / np.sum(state)
    if state_ratio[0] > invalid_threshold:
        return 0, state_ratio
    else:
        state = state[1:]
        state_ratio = state / np.sum(state)
        return np.argmax(state_ratio) + 1, state_ratio


def argmax_k_state_machine(mask, class_names, k=3, valid_threshold=0.25):
    state = np.bincount(mask.astype(np.uint8).flatten(), minlength=len(class_names))
    state_ratio = state / np.sum(state)
    if state_ratio[0] > invalid_threshold:
        return [0], state_ratio
    else:
        state = state[1:]
        state_ratio = state / np.sum(state)
        labels = state_ratio.argsort()[-k:][::-1]
        keep = np.where(state_ratio[labels] > valid_threshold)
        return (labels[keep] + 1).tolist(), state_ratio


def mask_label_to_machine_label(labels_list, class_names):
    if isinstance(labels_list, int):
        return label_map_dict[class_names[labels_list]]
    elif len(labels_list) == 1:
        return label_map_dict[class_names[labels_list[0]]]
    elif len(labels_list) > 1:
        main_label_name = class_names[labels_list[0]]
        aux_label_name = class_names[labels_list[1]]
        class_names_str = main_label_name + "," + aux_label_name
        if special_label_map_dict.__contains__(class_names_str):
            return special_label_map_dict[class_names_str]
        else:

            return max(label_map_dict[main_label_name], label_map_dict[aux_label_name])
