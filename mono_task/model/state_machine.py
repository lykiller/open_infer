import cv2
import os

import numpy as np

level_name_list = [
    "very low",
    "low",
    "normal",
    "high",
    "very_high"
]


class TerrainState:
    def __init__(
            self,
            main_class,
            new_main_class,
            aux_class,
            hardness,
            adhesion,
            wetness,
            evenness,
    ):
        self.main_class = main_class
        self.new_main_class = new_main_class
        self.aux_class = aux_class
        self.hardness = hardness
        self.adhesion = adhesion
        self.wetness = wetness
        self.evenness = evenness
        self.state_age = 0
        self.hit = 0

    def update(self, new_main_class, new_aux_class, hardness, adhesion, wetness, evenness):
        self.hardness = self.hardness * 0.8 + hardness * 0.2
        self.adhesion = self.adhesion * 0.8 + adhesion * 0.2
        self.wetness = self.wetness * 0.8 + wetness * 0.2
        self.evenness = self.evenness * 0.8 + evenness * 0.2

        if new_main_class == self.new_main_class:
            self.hit += 1
        else:
            self.new_main_class = new_main_class
            self.hit = 0

        if self.hit > 3:
            self.aux_class = new_aux_class

        if self.hit > 6 and self.state_age > 20 and self.new_main_class != self.main_class:
            self.main_class = self.new_main_class

            self.state_age = 0
        else:
            self.state_age += 1
        return self.describe_road()

    def describe_road(self):
        result = {
            "description": f"{self.main_class} road {self.aux_class}",
            "main_class": self.main_class,
            "hardness": level_name_list[int(self.hardness * 5)],
            "evenness": level_name_list[int(self.evenness * 5)],
            # "adhesion": level_name_list[int(self.adhesion * 5)],
            # "wetness": level_name_list[min(int(self.wetness * 5), 4)],
        }
        return result


class TerrainStateMachine:
    def __init__(
            self,
            class_names,
            driving_area_mask,
            mask_valid_threshold,

    ):
        self.class_names = class_names
        self.class_num = len(class_names)
        self.mask_valid_threshold = mask_valid_threshold
        self.driving_area_mask = driving_area_mask

        # 1 人工 # 2 砾石 # 3 土 # 4 草 # 5 泥
        # 6 水 # 7 岩石 # 8 沙 # 9 浅雪 # 10 深雪 # 11 其他
        self.aux_class_idx_list = [1, 3, 5, 7, 8]
        self.attribute_weights = {
            "hardness": np.array([1.0, 1.0, 0.8, 0.5, 0.2, 0.0, 1.0, 0.2, 0.5, 0.5, 0.9]),
            "adhesion": np.array([1.0, 0.5, 0.9, 0.3, 0.3, 0.0, 0.9, 0.3, 0.1, 0.1, 0.7]),
            "evenness": np.array([1.0, 0.2, 0.9, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            "wetness": np.array([0.0, 0.0, 0.0, 0.2, 0.6, 2.0, 0.0, 0.0, 0.6, 1.2, 0.0])
        }

        self.main_class_weights = {
            "rock": np.array([0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0]),
            "mud": np.array([0.0, 0.0, 0.0, 0.0, 1.5, 0.5, 0.0, 0.0, 0.2, 0.2, 0.0]),
            "snow": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 1.0, 0.0]),
            "sand": np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0]),
            "wading": np.array([0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "earth": np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "asphalt": np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.8])
        }
        self._init_state()

    def _init_state(self):
        self.last_mask_ratio = None
        self.history_mask_ratio = None
        self.terrain_state = TerrainState(
            main_class="unknown",
            new_main_class="unknown",
            aux_class="unknown",
            hardness=0.8,
            evenness=0.8,
            wetness=0.2,
            adhesion=0.8
        )

    def inference(self, mask):
        mask_ratio, if_valid = self.calculate_state_ratio(mask)
        if if_valid:
            self.update_mask_ratio(mask_ratio)
            self.update_terrain_state()

        return self.terrain_state.describe_road()

    def update_terrain_state(self):
        attribute_dict = dict()
        for attr, weight in self.attribute_weights.items():
            attribute_dict[attr] = np.dot(self.history_mask_ratio, weight)

        new_main_class = "unknown"

        for main_class_name, weight in self.main_class_weights.items():
            score = np.dot(self.history_mask_ratio, weight)
            if score > 0.45:
                new_main_class = main_class_name
                break
        if new_main_class == "unknown":
            return

        aux_class_list2 = []
        aux_class_list1 = []
        for aux_idx in self.aux_class_idx_list:
            if self.history_mask_ratio[aux_idx] > 0.2:
                aux_class_list2.append(self.class_names[aux_idx])
            elif self.history_mask_ratio[aux_idx] > 0.1:
                aux_class_list1.append(self.class_names[aux_idx])
        new_aux_class = ""
        if len(aux_class_list2):
            new_aux_class += f"with lots of {','.join(aux_class_list2)}"
        if len(aux_class_list1):
            if len(new_aux_class):
                new_aux_class += f" and little {','.join(aux_class_list1)}"
            else:
                new_aux_class += f"with little {','.join(aux_class_list1)}"

        self.terrain_state.update(
            new_main_class=new_main_class,
            new_aux_class=new_aux_class,
            hardness=attribute_dict["hardness"],
            evenness=attribute_dict["evenness"],
            adhesion=attribute_dict["adhesion"],
            wetness=attribute_dict["wetness"]
        )

    def calculate_distance(self, new_mask_ratio):
        distance = 1 - np.dot(self.last_mask_ratio, new_mask_ratio)
        return distance

    def calculate_state_ratio(self, mask):
        state = np.bincount(mask.astype(np.uint8).flatten(), minlength=self.class_num + 1)
        mask_ratio = state / np.sum(state)
        if mask_ratio[0] > 1 - self.mask_valid_threshold:
            return mask_ratio[1:] / np.sum(mask_ratio[1:]), False
        else:
            return mask_ratio[1:] / np.sum(mask_ratio[1:]), True

    def update_mask_ratio(self, new_mask_ratio):

        if self.last_mask_ratio is None:
            self.history_mask_ratio = new_mask_ratio
            self.last_mask_ratio = new_mask_ratio
        else:
            distance = 1 - self.calculate_distance(new_mask_ratio)
            self.last_mask_ratio = new_mask_ratio
            weight_decay = 0.04 * (1.25 - distance)
            self.history_mask_ratio = self.history_mask_ratio * (1 - weight_decay) + new_mask_ratio * weight_decay
