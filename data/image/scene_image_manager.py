import os
import cv2


class SceneImageLoader:
    def __init__(
            self,
            sample_path,
            sweep_path,
            sensor_name_list,
            scene_name,
            max_frame=300
    ):
        self.sensor_name_list = sensor_name_list
        self.scene_name = scene_name
        self.image_path_list = []
        self.cur_frame = 0
        self.max_frame = max_frame

        for sensor_name in sensor_name_list:
            per_sample_path = os.path.join(sample_path, sensor_name)
            per_sweep_path = os.path.join(sweep_path, sensor_name)

            image_name_list = os.listdir(per_sample_path) + os.listdir(per_sweep_path)
            image_name_list = sorted(image_name_list)[:max_frame]

            image_path_list = []
            for idx, image_name in enumerate(image_name_list):
                if idx % 5 == 0:
                    image_path_list.append(os.path.join(per_sample_path, image_name))
                else:
                    image_path_list.append(os.path.join(per_sweep_path, image_name))

            self.image_path_list.append(image_path_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_frame >= self.max_frame:
            raise StopIteration

        else:
            result = [self.image_path_list[x][self.cur_frame] for x in range(len(self.sensor_name_list))]
            self.cur_frame += 1
            return result
