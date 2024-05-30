import os

import json
import numpy as np


class Camera:
    def __init__(
            self,
            intrinsic,
            rotation,
            translation,
            sensor_name,
            image_size
    ):
        self.intrinsic = intrinsic
        self.rotation = rotation
        self.translation = translation
        self.sensor_name = sensor_name
        self.image_size = image_size

    def uv_to_camera_xyz(self, u_np, v_np, depth_np):
        camera_xyz = np.matmul(
            np.linalg.inv(self.intrinsic), np.stack([u_np * depth_np, v_np * depth_np, depth_np]))
        return camera_xyz

    def camera_xyz_to_ego_xyz(self, camera_xyz):
        return self.rotation @ camera_xyz + self.translation.repeat(camera_xyz.shape[1]).reshape(3, -1)

    def uv_to_ego_xyz(self, u_np, v_np, depth_np):
        return self.camera_xyz_to_ego_xyz(self.uv_to_camera_xyz(u_np, v_np, depth_np))


class MultiCamera:
    def __init__(self, root_path):
        if os.path.exists(root_path):
            json_list = os.listdir(root_path)
            for json_name in json_list:
                if ".json" in json_name:
                    self.set_camera_from_json(os.path.join(root_path, json_name))

        else:
            print(f"no such path:{root_path}")

    def set_camera_from_json(self, json_path):
        data = json.load(open(json_path))

        camera = Camera(
            intrinsic=np.array(data["undistort_intrinsic"]).reshape(3, 3),
            rotation=np.array(data["rotation"]).reshape(3, 3),
            translation=np.array(data["translation"]),
            image_size=data["image_size"],
            sensor_name=data["channel"]
        )

        setattr(self, data["channel"], camera)
