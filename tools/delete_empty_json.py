import os
import shutil

from core.utils import read_json
if __name__ == "__main__":
    json_path = r"E:\fullimg\train\label"
    dst_path = r"E:\fullimg\train\empty_label"
    for json_name in os.listdir(json_path):
        print(json_name)
        data = read_json(os.path.join(json_path, json_name))["parking_slots"]
        if len(data) == 0 and "avm__00" in json_name:
            shutil.move(os.path.join(json_path, json_name), os.path.join(dst_path, json_name))
