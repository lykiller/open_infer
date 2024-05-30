import os

from core.utils import read_json, save_json

if __name__ == "__main__":
    old_json_dir_path = r"D:\文档\自动泊车\fullimg\fullimg\train\old_label"
    new_json_dir_path = old_json_dir_path.replace("old_label", "label")

    for label_name in os.listdir(old_json_dir_path):
        old_data = read_json(os.path.join(old_json_dir_path, label_name))
        new_data = {
            "parking_slots": []
        }
        for x_list, y_list, occupy in zip(old_data["points_x"], old_data["points_y"], old_data["isOccupied"]):
            per_parking_slot = {
                "points": [],
                "occupy": occupy
            }
            for x, y in zip(x_list, y_list):
                per_parking_slot["points"].append([x, y])
            new_data["parking_slots"].append(per_parking_slot)

        save_json(os.path.join(new_json_dir_path, label_name), new_data)
