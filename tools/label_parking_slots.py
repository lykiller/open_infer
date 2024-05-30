import json
import os
import sys
import cv2 as cv
import numpy as np

MODE_ADD = 0
MODE_EDIT = 1
RED = (0, 0, 255)  # ADD CAR COLOR
BLUE = (255, 0, 0)  # SELECT CAR COLOR
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)  # OCCUPY CAR COLOR
GREEN = (0, 255, 0)  # NOT OCCUPY CAR COLOR

# 全局变量, 获取mouse交互状态
ctrl_state = {"pos": None, "points": [], "mode": MODE_ADD}


def reset_ctrl_state():
    global ctrl_state
    ctrl_state = {"pos": None, "points": [], "mode": MODE_ADD}


# 读取json文件
def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


# 解析json文件
def parse_json(data):
    parking_slots = data["parking_slots"]

    parking_slots_points = []
    for parking_slot in parking_slots:
        positions = parking_slot["positions"]
        parking_slots_points.append(positions)
        occupy = parking_slot["occupy"]


def select_parking_slot(data):
    if ctrl_state["mode"] != MODE_EDIT:
        return -1
    parking_slots = data["parking_slots"]
    selected_parking_slot = -1
    for (i, parking_slot) in enumerate(parking_slots):
        points = parking_slot["points"]
        if len(points) > 0:
            distance = cv.pointPolygonTest(np.array(points), ctrl_state["pos"], False)
            if distance > 0:
                selected_parking_slot = i
    return selected_parking_slot


# 在对应的图片上画出车道线点
def draw_parking_slots(img, data, select_idx=-1):
    if ctrl_state["mode"] == MODE_ADD:
        draw_color = RED
        points = ctrl_state["points"]
        for j in range(len(points)):
            cv.circle(img, (points[j][0], points[j][1]), 2, draw_color, 4)
            cv.putText(img, str(j), (points[j][0], points[j][1]), cv.FONT_HERSHEY_SIMPLEX, 1.5, draw_color, 1)
            if j == len(points) - 1:
                if len(points) == 4:
                    cv.line(img, (points[j][0], points[j][1]), (points[0][0], points[0][1]), draw_color, 2)
            else:
                cv.line(img, (points[j][0], points[j][1]), (points[j + 1][0], points[j + 1][1]), draw_color, 2)

    parking_slots = data["parking_slots"]
    for i in range(len(parking_slots)):
        parking_slot = parking_slots[i]
        points = parking_slot["points"]
        occupy = parking_slot["occupy"]

        if occupy:
            draw_color = BLACK
        else:
            draw_color = GREEN
        if ctrl_state["mode"] == MODE_EDIT and select_idx == i:
            draw_color = BLUE

        for j in range(len(points)):
            cv.circle(img, (points[j][0], points[j][1]), 2, draw_color, 4)
            cv.putText(img, str(j), (points[j][0], points[j][1]), cv.FONT_HERSHEY_SIMPLEX, 1, draw_color, 1)
            if j == len(points) - 1:
                cv.line(img, (points[j][0], points[j][1]), (points[0][0], points[0][1]), draw_color, 2)
            else:
                cv.line(img, (points[j][0], points[j][1]), (points[j + 1][0], points[j + 1][1]), draw_color, 2)


def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        ctrl_state["points"].append([x, y])
        if len(ctrl_state["points"]) > 4:
            ctrl_state["points"] = []
        ctrl_state["pos"] = None
        ctrl_state["mode"] = MODE_ADD
    elif event == cv.EVENT_RBUTTONDOWN:
        ctrl_state["points"] = []
        ctrl_state["pos"] = (x, y)
        ctrl_state["mode"] = MODE_EDIT


def get_json_files(json_path):
    json_files = [os.path.join(json_path, filename) for filename in filter(
        lambda x: x.endswith('.json'), os.listdir(json_path))]
    return json_files


def get_image_files(img_path: str):
    img_files = [os.path.join(img_path, filename) for filename in filter(
        lambda x: x.endswith('.jpg') or x.endswith('.png'), os.listdir(img_path))]
    return img_files


def help_lane_string():
    return "left click: add, right click: select, d: delete, 0: occupy, 1: unoccupy"


def label_run(json_path, img_path):
    cv.namedWindow("image")
    cv.setMouseCallback("image", mouse_callback)

    img_files = get_image_files(img_path)
    i = 0
    while i < len(img_files) and i >= 0:
        img_file = img_files[i]
        json_file = os.path.basename(img_file).split('.')[0] + '.json'
        json_file = os.path.join(json_path, json_file)
        if os.path.exists(json_file):
            print(json_file)
            data = read_json(json_file)

        else:
            data = {"parking_slots": []}

        cv.setWindowTitle("image", "{}-{}/{}".format(img_file, i + 1, len(img_files)))
        img = cv.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
        if i == 0:
            cv.putText(img, "This image is the first one", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        elif i == len(img_files) - 1:
            cv.putText(img, "This image is the last one", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv.putText(img, help_lane_string(), (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv.putText(img, "n: next picture, b: back picture, s: save json", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 0, 255), 1)
        reset_ctrl_state()
        cv.imshow("image", img)

        while True:
            img_copy = img.copy()
            select_key = -1
            select_idx = select_parking_slot(data)
            key = cv.waitKeyEx(10)
            if key == ord('n') or key == 0x270000:  # 右箭头
                i += 1
                break
            elif key == ord('b') or key == 0x250000:  # 左箭头
                i -= 1
                break
            elif key == ord('s'):
                break
            elif key == ord('q'):
                i = -1
                break
            elif key >= ord('0') and key <= ord('9'):
                select_key = key - ord('0')

            if ctrl_state["mode"] == MODE_ADD and len(ctrl_state["points"]) == 4:
                if select_key == 0:
                    data["parking_slots"].append({"points": ctrl_state["points"], "occupy": 0})
                    reset_ctrl_state()
                elif select_key == 1:
                    data["parking_slots"].append({"points": ctrl_state["points"], "occupy": 1})
                    reset_ctrl_state()
                elif key == ord('d'):
                    reset_ctrl_state()
            if ctrl_state["mode"] == MODE_EDIT and select_idx >= 0:
                if select_key == 0:
                    data["parking_slots"][select_idx]["occupy"] = 0
                    reset_ctrl_state()
                elif select_key == 1:
                    data["parking_slots"][select_idx]["occupy"] = 1
                    reset_ctrl_state()
                elif key == ord('d'):
                    data["parking_slots"].pop(select_idx)
                    reset_ctrl_state()
            draw_parking_slots(img_copy, data, select_idx)

            cv.imshow("image", img_copy)

        save_json(json_file, data)

    cv.destroyAllWindows()


def get_file_path(img_path, filename):
    if isinstance(img_path, str):
        path = os.path.join(img_path, filename)
        if os.path.isfile(path):
            return path
        else:
            return None
    for each_path in img_path:
        path = os.path.join(each_path, filename)
        if os.path.isfile(path):
            return path
    return None


def prepare_data(img_path, json_data):
    lane_points, raw_file, lane_types = parse_json(json_data)
    raw_file = get_file_path(img_path, raw_file)
    if raw_file is None:
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv.putText(img, "json error, raw_file {} not exist".format(
            raw_file), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        img = cv.imdecode(np.fromfile(raw_file, dtype=np.uint8), -1)

    if lane_types is None:
        cv.putText(img, "json error, lane_types number is not equal to lanes",
                   (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return img, lane_points, lane_types


if __name__ == '__main__':
    json_path = r'E:\fullimg\train\label'
    img_path = r'E:\fullimg\train\image'

    label_run(json_path, img_path)
