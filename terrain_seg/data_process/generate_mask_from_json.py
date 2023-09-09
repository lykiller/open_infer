import json
import os
import numpy as np
import cv2
import chardet
from utils import imwrite, imread

class_name_to_id_map = {
    "Background": 0,
    "background": 0,
    "背景": 0,

    "asphalt": 1,
    "as": 1,
    "人工材料": 1,

    "gravel": 2,
    "g": 2,
    "grasvel": 2,
    "砾石": 2,

    "earth": 3,
    "土": 3,

    "grass": 4,
    "草": 4,

    "mud": 5,
    "泥": 5,

    "water": 6,
    "wea": 6,
    "水": 6,

    "rock": 7,
    "岩石": 7,

    "sand": 8,
    "sad": 8,
    "沙": 8,

    "ice": 9,
    "sice": 9,
    "冰": 9,

    "snow": 10,
    "雪": 10,

    "others": 11,
    "other": 11,
    "其他元素": 11
}
# BGR顺序
mask_color = [
    [0, 0, 0],  # background 背景 0
    [255, 255, 0],  # asphalt 人工材料 1
    [255, 0, 0],  # gravel 沥青 2
    [192, 192, 192],  # earth 土 3
    [0, 255, 0],  # grassland 草 4
    [42, 42, 128],  # mud 泥 5
    [0, 91, 255],  # water 水 6
    [34, 139, 34],  # rock 岩石 7
    [33, 145, 237],  # sand 沙 8
    [205, 235, 255],  # light_snow 冰 9
    [240, 32, 160],  # deep_snow 雪 10
    [0, 255, 255],  # others 其他 11
]

color_to_idx = {}
for idx, color in enumerate(mask_color):
    color_to_idx[tuple(color)] = idx

class_names = ["background",
               "asphalt", "gravel", "earth", "grassland", "mud", "wading", "rock", "sand",
               "light_snow", "deep_snow", "others"]

mask_color_path = r"../doc/color_mask.png"

color_mask_map = cv2.imdecode(np.fromfile(mask_color_path, dtype=np.uint8), 1)


def get_image_with_mask(image, mask, color_list):
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3))
    for idx, color in enumerate(color_list):
        color_mask[mask == idx] = tuple(color)

    return cv2.addWeighted(image, 0.7, color_mask.astype(np.uint8), 0.3, 0)


def check_charset(file_path):
    with open(file_path, "rb") as f:
        data = f.read(4)
        charset = chardet.detect(data)['encoding']
    return charset


def is_in_poly(p, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    px, py = p
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


def is_polygon_inside(polygon1, polygon2):
    # 创建一个包围polygon2的矩形边界框
    # rect = cv2.boundingRect(polygon2)

    for point in polygon1:
        # 判断多边形1中的点是否在多边形2内部
        inside = is_in_poly(point, polygon2)

        # 如果有任意一个点在多边形2外部，则多边形1不完全位于多边形2内部
        if inside:
            return False

    return True


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def color_mask_to_mask():
    src_dir = r"D:\dataset\terrain_data\need_label_dianwo_0718_and_0721"
    color_mask_dir = os.path.join(src_dir, "color_masks")
    mask_dir = os.path.join(src_dir, "masks")
    visual_mask_dir = os.path.join(src_dir, "labeled_visual_masks")
    image_dir = os.path.join(src_dir, "images")
    # image_dir = r"D:\dataset\terrain_data\images_in_server\images"
    mkdir(mask_dir)
    mkdir(visual_mask_dir)

    if_show = False
    for idx, color_mask_name in enumerate(os.listdir(color_mask_dir)):
        # color_mask_name = os.path.join(color_mask_dir, color_mask_name)
        mask_save_name = color_mask_name
        src_image_name = color_mask_name.replace(".png", ".jpg")
        print(idx + 1, color_mask_name)
        if not os.path.exists(os.path.join(image_dir, src_image_name)):
            continue
        # if os.path.exists(os.path.join(mask_dir, mask_save_name)):
        #     continue
        src_image = imread(os.path.join(image_dir, src_image_name))
        color_mask = imread(os.path.join(color_mask_dir, color_mask_name))
        height, width, _ = src_image.shape
        if color_mask.shape != src_image.shape:
            color_mask = cv2.resize(color_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        mask = np.zeros((height, width)).astype(np.uint8)
        for k, v in color_to_idx.items():
            flag = (color_mask == k)
            mask[flag[:, :, 0] * flag[:, :, 1] * flag[:, :, 2]] = v
        # for h_idx in range(height):
        #     for w_idx in range(width):
        #         # print(h_idx, w_idx)
        #         mask[h_idx, w_idx] = color_to_idx[tuple(color_mask[h_idx, w_idx, :].tolist())]
        # image_with_mask = get_image_with_mask(src_image, mask, mask_color)
        image_with_mask = cv2.addWeighted(src_image, 0.7, color_mask.astype(np.uint8), 0.3, 0)
        h, w, _ = color_mask_map.shape
        image_with_mask[:h, :w, ] = color_mask_map
        percent = np.bincount(mask.flatten(), minlength=len(class_names)) / (height * width)
        for index, per in enumerate(percent):
            cv2.putText(
                image_with_mask,
                "%.1f" % (per * 100),
                (w + 10, h * (index + 1) // len(class_names)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255)
            )
        if if_show:
            cv2.imshow("image", image_with_mask)
            cv2.waitKey(0)
        else:
            imwrite(os.path.join(mask_dir, mask_save_name), mask)
            imwrite(os.path.join(visual_mask_dir, src_image_name), image_with_mask)


def label_me_json():
    # json_dir = r"D:\dataset\terrain_data\images_no_label\pt\20230629093537195297590291"
    # image_dir = r"D:\dataset\terrain_data\images_no_label\need_label_images_dianwo_0626"
    # mask_dir = r"D:\dataset\terrain_data\images_no_label\masks12_dianwo_0626"
    # visual_mask_dir = r"D:\dataset\terrain_data\images_no_label\visual_mask_dianwo_0626"

    src_dir = r"D:\dataset\terrain_data\need_label_dianwo_0718_and_0721"
    json_dir = os.path.join(src_dir, "jsons")
    mask_dir = os.path.join(src_dir, "masks")
    visual_mask_dir = os.path.join(src_dir, "labeled_visual_masks")
    image_dir = os.path.join(src_dir, "images")
    # image_dir = r"D:\dataset\terrain_data\images_in_server\images"
    mkdir(mask_dir)
    mkdir(visual_mask_dir)

    if_show = False
    for json_name in os.listdir(json_dir):

        json_path = os.path.join(json_dir, json_name)
        mask_save_name = json_name.replace(".json", ".png")
        src_image_name = json_name.replace(".json", ".jpg")

        if not os.path.exists(os.path.join(image_dir, src_image_name)):
            continue

        # if os.path.exists(os.path.join(mask_dir, mask_save_name)):
        #     continue

        print(src_image_name)
        src_image = cv2.imdecode(np.fromfile(os.path.join(image_dir, src_image_name), dtype=np.uint8), 1)
        height, width, _ = src_image.shape
        with open(json_path, "rb") as f:
            data = json.load(f)
            points_list = data["shapes"]
            mask = np.zeros((height, width)).astype(np.uint8)
            for points in points_list:
                label_name = points["label"]
                points_xy = points["points"]
                labelindex = class_name_to_id_map[label_name]  # class_names.index(label)

                if not isinstance(points_xy[0][0], list):

                    cv2.fillPoly(mask, [np.array(points_xy).astype(int)], labelindex)

                else:

                    for i, p in enumerate(points_xy):
                        if p == 0: continue

                        if not isinstance(p, list): continue

                        if p[0] and isinstance(p[0][0], list):

                            cv2.fillPoly(mask, [np.array(p[0]).astype(int)], labelindex)
                            for j, pp in enumerate(p):
                                if j == 0: continue
                                if pp == 0:
                                    continue

                                if not isinstance(pp[0], list):
                                    continue

                                if isinstance(pp[0][0], list):

                                    if not isinstance(pp[0][0][0], list): continue


                                    for jj, ppp in enumerate(pp):
                                        if ppp == 0:
                                            continue
                                        cv2.fillPoly(mask, [np.array(ppp).astype(int)], labelindex)

                                else:
                                    if is_polygon_inside(pp, np.array(p[0], dtype=np.int32)):
                                        print(pp, label_name)
                                        cv2.fillPoly(mask, [np.array(pp).astype(int)], color=0)
                                    else:
                                        cv2.fillPoly(mask, [np.array(pp).astype(int)], labelindex)
                        else:
                            cv2.fillPoly(mask, [np.array(p).astype(int)], labelindex)

            imwrite(os.path.join(mask_dir, mask_save_name), mask)

            image_with_mask = get_image_with_mask(src_image, mask, mask_color)
            h, w, _ = color_mask_map.shape
            image_with_mask[:h, :w, ] = color_mask_map
            percent = np.bincount(mask.flatten(), minlength=len(class_names)) / (height * width)
            for index, per in enumerate(percent):
                cv2.putText(
                    image_with_mask,
                    "%.1f" % (per * 100),
                    (w + 10, h * (index + 1) // len(class_names)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255)
                )
            if if_show:
                cv2.imshow("image", image_with_mask)
                cv2.waitKey(0)

            imwrite(os.path.join(visual_mask_dir, src_image_name), image_with_mask)


if __name__ == "__main__":
    # label_me_json()
    color_mask_to_mask()
