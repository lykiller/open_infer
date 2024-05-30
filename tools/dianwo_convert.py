import cv2
import json
import numpy as np

mask_color = [
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
class_names = [
    "background",
    "asphalt",
    "gravel",
    "earth",
    "grass",
    "mud",
    "water",
    "rock",
    "sand",
    "ice",
    "snow",
    "others"
]


def is_polygon_inside(polygon1, polygon2):
    # 创建一个包围polygon2的矩形边界框
    rect = cv2.boundingRect(polygon2)

    for point in polygon1:
        # 判断多边形1中的点是否在多边形2内部
        inside = cv2.pointPolygonTest(polygon2, tuple(point), measureDist=False)

        # 如果有任意一个点在多边形2外部，则多边形1不完全位于多边形2内部
        if inside < 0:
            return False

    return True


def labelme_to_hw3_mask(labelme_json_path):
    with open(labelme_json_path, 'r') as f:
        labelme_data = json.load(f)

    # 读取图像大小信息
    image_info = labelme_data['imageHeight'], labelme_data['imageWidth']

    # 创建空白掩膜图像
    mask = np.zeros((image_info[0], image_info[1], 3), dtype=np.uint8)

    # 遍历每个标注对象
    for shape in labelme_data['shapes']:
        label = shape['label']

        # 查询除 label 在 class_names 中的索引
        labelindex = class_names.index(label)

        polygon = shape['points']

        # 判断polygon是否为 n x 2 的二维数组

        if not isinstance(polygon[0], list):
            polygon = [polygon]

            # 判断该points 之后所有的多边形是否在 ploygon[0] 里面

            for i in polygon:
                if i == 0: continue

                if is_polygon_inside(polygon[0], polygon[i]):
                    cv2.fillPoly(mask, [polygon], color=[0, 0, 0])

        else:

            polygon = np.array(polygon, dtype=np.int32)

            # 绘制多边形区域
            cv2.fillPoly(mask, [polygon], color=mask_color[labelindex])

    return mask


# 示例使用
labelme_json_path = r'D:\dataset\terrain_data\need_label_dianwo_0718_and_0721\jsons\061_2M_20230307_145924_ningxia_wuzhong_0H_0_5_1183.json'

mask = labelme_to_hw3_mask(labelme_json_path)

# 保存转换后的彩色掩膜
cv2.imwrite(r'D:\dataset\terrain_data\need_label_dianwo_0718_and_0721\labeled_visual_masks\061_2M_20230307_145924_ningxia_wuzhong_0H_0_5_1183.png', mask)
