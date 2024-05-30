import math
import numpy as np


def get_center_point(start_points, end_points):
    return (start_points[0] + end_points[0]) / 2, (start_points[1] + end_points[1]) / 2


def calculate_length_and_direction(start_points, end_points):
    dis_x = end_points[0] - start_points[0]
    dis_y = end_points[1] - start_points[1]
    length = math.sqrt(dis_x * dis_x + dis_y * dis_y)
    if dis_x >= 0:
        return length, (math.asin(dis_y / length), 1)
    else:
        return length, (math.asin(dis_y / length), -1)


def calculate_length(start_points, end_points):
    dis_x = start_points[0] - end_points[0]
    dis_y = start_points[1] - end_points[1]

    return math.sqrt(dis_x * dis_x + dis_y * dis_y)


def warp_key_points(key_points, M, width, height):
    n = len(key_points)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = key_points.reshape(n * 4, 2)
        xy = xy @ M.T
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 4, 2)

        xy[:, :, 0] = xy[:, :, 0].clip(0, width)
        xy[:, :, 1] = xy[:, :, 1].clip(0, height)
        return xy
    else:
        return key_points
