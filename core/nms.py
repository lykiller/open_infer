import numpy as np


def nms(labels, scores, bboxes, points_offsets=None, iou_thresh=0.2, score_thresh=0.2):
    score_keep = np.where(scores > score_thresh)
    bboxes = bboxes[score_keep]
    scores = scores[score_keep]
    labels = labels[score_keep]
    if points_offsets is not None:
        points_offsets = points_offsets[score_keep]
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    res = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]
        res.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        iou = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(iou <= iou_thresh)[0]
        index = index[idx+1]
    if points_offsets is not None:
        return labels[res], scores[res], bboxes[res], points_offsets[res]
    else:
        return labels[res], scores[res], bboxes[res]


def parking_slots_nms(labels, scores, bboxes, points_offsets, dis_thresh=35, score_thresh=0.7):
    score_keep = np.where(scores > score_thresh)
    bboxes = bboxes[score_keep]
    scores = scores[score_keep]
    labels = labels[score_keep]
    points_offsets = points_offsets[score_keep]
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    res = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]
        res.append(i)
        x_dis = center_x[i] - center_x[index[1:]]
        y_dis = center_y[i] - center_y[index[1:]]

        dis = abs(x_dis) + abs(y_dis)

        idx = np.where(dis >= dis_thresh)[0]
        index = index[idx+1]

    return labels[res], scores[res], bboxes[res], points_offsets[res]
