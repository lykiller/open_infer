import os
import cv2
import numpy as np

from kalman_filter import KalmanFilter
from scipy.optimize import linear_sum_assignment


class General2DMotModel:
    def __init__(
            self,
            dist_threshold=0.5,
    ):
        self.dist_threshold = 0.5

    def inference(self, det_bboxes, last_result):
        result = dict()
        for key in det_bboxes.keys():
            result[key] = list()
        if last_result.__contains__("mot_bboxes"):
            last_mot_bboxes = last_result["mot_bboxes"]
            for key, value in det_bboxes.items():
                matches, unmatched_det_bboxes, unmatched_tracks = self.associate_bboxes_to_tracks(
                    det_bboxes=det_bboxes[key],
                    last_mot_bboxes=last_mot_bboxes[key]
                )

    def associate_bboxes_to_tracks(self, det_bboxes, last_mot_bboxes):
        dist_matrix = self.calculate_distance(det_bboxes, last_mot_bboxes)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        matched_indices = np.stack([row_ind, col_ind], axis=1)

        unmatched_det_bboxes = list()
        unmatched_tracks = list()
        matches = list()
        for d, det in enumerate(det_bboxes):
            if d not in matched_indices[:, 0]:
                unmatched_det_bboxes.append(d)
        for t, trk in enumerate(last_mot_bboxes):
            if t not in matched_indices[:, 1]:
                unmatched_tracks.append(t)

        for m in matched_indices:
            if dist_matrix[m[0], m[1]] > self.dist_threshold:
                unmatched_det_bboxes.append(m[0])
                unmatched_tracks.append(m[1])
            else:
                matches.append(m.reshape[2])
        return matches, unmatched_det_bboxes, unmatched_tracks

    def calculate_distance(self, det_bboxes, last_mot_bboxes):
        dist_matrix = np.zeros(len(det_bboxes), len(last_mot_bboxes))
        for bbox_idx, bbox in enumerate(det_bboxes):
            for mot_idx, mot_bbox in enumerate(last_mot_bboxes):
                dist_matrix[bbox_idx, mot_idx] = 1 - mot_bbox.cal_iou_with_det_bbox(bbox)

        return dist_matrix
