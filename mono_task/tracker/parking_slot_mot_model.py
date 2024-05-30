import numpy as np

from core.key_points import calculate_length
from scipy.optimize import linear_sum_assignment


class ParkingSlotMotModel:
    def __init__(
            self,
            dist_threshold
    ):
        self.dist_threshold = dist_threshold

    def inference(self, new_result, last_result):
        pass

    def associate(self, new_result, last_result):
        dist_matrix = self.calculate_distance(new_result, last_result)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        matched_indices = np.stack([row_ind, col_ind], axis=1)

        unmatched_det_bboxes = list()
        unmatched_tracks = list()
        matches = list()
        for d, det in enumerate(new_result):
            if d not in matched_indices[:, 0]:
                unmatched_det_bboxes.append(d)
        for t, trk in enumerate(last_result):
            if t not in matched_indices[:, 1]:
                unmatched_tracks.append(t)

        for m in matched_indices:
            if dist_matrix[m[0], m[1]] > self.dist_threshold:
                unmatched_det_bboxes.append(m[0])
                unmatched_tracks.append(m[1])
            else:
                matches.append(m.reshape[2])
        return matches, unmatched_det_bboxes, unmatched_tracks

    def calculate_distance(self, new_result, last_result):
        dist_matrix = np.zeros(len(new_result), len(last_result))
        for new_idx, new_parking_slot in enumerate(new_result):
            for last_idx, last_parking_slot in enumerate(last_result):
                dist_matrix[new_parking_slot, last_parking_slot] = calculate_length(
                    new_parking_slot.center, last_parking_slot.center)

        return dist_matrix
