import numpy as np
from tracker.match import find_matched_bbox


def bbox_to_xyah(bbox):
    x_min, y_min, x_max, y_max = bbox[:4]
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return x_center, y_center, w / h, h


def xyah_to_xyxy(xyah):
    x_center, y_center, ratio, h = xyah
    w = ratio * h
    return x_center - w / 2, y_center - h / 2, x_center + w / 2, y_center + h / 2


class TrackObject:
    def __init__(
            self,
            track_id,
            bbox,
            kalman_filter

    ):
        self.track_id = track_id
        self.bbox = bbox[:4]
        self.track_state = 0
        self.score = bbox[-1]
        xyah = bbox_to_xyah(bbox)
        mean, covariance = kalman_filter.initiate(np.array(xyah))
        self.mean = mean
        self.covariance = covariance

    def update(self, det_results, kalman_filter):

        self.mean, self.covariance = kalman_filter.predict(self.mean, self.covariance)

        match_bbox = find_matched_bbox(xyah_to_xyxy(self.mean[:4].tolist()), det_results)

        if len(match_bbox):

            self.mean, self.covariance = kalman_filter.update(self.mean, self.covariance, np.array(bbox_to_xyah(match_bbox)))
            self.bbox = xyah_to_xyxy(self.mean[:4].tolist())
            self.score += match_bbox[-1]
            self.score = min(self.score, 5)
            if self.score < 1:
                return False
        else:
            if self.score > 1:
                self.score -= 1
            else:
                return False

        return True

