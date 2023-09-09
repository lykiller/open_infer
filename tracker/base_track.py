from match import find_matched_bbox


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class TrackObject:
    def __init__(
            self,
            track_id,
            frame_id,
            bbox,

    ):
        self.track_id = track_id
        self.last_frame_id = frame_id
        self.start_frame_id = frame_id
        self.bbox = bbox
        self.track_state = 0
        self.score = bbox[-1]

    def update(self, det_results, frame_id):

        if self.track_state == 0:
            match_bbox = find_matched_bbox(self.bbox, det_results)
            if len(match_bbox):
                self.bbox = match_bbox
                self.last_frame_id = frame_id
            else:
                return False
        elif self.track_state == 1:
            match_bbox = find_matched_bbox(self.bbox, det_results)
        return
