from core.bbox import calculate_bbox_iou


class BBox2DMot:
    def __init__(
            self,
            x_min,
            x_max,
            y_min,
            y_max,
            ratio,
            score,
            vx=0,
            vy=0,
            vw=0,
            vh=0,
            vr=0
    ):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        self.x_center = (x_min + x_max) / 2
        self.y_center = (y_min + y_max) / 2
        self.width = x_max - x_min
        self.height = y_max - y_min
        self.ratio = ratio
        self.score = score
        self.vx = vx
        self.vy = vy
        self.vw = vw
        self.vh = vh
        self.vr = vr

    def cal_iou_with_det_bbox(self, det_bbox):
        return calculate_bbox_iou([self.x_min, self.y_min, self.x_max, self.y_max], det_bbox[:4])

    def cal_distance_with_det_bbox(self, det_bbox):
        det_x_center = (det_bbox[0] + det_bbox[2]) / 2
        det_y_center = (det_bbox[1] + det_bbox[3]) / 2
        return abs(self.x_center - det_x_center) + abs(self.y_center - det_y_center)

