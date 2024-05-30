from core.key_points import calculate_length, calculate_length_and_direction, get_center_point

STANDARD_ENTRY_LINE_LENGTH = 230
EGO_POINT = (512, 512)


class ParkingSlot:
    def __init__(
            self,
            occupy,
            score,
            bbox,
            key_point,

    ):
        self.occupy = occupy
        self.score = score
        x_min, y_min, x_max, y_max = bbox
        pa, pb, pc, pd = key_point
        self.sepa_line_length, self.sepa_direction = calculate_length_and_direction(pa, pd)
        self.sepb_line_length, self.sepb_direction = calculate_length_and_direction(pb, pc)

        self.entry_line_length = calculate_length(pa, pb)
        self.end_line_length = calculate_length(pc, pd)
        self.entry_center = get_center_point(pa, pb)
        self.end_center = get_center_point(pc, pd)
        self.center = ((x_min + x_max) / 2, (y_min + y_max) / 2)

        self.entry_center = (int(self.entry_center[0]), int(self.entry_center[1]))
        self.end_center = (int(self.end_center[0]), int(self.end_center[1]))

        self.pa = (int(pa[0]), int(pa[1]))
        self.pb = (int(pb[0]), int(pb[1]))
        self.pc = (int(pc[0]), int(pc[1]))
        self.pd = (int(pd[0]), int(pd[1]))

        self.valid = self.update_valid()
        self.pmin = (int(x_min), int(y_min))
        self.pmax = (int(x_max), int(y_max))

    def update_valid(self):

        if self.entry_line_length < STANDARD_ENTRY_LINE_LENGTH - 30:
            return False

        if self.sepa_direction[0] - self.sepb_direction[0] > 0.2:
            return False
        return True

    def calculate_cost(self, prediction):
        pass

    def birth(self, track_id):
        pass
