import os


class ModelManager:
    def __init__(
            self,
            det_model=None,
            tsr_model=None,
            tlr_model=None,
            mot_model=None,
            terrain_seg_model=None,
            terrain_state_machine=None,
            apa_model=None,
            elevation_model=None,

    ):
        self.det_model = det_model
        self.tsr_model = tsr_model
        self.tlr_model = tlr_model
        self.mot_model = mot_model

        self.terrain_seg_model = terrain_seg_model
        self.terrain_state_machine = terrain_state_machine
        self.apa_model = apa_model
        self.elevation_model = elevation_model

    def inference(self, image, last_result):
        result = dict()
        if self.det_model:
            det_bboxes = self.det_model.inference(image)
            result["det_bboxes"] = det_bboxes
            if self.tsr_model:
                traffic_sign = result["det_bboxes"]["traffic_sign"]
                tsr_result = self.tsr_model.inference(image, traffic_sign)
                result["det_bboxes"]["traffic_sign"] = tsr_result

            if self.mot_model:
                result["mot_bboxes"] = self.mot_model.inference(det_bboxes, last_result)

        if self.terrain_seg_model:
            terrain_seg = self.terrain_seg_model.inference(image)
            result["terrain_seg"] = dict()
            result["terrain_seg"]["mask"] = terrain_seg
            result["terrain_seg"]["crop"] = self.terrain_seg_model.crop
            if self.terrain_state_machine:
                if len(last_result["terrain_seg"]["state"]) == 0:
                    self.terrain_state_machine._init_state()
                result["terrain_seg"]["state"] = self.terrain_state_machine.inference(terrain_seg)
            else:
                result["terrain_seg"]["state"] = dict()

        if self.apa_model:
            result["parking_slots"] = self.apa_model.inference(image)

        return result
