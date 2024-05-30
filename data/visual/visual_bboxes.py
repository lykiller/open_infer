import cv2
import numpy as np

from core.utils import _COLORS


def det_bboxes_to_visual_bboxes(det_bboxes, score_thresh=0.3):
    visual_bboxes = []
    for label_idx, label_name in enumerate(det_bboxes.keys()):
        if "like" not in label_name:
            for bbox in det_bboxes[label_name]:
                score = bbox[4]
                if score > score_thresh:
                    x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                    if len(bbox) > 5:
                        visual_bboxes.append([bbox[-1], label_idx, x0, y0, x1, y1, score])
                    else:
                        visual_bboxes.append([label_name, label_idx, x0, y0, x1, y1, score])
    visual_bboxes.sort(key=lambda v: v[-1])
    return visual_bboxes


def overlay_bbox_cv(img, dets):
    visual_bboxes = det_bboxes_to_visual_bboxes(dets)
    for box in visual_bboxes:
        label, label_idx, x0, y0, x1, y1, score = box
        # color = self.cmap(i)[:3]
        color = (_COLORS[label_idx] * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(label, score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[label_idx]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        cv2.rectangle(
            img,
            (x0, y0 - txt_size[1] - 1),
            (x0 + txt_size[0] + txt_size[1], y0 - 1),
            color,
            -1,
        )
        cv2.putText(img, text, (x0, y0 - 1), font, 0.5, txt_color, thickness=1)
    return img
