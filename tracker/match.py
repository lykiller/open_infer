from detectron2d.bbox import calculate_bbox_iou


def find_matched_bbox(bbox, new_bbox_list):
    best_iou = 0.1
    best_match = []
    best_idx = -1
    for idx, new_bbox in enumerate(new_bbox_list):
        iou = calculate_bbox_iou(bbox, new_bbox)
        if iou > best_iou:
            best_iou = iou
            best_match = new_bbox
            best_idx = idx
    if best_idx > -1:
        new_bbox_list.pop(best_idx)

    return best_match


