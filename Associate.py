import numpy as np
from scipy.optimize import linear_sum_assignment

from IoU import CalcIou


def associate(predicted_boxes, detections, iou_threshold = 0.3):

    if len(predicted_boxes) == 0 or len(detections) == 0:
        return [], list(range(len(predicted_boxes))), list(range(len(detections))), np.zeros((len(predicted_boxes), len(detections)), dtype=np.float32)
    

    cost_matrix = np.zeros((len(predicted_boxes), len(detections)), dtype=np.float32)
    for t, pred_box in enumerate(predicted_boxes):
        for d, det_box in enumerate(detections):
            iou = CalcIou(pred_box, det_box)
            cost_matrix[t, d] = 1 - iou

    track_indices, detection_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = list(range(len(predicted_boxes)))
    unmatched_detections = list(range(len(detections)))

    for t, d in zip(track_indices, detection_indices):

        t, d = int(t), int(d)

        if cost_matrix[t, d] < (1 - iou_threshold):
            matches.append((t, d))
            unmatched_tracks.remove(t)
            unmatched_detections.remove(d)

    return matches, unmatched_tracks, unmatched_detections, cost_matrix