from scipy.optimize import linear_sum_assignment
from IoU import CalcIou
import numpy as np
from tracker import Tracker

class MultiObjectTracker:
    def __init__(self):

        self.tracks = {}
        self.next_id = 0

    def predict(self):

        predictions = {}
        for track_id, track in self.tracks.items():
            predicted_bbox = track.predict()
            predictions[track_id] = predicted_bbox

        return predictions
    
    def calculate_iou_matrix(self, predictions, detections):
        iou_matrix = []
        for track_id, predicted_bbox in predictions.items():
            iou_row = []
            for detection_bbox in detections:
                iou = CalcIou(predicted_bbox, detection_bbox)
                iou_row.append(iou)
            iou_matrix.append(iou_row)
        return iou_matrix
    

    def assign_detections_to_tracks(self, iou_matrix, iou_threshold=0.3):
        if len(iou_matrix) == 0 or len(iou_matrix[0]) == 0:
            return []
    
        iou_matrix = np.array(iou_matrix)
        cost_matrix = 1 - iou_matrix                                     
        row_indices, col_indices = linear_sum_assignment(cost_matrix)



        matches = []
        unmatched_tracks = list(range(len(iou_matrix)))
        unmatched_detections = list(range(len(iou_matrix[0])))

        for row, col in zip(row_indices, col_indices):

            row, col = int(row), int(col)

            if iou_matrix[row, col] >= iou_threshold:
                matches.append((row, col))
                unmatched_tracks.remove(row)
                unmatched_detections.remove(col)

        return matches, unmatched_tracks, unmatched_detections
    
    def create_new_track(self, unmatched_detections, detections, dt = 1.0):

        for detection_idx in unmatched_detections:

            bbox = detections[detection_idx]
            new_track = Tracker(
                self.next_id, 
                bbox, dt)
            
            self.tracks[self.next_id] = new_track
            self.next_id += 1

    def missed_tracks(self, unmatched_tracks, max_missed = 5):

        for unmatched_id in unmatched_tracks:
            track_object = self.tracks[unmatched_id]
            track_object.missed_frames += 1

            if track_object.missed_frames >= max_missed:
                self.tracks.pop(unmatched_id, None)

























