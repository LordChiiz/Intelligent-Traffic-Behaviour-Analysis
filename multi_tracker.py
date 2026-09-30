from tracker import Tracker
from Associate import associate

class MultiObjectTracker:

    def __init__(self, iou_threshold = 0.3, max_missed = 5, dt = 1.0):

        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.dt = dt    

    def update(self, detections):

        track_ids = list(self.tracks.keys())
        predicted_boxes = [self.tracks[track_id].predict() for track_id in track_ids]

        matches, unmatched_tracks, unmatched_detections = associate(predicted_boxes, detections, self.iou_threshold)

        for track_pos, detect_idx in matches:
            
            track_id = track_ids[track_pos]
            self.tracks[track_id].update(detections[detect_idx])

        for u_track_pos in unmatched_tracks:
            unmatched_id = track_ids[u_track_pos]
            self._age_and_remove([unmatched_id])

        new_box = [detections[i] for i in unmatched_detections]

        self._create_new_tracks(new_box)

        return {tid: track.bbox for tid, track in self.tracks.items()}






    def _age_and_remove(self, unmatched_track_ids):
        for track_id in unmatched_track_ids:
            track = self.tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.max_missed:
                del self.tracks[track_id]

    
    def _create_new_tracks(self, new_boxes):
        for bbox in new_boxes:
            self.tracks[self.next_id] = Tracker(self.next_id, bbox, dt=self.dt)
            self.next_id += 1

