from kalman_filter import KalmanFilter


    def __init__(self, track_id, bbox, dt=1.0):

        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.missed_frames = 0

        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0


        self.kf = KalmanFilter(cx, cy, dt)

    
    def predict(self):
        
        x1, y1, x2, y2 = self.bbox

        width = x2 - x1
        height = y2 - y1

        pred_center = self.kf.predict()

        cx = pred_center[0, 0]
        cy = pred_center[1, 0]

        new_x1 = cx - width / 2.0
        new_y1 = cy - height / 2.0 
        new_x2 = cx + width / 2.0
        new_y2 = cy + height / 2.0

        self.bbox = [new_x1, new_y1, new_x2, new_y2]

        return self.bbox


    def update(self, bbox):

        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        width = x2 - x1 
        height = y2 - y1

        updated_state = self.kf.update(cx, cy)

        new_cx = updated_state[0, 0]
        new_cy = updated_state[1, 0]

        new_x1 = new_cx - width / 2.0
        new_y1 = new_cy - height / 2.0
        new_x2 = new_cx + width / 2.0
        new_y2 = new_cy + height / 2.0

        self.bbox = [new_x1, new_y1, new_x2, new_y2]

        self.missed_frames = 0

        return self.bbox

        



