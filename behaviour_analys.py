import time
import math


class BehaviorAnalyzer:

    def __init__(self):

        self.prev_positions = {}
        self.prev_times = {}
        self.speeds = {}

    def estimate_speed(self, track_id, center):

        current_time = time.time()

        if track_id not in self.prev_positions:

            self.prev_positions[track_id] = center
            self.prev_times[track_id] = current_time
            self.speeds[track_id] = 0
            return 0

        prev_x, prev_y = self.prev_positions[track_id]
        curr_x, curr_y = center

        distance = math.sqrt((curr_x-prev_x)**2 + (curr_y-prev_y)**2)

        time_diff = current_time - self.prev_times[track_id]

        if time_diff == 0:
            speed = 0
        else:
            speed = distance / time_diff

        self.prev_positions[track_id] = center
        self.prev_times[track_id] = current_time

        self.speeds[track_id] = speed

        return speed


    def average_speed(self):

        if len(self.speeds) == 0:
            return 0

        return sum(self.speeds.values()) / len(self.speeds)