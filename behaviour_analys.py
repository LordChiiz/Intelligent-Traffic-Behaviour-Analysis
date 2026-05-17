import time
import math


class BehaviorAnalyzer:

    def __init__(self):

        self.prev_positions = {}
        self.prev_times = {}

        self.speeds = {}

        self.stopped_vehicles = set()
        self.stop_start_time = {}


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

        self.detect_stopped_vehicle(track_id, speed)

        return speed


    def detect_stopped_vehicle(self, track_id, speed):

        current_time = time.time()

        SPEED_THRESHOLD = 2  # pixels per second
        STOP_TIME = 3  # seconds

        if speed < SPEED_THRESHOLD:

            if track_id not in self.stop_start_time:
                self.stop_start_time[track_id] = current_time

            elapsed = current_time - self.stop_start_time[track_id]

            if elapsed > STOP_TIME:
                self.stopped_vehicles.add(track_id)

        else:

            if track_id in self.stop_start_time:
                del self.stop_start_time[track_id]  # Reset stop timer

            if track_id in self.stopped_vehicles:
                self.stopped_vehicles.remove(track_id)


    def average_speed(self):

        moving_speeds = [
            speed for track_id, speed in self.speeds.items()
            if track_id not in self.stopped_vehicles
        ]

        if len(moving_speeds) == 0:
            return 0

        return sum(moving_speeds) / len(moving_speeds)


    def active_vehicle_count(self):

        moving = [
            track_id for track_id in self.speeds
            if track_id not in self.stopped_vehicles
        ]

        return len(moving)


    def stopped_vehicle_count(self):

        return len(self.stopped_vehicles)