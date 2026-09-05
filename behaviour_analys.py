import math

from calibration import pixel_to_world

class BehaviorAnalyzer:

    def __init__(self):

        self.prev_world_positions = {}
        self.prev_frame_times = {}

        self.speeds = {}   #km/h

        self.stopped_vehicles = set()
        self.stop_start_time = {}


    def estimate_speed(self, track_id, ground_pixel_point, frame_time):

       

        world_point = pixel_to_world(ground_pixel_point)

        if track_id not in self.prev_world_positions:

            self.prev_world_positions[track_id] = world_point
            self.prev_frame_times[track_id] = frame_time
            self.speeds[track_id] = 0
            return 0


        prev_x, prev_y = self.prev_world_positions[track_id]
        curr_x, curr_y = world_point

        distance_m = math.sqrt((curr_x-prev_x)**2 + (curr_y-prev_y)**2)

        time_diff = frame_time - self.prev_frame_times[track_id]

        if time_diff <= 0:
            speed_kmh  = self.speeds.get(track_id, 0)
        else:
            raw_speed_kmh = (distance_m / time_diff) * 3.6


            #EMA
            ALPHA = 0.20
            prev_speed = self.speeds.get(track_id, raw_speed_kmh)
            speed_kmh = (ALPHA * raw_speed_kmh) + ((1 - ALPHA) * prev_speed)



        self.prev_world_positions[track_id] = world_point                    #save
        self.prev_frame_times[track_id] = frame_time

        self.speeds[track_id] = speed_kmh

        self.detect_stopped_vehicle(track_id, speed_kmh, frame_time)


        return speed_kmh


    def detect_stopped_vehicle(self, track_id, speed_kmh, frame_time):

        current_time = frame_time

        SPEED_THRESHOLD = 10  # km/h
        STOP_TIME = 3  # seconds

        if speed_kmh < SPEED_THRESHOLD:

            if track_id not in self.stop_start_time:
                self.stop_start_time[track_id] = current_time

            elapsed = current_time - self.stop_start_time[track_id]

            if elapsed > STOP_TIME:
                self.stopped_vehicles.add(track_id)

        else:

            if track_id in self.stop_start_time:
                del self.stop_start_time[track_id]  # Reset timer

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
    
    def detect_congestion(self):

        active_vehicles = self.active_vehicle_count()
        avg_speed = self.average_speed()
    
        VEHICLE_THRESHOLD = 5
        SPEED_THRESHOLD = 10 #km/h
    
        if active_vehicles >= VEHICLE_THRESHOLD and avg_speed <= SPEED_THRESHOLD:
            return "Congested"
    
        elif active_vehicles >= VEHICLE_THRESHOLD and avg_speed <= SPEED_THRESHOLD * 2:
            return "Slow Traffic"
    
        else:
            return "Normal"