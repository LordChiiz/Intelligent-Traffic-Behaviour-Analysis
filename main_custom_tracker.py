from behaviour_analys import BehaviorAnalyzer
from multi_tracker import MultiObjectTracker

import cv2
from ultralytics import YOLO
from calibration import ground_point

model = YOLO("yolov8n.pt")


mot = MultiObjectTracker()

video_path = "traffic.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30 # if it cant get it assume 30 fps

analyzer = BehaviorAnalyzer(fps=fps)

frame_index = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_time = frame_index / fps
    frame_index += 1

    results = model(frame)

    boxes = results[0].boxes

    visible_ids = set()
    annotated_frame = frame.copy()




    detections = []

    if boxes is not None:

        for box in boxes:

            cls = int(box.cls.item())

            if cls not in [2, 5, 7]:
                continue

            x1, y1, x2, y2 = box.xyxy[0]

            detections.append([float(x1), float(y1), float(x2), float(y2)])


        tracks = mot.update(detections)



        for track_id, bbox in tracks.items():

            x1, y1, x2, y2 = bbox
            visible_ids.add(track_id)

            tid_label = f"ID: {track_id}"
            (text_w, text_h), baseline = cv2.getTextSize(tid_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            cv2.rectangle(annotated_frame, (int(x1), int(y1) - text_h - baseline), (int(x1) + text_w, int(y1)), (233, 255, 233), -1)
            cv2.putText(annotated_frame, tid_label, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 2)

    

            gp = ground_point(float(x1), float(y1), float(x2), float(y2))

            # speed = analyzer.estimate_speed(track_id, gp, frame_time)
            kalman_speed = analyzer.kalman_filter_speed_estimation(track_id, gp, frame_time)

            # speed_label = f"{speed:.1f} km/h"
            kalman_speed_label = f"{kalman_speed:.1f} km/h"

            (text_w, text_h), baseline = cv2.getTextSize(kalman_speed_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (int(x1), int(y2)), (int(x1) + text_w, int(y2) + text_h + baseline), (233, 255, 233), -1)
            cv2.putText(annotated_frame, kalman_speed_label, (int(x1), int(y2) + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 2)

            # (text_w, text_h), baseline = cv2.getTextSize(kalman_speed_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            # cv2.rectangle(annotated_frame, (int(x1), int(y2) + text_h + baseline + baseline), (int(x1) + text_w, int(y2) + text_h + baseline + text_h + baseline), (233, 255, 233), -1)
            # cv2.putText(annotated_frame, speed_label, (int(x1), int(y2) + text_h + baseline + text_h + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 2)

            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)

    analyzer.cleanup_inactive_ids(visible_ids)

    stopped = analyzer.stopped_vehicle_count()

    traffic_stats = analyzer.detect_congestion()

    avg_speed = analyzer.average_speed()

    cv2.rectangle(annotated_frame,(20,20),(450,170),(0,0,0),-1)

    cv2.putText(annotated_frame,f"Active Vehicles: {analyzer.active_vehicle_count()}",
                (30,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.putText(annotated_frame,f"Average Speed: {avg_speed:.2f}",
                (30,130),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    
    cv2.putText(annotated_frame,f"Stopped Vehicles: {stopped}",
                (30,85),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    
    cv2.putText(annotated_frame,f"Traffic Status: {traffic_stats}",
            (30,160),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

    cv2.putText(annotated_frame,f"Press 'q' to Quit",
                (1400,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)



    cv2.imshow("Traffic Behavior Analysis",annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):
        break


cap.release()
cv2.destroyAllWindows()
