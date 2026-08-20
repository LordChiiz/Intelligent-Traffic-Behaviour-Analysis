from ultralytics import YOLO
import cv2
from behaviour_analys import BehaviorAnalyzer
from calibration import ground_point


model = YOLO("yolov8n.pt")

analyzer = BehaviorAnalyzer()

video_path = "traffic.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30 # if it cant get it assume 30 fps



frame_index = 0


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_time = frame_index / fps
    frame_index += 1 

    results = model.track(frame, persist=True)

    boxes = results[0].boxes
    annotated_frame = results[0].plot() #so we can add the speed to label

    if boxes is not None:

        for box in boxes:

            if box.id is None:
                continue

            track_id = int(box.id.item())
            cls = int(box.cls.item())

            if cls not in [2,5,7]:
                continue

            x1, y1, x2, y2 = box.xyxy[0]

            gp = ground_point(float(x1), float(y1), float(x2), float(y2))


            speed = analyzer.estimate_speed(track_id, gp, frame_time)

            label = f"{speed:.1f} km/h"
            cv2.putText(annotated_frame, label, (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)



    # annotated_frame = results[0].plot()

    # stopped = analyzer.stopped_vehicle_count()

    # traffic_stats = analyzer.detect_congestion()

    # avg_speed = analyzer.average_speed()

    # cv2.rectangle(annotated_frame,(20,20),(450,170),(0,0,0),-1)

    # cv2.putText(annotated_frame,f"Active Vehicles: {analyzer.active_vehicle_count()}",
    #             (30,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    # cv2.putText(annotated_frame,f"Average Speed: {avg_speed:.2f}",
    #             (30,130),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    
    # cv2.putText(annotated_frame,f"Stopped Vehicles: {stopped}",
    #             (30,85),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    
    # cv2.putText(annotated_frame,f"Traffic Status: {traffic_stats}",
    #         (30,160),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

    cv2.putText(annotated_frame,f"Press 'q' to Quit",
                (1400,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)



    cv2.imshow("Traffic Behavior Analysis",annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):
        break


cap.release()
cv2.destroyAllWindows()
