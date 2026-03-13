from ultralytics import YOLO
import cv2
from behaviour_analys import BehaviorAnalyzer


model = YOLO("yolov8n.pt")

analyzer = BehaviorAnalyzer()

video_path = "traffic.mp4"
cap = cv2.VideoCapture(video_path)


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(frame, persist=True)

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            if box.id is None:
                continue

            track_id = int(box.id.item())
            cls = int(box.cls.item())

            if cls not in [2,5,7]:
                continue

            x1, y1, x2, y2 = box.xyxy[0]

            center_x = int((x1+x2)/2)
            center_y = int((y1+y2)/2)

            speed = analyzer.estimate_speed(track_id,(center_x,center_y))


    annotated_frame = results[0].plot()


    avg_speed = analyzer.average_speed()

    cv2.rectangle(annotated_frame,(20,20),(350,120),(0,0,0),-1)

    cv2.putText(annotated_frame,f"Active Vehicles: {len(analyzer.speeds)}",
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2)

    cv2.putText(annotated_frame,f"Average Speed: {avg_speed:.2f}",
                (30,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2)


    cv2.imshow("Traffic Behavior Analysis",annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
