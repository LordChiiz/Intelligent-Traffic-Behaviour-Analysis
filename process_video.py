import subprocess

import cv2
import imageio_ffmpeg
from ultralytics import YOLO

from multi_tracker import MultiObjectTracker
from behaviour_analys import BehaviorAnalyzer
from calibration import ground_point, H as DEFAULT_H

def process_video(input_path, raw_output_path, final_output_path, H=None):

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mot = MultiObjectTracker(iou_threshold=0.2, max_missed=5, dt=1.0/fps)

    if H is None:
        H = DEFAULT_H
    analyzer = BehaviorAnalyzer(fps=fps, H=H)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    frame_index = 0
    unique_track_ids = set()
    speed_samples = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_time = frame_index / fps
        frame_index += 1

        results = model(frame)[0]
        boxes = results[0].boxes

        detections = []
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls.item())
                if cls not in [2, 5, 7]:
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                detections.append((float(x1), float(y1), float(x2), float(y2)))
        tracked = mot.update(detections)
 
        annotated_frame = frame.copy()
        visible_ids = set()
 
        for track_id, bbox in tracked.items():
            x1, y1, x2, y2 = bbox
            visible_ids.add(track_id)
            unique_track_ids.add(track_id)
 
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"id:{track_id}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
 
            gp = ground_point(x1, y1, x2, y2)
            speed = analyzer.kalman_filter_speed_estimation(track_id, gp, frame_time)
            speed_samples.append(speed)
            cv2.putText(annotated_frame, f"{speed:.1f} km/h", (int(x1), int(y2) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
 
        analyzer.cleanup_inactive_ids(visible_ids)
 
        stopped = analyzer.stopped_vehicle_count()
        traffic_stats = analyzer.detect_congestion()
        avg_speed = analyzer.average_speed()
 
        cv2.rectangle(annotated_frame, (20, 20), (450, 170), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Active Vehicles: {analyzer.active_vehicle_count()}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Average Speed: {avg_speed:.2f}",
                    (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Stopped Vehicles: {stopped}",
                    (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Traffic Status: {traffic_stats}",
                    (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
 
        writer.write(annotated_frame)
 
    cap.release()
    writer.release()
 
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg_exe, "-y", "-i", raw_output_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "error",
         final_output_path],
        check=True,
    )
 
    return {
        "frames_processed": frame_index,
        "unique_vehicles_tracked": len(unique_track_ids),
        "average_speed_kmh": round(sum(speed_samples) / len(speed_samples), 1) if speed_samples else 0,
    }