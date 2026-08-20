import cv2

VIDEO_PATH = "traffic.mp4"
FRAME_NUMBER = 0  

points = []


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow(WINDOW_NAME, frame)


WINDOW_NAME = "Click 4 points: TL, TR, BR, BL  (press q when done)"

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError(f"Could not read frame {FRAME_NUMBER} from {VIDEO_PATH}")

cv2.imshow(WINDOW_NAME, frame)
cv2.setMouseCallback(WINDOW_NAME, on_click)

print("Click the 4 corners of your reference rectangle, in order: TL, TR, BR, BL. Click point that you can measure or estimate using information in video data")
print("Press 'q' when done.")

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

if len(points) == 4:
    formatted = ",\n    ".join(f"[{x}, {y}]" for x, y in points)
    snippet = f"IMAGE_POINTS = np.array([\n    {formatted},\n], dtype=np.float32)"
    with open("calibration_points.txt", "w") as f:
        f.write(snippet + "\n")
else:
    print(f"\nOnly got {len(points)} points -- run again and click exactly 4.")