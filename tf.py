from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np

# Load class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Set line for counting
limits = [100, 275, 450, 277]
totalCount = []

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
video_path = "video3.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error opening video file.")
    exit()

# Get width and height for output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output writer
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while True:
    success, img = cap.read()
    if not success or img is None:
        print("✅ Done! No image/frame received. Exiting loop.")
        break

    detections = np.empty((0, 5))

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            className = classNames[cls]

            if className in ["car", "truck", "bus", "motorbike"] and conf > 0.4:
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

    # Update tracker
    resultstracker = tracker.update(detections)

    # Draw counting line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultstracker:
        x1, y1, x2, y2, track_id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # Draw bounding box and label
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'ID: {track_id}', (max(0, x1), max(35, y1)),
                           scale=1, thickness=2, offset=3)

        cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)

        # Count vehicle crossing the line
        if limits[0] < cx < limits[2] and (limits[1] - 15) < cy < (limits[1] + 15):
            if track_id not in totalCount:
                totalCount.append(track_id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
                with open("traffic_jam.txt", "a") as f:
                    f.write(f"{track_id}\n")

    # Display count
    cv2.putText(img, f'Count: {len(totalCount)}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Show & save
    cv2.imshow("Image", img)
    out.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
out.release()
cv2.destroyAllWindows()
