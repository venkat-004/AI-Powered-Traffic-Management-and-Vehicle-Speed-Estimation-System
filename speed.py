import cv2
import math
import time
from ultralytics import YOLO
from collections import defaultdict
import cvzone

# Load your YOLOv8 model
model = YOLO("yolov8n.pt")  # Replace with your trained model if needed

# Define vehicle class IDs (use correct IDs for your model)
vehicle_classes = [2, 3, 5, 7]  # Example: car, motorcycle, bus, truck (COCO)

# Open video
cap = cv2.VideoCapture("video3.mp4")  # Replace with your input file

# Get video properties
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# Store previous positions of tracked objects
prev_positions = {}
prev_times = {}

# Define meters per pixel (adjust for your video)
meters_per_pixel = 0.05  # Example: 0.05 meter = 1 pixel

while True:
    success, frame = cap.read()
    if not success:
        print("No image/frame received. Exiting loop.")
        break

    current_time = time.time()
    
    # Run detection
    results = model.track(frame, persist=True, conf=0.3, iou=0.5, classes=vehicle_classes)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box.astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point

            # Draw box and ID
            cvzone.putTextRect(frame, f'ID: {track_id}', (x1, y1 - 10), scale=1, thickness=2, offset=3)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Speed calculation
            if track_id in prev_positions:
                prev_cx, prev_cy = prev_positions[track_id]
                prev_time = prev_times[track_id]
                distance_pixels = math.hypot(cx - prev_cx, cy - prev_cy)
                time_diff = current_time - prev_time

                if time_diff > 0:
                    speed_mps = distance_pixels * meters_per_pixel / time_diff
                    speed_kmph = speed_mps * 3.6

                    # Display speed in km/h
                    cvzone.putTextRect(frame, f'Speed: {int(speed_kmph)} km/h', (x1, y1 - 40), scale=1, thickness=2, offset=3)

            # Update positions and times
            prev_positions[track_id] = (cx, cy)
            prev_times[track_id] = current_time

    # Write output frame
    out.write(frame)

    # Optional: show live output
    cv2.imshow("Speed Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
