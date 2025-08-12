# traffic_report_full.py
import os
import time
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from collections import defaultdict
from ultralytics import YOLO
from sort import Sort
import cvzone

# ----------------- CONFIG -----------------
VIDEO_IN = "video3.mp4"           # input video
VIDEO_OUT = "output_annotated.mp4" # annotated output
MODEL_PATH = "yolov8n.pt"         # yolov8 weights
OUTPUT_PDF = "traffic_report.pdf"
OUTPUT_CSV = "traffic_log.csv"
OVERSPEED_DIR = "overspeed_snapshots"

meters_per_pixel = 0.05   # <-- calibrate this value (meters per pixel)
speed_limit_kmph = 40.0   # speed limit for overspeed detection
CONF_THRESH = 0.35
IOU_MATCH_THRESH = 0.5

os.makedirs(OVERSPEED_DIR, exist_ok=True)

# ----------------- HELPERS -----------------
def iou(boxA, boxB):
    # boxes: (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea/union if union>0 else 0

# ----------------- INIT MODEL / IO -----------------
model = YOLO(MODEL_PATH)  # loads model and class names
# Some ultralytics versions expose names as model.names
try:
    names = model.model.names
except Exception:
    # fallback list for COCO if model doesn't provide names
    names = {i: n for i, n in enumerate([
        "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
        "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
        "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
        "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
        "carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
        "diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone",
        "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
        "teddy bear","hair drier","toothbrush"])}

cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video file: {VIDEO_IN}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (width, height))

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Data structures
prev_pos = dict()  # track_id -> (cx,cy,t)
vehicle_records = defaultdict(lambda: {
    "class": None, "speeds": [], "first_frame": None, "last_frame": None, "snapshots": []
})
overspeed_events = []

frame_idx = 0
print("Processing video... (press 'q' window to stop early)")

while True:
    success, frame = cap.read()
    if not success:
        break
    frame_idx += 1
    t_now = time.time()

    # Run detection
    detections = np.empty((0,5))
    dets_info = []  # parallel list to hold class & box for each detection row
    results = model(frame, stream=True, conf=CONF_THRESH, iou=0.45)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cname = names.get(cls, str(cls))
            # Only track vehicle classes
            if cname in {"car","truck","bus","motorbike","motorcycle","bicycle"} or cname.startswith("bus") or cname.startswith("truck"):
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
                dets_info.append({"box":(x1,y1,x2,y2),"class":cname})

    # Update SORT tracker with detections (bbox + conf)
    tracks = tracker.update(detections)

    # For each track, attempt to assign a class by matching to detections in same frame via IoU
    for tr in tracks:
        tx1, ty1, tx2, ty2, tid = map(int, tr)
        w, h = tx2-tx1, ty2-ty1
        cx, cy = tx1 + w//2, ty1 + h//2

        # match detection with highest IoU
        matched_class = vehicle_records[tid]["class"]
        best_iou = 0
        for di, di_info in enumerate(dets_info):
            dbox = di_info["box"]
            val = iou((tx1,ty1,tx2,ty2), dbox)
            if val > best_iou:
                best_iou = val
                matched_class = di_info["class"]

        # update vehicle_records first/last seen
        rec = vehicle_records[tid]
        if rec["first_frame"] is None:
            rec["first_frame"] = frame_idx
        rec["last_frame"] = frame_idx
        rec["class"] = matched_class

        # compute instantaneous speed (km/h)
        speed_kmph = 0.0
        if tid in prev_pos:
            prev_cx, prev_cy, prev_t = prev_pos[tid]
            pixel_dist = math.hypot(cx - prev_cx, cy - prev_cy)
            time_diff = t_now - prev_t
            if time_diff <= 0:
                time_diff = 1.0/fps
            speed_mps = (pixel_dist * meters_per_pixel) / time_diff
            speed_kmph = speed_mps * 3.6
            rec["speeds"].append(speed_kmph)
        prev_pos[tid] = (cx,cy,t_now)

        # annotate frame
        label = f"ID:{tid} {matched_class} {speed_kmph:.1f} km/h"
        cvzone.cornerRect(frame, (tx1, ty1, w, h), l=9, rt=2, colorR=(255,0,255))
        cvzone.putTextRect(frame, label, (max(0,tx1), max(25,ty1)),
                           scale=0.7, thickness=2, offset=2)

        # overspeed handling
        if speed_kmph > speed_limit_kmph:
            snapshot_name = os.path.join(OVERSPEED_DIR, f"overspeed_ID{tid}_F{frame_idx}.jpg")
            # save the cropped vehicle region for clarity
            crop = frame[max(0,ty1-5):min(height,ty2+5), max(0,tx1-5):min(width,tx2+5)].copy()
            cv2.putText(crop, f"{speed_kmph:.1f} km/h", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imwrite(snapshot_name, crop)
            rec["snapshots"].append(snapshot_name)
            overspeed_events.append({"id":tid, "speed":speed_kmph, "frame":frame_idx, "image":snapshot_name})

    # show totals for debug
    active_ids = set(int(t[4]) for t in tracks)
    cv2.putText(frame, f"Active tracks: {len(active_ids)}", (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    out.write(frame)
    cv2.imshow("Traffic", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release IO
cap.release()
out.release()
cv2.destroyAllWindows()

# ----------------- AGGREGATE STATS -----------------
rows = []
all_speeds = []
for tid, rec in vehicle_records.items():
    speeds = rec["speeds"]
    if len(speeds) == 0:
        avg_s = max_s = min_s = 0.0
    else:
        avg_s = float(np.mean(speeds))
        max_s = float(np.max(speeds))
        min_s = float(np.min(speeds))
        all_speeds.extend(speeds)
    rows.append({
        "track_id": tid,
        "class": rec["class"] if rec["class"] else "unknown",
        "avg_kmph": avg_s,
        "max_kmph": max_s,
        "min_kmph": min_s,
        "first_frame": rec["first_frame"],
        "last_frame": rec["last_frame"],
        "snapshots": ";".join(rec["snapshots"])
    })

df = pd.DataFrame(rows).sort_values("track_id")
df.to_csv(OUTPUT_CSV, index=False)

total_tracks = len(df)
per_class_counts = df['class'].value_counts().to_dict()
avg_speed_all = float(np.mean(all_speeds)) if all_speeds else 0.0
max_speed_all = float(np.max(all_speeds)) if all_speeds else 0.0

# ----------------- PLOTS -----------------
hist_path = "speed_hist.png"
plt.figure(figsize=(6,4))
if all_speeds:
    plt.hist(all_speeds, bins=20)
    plt.title("Speed Distribution (km/h)")
    plt.xlabel("km/h"); plt.ylabel("count")
else:
    plt.text(0.5,0.5,"No speed data", ha='center')
plt.tight_layout()
plt.savefig(hist_path)
plt.close()

bar_path = "class_counts.png"
plt.figure(figsize=(6,4))
if not df.empty:
    vc = df['class'].value_counts()
    vc.plot(kind='bar')
    plt.title("Class Counts")
    plt.xlabel("class"); plt.ylabel("count")
else:
    plt.text(0.5,0.5,"No vehicle data", ha='center')
plt.tight_layout()
plt.savefig(bar_path)
plt.close()

# ----------------- BUILD PDF REPORT -----------------
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=12)
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 8, "Traffic Monitoring Report", ln=1, align="C")
pdf.ln(4)
pdf.set_font("Arial", size=11)
pdf.cell(0, 6, f"Source video: {VIDEO_IN}", ln=1)
pdf.cell(0, 6, f"Annotated video: {VIDEO_OUT}", ln=1)
pdf.cell(0, 6, f"Total tracked vehicles: {total_tracks}", ln=1)
pdf.cell(0, 6, f"Overspeed events: {len(overspeed_events)}", ln=1)
pdf.cell(0, 6, f"Average speed (samples): {avg_speed_all:.2f} km/h", ln=1)
pdf.cell(0, 6, f"Max speed observed: {max_speed_all:.2f} km/h", ln=1)
pdf.ln(6)

# insert plots
if os.path.exists(hist_path):
    pdf.image(hist_path, w=180)
    pdf.ln(6)
if os.path.exists(bar_path):
    pdf.image(bar_path, w=180)
    pdf.ln(6)

# per-class counts
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 6, "Counts by class:", ln=1)
pdf.set_font("Arial", size=11)
for cls, ct in per_class_counts.items():
    pdf.cell(0, 6, f"{cls}: {ct}", ln=1)
pdf.ln(6)

# table header
pdf.set_font("Arial", "B", 11)
pdf.cell(18, 8, "ID", 1)
pdf.cell(42, 8, "Class", 1)
pdf.cell(30, 8, "Avg(km/h)", 1)
pdf.cell(30, 8, "Max(km/h)", 1)
pdf.cell(30, 8, "Frames", 1)
pdf.ln()

pdf.set_font("Arial", size=10)
for _, r in df.iterrows():
    pdf.cell(18, 8, str(int(r['track_id'])), 1)
    pdf.cell(42, 8, str(r['class'])[:22], 1)
    pdf.cell(30, 8, f"{r['avg_kmph']:.1f}", 1)
    pdf.cell(30, 8, f"{r['max_kmph']:.1f}", 1)
    pdf.cell(30, 8, f"{int(r['last_frame'])-int(r['first_frame']) if r['first_frame'] and r['last_frame'] else 0}", 1)
    pdf.ln()

# Overspeed page
pdf.add_page()
pdf.set_font("Arial", "B", 13)
pdf.cell(0, 8, "Overspeed Events (snapshots)", ln=1)
pdf.set_font("Arial", size=11)
if overspeed_events:
    for ev in overspeed_events[:12]:  # limit to first 12 to keep size reasonable
        pdf.cell(0,6, f"ID: {int(ev['id'])}  Speed: {ev['speed']:.1f} km/h  Frame: {ev['frame']}", ln=1)
        if os.path.exists(ev['image']):
            pdf.image(ev['image'], w=160)
            pdf.ln(4)
else:
    pdf.cell(0,6,"No overspeed events detected.", ln=1)

pdf.output(OUTPUT_PDF)
print("Done.")
print(f"Annotated video: {VIDEO_OUT}")
print(f"CSV log: {OUTPUT_CSV}")
print(f"PDF report: {OUTPUT_PDF}")
print(f"Overspeed snapshots: {OVERSPEED_DIR}/")
