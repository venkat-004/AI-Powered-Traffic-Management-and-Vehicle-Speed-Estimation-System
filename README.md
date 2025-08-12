🚗 AI-Powered Vehicle Detection, Speed Estimation & Traffic Reporting
📌 Project Overview
This project implements an AI-based traffic monitoring system using YOLOv8 and ByteTrack to detect, classify, and track vehicles in real time from video or live camera feeds. It estimates vehicle speeds, counts vehicles crossing lanes, and generates PDF reports summarizing traffic statistics.

The system is ideal for smart city applications, traffic enforcement, and transportation analytics.

🎯 Features
✅ Real-time vehicle detection using YOLOv8
✅ Vehicle classification (Car, Truck, Bus, Bike, etc.)
✅ Persistent tracking with ByteTrack for unique vehicle IDs
✅ Lane-specific counting for traffic monitoring
✅ Speed estimation based on pixel-to-real distance mapping
✅ Automatic PDF report generation with detailed stats
✅ Output video with annotations (ID, speed, vehicle type)

📂 Output Files
output_video.mp4 – Annotated processed video

traffic_report.pdf – Generated traffic statistics report

Sample PDF Contents:

Date & Time of recording

Total vehicle count

Count by vehicle type

Average speed by type

Lane-specific statistics

🛠 Tech Stack
Language: Python 3.8+

Detection: YOLOv8

Tracking: ByteTrack

Video Processing: OpenCV

PDF Generation: FPDF / ReportLab

📦 Installation
1️⃣ Clone the repository
bash
Copy
Edit
git clone (https://github.com/venkat-004/AI-Powered-Traffic-Management-and-Vehicle-Speed-Estimation-System.git)
cd traffic-monitoring
2️⃣ Create a virtual environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
3️⃣ Install dependencies
bash
Copy
Edit
pip install ultralytics opencv-python fpdf filterpy lap onnxruntime
📊 Dataset
You can use:

Custom dataset (annotated with vehicle classes)

Public datasets like:

UA-DETRAC

BIT Vehicle Dataset

If using a custom dataset:

bash
Copy
Edit
yolo task=detect mode=train data=custom.yaml model=yolov8n.pt epochs=50 imgsz=640
🖥️ Usage
1️⃣ Run detection, tracking & speed estimation
bash
Copy
Edit
python traffic_monitor.py --video input_video.mp4 --output output_video.mp4 --report traffic_report.pdf
Arguments:

--video → Input video file (or camera index)

--output → Output annotated video

--report → Output PDF report

⚙️ How It Works
Detection

YOLOv8 detects vehicles frame-by-frame.

Tracking

ByteTrack assigns unique IDs to vehicles for consistent tracking.

Lane Counting

Predefined lane lines check when a vehicle crosses, triggering a count.

Speed Estimation

Speed is calculated from distance traveled between frames using FPS and scale factor (pixels to meters).

PDF Report Generation

FPDF creates a professional summary with counts, speed, and classifications.

📄 Example PDF Report
Traffic Report – 12 Aug 2025

Total Vehicles: 78

Cars: 54 | Avg Speed: 42 km/h

Trucks: 12 | Avg Speed: 38 km/h

Bikes: 8 | Avg Speed: 45 km/h

Lane 1 Count: 60

Lane 2 Count: 18


📄 PDF Report
After processing, a PDF report will be generated automatically in the reports/ folder containing:

Sample PDF Layout
Vehicle ID	Type	Lane	Speed (km/h)	Timestamp
1	Car	2	58	00:00:12
2	Truck	1	42	00:00:15
3	Bike	3	65	00:00:17
...	...	...	...	...


🔮 Future Enhancements
License Plate Recognition (ANPR)

Violation detection (Red-light jumping, Overspeeding)

Cloud dashboard integration

Live traffic analytics API
