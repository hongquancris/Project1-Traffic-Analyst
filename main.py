# File chạy chính: đọc video, gọi detector & tracker, vẽ quỹ đạo

import cv2
import numpy as np
from core.detector import VehicleDetector
from core.tracker import CustomVehicleTracker
from core.stabilizer import VideoStabilizer

VEHICLE_CLASSES = {2: 'Car', 3: 'Motorbike', 5: 'Bus', 7: 'Truck'}
COLORS = {2: (255, 180, 0), 3: (0, 255, 128), 5: (0, 140, 255), 7: (180, 0, 255)}

def resize_if_needed(frame, max_width=1280, max_height=720):
    h, w = frame.shape[:2]
    if w <= max_width and h <= max_height:
        return frame
    scale = min(max_width / w, max_height / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def main():
    detector = VehicleDetector(model_path="yolov8n.pt", conf_threshold=0.35)
    tracker = CustomVehicleTracker(max_disappeared=15, max_distance=80, history_len=30)
    stabilizer = VideoStabilizer(border_crop_ratio=0.04)

    cap = cv2.VideoCapture("data/sample_traffic.mp4")
    if not cap.isOpened():
        print("Lỗi: Không mở được file data/sample_traffic.mp4!")
        return

    print("Đang chạy... Nhấn phím 'q' trên cửa sổ video để dừng.")
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        raw_frame = resize_if_needed(raw_frame, max_width=1280, max_height=720)
        frame = stabilizer.stabilize(raw_frame)

        detections = detector.detect(frame, min_y2=220)
        tracked_vehicles = tracker.update(detections)

        for vid, info in tracked_vehicles.items():
            if info['disappeared'] > 0:
                continue

            x1, y1, x2, y2 = info['bbox']
            cls_id = info['class_id']
            label = f"ID:{vid} {VEHICLE_CLASSES.get(cls_id, 'Vehicle')}"
            color = COLORS.get(cls_id, (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            pts = list(info['trajectory'])
            for i in range(1, len(pts)):
                thickness = int(np.sqrt(30 / float(len(pts) - i + 1)) * 1.5)
                cv2.line(frame, pts[i - 1], pts[i], color, max(1, thickness))

        cv2.imshow("Project 1 - Traffic Analyst (Week 8)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()