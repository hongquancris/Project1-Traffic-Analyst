from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.35):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = [2, 3, 5, 7]

    def detect(self, frame, min_y2=220):
        results = self.model(
            frame, classes=self.target_classes, conf=self.conf_threshold, verbose=False
        )[0]
        detections = []
        for box in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls_id = box
            if y2 > min_y2:  # Bỏ qua các chấm xe quá nhỏ ở sát chân trời
                detections.append([int(x1), int(y1), int(x2), int(y2), int(cls_id)])
        return detections