import math
from collections import deque

class CustomVehicleTracker:
    def __init__(self, max_disappeared=15, max_distance=80, iou_threshold=0.25, history_len=30):
        self.next_object_id = 1
        self.vehicles = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.iou_threshold = iou_threshold
        self.history_len = history_len

    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def _register(self, bbox, class_id):
        cx, cy = self._get_centroid(bbox)
        self.vehicles[self.next_object_id] = {
            'bbox': bbox,
            'class_id': class_id,
            'disappeared': 0,
            'trajectory': deque([(cx, cy)], maxlen=self.history_len)
        }
        self.next_object_id += 1

    def update(self, detections):
        if len(detections) == 0:
            for vid in list(self.vehicles.keys()):
                self.vehicles[vid]['disappeared'] += 1
                if self.vehicles[vid]['disappeared'] > self.max_disappeared:
                    del self.vehicles[vid]
            return self.vehicles

        if len(self.vehicles) == 0:
            for det in detections:
                self._register(det[:4], int(det[4]))
            return self.vehicles

        vehicle_ids = list(self.vehicles.keys())
        used_rows = set()
        used_cols = set()
        matches = []

        for r, vid in enumerate(vehicle_ids):
            prev_bbox = self.vehicles[vid]['bbox']
            prev_cx, prev_cy = self.vehicles[vid]['trajectory'][-1]

            for c, det in enumerate(detections):
                curr_bbox = det[:4]
                curr_cx, curr_cy = self._get_centroid(curr_bbox)
                dist = math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)
                iou = self._compute_iou(prev_bbox, curr_bbox)

                if dist <= self.max_distance or iou >= self.iou_threshold:
                    score = dist - (iou * 50.0)
                    matches.append((score, r, c))

        matches.sort(key=lambda x: x[0])

        for _, r, c in matches:
            if r in used_rows or c in used_cols:
                continue
            vid = vehicle_ids[r]
            curr_bbox = detections[c][:4]
            curr_cls = int(detections[c][4])
            cx, cy = self._get_centroid(curr_bbox)

            self.vehicles[vid]['bbox'] = curr_bbox
            self.vehicles[vid]['class_id'] = curr_cls
            self.vehicles[vid]['disappeared'] = 0
            self.vehicles[vid]['trajectory'].append((cx, cy))

            used_rows.add(r)
            used_cols.add(c)

        for r, vid in enumerate(vehicle_ids):
            if r not in used_rows:
                self.vehicles[vid]['disappeared'] += 1
                if self.vehicles[vid]['disappeared'] > self.max_disappeared:
                    del self.vehicles[vid]

        for c in range(len(detections)):
            if c not in used_cols:
                self._register(detections[c][:4], int(detections[c][4]))

        return self.vehicles