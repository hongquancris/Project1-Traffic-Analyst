# Project1-Traffic-Analyst

## Cấu trúc dự án

```text
Project1-Traffic-Analyst/
├── data/
│   └── sample_traffic.mp4       # Tải 1 video giao thông góc cao (30s - 1p) bỏ vào đây
├── core/
│   ├── __init__.py
│   ├── detector.py              # Gọi YOLOv8 lấy tọa độ [x1, y1, x2, y2] của 4 loại xe
│   └── tracker.py               # (Bạn tự code) Thuật toán nối ID bằng IoU + Khoảng cách tâm
├── main.py                      # File chạy chính: đọc video, gọi detector & tracker, vẽ quỹ đạo
├── requirements.txt             # Danh sách thư viện: ultralytics, opencv-python, numpy
└── README.md
```

## Cấu trúc dự án

```text
Project1-Traffic-Analyst/
├── data/
│   └── sample_traffic.mp4       # Tải 1 video giao thông góc cao (30s - 1p) bỏ vào đây
├── core/
│   ├── __init__.py
│   ├── detector.py              # Gọi YOLOv8 lấy tọa độ [x1, y1, x2, y2] của 4 loại xe
│   └── tracker.py               # (Bạn tự code) Thuật toán nối ID bằng IoU + Khoảng cách tâm
├── main.py                      # File chạy chính: đọc video, gọi detector & tracker, vẽ quỹ đạo
├── requirements.txt             # Danh sách thư viện: ultralytics, opencv-python, numpy
└── README.md
```
