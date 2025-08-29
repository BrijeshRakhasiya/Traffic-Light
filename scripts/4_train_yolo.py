from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Or yolov8s.pt, yolov8m.pt, etc.
results = model.train(
    data='config/traffic_light.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='traffic_light_yolov8'
)
