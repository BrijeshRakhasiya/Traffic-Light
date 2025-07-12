from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

model_path = 'runs/detect/traffic_light_yolov83/weights/best.pt'
test_dir = Path('test_images')
output_dir = Path('test_results')
output_dir.mkdir(exist_ok=True)
report_path = output_dir / 'test_report.txt'

model = YOLO(model_path)

with open(report_path, "w") as report:
    report.write("YOLOv8 Traffic Light Test Report\n")
    report.write("="*60 + "\n")
    report.write(f"Model: {model_path}\n")
    report.write(f"Test images folder: {test_dir.resolve()}\n")
    report.write(f"Report generated: {datetime.now()}\n\n")

    for img_path in test_dir.glob('*.*'):
        results = model(img_path)
        result_img_path = output_dir / img_path.name
        results[0].save(filename=str(result_img_path))
        report.write(f"Image: {img_path.name}\n")
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                report.write(f"  Class: {model.names[cls]}, Confidence: {conf:.2f}, BBox: {xyxy}\n")
        else:
            report.write("  No objects detected.\n")
        report.write("\n")

print(f"Test report saved to {report_path.resolve()}")
