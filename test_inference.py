from ultralytics import YOLO
from pathlib import Path

model = YOLO('runs/detect/traffic_light_yolov83/weights/best.pt')
val_images = Path('data/yolo_dataset/val/images')

output_dir = Path('test_results')
output_dir.mkdir(exist_ok=True)

# Test on 5 random validation images
test_imgs = list(val_images.glob('*.*'))[:5]  # Change number as needed

for img_path in test_imgs:
    results = model(img_path)
    results[0].save(filename=str(output_dir / img_path.name))
    print(f"Processed {img_path.name}:")
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  Class: {model.names[cls]}, Confidence: {conf:.2f}")

print(f"Results saved in {output_dir.resolve()}")
