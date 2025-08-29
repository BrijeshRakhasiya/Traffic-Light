import yaml
from pathlib import Path

ANNOT_PATH = Path("data/raw/additional_train.yaml")
IMAGES_ROOT = Path("data/images")
LABELS_DIR = Path("data/labels")
LABELS_DIR.mkdir(parents=True, exist_ok=True)

w, h = 1280, 720  # Bosch dataset standard size

with open(ANNOT_PATH) as f:
    data = yaml.safe_load(f)

for entry in data:
    img_path = IMAGES_ROOT / entry['path']
    label_file = LABELS_DIR / (img_path.stem + ".txt")
    if not 'boxes' in entry or not entry['boxes']:
        continue
    with open(label_file, 'w') as lf:
        for box in entry['boxes']:
            class_map = {'Red': 0, 'Yellow': 1, 'Green': 2, 'Off': 3}
            cls = class_map.get(box['label'], 3)
            x_center = ((box['x_min'] + box['x_max']) / 2) / w
            y_center = ((box['y_min'] + box['y_max']) / 2) / h
            width = (box['x_max'] - box['x_min']) / w
            height = (box['y_max'] - box['y_min']) / h
            lf.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
print("Converted annotations to YOLO format.")
