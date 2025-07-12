import random
import shutil
from pathlib import Path

IMG_DIR = Path("data/images")
LBL_DIR = Path("data/labels")
OUT_DIR = Path("data/yolo_dataset")

# Recursively find all images in subfolders
images = list(IMG_DIR.rglob("*.png")) + list(IMG_DIR.rglob("*.jpg"))
random.shuffle(images)
split_idx = int(len(images) * 0.8)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

for split, split_imgs in zip(['train', 'val'], [train_imgs, val_imgs]):
    (OUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
    (OUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    for img in split_imgs:
        shutil.copy(img, OUT_DIR / split / 'images' / img.name)
        label = LBL_DIR / (img.stem + ".txt")
        if label.exists():
            shutil.copy(label, OUT_DIR / split / 'labels' / label.name)
