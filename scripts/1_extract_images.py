import zipfile
from pathlib import Path

zip_path = Path("data/raw/dataset_additional_rgb.zip")
out_dir = Path("data/images")
out_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(out_dir)

print(f"Extracted images to {out_dir.resolve()}")
