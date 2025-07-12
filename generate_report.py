import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

run_dir = Path("runs/detect/traffic_light_yolov8")  # Adjust if your run folder is different

# 1. Load training arguments
args_path = run_dir / "args.yaml"
args = {}
if args_path.exists():
    with open(args_path) as f:
        args = yaml.safe_load(f)

# 2. Load results.csv (training metrics per epoch)
results_path = run_dir / "results.csv"
df = pd.read_csv(results_path) if results_path.exists() else None

# 3. Get final and best epoch metrics
final_metrics = df.iloc[-1].to_dict() if df is not None else {}
best_epoch = df['metrics/mAP_0.5:0.95'].idxmax() if df is not None else None
best_metrics = df.iloc[best_epoch].to_dict() if best_epoch is not None else {}

# 4. Load per-class metrics if available
per_class_path = run_dir / "metrics" / "per_class_metrics.csv"
per_class = pd.read_csv(per_class_path) if per_class_path.exists() else None

# 5. Prepare evaluation results (if available)
val_dir = next(run_dir.parent.glob("val*"), None)
val_metrics = {}
if val_dir is not None and (val_dir / "results.csv").exists():
    val_df = pd.read_csv(val_dir / "results.csv")
    val_metrics = val_df.iloc[-1].to_dict()

# 6. Write report
report_path = run_dir / "training_report.txt"
with open(report_path, "w") as f:
    f.write("YOLOv8 Traffic Light Detection Training Report\n")
    f.write("="*60 + "\n")
    f.write(f"Report generated: {datetime.now()}\n\n")

    f.write("Model and Training Configuration:\n")
    for k, v in args.items():
        f.write(f"  {k}: {v}\n")
    f.write("\n")

    if final_metrics:
        f.write("Final Epoch Metrics:\n")
        for k, v in final_metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

    if best_metrics:
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write("Best Epoch Metrics:\n")
        for k, v in best_metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

    if per_class is not None:
        f.write("Per-Class Metrics:\n")
        f.write(per_class.to_string(index=False))
        f.write("\n\n")
    else:
        f.write("Per-class metrics file not found.\n\n")

    if val_metrics:
        f.write("Validation Set Evaluation (last validation run):\n")
        for k, v in val_metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

    f.write("Confusion matrix image saved as: confusion_matrix.png\n")
    f.write("Other plots (PR, F1, metrics curves) are in the run directory.\n")
    f.write("="*60 + "\n")

print(f"Report saved to {report_path}")
