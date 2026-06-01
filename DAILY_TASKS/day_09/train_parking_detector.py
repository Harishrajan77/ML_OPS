import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO


VIDEO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = VIDEO_DIR / "parking_occupancy_yolo" / "data.yaml"
DEFAULT_OUTPUT = VIDEO_DIR / "runs" / "parking_detector"
DEFAULT_PRETRAINED_DIR = VIDEO_DIR / "pretrained"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 parking slot occupancy detector.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Prepared YOLO data.yaml path.")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO base model. yolov8s is a strong accuracy/speed balance.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size. Higher helps small parking slots.")
    parser.add_argument("--batch", type=float, default=8, help="Batch size. Use a smaller value if GPU memory is low.")
    parser.add_argument("--device", default="0", help="CUDA device id, 'cpu', or '0'.")
    parser.add_argument("--project", type=Path, default=DEFAULT_OUTPUT, help="Training output project folder.")
    parser.add_argument("--name", default="parksight_yolov8s", help="Run name.")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers. 0 is safest on Windows.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def main() -> None:
    args = parse_args()
    data_path = resolve(args.data)
    project_path = resolve(args.project)

    if not data_path.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_path}. Run prepare_yolo_dataset.py first.")

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. Use --device cpu only if you intentionally want CPU training.")

    if args.device != "cpu":
        print(f"Using GPU: {torch.cuda.get_device_name(int(args.device))}")

    DEFAULT_PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    os.chdir(DEFAULT_PRETRAINED_DIR)
    model = YOLO(args.model)

    batch = int(args.batch) if float(args.batch).is_integer() else args.batch

    try:
        results = model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            device=args.device,
            project=str(project_path),
            name=args.name,
            patience=30,
            cache=False,
            workers=args.workers,
            plots=True,
            save=True,
            save_period=10,
            cos_lr=True,
            close_mosaic=10,
            amp=True,
            optimizer="auto",
            seed=42,
            exist_ok=True,
        )
    finally:
        os.chdir(old_cwd)

    best_weights = project_path / args.name / "weights" / "best.pt"
    print(f"Training complete: {results}")
    print(f"Best weights: {best_weights}")


if __name__ == "__main__":
    main()
