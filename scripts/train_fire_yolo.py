import argparse

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fast YOLO fire detector.")
    parser.add_argument("--data", default="data/ts_satfire_yolo/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default="ts-satfire-fast")
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        project="runs/fire",
        name=args.name,
        verbose=False,
    )


if __name__ == "__main__":
    main()
