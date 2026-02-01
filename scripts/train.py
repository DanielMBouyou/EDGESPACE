"""
Script d'entraînement avancé pour le modèle de détection de feux.
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import FireTrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train fire detection model for edge deployment"
    )
    
    # Dataset
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/merged_yolo/data.yaml"),
        help="Path to data.yaml",
    )
    
    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="Base model (nano recommended for edge)",
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.01)
    
    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, 0, 0,1",
    )
    parser.add_argument("--workers", type=int, default=8)
    
    # Output
    parser.add_argument(
        "--project",
        type=str,
        default="runs/fire",
        help="Project directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Run name",
    )
    
    # Export
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export to TensorRT after training",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="engine",
        choices=["engine", "onnx", "tflite"],
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Use INT8 quantization for export",
    )
    
    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SpaceEdge AI - Fire Detection Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print("=" * 60)
    
    # Configuration
    config = TrainingConfig(
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        learning_rate=args.lr,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume is not None,
        resume_path=args.resume,
    )
    
    # Entraînement
    trainer = FireTrainer(config=config, data_yaml=args.data)
    metrics = trainer.train()
    
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Export si demandé
    if args.export:
        print("\n" + "=" * 60)
        print(f"Exporting to {args.export_format}...")
        print("=" * 60)
        
        export_path = trainer.export_for_edge(
            format=args.export_format,
            half=not args.int8,
            int8=args.int8,
            imgsz=args.imgsz,
        )
        print(f"✅ Exported to: {export_path}")
    
    # Copier le meilleur modèle
    best_weights = trainer.get_best_weights()
    print(f"\n✅ Best weights saved at: {best_weights}")
    
    # Copier vers models/
    import shutil
    output_path = Path("models/fire_best.pt")
    output_path.parent.mkdir(exist_ok=True)
    shutil.copy(best_weights, output_path)
    print(f"✅ Copied to: {output_path}")


if __name__ == "__main__":
    main()
