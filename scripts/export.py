"""
Script d'export pour déploiement edge (TensorRT, ONNX, etc.)
"""

import argparse
from pathlib import Path
import shutil
import sys

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Export fire detection model for edge deployment"
    )
    
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/fire_best.pt"),
        help="Path to PyTorch model",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="engine",
        choices=["engine", "onnx", "tflite", "coreml", "openvino"],
        help="Export format",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for export",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="Use FP16 (default: True)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Use INT8 quantization",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="TensorRT workspace size (GB)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SpaceEdge AI - Model Export")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Format: {args.format}")
    print(f"Image size: {args.imgsz}")
    print(f"FP16: {args.half and not args.int8}")
    print(f"INT8: {args.int8}")
    print("=" * 60)
    
    # Charger le modèle
    model = YOLO(str(args.model))
    
    # Export
    export_path = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half and not args.int8,
        int8=args.int8,
        workspace=args.workspace,
        simplify=True,
        dynamic=False,
    )
    
    print(f"\n✅ Exported to: {export_path}")
    
    # Copier vers output si spécifié
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(export_path, args.output)
        print(f"✅ Copied to: {args.output}")
    
    # Afficher les infos
    import os
    size_mb = os.path.getsize(export_path) / (1024 * 1024)
    print(f"\nModel size: {size_mb:.1f} MB")
    
    # Recommandations
    print("\n" + "=" * 60)
    print("Deployment Recommendations")
    print("=" * 60)
    
    if args.format == "engine":
        print("TensorRT Engine:")
        print("  - Best for NVIDIA Jetson (Nano, Xavier, Orin)")
        print("  - Requires same GPU architecture for deployment")
        print("  - Loft Orbital: Compatible with Hubble Interface")
        
    elif args.format == "onnx":
        print("ONNX:")
        print("  - Portable across platforms")
        print("  - Can be converted to TensorRT/Vitis AI later")
        print("  - Good for D-Orbit ION with Unibap processors")
        
    elif args.format == "tflite":
        print("TensorFlow Lite:")
        print("  - Good for low-power edge devices")
        print("  - Can run on ARM Cortex processors")


if __name__ == "__main__":
    main()
