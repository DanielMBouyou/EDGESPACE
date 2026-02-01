"""
Script de benchmark pour mesurer les performances edge.
"""

import argparse
from pathlib import Path
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import FireDetector
from src.config import SpaceEdgeConfig, JETSON_NANO_CONFIG, JETSON_ORIN_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fire detection model for edge deployment"
    )
    
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/fire_best.pt"),
        help="Path to model weights",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="cpu",
        choices=["cpu", "jetson_nano", "jetson_orin", "cuda"],
        help="Target platform",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SpaceEdge AI - Performance Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Platform: {args.platform}")
    print(f"Image size: {args.imgsz}")
    print(f"Benchmark runs: {args.runs}")
    print("=" * 60)
    
    # Sélectionner la configuration
    if args.platform == "jetson_nano":
        config = JETSON_NANO_CONFIG
    elif args.platform == "jetson_orin":
        config = JETSON_ORIN_CONFIG
    else:
        config = SpaceEdgeConfig()
        config.hardware.platform = args.platform
    
    config.model.input_size = args.imgsz
    
    # Créer le détecteur
    detector = FireDetector(
        model_path=args.model,
        config=config,
    )
    
    # Benchmark
    print("\nRunning benchmark...")
    results = detector.benchmark(num_runs=args.runs)
    
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"  Mean inference time: {results['mean_ms']:.2f} ms")
    print(f"  Std deviation: {results['std_ms']:.2f} ms")
    print(f"  Min time: {results['min_ms']:.2f} ms")
    print(f"  Max time: {results['max_ms']:.2f} ms")
    print(f"  P50 (median): {results['p50_ms']:.2f} ms")
    print(f"  P95: {results['p95_ms']:.2f} ms")
    print(f"  P99: {results['p99_ms']:.2f} ms")
    print(f"  FPS: {results['fps']:.1f}")
    print("=" * 60)
    
    # Analyse pour edge
    target_fps = config.pipeline.target_fps
    if results['fps'] >= target_fps:
        print(f"✅ Target FPS ({target_fps}) achieved!")
    else:
        print(f"⚠️ Below target FPS ({target_fps}). Consider:")
        print("   - Reducing image size (--imgsz 416)")
        print("   - Using INT8 quantization")
        print("   - Using a smaller model (yolov8n)")
    
    # Sauvegarder les résultats
    if args.output:
        results["platform"] = args.platform
        results["model"] = str(args.model)
        results["image_size"] = args.imgsz
        
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
