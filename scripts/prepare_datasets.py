"""
Script principal pour préparer les datasets multi-sources.
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dataset_manager import DatasetManager


def main():
    parser = argparse.ArgumentParser(
        description="Prepare fire detection datasets from multiple sources"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to download (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum samples per dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Training split ratio",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split ratio",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SpaceEdge AI - Dataset Preparation")
    print("=" * 60)
    
    manager = DatasetManager(data_dir=args.data_dir, seed=args.seed)
    
    try:
        merged_path = manager.prepare_all(
            datasets=args.datasets,
            max_samples_per_dataset=args.max_samples,
        )
        
        print(f"\n✅ Dataset prepared successfully!")
        print(f"   Location: {merged_path}")
        print(f"   data.yaml: {merged_path / 'data.yaml'}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
