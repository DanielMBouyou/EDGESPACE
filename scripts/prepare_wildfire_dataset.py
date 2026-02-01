"""
Script pour préparer le dataset Wildfire Prediction (classification)
pour l'entraînement YOLO (détection).

Ce dataset contient des images 350x350 classifiées en:
- wildfire: images contenant un feu
- nowildfire: images sans feu

Stratégie: Pour les images de feu, on crée une bounding box
qui couvre la zone centrale (où le feu est généralement visible).
Pour un vrai dataset de production, on utiliserait des annotations manuelles.
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image


def detect_fire_region(image_path: Path) -> list[tuple[float, float, float, float]]:
    """
    Détecte les régions de feu dans une image en utilisant
    la segmentation par couleur (rouge/orange).
    
    Returns:
        Liste de bounding boxes normalisées (x_center, y_center, width, height)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    h, w = img.shape[:2]
    
    # Convertir en HSV pour détecter les couleurs de feu
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Masque pour les couleurs de feu (rouge/orange/jaune)
    # Rouge bas
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    # Rouge haut
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Orange/Jaune (feu)
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([35, 255, 255])
    mask3 = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Combiner les masques
    fire_mask = mask1 | mask2 | mask3
    
    # Morphologie pour nettoyer
    kernel = np.ones((5, 5), np.uint8)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    
    # Trouver les contours
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    min_area = (h * w) * 0.01  # Au moins 1% de l'image
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # Élargir légèrement la box
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        bw = min(w - x, bw + 2 * padding)
        bh = min(h - y, bh + 2 * padding)
        
        # Normaliser
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        width = bw / w
        height = bh / h
        
        boxes.append((x_center, y_center, width, height))
    
    # Si aucune box trouvée mais c'est une image de feu,
    # créer une box centrale par défaut
    if not boxes:
        # Box centrale couvrant 60% de l'image
        boxes.append((0.5, 0.5, 0.6, 0.6))
    
    return boxes


def process_image(args: tuple) -> bool:
    """Traite une seule image."""
    src_path, dst_img_dir, dst_label_dir, class_id = args
    
    try:
        # Copier l'image
        img_name = src_path.stem + ".jpg"
        dst_img_path = dst_img_dir / img_name
        
        # Charger et sauvegarder en JPG
        img = Image.open(src_path).convert("RGB")
        img.save(dst_img_path, "JPEG", quality=92)
        
        # Créer le label
        label_path = dst_label_dir / (src_path.stem + ".txt")
        
        if class_id == 0:  # wildfire
            boxes = detect_fire_region(src_path)
            lines = [f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b in boxes]
            label_path.write_text("\n".join(lines))
        else:  # nowildfire - fichier vide (image négative)
            label_path.write_text("")
            
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def prepare_dataset(
    raw_dir: Path,
    output_dir: Path,
    max_samples: int = 5000,
    include_negative: bool = True,
    negative_ratio: float = 0.3,
    seed: int = 42,
):
    """
    Prépare le dataset au format YOLO.
    
    Args:
        raw_dir: Répertoire contenant train/valid/test avec wildfire/nowildfire
        output_dir: Répertoire de sortie
        max_samples: Nombre maximum d'images de feu à utiliser
        include_negative: Inclure les images sans feu
        negative_ratio: Ratio d'images négatives vs positives
        seed: Graine aléatoire
    """
    random.seed(seed)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer la structure YOLO
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Collecter les images de feu
    fire_images = []
    for split_dir in ["train", "valid", "test"]:
        fire_dir = raw_dir / split_dir / "wildfire"
        if fire_dir.exists():
            fire_images.extend(list(fire_dir.glob("*.jpg")))
    
    print(f"Found {len(fire_images)} fire images")
    
    # Limiter et mélanger
    random.shuffle(fire_images)
    fire_images = fire_images[:max_samples]
    
    # Collecter les images sans feu si demandé
    nofire_images = []
    if include_negative:
        for split_dir in ["train", "valid", "test"]:
            nofire_dir = raw_dir / split_dir / "nowildfire"
            if nofire_dir.exists():
                nofire_images.extend(list(nofire_dir.glob("*.jpg")))
        
        random.shuffle(nofire_images)
        n_negative = int(len(fire_images) * negative_ratio)
        nofire_images = nofire_images[:n_negative]
        print(f"Including {len(nofire_images)} negative images")
    
    # Split train/val (80/20)
    n_fire_train = int(len(fire_images) * 0.8)
    n_nofire_train = int(len(nofire_images) * 0.8)
    
    tasks = []
    
    # Fire images
    for i, img_path in enumerate(fire_images):
        split = "train" if i < n_fire_train else "val"
        tasks.append((
            img_path,
            output_dir / "images" / split,
            output_dir / "labels" / split,
            0  # class_id for fire
        ))
    
    # No-fire images
    for i, img_path in enumerate(nofire_images):
        split = "train" if i < n_nofire_train else "val"
        tasks.append((
            img_path,
            output_dir / "images" / split,
            output_dir / "labels" / split,
            1  # class_id for no-fire (empty label)
        ))
    
    # Traiter en parallèle
    print(f"Processing {len(tasks)} images...")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_image, tasks))
    
    success = sum(results)
    print(f"Successfully processed {success}/{len(tasks)} images")
    
    # Compter les images par split
    train_count = len(list((output_dir / "images" / "train").glob("*.jpg")))
    val_count = len(list((output_dir / "images" / "val").glob("*.jpg")))
    
    # Créer data.yaml
    yaml_content = f"""# SpaceEdge AI - Wildfire Detection Dataset
# Source: Kaggle - abdelghaniaaba/wildfire-prediction-dataset
# Processed for YOLO object detection

path: {output_dir.absolute().as_posix()}
train: images/train
val: images/val

# Classes
names:
  0: fire

# Dataset info
# Train images: {train_count}
# Val images: {val_count}
# Total: {train_count + val_count}
"""
    
    (output_dir / "data.yaml").write_text(yaml_content)
    
    print(f"\n✅ Dataset prepared at {output_dir}")
    print(f"   Train: {train_count} images")
    print(f"   Val: {val_count} images")
    print(f"   data.yaml: {output_dir / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Wildfire Prediction Dataset for YOLO training"
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw/wildfire_canada"),
        help="Raw dataset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/wildfire_yolo"),
        help="Output directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum fire images to use",
    )
    parser.add_argument(
        "--no-negative",
        action="store_true",
        help="Don't include negative (no-fire) images",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.3,
        help="Ratio of negative to positive images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        raw_dir=args.raw,
        output_dir=args.output,
        max_samples=args.max_samples,
        include_negative=not args.no_negative,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
