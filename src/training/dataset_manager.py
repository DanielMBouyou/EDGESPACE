"""
Gestionnaire de datasets multi-sources pour l'entraînement.
Supporte les datasets Kaggle et les formats YOLO/COCO.
"""

import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tifffile as tiff
from PIL import Image

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    HAS_KAGGLE = True
except ImportError:
    HAS_KAGGLE = False


@dataclass
class DatasetInfo:
    """Informations sur un dataset."""
    name: str
    source: str  # "kaggle", "local", "url"
    dataset_id: str  # ID Kaggle ou chemin
    format: str  # "yolo", "coco", "tiff_pairs", "images"
    num_samples: int = 0
    classes: list[str] = None


class DatasetManager:
    """
    Gestionnaire centralisé pour télécharger, préparer et fusionner
    les datasets de détection de feux.
    """
    
    # Datasets Kaggle recommandés pour la détection de feux
    KAGGLE_FIRE_DATASETS = [
        DatasetInfo(
            name="TS-SatFire",
            source="kaggle",
            dataset_id="z789456sx/ts-satfire",
            format="tiff_pairs",
            classes=["fire"],
        ),
        DatasetInfo(
            name="Wildfire Dataset",
            source="kaggle",
            dataset_id="elmadafri/the-wildfire-dataset",
            format="yolo",
            classes=["fire", "smoke"],
        ),
        DatasetInfo(
            name="Fire & Smoke YOLO",
            source="kaggle",
            dataset_id="azimjaan21/fire-and-smoke-dataset-object-detection-yolo",
            format="yolo",
            classes=["fire", "smoke"],
        ),
        DatasetInfo(
            name="Forest Fire Dataset",
            source="kaggle",
            dataset_id="alik05/forest-fire-dataset",
            format="images",
            classes=["fire", "no_fire"],
        ),
    ]
    
    def __init__(
        self,
        data_dir: Path = Path("data"),
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Répertoire racine des données
            seed: Graine aléatoire pour reproductibilité
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.merged_dir = self.data_dir / "merged_yolo"
        self.seed = seed
        
        self._kaggle_api: Optional[KaggleApi] = None
        
        # Créer les répertoires
        for d in [self.raw_dir, self.processed_dir, self.merged_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
    @property
    def kaggle_api(self) -> KaggleApi:
        """Lazy initialization de l'API Kaggle."""
        if self._kaggle_api is None:
            if not HAS_KAGGLE:
                raise ImportError("kaggle package not installed")
            self._kaggle_api = KaggleApi()
            self._kaggle_api.authenticate()
        return self._kaggle_api
    
    def download_dataset(
        self,
        dataset_info: DatasetInfo,
        max_files: int = 1000,
    ) -> Path:
        """
        Télécharge un dataset depuis Kaggle.
        
        Args:
            dataset_info: Informations sur le dataset
            max_files: Nombre maximum de fichiers
            
        Returns:
            Chemin vers le dataset téléchargé
        """
        output_dir = self.raw_dir / dataset_info.name.replace(" ", "_").lower()
        
        if output_dir.exists() and any(output_dir.iterdir()):
            print(f"Dataset {dataset_info.name} already exists at {output_dir}")
            return output_dir
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {dataset_info.name} from Kaggle...")
        
        self.kaggle_api.dataset_download_files(
            dataset_info.dataset_id,
            path=str(output_dir),
            unzip=True,
            quiet=False,
        )
        
        return output_dir
    
    def process_ts_satfire(
        self,
        raw_dir: Path,
        output_dir: Path,
        viirs_bands: list[int] = [0, 1, 2],
        fire_threshold: float = 0.5,
        min_area: int = 20,
        max_pairs: int = 100,
    ) -> int:
        """
        Traite le dataset TS-SatFire (images TIFF satellites).
        
        Args:
            raw_dir: Répertoire des données brutes
            output_dir: Répertoire de sortie YOLO
            viirs_bands: Bandes à utiliser pour RGB
            fire_threshold: Seuil pour la détection de feu
            min_area: Surface minimale d'un feu (pixels)
            max_pairs: Nombre maximum de paires à traiter
            
        Returns:
            Nombre d'images traitées
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "images").mkdir(exist_ok=True)
        (output_dir / "labels").mkdir(exist_ok=True)
        
        # Trouver les paires VIIRS_Day / FirePred
        pairs = self._find_ts_satfire_pairs(raw_dir)
        
        if not pairs:
            print(f"No VIIRS/FirePred pairs found in {raw_dir}")
            return 0
            
        random.Random(self.seed).shuffle(pairs)
        pairs = pairs[:max_pairs]
        
        count = 0
        for pair in pairs:
            try:
                success = self._process_ts_satfire_pair(
                    pair, output_dir, viirs_bands, fire_threshold, min_area, count
                )
                if success:
                    count += 1
            except Exception as e:
                print(f"Error processing pair: {e}")
                
        return count
    
    def _find_ts_satfire_pairs(self, raw_dir: Path) -> list[dict]:
        """Trouve les paires VIIRS/FirePred dans le dataset."""
        pairs = []
        
        for event_dir in raw_dir.iterdir():
            if not event_dir.is_dir():
                continue
                
            viirs_dir = event_dir / "VIIRS_Day"
            firepred_dir = event_dir / "FirePred"
            
            if not viirs_dir.exists() or not firepred_dir.exists():
                continue
                
            # Matcher les fichiers par date
            for viirs_file in viirs_dir.glob("*.tif"):
                date = viirs_file.stem.split("_")[0]
                firepred_file = firepred_dir / f"{date}_FirePred.tif"
                
                if not firepred_file.exists():
                    # Essayer d'autres patterns
                    candidates = list(firepred_dir.glob(f"{date}*.tif"))
                    if candidates:
                        firepred_file = candidates[0]
                    else:
                        continue
                        
                pairs.append({
                    "event_id": event_dir.name,
                    "date": date,
                    "viirs": viirs_file,
                    "firepred": firepred_file,
                })
                
        return pairs
    
    def _process_ts_satfire_pair(
        self,
        pair: dict,
        output_dir: Path,
        viirs_bands: list[int],
        fire_threshold: float,
        min_area: int,
        index: int,
    ) -> bool:
        """Traite une paire VIIRS/FirePred."""
        viirs = tiff.imread(pair["viirs"])
        firepred = tiff.imread(pair["firepred"])
        
        if viirs.ndim != 3:
            return False
            
        # Extraire RGB
        rgb_channels = []
        for b in viirs_bands:
            if b < viirs.shape[2]:
                ch = viirs[:, :, b].astype(np.float32)
                ch = self._normalize_channel(ch)
                rgb_channels.append(ch)
            else:
                rgb_channels.append(np.zeros(viirs.shape[:2], dtype=np.uint8))
                
        rgb = np.stack(rgb_channels, axis=2)
        
        # Créer le masque de feu
        if firepred.ndim == 3:
            mask = firepred[:, :, 0]  # Première bande
        else:
            mask = firepred
            
        mask = np.nan_to_num(mask, nan=0.0)
        
        # Redimensionner si nécessaire
        if mask.shape != rgb.shape[:2]:
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]))
            
        binary = (mask > fire_threshold).astype(np.uint8)
        
        # Trouver les bounding boxes
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        
        yolo_lines = []
        for lbl in range(1, num_labels):
            x, y, w, h, area = stats[lbl]
            if area < min_area:
                continue
                
            x_center = (x + w / 2.0) / rgb.shape[1]
            y_center = (y + h / 2.0) / rgb.shape[0]
            w_norm = w / rgb.shape[1]
            h_norm = h / rgb.shape[0]
            
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # Sauvegarder
        img_name = f"{pair['event_id']}_{pair['date']}_{index:05d}.jpg"
        img_path = output_dir / "images" / img_name
        Image.fromarray(rgb).save(img_path, quality=92)
        
        label_path = output_dir / "labels" / img_name.replace(".jpg", ".txt")
        label_path.write_text("\n".join(yolo_lines))
        
        return True
    
    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """Normalise un canal en uint8."""
        finite = np.isfinite(channel)
        if not finite.any():
            return np.zeros(channel.shape, dtype=np.uint8)
            
        minv = channel[finite].min()
        maxv = channel[finite].max()
        
        if maxv <= minv:
            return np.zeros(channel.shape, dtype=np.uint8)
            
        filled = np.nan_to_num(channel, nan=minv)
        norm = (filled - minv) / (maxv - minv)
        
        return (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    
    def merge_datasets(
        self,
        datasets: list[Path],
        output_dir: Optional[Path] = None,
        train_split: float = 0.7,
        val_split: float = 0.15,
        class_mapping: Optional[dict[str, int]] = None,
    ) -> Path:
        """
        Fusionne plusieurs datasets YOLO en un seul.
        
        Args:
            datasets: Liste des répertoires de datasets YOLO
            output_dir: Répertoire de sortie
            train_split: Proportion d'entraînement
            val_split: Proportion de validation
            class_mapping: Mapping des noms de classes vers IDs
            
        Returns:
            Chemin vers le dataset fusionné
        """
        output_dir = output_dir or self.merged_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if class_mapping is None:
            class_mapping = {"fire": 0, "smoke": 1}
            
        # Créer la structure
        for split in ["train", "val", "test"]:
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            
        all_samples = []
        
        for dataset_path in datasets:
            dataset_path = Path(dataset_path)
            
            # Trouver les images
            images_dir = dataset_path / "images"
            labels_dir = dataset_path / "labels"
            
            if not images_dir.exists():
                continue
                
            for img_path in images_dir.rglob("*.jpg"):
                label_path = labels_dir / img_path.relative_to(images_dir).with_suffix(".txt")
                
                if label_path.exists():
                    all_samples.append((img_path, label_path))
                    
            for img_path in images_dir.rglob("*.png"):
                label_path = labels_dir / img_path.relative_to(images_dir).with_suffix(".txt")
                
                if label_path.exists():
                    all_samples.append((img_path, label_path))
        
        # Shuffle et split
        random.Random(self.seed).shuffle(all_samples)
        
        n = len(all_samples)
        n_train = int(n * train_split)
        n_val = int(n * val_split)
        
        splits = {
            "train": all_samples[:n_train],
            "val": all_samples[n_train:n_train + n_val],
            "test": all_samples[n_train + n_val:],
        }
        
        # Copier les fichiers
        for split_name, samples in splits.items():
            for i, (img_path, label_path) in enumerate(samples):
                new_name = f"{i:06d}{img_path.suffix}"
                
                shutil.copy(img_path, output_dir / "images" / split_name / new_name)
                shutil.copy(
                    label_path, 
                    output_dir / "labels" / split_name / new_name.replace(img_path.suffix, ".txt")
                )
        
        # Créer data.yaml
        yaml_content = f"""path: {output_dir.absolute().as_posix()}
train: images/train
val: images/val
test: images/test

names:
"""
        for name, idx in sorted(class_mapping.items(), key=lambda x: x[1]):
            yaml_content += f"  {idx}: {name}\n"
            
        (output_dir / "data.yaml").write_text(yaml_content)
        
        print(f"Merged dataset created at {output_dir}")
        print(f"  Train: {len(splits['train'])} samples")
        print(f"  Val: {len(splits['val'])} samples")
        print(f"  Test: {len(splits['test'])} samples")
        
        return output_dir
    
    def prepare_all(
        self,
        datasets: list[str] = None,
        max_samples_per_dataset: int = 500,
    ) -> Path:
        """
        Télécharge et prépare tous les datasets.
        
        Args:
            datasets: Liste des noms de datasets (ou tous par défaut)
            max_samples_per_dataset: Max samples par dataset
            
        Returns:
            Chemin vers le dataset fusionné final
        """
        processed_dirs = []
        
        for ds_info in self.KAGGLE_FIRE_DATASETS:
            if datasets and ds_info.name not in datasets:
                continue
                
            try:
                raw_path = self.download_dataset(ds_info)
                
                processed_path = self.processed_dir / ds_info.name.replace(" ", "_").lower()
                
                if ds_info.format == "tiff_pairs":
                    count = self.process_ts_satfire(
                        raw_path, processed_path, max_pairs=max_samples_per_dataset
                    )
                    if count > 0:
                        processed_dirs.append(processed_path)
                elif ds_info.format == "yolo":
                    processed_dirs.append(raw_path)
                    
            except Exception as e:
                print(f"Error processing {ds_info.name}: {e}")
                
        if not processed_dirs:
            raise RuntimeError("No datasets were successfully processed")
            
        return self.merge_datasets(processed_dirs)
