"""
Entraîneur optimisé pour YOLOv8 avec support multi-GPU et export edge.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

from ultralytics import YOLO


@dataclass
class TrainingConfig:
    """Configuration d'entraînement."""
    
    # Modèle de base
    base_model: str = "yolov8n.pt"  # nano pour edge
    
    # Hyperparamètres
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    
    # Optimiseur
    optimizer: str = "AdamW"
    weight_decay: float = 0.0005
    momentum: float = 0.937
    
    # Learning rate scheduler
    lr_scheduler: str = "cosine"  # "linear", "cosine"
    warmup_epochs: int = 3
    
    # Augmentation
    augment: bool = True
    mosaic: float = 1.0
    mixup: float = 0.15
    copy_paste: float = 0.1
    
    # Régularisation
    dropout: float = 0.0
    label_smoothing: float = 0.0
    
    # Multi-scale training
    multi_scale: bool = True
    
    # Early stopping
    patience: int = 20
    
    # Hardware
    device: str = "auto"  # "auto", "cpu", "cuda", "0", "0,1"
    workers: int = 8
    amp: bool = True  # Automatic Mixed Precision
    
    # Sauvegarde
    project: str = "runs/fire"
    name: str = "train"
    save_period: int = 10  # Sauvegarder tous les N epochs
    
    # Resume
    resume: bool = False
    resume_path: Optional[str] = None


class FireTrainer:
    """
    Entraîneur spécialisé pour la détection de feux.
    
    Features:
    - Support multi-dataset
    - Augmentation avancée (mosaic, mixup)
    - Export automatique pour edge devices
    - Validation avec métriques satellites
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        data_yaml: Optional[Path] = None,
    ):
        """
        Args:
            config: Configuration d'entraînement
            data_yaml: Chemin vers data.yaml du dataset
        """
        self.config = config or TrainingConfig()
        self.data_yaml = data_yaml
        self.model: Optional[YOLO] = None
        self.results = None
        
    def setup(self) -> None:
        """Initialise le modèle."""
        self.model = YOLO(self.config.base_model)
        
        # Vérifier le device
        if self.config.device == "auto":
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
            
        print(f"Training device: {self.device}")
        print(f"Base model: {self.config.base_model}")
        
    def train(
        self,
        data_yaml: Optional[Path] = None,
        resume: bool = False,
    ) -> dict:
        """
        Lance l'entraînement.
        
        Args:
            data_yaml: Chemin vers data.yaml (override)
            resume: Reprendre l'entraînement
            
        Returns:
            Métriques finales
        """
        if self.model is None:
            self.setup()
            
        data_path = data_yaml or self.data_yaml
        if data_path is None:
            raise ValueError("data_yaml must be provided")
            
        # Arguments d'entraînement
        train_args = {
            "data": str(data_path),
            "epochs": self.config.epochs,
            "batch": self.config.batch_size,
            "imgsz": self.config.image_size,
            "device": self.device,
            "workers": self.config.workers,
            "project": self.config.project,
            "name": self.config.name,
            "exist_ok": True,
            "pretrained": True,
            "optimizer": self.config.optimizer,
            "lr0": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "momentum": self.config.momentum,
            "warmup_epochs": self.config.warmup_epochs,
            "patience": self.config.patience,
            "save_period": self.config.save_period,
            "amp": self.config.amp,
            "augment": self.config.augment,
            "mosaic": self.config.mosaic,
            "mixup": self.config.mixup,
            "copy_paste": self.config.copy_paste,
            "multi_scale": self.config.multi_scale,
            "dropout": self.config.dropout,
            "label_smoothing": self.config.label_smoothing,
            "verbose": True,
            "val": True,
            "plots": True,
        }
        
        if resume and self.config.resume_path:
            train_args["resume"] = self.config.resume_path
            
        self.results = self.model.train(**train_args)
        
        return self._get_metrics()
    
    def validate(self, data_yaml: Optional[Path] = None) -> dict:
        """Valide le modèle sur le dataset de validation."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call setup() or train() first.")
            
        data_path = data_yaml or self.data_yaml
        
        results = self.model.val(
            data=str(data_path),
            imgsz=self.config.image_size,
            batch=self.config.batch_size,
            device=self.device,
        )
        
        return {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
        }
    
    def export_for_edge(
        self,
        format: str = "engine",
        half: bool = True,
        int8: bool = False,
        imgsz: int = 640,
    ) -> Path:
        """
        Exporte le modèle pour déploiement edge.
        
        Args:
            format: Format d'export ("engine", "onnx", "tflite")
            half: FP16 precision
            int8: INT8 quantization
            imgsz: Taille d'image pour export
            
        Returns:
            Chemin vers le modèle exporté
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        export_path = self.model.export(
            format=format,
            half=half,
            int8=int8,
            imgsz=imgsz,
            simplify=True,
            dynamic=False,
        )
        
        return Path(export_path)
    
    def _get_metrics(self) -> dict:
        """Extrait les métriques d'entraînement."""
        if self.results is None:
            return {}
            
        return {
            "best_mAP50": self.results.results_dict.get("metrics/mAP50(B)", 0),
            "best_mAP50-95": self.results.results_dict.get("metrics/mAP50-95(B)", 0),
            "best_precision": self.results.results_dict.get("metrics/precision(B)", 0),
            "best_recall": self.results.results_dict.get("metrics/recall(B)", 0),
            "best_epoch": self.results.results_dict.get("epoch", 0),
        }
    
    def get_best_weights(self) -> Path:
        """Retourne le chemin vers les meilleurs poids."""
        run_dir = Path(self.config.project) / self.config.name
        best_path = run_dir / "weights" / "best.pt"
        
        if not best_path.exists():
            raise FileNotFoundError(f"Best weights not found at {best_path}")
            
        return best_path


def train_fire_model(
    data_yaml: Path,
    output_dir: Path = Path("models"),
    epochs: int = 50,
    batch_size: int = 16,
    image_size: int = 640,
    base_model: str = "yolov8n.pt",
) -> Path:
    """
    Fonction utilitaire pour entraîner rapidement un modèle.
    
    Args:
        data_yaml: Chemin vers data.yaml
        output_dir: Répertoire de sortie
        epochs: Nombre d'epochs
        batch_size: Taille du batch
        image_size: Taille d'image
        base_model: Modèle de base
        
    Returns:
        Chemin vers le meilleur modèle
    """
    config = TrainingConfig(
        base_model=base_model,
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
    )
    
    trainer = FireTrainer(config=config, data_yaml=data_yaml)
    metrics = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Best mAP50: {metrics.get('best_mAP50', 0):.4f}")
    print(f"Best mAP50-95: {metrics.get('best_mAP50-95', 0):.4f}")
    
    # Copier le meilleur modèle
    best_weights = trainer.get_best_weights()
    final_path = output_dir / "fire_best.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import shutil
    shutil.copy(best_weights, final_path)
    
    return final_path
