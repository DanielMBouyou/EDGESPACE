"""
Configuration centrale pour SpaceEdge AI.
Paramètres optimisés pour le déploiement edge spatial.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class HardwareConfig:
    """Configuration hardware pour différentes plateformes spatiales."""
    
    # Plateforme cible
    platform: Literal["jetson_nano", "jetson_xavier", "jetson_orin", "xilinx_zynq", "cpu"] = "cpu"
    
    # Paramètres de quantification
    quantization: Literal["fp32", "fp16", "int8"] = "fp16"
    
    # Mémoire maximale (Mo)
    max_memory_mb: int = 2048
    
    # Threads CPU disponibles
    cpu_threads: int = 4
    
    # Activer TensorRT (NVIDIA) ou Vitis AI (Xilinx)
    use_tensorrt: bool = True
    use_vitis: bool = False


@dataclass
class ModelConfig:
    """Configuration du modèle de détection."""
    
    # Architecture du modèle
    model_type: Literal["yolov8n", "yolov8s", "efficientdet_lite", "unet_tiny"] = "yolov8n"
    
    # Taille d'entrée (doit être multiple de 32)
    input_size: int = 640
    
    # Seuil de confiance pour les détections
    confidence_threshold: float = 0.35
    
    # Seuil NMS (Non-Maximum Suppression)
    nms_threshold: float = 0.45
    
    # Classes à détecter
    classes: list[str] = field(default_factory=lambda: ["fire", "smoke"])
    
    # Chemin du modèle
    weights_path: Path = Path("models/fire_best.pt")
    
    # Export TensorRT
    tensorrt_path: Path = Path("models/fire_best.engine")


@dataclass
class PipelineConfig:
    """Configuration du pipeline de traitement."""
    
    # Activer le masquage des nuages
    enable_cloud_masking: bool = True
    
    # Seuil de couverture nuageuse (ignorer si > seuil)
    cloud_threshold: float = 0.7
    
    # Nombre de bandes spectrales à utiliser
    spectral_bands: list[int] = field(default_factory=lambda: [0, 1, 2])  # RGB
    
    # Bandes thermiques (si disponibles)
    thermal_bands: list[int] = field(default_factory=lambda: [3, 4])  # SWIR/TIR
    
    # Fusionner RGB + Thermal
    use_multispectral: bool = True
    
    # Taille minimale de zone de feu (pixels)
    min_fire_area: int = 16
    
    # FPS cible pour le traitement temps réel
    target_fps: int = 15


@dataclass
class DownlinkConfig:
    """Configuration pour la transmission des alertes."""
    
    # Format de sortie
    output_format: Literal["json", "protobuf", "cbor"] = "json"
    
    # Compression des métadonnées
    compress: bool = True
    
    # Inclure les miniatures (très basse résolution)
    include_thumbnails: bool = False
    thumbnail_size: tuple[int, int] = (64, 64)
    
    # Priorité des alertes
    priority_levels: dict[str, int] = field(default_factory=lambda: {
        "fire_detected": 1,
        "smoke_detected": 2,
        "anomaly": 3
    })
    
    # Réseau de communication
    network: Literal["iridium_sbd", "viasat", "ksat", "leafspace"] = "iridium_sbd"
    
    # Taille max du message (bytes)
    max_message_size: int = 1960  # Iridium SBD limit


@dataclass
class DatasetConfig:
    """Configuration des datasets pour l'entraînement."""
    
    # Datasets Kaggle à utiliser
    kaggle_datasets: list[str] = field(default_factory=lambda: [
        "z789456sx/ts-satfire",
        "abdelghaniaaba/wildfire-prediction-dataset",
        "elmadafri/the-wildfire-dataset",
        "vijayveersingh/nasa-firms-active-fire-dataset-modisviirs",
    ])
    
    # Répertoire des données brutes
    raw_dir: Path = Path("data/raw")
    
    # Répertoire des données traitées
    processed_dir: Path = Path("data/processed")
    
    # Split train/val/test
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Augmentation des données
    augmentation: bool = True
    
    # Taille max du dataset
    max_samples: int = 10000


@dataclass
class SpaceEdgeConfig:
    """Configuration globale SpaceEdge AI."""
    
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    downlink: DownlinkConfig = field(default_factory=DownlinkConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    # Mode de fonctionnement
    mode: Literal["inference", "training", "benchmark"] = "inference"
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # Répertoire de sortie
    output_dir: Path = Path("outputs")


# Configuration par défaut pour différentes plateformes
JETSON_NANO_CONFIG = SpaceEdgeConfig(
    hardware=HardwareConfig(
        platform="jetson_nano",
        quantization="int8",
        max_memory_mb=4096,
        cpu_threads=4,
    ),
    model=ModelConfig(
        model_type="yolov8n",
        input_size=416,  # Plus petit pour Nano
        confidence_threshold=0.4,
    ),
    pipeline=PipelineConfig(
        target_fps=10,
    ),
)

JETSON_ORIN_CONFIG = SpaceEdgeConfig(
    hardware=HardwareConfig(
        platform="jetson_orin",
        quantization="fp16",
        max_memory_mb=32768,
        cpu_threads=12,
    ),
    model=ModelConfig(
        model_type="yolov8n",
        input_size=640,
        confidence_threshold=0.35,
    ),
    pipeline=PipelineConfig(
        target_fps=30,
    ),
)

XILINX_ZYNQ_CONFIG = SpaceEdgeConfig(
    hardware=HardwareConfig(
        platform="xilinx_zynq",
        quantization="int8",
        max_memory_mb=2048,
        use_tensorrt=False,
        use_vitis=True,
    ),
    model=ModelConfig(
        model_type="yolov8n",
        input_size=320,
        confidence_threshold=0.4,
    ),
    pipeline=PipelineConfig(
        target_fps=15,
    ),
)
