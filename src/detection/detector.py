"""
Détecteur de feux complet avec pipeline intégré.
Combine prétraitement, inférence et post-traitement.
"""

import time
import numpy as np
from pathlib import Path
from typing import Optional, Union
import uuid

from ..config import SpaceEdgeConfig, ModelConfig, PipelineConfig
from ..preprocessing import CloudMasker, SpectralProcessor, ImageNormalizer
from .inference import InferenceEngine, InferenceConfig
from .results import Detection, DetectionResult


class FireDetector:
    """
    Détecteur de feux optimisé pour l'edge computing spatial.
    
    Pipeline complet:
    1. Cloud Masking (économie de cycles GPU)
    2. Traitement spectral (fusion RGB + thermal)
    3. Normalisation
    4. Inférence YOLO
    5. Post-traitement et génération d'alertes
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config: Optional[SpaceEdgeConfig] = None,
    ):
        """
        Args:
            model_path: Chemin vers le modèle de détection
            config: Configuration globale (ou par défaut)
        """
        self.config = config or SpaceEdgeConfig()
        self.model_path = Path(model_path)
        
        # Initialiser les composants du pipeline
        self.cloud_masker = CloudMasker(
            brightness_threshold=0.85,
            cloud_coverage_limit=self.config.pipeline.cloud_threshold,
        )
        
        self.spectral_processor = SpectralProcessor(
            rgb_bands=self.config.pipeline.spectral_bands,
            thermal_bands=self.config.pipeline.thermal_bands,
            enhance_fire=True,
        )
        
        self.normalizer = ImageNormalizer(
            target_size=self.config.model.input_size,
        )
        
        # Moteur d'inférence
        self.engine = InferenceEngine(
            model_path=self.model_path,
            config=InferenceConfig(
                confidence_threshold=self.config.model.confidence_threshold,
                nms_threshold=self.config.model.nms_threshold,
                device="cuda" if self.config.hardware.platform != "cpu" else "cpu",
                half_precision=self.config.hardware.quantization == "fp16",
            ),
        )
        
        # Statistiques
        self.total_processed = 0
        self.total_fires_detected = 0
        
    def detect(
        self,
        image: np.ndarray,
        thermal_band: Optional[np.ndarray] = None,
        image_id: Optional[str] = None,
        skip_cloud_mask: bool = False,
    ) -> DetectionResult:
        """
        Effectue la détection de feux sur une image.
        
        Args:
            image: Image RGB ou multispectrale (H, W, C)
            thermal_band: Bande thermique séparée (optionnel)
            image_id: Identifiant de l'image
            skip_cloud_mask: Ignorer le masquage des nuages
            
        Returns:
            DetectionResult avec toutes les détections et métriques
        """
        start_total = time.perf_counter()
        preprocessing_time = 0.0
        
        image_id = image_id or str(uuid.uuid4())[:8]
        timestamp = time.time()
        
        # Dimensions originales
        h, w = image.shape[:2]
        
        # 1. Cloud Masking (si activé)
        cloud_coverage = 0.0
        if self.config.pipeline.enable_cloud_masking and not skip_cloud_mask:
            start_cloud = time.perf_counter()
            cloud_result = self.cloud_masker.process(image, thermal_band)
            cloud_coverage = cloud_result.cloud_coverage
            preprocessing_time += (time.perf_counter() - start_cloud) * 1000
            
            # Si trop de nuages, retourner résultat vide
            if not cloud_result.is_usable:
                return DetectionResult(
                    image_id=image_id,
                    timestamp=timestamp,
                    detections=[],
                    cloud_coverage=cloud_coverage,
                    image_width=w,
                    image_height=h,
                    priority=5,
                )
        
        # 2. Traitement spectral (si image multispectrale)
        if image.shape[2] > 3 and self.config.pipeline.use_multispectral:
            start_spectral = time.perf_counter()
            image = self.spectral_processor.create_fire_composite(image)
            preprocessing_time += (time.perf_counter() - start_spectral) * 1000
        elif image.shape[2] > 3:
            # Garder seulement RGB
            image = image[:, :, :3]
            
        # Convertir en uint8 si nécessaire
        if image.dtype != np.uint8:
            image = self.spectral_processor.to_uint8(image)
        
        # 3. Inférence
        raw_detections, inference_time = self.engine.infer(image)
        
        # 4. Post-traitement
        detections = []
        for det in raw_detections:
            # Filtrer par taille minimale
            bbox = det["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < self.config.pipeline.min_fire_area:
                continue
                
            detection = Detection(
                class_id=det["class_id"],
                class_name=det["class_name"],
                confidence=det["confidence"],
                bbox=tuple(det["bbox"]),
                area_pixels=int(area),
            )
            detections.append(detection)
        
        # Créer le résultat
        result = DetectionResult(
            image_id=image_id,
            timestamp=timestamp,
            detections=detections,
            inference_time_ms=inference_time,
            preprocessing_time_ms=preprocessing_time,
            cloud_coverage=cloud_coverage,
            satellite_id=self.config.downlink.network.upper()[:10],
            image_width=w,
            image_height=h,
        )
        
        # Mettre à jour les stats
        self.total_processed += 1
        self.total_fires_detected += result.fire_count
        
        return result
    
    def detect_batch(
        self,
        images: list[np.ndarray],
        image_ids: Optional[list[str]] = None,
    ) -> list[DetectionResult]:
        """
        Détection sur un batch d'images.
        
        Args:
            images: Liste d'images
            image_ids: Liste d'identifiants (optionnel)
            
        Returns:
            Liste de DetectionResult
        """
        if image_ids is None:
            image_ids = [None] * len(images)
            
        results = []
        for img, img_id in zip(images, image_ids):
            result = self.detect(img, image_id=img_id)
            results.append(result)
            
        return results
    
    def get_stats(self) -> dict:
        """Retourne les statistiques de fonctionnement."""
        return {
            "total_processed": self.total_processed,
            "total_fires_detected": self.total_fires_detected,
            "avg_fires_per_image": (
                self.total_fires_detected / self.total_processed 
                if self.total_processed > 0 else 0
            ),
        }
    
    def benchmark(self, num_runs: int = 50) -> dict:
        """Benchmark le pipeline complet."""
        return self.engine.benchmark(
            image_size=(self.config.model.input_size, self.config.model.input_size),
            num_runs=num_runs,
        )
