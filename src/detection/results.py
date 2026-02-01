"""
Structures de données pour les résultats de détection.
Format léger optimisé pour le downlink spatial.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import numpy as np


@dataclass
class Detection:
    """Une détection individuelle de feu ou fumée."""
    
    # Classe détectée
    class_id: int
    class_name: str
    
    # Confiance (0-1)
    confidence: float
    
    # Bounding box (x1, y1, x2, y2) en pixels
    bbox: tuple[float, float, float, float]
    
    # Coordonnées géographiques (si géoréférencé)
    lat: Optional[float] = None
    lon: Optional[float] = None
    
    # Métriques additionnelles
    area_pixels: Optional[int] = None
    intensity: Optional[float] = None  # Basé sur les bandes thermiques
    
    def to_dict(self) -> dict:
        """Convertit en dictionnaire léger pour JSON."""
        d = {
            "cls": self.class_id,
            "conf": round(self.confidence, 3),
            "box": [round(x, 1) for x in self.bbox],
        }
        if self.lat is not None and self.lon is not None:
            d["geo"] = [round(self.lat, 6), round(self.lon, 6)]
        if self.area_pixels is not None:
            d["area"] = self.area_pixels
        if self.intensity is not None:
            d["int"] = round(self.intensity, 2)
        return d
    
    @property
    def center(self) -> tuple[float, float]:
        """Centre de la bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )
    
    @property
    def size(self) -> tuple[float, float]:
        """Taille (width, height) de la bounding box."""
        return (
            self.bbox[2] - self.bbox[0],
            self.bbox[3] - self.bbox[1],
        )


@dataclass
class DetectionResult:
    """Résultat complet d'une inférence."""
    
    # Identifiants
    image_id: str
    timestamp: float
    
    # Liste des détections
    detections: list[Detection] = field(default_factory=list)
    
    # Métadonnées de traitement
    inference_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    cloud_coverage: float = 0.0
    
    # Informations satellite (simulées ou réelles)
    satellite_id: str = "EDGE-SAT-01"
    orbit_pass: Optional[int] = None
    
    # Image dimensions
    image_width: int = 0
    image_height: int = 0
    
    # Priorité de l'alerte (1 = critique, 5 = info)
    priority: int = 5
    
    def has_fire(self) -> bool:
        """True si au moins un feu est détecté."""
        return any(d.class_name == "fire" for d in self.detections)
    
    def has_smoke(self) -> bool:
        """True si de la fumée est détectée."""
        return any(d.class_name == "smoke" for d in self.detections)
    
    @property
    def fire_count(self) -> int:
        """Nombre de feux détectés."""
        return sum(1 for d in self.detections if d.class_name == "fire")
    
    @property
    def max_confidence(self) -> float:
        """Confiance maximale parmi les détections."""
        if not self.detections:
            return 0.0
        return max(d.confidence for d in self.detections)
    
    def compute_priority(self) -> int:
        """Calcule la priorité en fonction des détections."""
        if not self.detections:
            return 5
        
        fire_count = self.fire_count
        max_conf = self.max_confidence
        
        if fire_count >= 3 and max_conf > 0.8:
            return 1  # Critique
        elif fire_count >= 1 and max_conf > 0.6:
            return 2  # Urgent
        elif fire_count >= 1:
            return 3  # Important
        elif self.has_smoke():
            return 4  # Attention
        return 5  # Info
    
    def to_alert_payload(self, compact: bool = True) -> dict:
        """
        Génère le payload d'alerte pour downlink.
        
        Args:
            compact: Format compact pour minimiser les bytes
            
        Returns:
            Dictionnaire prêt pour JSON
        """
        self.priority = self.compute_priority()
        
        if compact:
            # Format ultra-compact pour Iridium SBD (< 2KB)
            payload = {
                "sat": self.satellite_id[:10],
                "ts": int(self.timestamp),
                "p": self.priority,
                "n": len(self.detections),
                "det": [d.to_dict() for d in self.detections[:10]],  # Max 10
                "meta": {
                    "inf_ms": round(self.inference_time_ms, 1),
                    "cloud": round(self.cloud_coverage, 2),
                }
            }
        else:
            # Format complet
            payload = {
                "satellite_id": self.satellite_id,
                "image_id": self.image_id,
                "timestamp": self.timestamp,
                "priority": self.priority,
                "detections_count": len(self.detections),
                "detections": [d.to_dict() for d in self.detections],
                "metrics": {
                    "inference_time_ms": self.inference_time_ms,
                    "preprocessing_time_ms": self.preprocessing_time_ms,
                    "cloud_coverage": self.cloud_coverage,
                    "total_time_ms": self.inference_time_ms + self.preprocessing_time_ms,
                },
                "image": {
                    "width": self.image_width,
                    "height": self.image_height,
                }
            }
            
        return payload
    
    def to_json(self, compact: bool = True) -> str:
        """Sérialise en JSON."""
        payload = self.to_alert_payload(compact)
        if compact:
            return json.dumps(payload, separators=(',', ':'))
        return json.dumps(payload, indent=2)
    
    def payload_size_bytes(self, compact: bool = True) -> int:
        """Taille du payload en bytes."""
        return len(self.to_json(compact).encode('utf-8'))
