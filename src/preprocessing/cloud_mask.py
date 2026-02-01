"""
Module de masquage des nuages pour économiser les cycles GPU.
Élimine les zones couvertes par les nuages avant la détection de feu.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional


@dataclass
class CloudMaskResult:
    """Résultat du masquage des nuages."""
    mask: np.ndarray  # Masque binaire (1 = nuage, 0 = clair)
    cloud_coverage: float  # Pourcentage de couverture nuageuse
    is_usable: bool  # True si l'image est exploitable


class CloudMasker:
    """
    Détecteur de nuages léger pour images satellites.
    
    Utilise des heuristiques simples basées sur:
    - Luminosité élevée (nuages réfléchissent beaucoup)
    - Ratio entre bandes spectrales
    - Température (si bandes thermiques disponibles)
    """
    
    def __init__(
        self,
        brightness_threshold: float = 0.85,
        cloud_coverage_limit: float = 0.7,
        use_thermal: bool = True,
        thermal_threshold: float = 260.0,  # Kelvin
    ):
        """
        Args:
            brightness_threshold: Seuil de luminosité normalisée (0-1)
            cloud_coverage_limit: Limite de couverture nuageuse acceptable
            use_thermal: Utiliser les bandes thermiques si disponibles
            thermal_threshold: Température sous laquelle on considère un nuage (K)
        """
        self.brightness_threshold = brightness_threshold
        self.cloud_coverage_limit = cloud_coverage_limit
        self.use_thermal = use_thermal
        self.thermal_threshold = thermal_threshold
        
    def detect_clouds_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Détection de nuages basée sur RGB uniquement.
        
        Args:
            image: Image RGB normalisée (H, W, 3) ou uint8
            
        Returns:
            Masque binaire des nuages (H, W)
        """
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            
        # Moyenne des canaux (nuages = haute réflectance uniforme)
        brightness = np.mean(image, axis=2)
        
        # Écart-type faible (nuages = couleur uniforme)
        std_dev = np.std(image, axis=2)
        
        # Masque de nuages: brillant ET uniforme
        cloud_mask = (brightness > self.brightness_threshold) & (std_dev < 0.1)
        
        # Morphologie pour nettoyer le masque
        kernel = np.ones((5, 5), np.uint8)
        cloud_mask = cv2.morphologyEx(
            cloud_mask.astype(np.uint8), 
            cv2.MORPH_CLOSE, 
            kernel
        )
        cloud_mask = cv2.morphologyEx(
            cloud_mask, 
            cv2.MORPH_OPEN, 
            kernel
        )
        
        return cloud_mask.astype(bool)
    
    def detect_clouds_thermal(
        self, 
        thermal_band: np.ndarray,
        rgb_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Détection de nuages avec bande thermique.
        Nuages = température froide dans l'infrarouge.
        
        Args:
            thermal_band: Bande thermique (température en Kelvin ou normalisée)
            rgb_mask: Masque RGB pré-calculé (optionnel, pour fusion)
            
        Returns:
            Masque binaire des nuages
        """
        # Normaliser si nécessaire
        if thermal_band.max() <= 1.0:
            # Supposer que c'est normalisé, dénormaliser approximativement
            thermal_band = thermal_band * 100 + 200  # Rough estimate
            
        # Nuages = pixels froids
        cold_mask = thermal_band < self.thermal_threshold
        
        if rgb_mask is not None:
            # Fusion: nuage si (froid ET brillant) OU très brillant
            return (cold_mask & rgb_mask) | rgb_mask
        
        return cold_mask
    
    def process(
        self, 
        image: np.ndarray,
        thermal_band: Optional[np.ndarray] = None
    ) -> CloudMaskResult:
        """
        Pipeline complet de masquage des nuages.
        
        Args:
            image: Image RGB (H, W, 3)
            thermal_band: Bande thermique optionnelle (H, W)
            
        Returns:
            CloudMaskResult avec masque et métriques
        """
        # Détection RGB
        rgb_mask = self.detect_clouds_rgb(image)
        
        # Améliorer avec thermal si disponible
        if thermal_band is not None and self.use_thermal:
            final_mask = self.detect_clouds_thermal(thermal_band, rgb_mask)
        else:
            final_mask = rgb_mask
            
        # Calculer la couverture
        cloud_coverage = float(final_mask.sum()) / final_mask.size
        
        # L'image est utilisable si pas trop de nuages
        is_usable = cloud_coverage < self.cloud_coverage_limit
        
        return CloudMaskResult(
            mask=final_mask,
            cloud_coverage=cloud_coverage,
            is_usable=is_usable
        )
    
    def apply_mask(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        fill_value: int = 0
    ) -> np.ndarray:
        """
        Applique le masque de nuages sur l'image.
        
        Args:
            image: Image originale
            mask: Masque de nuages
            fill_value: Valeur pour les zones masquées
            
        Returns:
            Image avec zones nuageuses masquées
        """
        masked = image.copy()
        if masked.ndim == 3:
            for c in range(masked.shape[2]):
                masked[:, :, c][mask] = fill_value
        else:
            masked[mask] = fill_value
        return masked
