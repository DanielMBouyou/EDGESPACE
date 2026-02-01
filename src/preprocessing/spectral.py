"""
Traitement des données multispectrales.
Fusion RGB + bandes thermiques/SWIR pour améliorer la détection.
"""

import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SpectralBands:
    """Définition des bandes spectrales typiques pour satellites d'observation."""
    
    # VIIRS (Visible Infrared Imaging Radiometer Suite)
    VIIRS_I1 = 0   # Red (0.60–0.68 μm)
    VIIRS_I2 = 1   # NIR (0.85–0.88 μm)
    VIIRS_I3 = 2   # SWIR (1.58–1.64 μm)
    VIIRS_I4 = 3   # MWIR (3.55–3.93 μm) - Fire detection
    VIIRS_I5 = 4   # TIR (10.5–12.4 μm) - Temperature
    
    # Indices utiles pour la détection de feu
    FIRE_BANDS = [3, 4]  # MWIR + TIR
    RGB_BANDS = [0, 1, 2]


class SpectralProcessor:
    """
    Processeur spectral pour images satellites.
    Crée des composites optimisés pour la détection de feux.
    """
    
    def __init__(
        self,
        rgb_bands: list[int] = [0, 1, 2],
        thermal_bands: list[int] = [3, 4],
        normalize: bool = True,
        enhance_fire: bool = True,
    ):
        """
        Args:
            rgb_bands: Indices des bandes RGB
            thermal_bands: Indices des bandes thermiques
            normalize: Normaliser les bandes
            enhance_fire: Appliquer une enhancement pour le feu
        """
        self.rgb_bands = rgb_bands
        self.thermal_bands = thermal_bands
        self.normalize = normalize
        self.enhance_fire = enhance_fire
        
    def extract_rgb(self, multispectral: np.ndarray) -> np.ndarray:
        """
        Extrait les bandes RGB d'une image multispectrale.
        
        Args:
            multispectral: Image (H, W, C) avec C bandes
            
        Returns:
            Image RGB (H, W, 3)
        """
        if multispectral.ndim != 3:
            raise ValueError(f"Expected 3D array, got {multispectral.ndim}D")
            
        rgb = []
        for band_idx in self.rgb_bands:
            if band_idx < multispectral.shape[2]:
                rgb.append(self._normalize_band(multispectral[:, :, band_idx]))
            else:
                # Bande manquante, utiliser zéros
                rgb.append(np.zeros(multispectral.shape[:2], dtype=np.float32))
                
        return np.stack(rgb, axis=2)
    
    def extract_thermal(self, multispectral: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrait les bandes thermiques.
        
        Args:
            multispectral: Image multispectrale
            
        Returns:
            Bandes thermiques combinées ou None si non disponibles
        """
        if multispectral.shape[2] <= max(self.thermal_bands, default=0):
            return None
            
        thermal_stack = []
        for band_idx in self.thermal_bands:
            if band_idx < multispectral.shape[2]:
                thermal_stack.append(
                    self._normalize_band(multispectral[:, :, band_idx])
                )
                
        if not thermal_stack:
            return None
            
        # Moyenne des bandes thermiques
        return np.mean(thermal_stack, axis=0)
    
    def compute_fire_index(self, multispectral: np.ndarray) -> np.ndarray:
        """
        Calcule un indice de feu basé sur les bandes thermiques.
        
        Utilise le ratio MWIR/TIR qui est élevé pour les feux actifs.
        
        Args:
            multispectral: Image multispectrale
            
        Returns:
            Image de l'indice de feu (H, W)
        """
        if multispectral.shape[2] < 5:
            # Pas assez de bandes, retourner zéros
            return np.zeros(multispectral.shape[:2], dtype=np.float32)
            
        # MWIR (bande 3) et TIR (bande 4) pour VIIRS
        mwir = multispectral[:, :, 3].astype(np.float32)
        tir = multispectral[:, :, 4].astype(np.float32)
        
        # Éviter division par zéro
        tir = np.maximum(tir, 1e-6)
        
        # Ratio MWIR/TIR - élevé pour les feux
        fire_index = mwir / tir
        
        # Normaliser
        fire_index = np.clip(fire_index, 0, 10) / 10
        
        return fire_index
    
    def create_fire_composite(
        self, 
        multispectral: np.ndarray,
        output_channels: int = 3
    ) -> np.ndarray:
        """
        Crée un composite optimisé pour la détection de feux.
        
        Combine RGB + indice de feu dans une image à N canaux.
        
        Args:
            multispectral: Image multispectrale
            output_channels: Nombre de canaux de sortie (3 ou 4)
            
        Returns:
            Image composite (H, W, output_channels)
        """
        rgb = self.extract_rgb(multispectral)
        
        if output_channels == 3:
            # Fusionner l'indice de feu dans le canal rouge
            if self.enhance_fire and multispectral.shape[2] >= 5:
                fire_index = self.compute_fire_index(multispectral)
                # Boost du rouge là où l'indice de feu est élevé
                rgb[:, :, 0] = np.maximum(rgb[:, :, 0], fire_index * 0.5)
            return rgb
            
        elif output_channels == 4:
            # RGB + canal d'indice de feu séparé
            fire_index = self.compute_fire_index(multispectral)
            return np.dstack([rgb, fire_index])
            
        else:
            raise ValueError(f"Unsupported output_channels: {output_channels}")
    
    def _normalize_band(self, band: np.ndarray) -> np.ndarray:
        """Normalise une bande spectrale en 0-1."""
        if not self.normalize:
            return band.astype(np.float32)
            
        # Gérer les NaN et Inf
        finite_mask = np.isfinite(band)
        if not finite_mask.any():
            return np.zeros_like(band, dtype=np.float32)
            
        min_val = band[finite_mask].min()
        max_val = band[finite_mask].max()
        
        if max_val <= min_val:
            return np.zeros_like(band, dtype=np.float32)
            
        normalized = (band - min_val) / (max_val - min_val)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        
        return normalized.astype(np.float32)
    
    def to_uint8(self, image: np.ndarray) -> np.ndarray:
        """Convertit une image normalisée en uint8."""
        if image.dtype == np.uint8:
            return image
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)
