"""
Data augmentation spécifique pour la détection de feux.
Optimisé pour images satellites avec variations atmosphériques.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
import random


@dataclass
class AugmentationConfig:
    """Configuration des augmentations."""
    
    # Probabilités d'application
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    rotation_prob: float = 0.3
    
    # Transformations géométriques
    max_rotation_angle: float = 15.0  # degrés
    scale_range: Tuple[float, float] = (0.8, 1.2)
    
    # Transformations colorimétriques
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.7, 1.3)
    hue_shift_range: float = 10.0  # degrés
    
    # Simulation atmosphérique
    haze_prob: float = 0.2
    haze_intensity_range: Tuple[float, float] = (0.1, 0.3)
    
    # Bruit
    noise_prob: float = 0.15
    noise_std_range: Tuple[float, float] = (5, 20)
    
    # Cutout / mosaic
    cutout_prob: float = 0.1
    cutout_size_ratio: float = 0.2


class FireAugmentation:
    """
    Augmentation de données pour détection de feux satellites.
    
    Inclut des transformations spécifiques:
    - Simulation de brume/haze atmosphérique
    - Variations d'illumination (angle solaire)
    - Bruit de capteur satellite
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Applique les augmentations sur une image et ses bounding boxes.
        
        Args:
            image: Image BGR/RGB (H, W, 3) uint8
            bboxes: Bounding boxes (N, 5) format [class_id, x_center, y_center, w, h] normalisé
            
        Returns:
            (image_augmentée, bboxes_transformées)
        """
        img = image.copy()
        boxes = bboxes.copy() if bboxes is not None else None
        
        # Flips
        img, boxes = self._horizontal_flip(img, boxes)
        img, boxes = self._vertical_flip(img, boxes)
        
        # Rotation (avec transformation des boxes)
        img, boxes = self._random_rotation(img, boxes)
        
        # Transformations colorimétriques
        img = self._color_jitter(img)
        
        # Effets atmosphériques
        img = self._add_haze(img)
        
        # Bruit
        img = self._add_noise(img)
        
        # Cutout
        img = self._cutout(img)
        
        return img, boxes
    
    def _horizontal_flip(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Flip horizontal."""
        if random.random() > self.config.horizontal_flip_prob:
            return image, bboxes
            
        image = cv2.flip(image, 1)
        if bboxes is not None and len(bboxes) > 0:
            bboxes = bboxes.copy()
            bboxes[:, 1] = 1.0 - bboxes[:, 1]  # x_center = 1 - x_center
            
        return image, bboxes
    
    def _vertical_flip(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Flip vertical."""
        if random.random() > self.config.vertical_flip_prob:
            return image, bboxes
            
        image = cv2.flip(image, 0)
        if bboxes is not None and len(bboxes) > 0:
            bboxes = bboxes.copy()
            bboxes[:, 2] = 1.0 - bboxes[:, 2]  # y_center = 1 - y_center
            
        return image, bboxes
    
    def _random_rotation(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Rotation aléatoire."""
        if random.random() > self.config.rotation_prob:
            return image, bboxes
            
        angle = random.uniform(
            -self.config.max_rotation_angle,
            self.config.max_rotation_angle
        )
        
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        
        # Matrice de rotation
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Appliquer la rotation
        image = cv2.warpAffine(image, M, (w, h), borderValue=(114, 114, 114))
        
        # Transformer les bounding boxes
        if bboxes is not None and len(bboxes) > 0:
            # TODO: Implémentation complète de la transformation des boxes
            # Pour l'instant, on garde les boxes telles quelles (approximation)
            pass
            
        return image, bboxes
    
    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        """Variations de couleur."""
        # Brightness
        brightness = random.uniform(*self.config.brightness_range)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        # Contrast
        contrast = random.uniform(*self.config.contrast_range)
        mean = image.mean()
        image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        # Saturation & Hue (dans HSV)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        saturation = random.uniform(*self.config.saturation_range)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        
        hue_shift = random.uniform(
            -self.config.hue_shift_range,
            self.config.hue_shift_range
        )
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return image
    
    def _add_haze(self, image: np.ndarray) -> np.ndarray:
        """Simule la brume atmosphérique."""
        if random.random() > self.config.haze_prob:
            return image
            
        intensity = random.uniform(*self.config.haze_intensity_range)
        
        # Couleur de la brume (gris-bleuté)
        haze_color = np.array([200, 210, 220], dtype=np.float32)
        
        # Mélanger avec l'image
        image = image.astype(np.float32)
        image = image * (1 - intensity) + haze_color * intensity
        
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Ajoute du bruit de capteur."""
        if random.random() > self.config.noise_prob:
            return image
            
        std = random.uniform(*self.config.noise_std_range)
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        
        image = image.astype(np.float32) + noise
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def _cutout(self, image: np.ndarray) -> np.ndarray:
        """Applique un cutout aléatoire."""
        if random.random() > self.config.cutout_prob:
            return image
            
        h, w = image.shape[:2]
        size = int(min(h, w) * self.config.cutout_size_ratio)
        
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        
        image[y:y+size, x:x+size] = 114  # Gris YOLO
        
        return image


class MosaicAugmentation:
    """
    Augmentation Mosaic pour améliorer la détection de petits objets.
    Combine 4 images en une seule.
    """
    
    def __init__(self, image_size: int = 640):
        self.image_size = image_size
        
    def __call__(
        self,
        images: list[np.ndarray],
        bboxes_list: list[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée une image mosaic à partir de 4 images.
        
        Args:
            images: 4 images
            bboxes_list: 4 listes de bounding boxes
            
        Returns:
            (image_mosaic, bboxes_combinées)
        """
        assert len(images) == 4 and len(bboxes_list) == 4
        
        s = self.image_size
        mosaic = np.full((s, s, 3), 114, dtype=np.uint8)
        
        # Point central aléatoire
        cx = random.randint(s // 4, 3 * s // 4)
        cy = random.randint(s // 4, 3 * s // 4)
        
        all_bboxes = []
        
        # Placements: top-left, top-right, bottom-left, bottom-right
        placements = [
            (0, 0, cx, cy),
            (cx, 0, s, cy),
            (0, cy, cx, s),
            (cx, cy, s, s),
        ]
        
        for i, (x1, y1, x2, y2) in enumerate(placements):
            img = images[i]
            h, w = img.shape[:2]
            
            # Redimensionner pour remplir la zone
            new_w = x2 - x1
            new_h = y2 - y1
            img = cv2.resize(img, (new_w, new_h))
            
            mosaic[y1:y2, x1:x2] = img
            
            # Transformer les bounding boxes
            if bboxes_list[i] is not None and len(bboxes_list[i]) > 0:
                boxes = bboxes_list[i].copy()
                
                # Dénormaliser puis ajuster puis renormaliser
                boxes[:, 1] = (boxes[:, 1] * new_w + x1) / s
                boxes[:, 2] = (boxes[:, 2] * new_h + y1) / s
                boxes[:, 3] = boxes[:, 3] * new_w / s
                boxes[:, 4] = boxes[:, 4] * new_h / s
                
                all_bboxes.append(boxes)
        
        combined_bboxes = np.concatenate(all_bboxes, axis=0) if all_bboxes else np.array([])
        
        return mosaic, combined_bboxes
