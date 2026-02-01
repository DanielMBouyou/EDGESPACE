"""
Normalisation des images pour l'inférence du modèle.
Optimisé pour le traitement edge avec faible latence.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class ImageNormalizer:
    """
    Normalisation et préparation des images pour YOLOv8.
    Optimisé pour une exécution rapide sur edge devices.
    """
    
    def __init__(
        self,
        target_size: int = 640,
        normalize_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        normalize_std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        pad_value: int = 114,  # Gris YOLO par défaut
    ):
        """
        Args:
            target_size: Taille cible (carrée)
            normalize_mean: Moyenne pour normalisation (par canal)
            normalize_std: Écart-type pour normalisation
            pad_value: Valeur de padding pour letterbox
        """
        self.target_size = target_size
        self.normalize_mean = np.array(normalize_mean, dtype=np.float32)
        self.normalize_std = np.array(normalize_std, dtype=np.float32)
        self.pad_value = pad_value
        
    def letterbox(
        self,
        image: np.ndarray,
        auto: bool = False,
        scale_fill: bool = False,
        scale_up: bool = True,
        stride: int = 32,
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Redimensionne l'image avec padding pour maintenir le ratio.
        
        Args:
            image: Image source (H, W, C)
            auto: Ajuster au stride le plus proche
            scale_fill: Étirer sans padding
            scale_up: Autoriser l'agrandissement
            stride: Stride du modèle
            
        Returns:
            (image_padded, ratio, (pad_w, pad_h))
        """
        shape = image.shape[:2]  # H, W
        new_shape = (self.target_size, self.target_size)
        
        # Calculer le ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scale_up:
            r = min(r, 1.0)
            
        # Nouvelle taille non-paddée
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        
        # Padding nécessaire
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        
        if auto:
            dw = dw % stride
            dh = dh % stride
        elif scale_fill:
            dw, dh = 0, 0
            new_unpad = new_shape[::-1]
            r = new_shape[1] / shape[1], new_shape[0] / shape[0]
            
        # Diviser le padding
        dw /= 2
        dh /= 2
        
        # Redimensionner
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        # Ajouter le padding
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(
            image,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=(self.pad_value, self.pad_value, self.pad_value)
        )
        
        return image, r, (int(dw), int(dh))
    
    def preprocess(
        self,
        image: np.ndarray,
        to_tensor: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """
        Pipeline complet de prétraitement.
        
        Args:
            image: Image BGR/RGB (H, W, C)
            to_tensor: Convertir au format tensor (C, H, W)
            
        Returns:
            (image_processed, metadata)
        """
        original_shape = image.shape[:2]
        
        # Letterbox
        img, ratio, pad = self.letterbox(image)
        
        # BGR -> RGB si nécessaire (YOLO attend RGB)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        # Normaliser en float32
        img = img.astype(np.float32) / 255.0
        
        # Appliquer mean/std si configuré
        if np.any(self.normalize_mean != 0) or np.any(self.normalize_std != 1):
            img = (img - self.normalize_mean) / self.normalize_std
            
        # Convertir en format tensor (C, H, W) si demandé
        if to_tensor:
            img = img.transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            
        metadata = {
            "original_shape": original_shape,
            "ratio": ratio,
            "pad": pad,
            "processed_shape": img.shape,
        }
        
        return img, metadata
    
    def postprocess_boxes(
        self,
        boxes: np.ndarray,
        metadata: dict,
    ) -> np.ndarray:
        """
        Convertit les coordonnées des boîtes vers l'espace image original.
        
        Args:
            boxes: Boîtes (N, 4) en format xyxy
            metadata: Métadonnées du prétraitement
            
        Returns:
            Boîtes dans les coordonnées originales
        """
        if len(boxes) == 0:
            return boxes
            
        ratio = metadata["ratio"]
        pad = metadata["pad"]
        original_shape = metadata["original_shape"]
        
        # Retirer le padding
        boxes = boxes.copy()
        boxes[:, [0, 2]] -= pad[0]  # x
        boxes[:, [1, 3]] -= pad[1]  # y
        
        # Diviser par le ratio
        boxes /= ratio
        
        # Clipper aux limites de l'image
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_shape[0])
        
        return boxes
