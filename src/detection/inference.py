"""
Moteur d'inférence optimisé pour edge computing.
Supporte YOLOv8 avec optimisations TensorRT.
"""

import time
import numpy as np
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False


@dataclass
class InferenceConfig:
    """Configuration pour l'inférence."""
    confidence_threshold: float = 0.35
    nms_threshold: float = 0.45
    max_detections: int = 100
    device: str = "cpu"  # "cpu", "cuda", "cuda:0"
    half_precision: bool = False  # FP16
    dynamic_batch: bool = False


class InferenceEngine:
    """
    Moteur d'inférence abstrait pour modèles de détection.
    
    Supporte:
    - YOLOv8 (Ultralytics)
    - TensorRT engines (.engine)
    - ONNX Runtime (TODO)
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config: Optional[InferenceConfig] = None,
    ):
        """
        Args:
            model_path: Chemin vers le modèle (.pt, .engine, .onnx)
            config: Configuration d'inférence
        """
        self.model_path = Path(model_path)
        self.config = config or InferenceConfig()
        self.model = None
        self.model_type: str = ""
        
        self._load_model()
        
    def _load_model(self) -> None:
        """Charge le modèle en fonction de son extension."""
        suffix = self.model_path.suffix.lower()
        
        if suffix == ".pt":
            self._load_pytorch()
        elif suffix == ".engine":
            self._load_tensorrt()
        elif suffix == ".onnx":
            self._load_onnx()
        else:
            raise ValueError(f"Unsupported model format: {suffix}")
            
    def _load_pytorch(self) -> None:
        """Charge un modèle PyTorch (YOLO)."""
        if not HAS_ULTRALYTICS:
            raise ImportError("ultralytics not installed")
            
        self.model = YOLO(str(self.model_path))
        self.model_type = "yolo"
        
        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        
    def _load_tensorrt(self) -> None:
        """Charge un engine TensorRT."""
        if not HAS_TENSORRT:
            raise ImportError("tensorrt not installed")
        # TODO: Implémenter le chargement TensorRT
        raise NotImplementedError("TensorRT loading not yet implemented")
        
    def _load_onnx(self) -> None:
        """Charge un modèle ONNX."""
        # TODO: Implémenter le chargement ONNX Runtime
        raise NotImplementedError("ONNX loading not yet implemented")
        
    def infer(
        self,
        image: np.ndarray,
        return_raw: bool = False,
    ) -> tuple[list[dict], float]:
        """
        Effectue l'inférence sur une image.
        
        Args:
            image: Image RGB (H, W, 3)
            return_raw: Retourner les résultats bruts
            
        Returns:
            (detections, inference_time_ms)
        """
        start = time.perf_counter()
        
        if self.model_type == "yolo":
            results = self.model(
                image,
                conf=self.config.confidence_threshold,
                iou=self.config.nms_threshold,
                max_det=self.config.max_detections,
                verbose=False,
                device=self.config.device,
                half=self.config.half_precision,
            )
            
            inference_time = (time.perf_counter() - start) * 1000
            
            if return_raw:
                return results, inference_time
                
            # Parser les résultats
            detections = []
            for r in results:
                for box in r.boxes:
                    det = {
                        "class_id": int(box.cls[0]),
                        "class_name": self.model.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist(),
                    }
                    detections.append(det)
                    
            return detections, inference_time
        else:
            raise NotImplementedError(f"Inference not implemented for {self.model_type}")
            
    def get_class_names(self) -> dict[int, str]:
        """Retourne le mapping des classes."""
        if self.model_type == "yolo":
            return self.model.names
        return {}
    
    def export_tensorrt(
        self,
        output_path: Optional[Path] = None,
        half: bool = True,
        int8: bool = False,
        workspace_gb: int = 4,
    ) -> Path:
        """
        Exporte le modèle en TensorRT.
        
        Args:
            output_path: Chemin de sortie
            half: Utiliser FP16
            int8: Utiliser INT8 (nécessite calibration)
            workspace_gb: Mémoire workspace GPU en GB
            
        Returns:
            Chemin vers le fichier .engine
        """
        if self.model_type != "yolo":
            raise ValueError("Export only supported for YOLO models")
            
        if output_path is None:
            output_path = self.model_path.with_suffix(".engine")
            
        self.model.export(
            format="engine",
            half=half,
            int8=int8,
            workspace=workspace_gb,
        )
        
        return output_path
    
    def benchmark(
        self,
        image_size: tuple[int, int] = (640, 640),
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> dict:
        """
        Benchmark les performances du modèle.
        
        Args:
            image_size: Taille de l'image de test
            num_runs: Nombre d'itérations
            warmup_runs: Nombre d'itérations de warmup
            
        Returns:
            Statistiques de performance
        """
        dummy = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(warmup_runs):
            self.infer(dummy)
            
        # Benchmark
        times = []
        for _ in range(num_runs):
            _, t = self.infer(dummy)
            times.append(t)
            
        times = np.array(times)
        
        return {
            "mean_ms": float(times.mean()),
            "std_ms": float(times.std()),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
            "fps": float(1000 / times.mean()),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
        }
