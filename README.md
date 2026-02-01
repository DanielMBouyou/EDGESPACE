# SpaceEdge AI 🛰️🔥

**Détection rapide des feux de forêt à partir d'images satellites avec YOLOv8 et edge computing spatial.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Objectif

Système de détection de feux de forêt **en temps réel** à bord de satellites.

### Comparaison avec l'approche traditionnelle

| Métrique | Bent Pipe (classique) | **SpaceEdge AI** |
|----------|----------------------|------------------|
| Volume downlink | 100% des images | **< 1%** (alertes JSON) |
| Temps de réaction | 4-24 heures | **< 30 minutes** |
| Coût downlink | Très élevé (€/Go) | Minimal (€/message) |

## 🚀 Plateformes Cibles

- **Loft Orbital**: NVIDIA Jetson AGX Orin durci - 30+ FPS
- **D-Orbit ION**: Unibap iX5-100 / Xilinx Zynq - 15+ FPS

## 📁 Structure

```
├── app.py                  # Demo Streamlit
├── models/fire_best.pt     # Modèle entraîné
├── src/                    # Code source
│   ├── config.py          # Configuration
│   ├── preprocessing/     # Cloud masking, spectral
│   ├── detection/         # Détecteur YOLO
│   └── training/          # Entraînement
└── scripts/               # Scripts CLI
```

## 🛠️ Installation

```bash
git clone https://github.com/DanielMBouyou/EDGESPACE.git
cd EDGESPACE
uv sync  # ou pip install -e .
```

## 🏋️ Utilisation

```bash
# Demo
streamlit run app.py

# Entraîner
python scripts/train.py --epochs 100

# Benchmark
python scripts/benchmark.py --platform jetson_nano

# Export TensorRT
python scripts/export.py --format engine --int8
```

## ⚡ Pipeline

```
Image → Cloud Mask → YOLOv8-nano → JSON Alert → Downlink
```

## 📡 Format d'Alerte

```json
{"sat":"EDGE-SAT-01","ts":1706745600,"p":1,"det":[{"cls":0,"conf":0.92,"box":[120,80,180,140]}]}
```

## 📈 Performance

| Plateforme | FPS | Latence |
|------------|-----|---------|
| Jetson Orin | 45 | 28ms |
| Jetson Nano | 12 | 95ms |

## 📜 License

MIT - **DanielMBouyou**
