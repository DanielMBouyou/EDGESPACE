"""
SpaceEdge AI - Application de démonstration Streamlit
Détection de feux de forêt optimisée pour edge computing spatial

Simulateur des environnements:
- Loft Orbital (NVIDIA Jetson via Hubble Interface)
- D-Orbit ION (Unibap iX5-100 / Xilinx Zynq)
"""

import streamlit as st
import numpy as np
import cv2
import json
import time
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# --- CONFIGURATION ---
st.set_page_config(
    page_title="SpaceEdge AI - Wildfire Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #ff6b6b, #ffa500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .alert-critical {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
    }
    .alert-warning {
        background: linear-gradient(135deg, #ffaa00, #ff8800);
        color: white;
        padding: 1rem;
        border-radius: 10px;
    }
    .satellite-info {
        background: #0d1b2a;
        color: #00ff88;
        font-family: monospace;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #00ff88;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Charge le modèle YOLOv8."""
    model_path = Path("models/fire_best.pt")
    if not model_path.exists():
        model_path = Path("yolov8n.pt")
    return YOLO(str(model_path))


def draw_detections(image: np.ndarray, results) -> tuple[np.ndarray, list]:
    """Dessine les détections et retourne les infos."""
    img = image.copy()
    detections = []
    
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.25:
                continue
                
            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0])
            cls_name = r.names.get(cls_id, "fire")
            
            # Couleur
            color = (0, 0, 255) if "fire" in cls_name.lower() else (128, 128, 128)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            detections.append({
                "cls": cls_id,
                "name": cls_name,
                "conf": round(conf, 3),
                "box": [x1, y1, x2, y2],
                "area": (x2 - x1) * (y2 - y1)
            })
    
    return img, detections


# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Configuration Edge")
    
    platform = st.selectbox(
        "🖥️ Plateforme",
        ["CPU (Demo)", "Jetson Nano", "Jetson Orin", "CUDA"],
    )
    
    confidence = st.slider("Seuil de confiance", 0.1, 0.9, 0.35, 0.05)
    
    st.subheader("📡 Réseau Downlink")
    network = st.selectbox(
        "Réseau",
        ["Iridium SBD", "KSAT Ground", "LeafSpace", "SpaceDataHighway"],
    )
    
    st.markdown("---")
    st.markdown("""
    <div class="satellite-info">
    SAT_ID: EDGE-SAT-01<br>
    ORBIT: LEO 550km<br>
    POWER: 87% ████████░░<br>
    NEXT_GS: 23min
    </div>
    """, unsafe_allow_html=True)


# --- MAIN ---
st.markdown('<p class="main-header">🛰️ SpaceEdge AI - Wildfire Detection</p>', unsafe_allow_html=True)

st.markdown("""
**Détection en temps réel de feux de forêt depuis l'espace**  
L'IA traite les images à bord et n'envoie que des alertes légères via le downlink.
""")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔥 Détection", "📡 Downlink", "📖 Docs"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📷 Image Satellite")
        img_file = st.file_uploader("Upload image", type=['jpg', 'png', 'jpeg', 'tif'])
        
        if img_file:
            img = Image.open(img_file).convert("RGB")
            img_array = np.array(img)
            st.image(img_array, caption="Image brute", use_container_width=True)
    
    with col2:
        st.header("🧠 Analyse Edge AI")
        
        if img_file:
            model = load_model()
            
            start = time.perf_counter()
            results = model(img_array, conf=confidence, verbose=False)
            inference_time = (time.perf_counter() - start) * 1000
            
            annotated, detections = draw_detections(img_array, results)
            
            # Métriques
            c1, c2, c3 = st.columns(3)
            c1.metric("⏱️ Inférence", f"{inference_time:.1f} ms")
            c2.metric("🔥 Détections", len(detections))
            c3.metric("📈 FPS", f"{1000/inference_time:.1f}")
            
            st.image(annotated, caption="Résultat IA", use_container_width=True)
            
            if detections:
                st.markdown('<div class="alert-critical">🚨 FEU DÉTECTÉ!</div>', unsafe_allow_html=True)

with tab2:
    st.header("📡 Payload Downlink")
    
    if img_file and 'detections' in dir():
        # Créer le payload
        payload = {
            "sat": "EDGE-SAT-01",
            "ts": int(time.time()),
            "p": 1 if detections else 5,
            "n": len(detections),
            "det": detections[:10],
            "meta": {"inf_ms": round(inference_time, 1)}
        }
        
        compact_json = json.dumps(payload, separators=(',', ':'))
        size_bytes = len(compact_json.encode())
        
        c1, c2 = st.columns(2)
        with c1:
            raw_size = img_array.shape[0] * img_array.shape[1] * 3
            st.metric("🖼️ Image brute", f"{raw_size/1024:.1f} KB")
            st.metric("📄 Alerte JSON", f"{size_bytes} bytes")
            st.metric("💾 Réduction", f"-{(1-size_bytes/raw_size)*100:.1f}%")
            
            if size_bytes <= 1960:
                st.success("✅ Compatible Iridium SBD")
        
        with c2:
            st.code(json.dumps(payload, indent=2), language="json")
    else:
        st.info("Effectuez d'abord une détection")

with tab3:
    st.header("📖 Architecture")
    st.markdown("""
    ## Pipeline
    ```
    Image → Cloud Mask → YOLOv8-nano → JSON Alert → Downlink
    ```
    
    ## Plateformes
    | Plateforme | Hardware | FPS cible |
    |------------|----------|-----------|
    | Loft Orbital | Jetson Orin | 30+ |
    | D-Orbit ION | Xilinx Zynq | 15+ |
    
    ## Scripts
    ```bash
    python scripts/train.py --epochs 100
    python scripts/benchmark.py --platform jetson_nano
    python scripts/export.py --format engine --int8
    ```
    """)

st.markdown("---")
st.markdown("<center><small>SpaceEdge AI v2.0 | <a href='https://github.com/DanielMBouyou/EDGESPACE'>GitHub</a></small></center>", unsafe_allow_html=True)
