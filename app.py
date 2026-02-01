import streamlit as st
from ultralytics import YOLO
import cv2
import json
import time
from PIL import Image
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
st.set_page_config(page_title="SpaceEdge AI - Fire Detection", layout="wide")
st.title("🛰️ SpaceEdge AI : Démo Onboard Loft Orbital/D-Orbit")

# Simulation du matériel embarqué
device = "cpu" # Sur le satellite, ce serait 'cuda' (Nvidia Jetson)

@st.cache_resource
def load_model():
    # On télécharge un modèle pré-entraîné spécifique au feu (poids légers ~6MB)
    custom = Path("models/fire_best.pt")
    return YOLO(str(custom) if custom.exists() else "yolov8n.pt") # Pour la démo, on utilise nano

model = load_model()

col1, col2 = st.columns(2)

with col1:
    st.header("📷 Flux Satellite (Entrée)")
    img_file = st.file_uploader("Uploader une image satellite (TIF/JPG)", type=['jpg', 'png', 'jpeg'])
    
    if img_file:
        img = Image.open(img_file)
        st.image(img, caption="Image brute acquise par le capteur optique", use_container_width=True)

with col2:
    st.header("🧠 Traitement Edge AI")
    if img_file:
        # 1. Inférence
        start_time = time.time()
        results = model(img)
        end_time = time.time()
        
        # 2. Simulation de détection (Force une détection pour la démo si besoin)
        # On va parser les résultats
        detections = []
        for r in results:
            for box in r.boxes:
                # Simuler une détection de feu (Class 0 par défaut pour le test)
                conf = float(box.conf[0])
                if conf > 0.25:
                    coords = box.xyxy[0].tolist()
                    detections.append({"label": "FIRE_DETECTED", "conf": conf, "bbox": coords})

        st.write(f"**Temps d'inférence onboard :** {(end_time - start_time)*1000:.2f} ms")
        
        # Affichage de l'image analysée
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Analyse IA en temps réel", use_container_width=True)

st.header("📡 Downlink : Alerte Temps Réel (Format JSON Iridium/Viasat)")

if img_file and detections:
    # C'est ici que la magie opère : on n'envoie PAS l'image, juste ça :
    alert_payload = {
        "sat_id": "LOFT-YAM-01",
        "timestamp": time.time(),
        "alerts": detections,
        "status": "CRITICAL",
        "priority": 1
    }
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric("Poids des données brutes", "2.4 MB")
        st.metric("Poids de l'alerte Downlink", f"{len(json.dumps(alert_payload))} Bytes", delta="-99.9%")
    
    with col_b:
        st.code(json.dumps(alert_payload, indent=2), language="json")
        st.success("✅ Alerte envoyée via Viasat IDRS (Latence estimée : 22s)")
else:
    st.info("Aucune anomalie détectée. Consommation downlink : 0 octets.")
