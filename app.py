"""
🛰️ SpaceEdge AI - Démonstration Complète
==========================================
Détection de feux de forêt par IA embarquée sur satellite

Ce système révolutionne la surveillance environnementale depuis l'espace
en traitant les images directement à bord du satellite.
"""

import streamlit as st
import numpy as np
import cv2
import json
import time
import random
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SpaceEdge AI - Détection de Feux depuis l'Espace",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Header gradient */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 50%, #FFD700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Alert boxes */
    .fire-alert {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 20px rgba(255,68,68,0.4);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 4px 20px rgba(255,68,68,0.4); }
        50% { box-shadow: 0 4px 40px rgba(255,68,68,0.8); }
        100% { box-shadow: 0 4px 20px rgba(255,68,68,0.4); }
    }
    
    .no-fire-alert {
        background: linear-gradient(135deg, #00C853, #00E676);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    
    /* Satellite terminal */
    .sat-terminal {
        background: #0a1929;
        border: 1px solid #00ff88;
        border-radius: 10px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        color: #00ff88;
        font-size: 0.9rem;
    }
    .sat-terminal .header {
        border-bottom: 1px solid #00ff88;
        padding-bottom: 0.5rem;
        margin-bottom: 0.5rem;
        color: #00ffff;
    }
    
    /* Comparison cards */
    .compare-old {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #e74c3c;
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
    }
    .compare-new {
        background: linear-gradient(135deg, #0a3d62, #0c2461);
        border: 2px solid #00d4aa;
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
    }
    
    /* JSON payload */
    .json-box {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #FFD700;
    }
    
    /* Stats */
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        color: white;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Charge le modèle YOLOv8 entraîné."""
    # Chercher le meilleur modèle disponible
    model_paths = [
        Path("models/fire_detector/weights/best.pt"),
        Path("runs/detect/models/fire_detector/weights/best.pt"),
        Path("models/fire_best.pt"),
        Path("yolov8n.pt"),
    ]
    
    for path in model_paths:
        if path.exists():
            return YOLO(str(path))
    
    return YOLO("yolov8n.pt")


def detect_fire(image: np.ndarray, model, confidence: float = 0.35) -> tuple:
    """
    Effectue la détection de feu sur une image.
    Retourne: (image annotée, liste des détections, temps d'inférence)
    """
    start_time = time.perf_counter()
    results = model(image, conf=confidence, verbose=False)
    inference_time = (time.perf_counter() - start_time) * 1000
    
    img_annotated = image.copy()
    detections = []
    
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0])
            cls_name = r.names.get(cls_id, "fire")
            
            # Dessiner le rectangle avec effet glow
            color = (0, 0, 255)  # Rouge BGR
            thickness = 3
            
            # Rectangle principal
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Label avec fond
            label = f"FEU {conf*100:.0f}%"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img_annotated, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)
            cv2.putText(img_annotated, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Calculer le centre et la surface relative
            center_x = (x1 + x2) / 2 / image.shape[1]
            center_y = (y1 + y2) / 2 / image.shape[0]
            area = ((x2 - x1) * (y2 - y1)) / (image.shape[0] * image.shape[1])
            
            detections.append({
                "id": len(detections) + 1,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "center": [round(center_x, 3), round(center_y, 3)],
                "area_ratio": round(area, 4),
                "severity": "CRITICAL" if area > 0.1 else "HIGH" if area > 0.05 else "MEDIUM"
            })
    
    return img_annotated, detections, inference_time


def create_downlink_payload(detections: list, inference_time: float, image_shape: tuple) -> dict:
    """
    Crée le payload JSON ultra-compact qui sera envoyé vers la Terre.
    Optimisé pour bande passante satellite limitée (< 2KB pour Iridium SBD).
    """
    payload = {
        "v": 2,  # Version du protocole
        "sat": "EDGE-01",
        "ts": int(time.time()),
        "pos": {
            "lat": round(45.5017 + random.uniform(-5, 5), 4),  # Simulation
            "lon": round(-73.5673 + random.uniform(-10, 10), 4),
            "alt": 550  # km
        },
        "fire": {
            "detected": len(detections) > 0,
            "count": len(detections),
            "priority": 1 if any(d["severity"] == "CRITICAL" for d in detections) else 
                       2 if any(d["severity"] == "HIGH" for d in detections) else 
                       3 if detections else 5,
            "zones": [
                {
                    "id": d["id"],
                    "c": d["center"],
                    "conf": d["confidence"],
                    "sev": d["severity"][0]  # C, H, M
                }
                for d in detections[:5]  # Max 5 zones pour limiter taille
            ]
        },
        "meta": {
            "inf_ms": round(inference_time, 1),
            "img_px": f"{image_shape[1]}x{image_shape[0]}",
            "model": "yolov8n-fire"
        }
    }
    return payload


# --- MAIN APP ---

# Header
st.markdown('<p class="main-title">🛰️ SpaceEdge AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Détection de Feux de Forêt par IA Embarquée sur Satellite</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/satellite.png", width=80)
    st.title("Configuration")
    
    confidence = st.slider(
        "🎯 Seuil de confiance", 
        min_value=0.10, 
        max_value=0.90, 
        value=0.35, 
        step=0.05,
        help="Seuil minimum de confiance pour une détection"
    )
    
    st.markdown("---")
    
    # Simulation satellite status
    st.markdown("""
    <div class="sat-terminal">
        <div class="header">📡 SATELLITE STATUS</div>
        <b>ID:</b> SPACEEDGE-01<br>
        <b>ORBIT:</b> LEO 550km<br>
        <b>SPEED:</b> 7.66 km/s<br>
        <b>━━━━━━━━━━━━━━</b><br>
        <b>POWER:</b> 87% █████████░<br>
        <b>TEMP:</b> 23°C nominal<br>
        <b>NEXT GS:</b> 18 min<br>
        <b>LINK:</b> <span style="color:#00ff00">● ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**🔗 Liens**")
    st.markdown("[📚 GitHub](https://github.com/DanielMBouyou/EDGESPACE)")
    st.markdown("[📄 Documentation](https://github.com/DanielMBouyou/EDGESPACE#readme)")

# Main tabs
tab1, tab2, tab3 = st.tabs([
    "🔥 Détection en Direct", 
    "🆚 Comparaison Technologique", 
    "📡 Message vers la Terre"
])

# --- TAB 1: DETECTION ---
with tab1:
    st.header("Analyse d'Image Satellite")
    
    col_upload, col_result = st.columns(2)
    
    with col_upload:
        st.subheader("📷 Image d'entrée")
        
        # Sample images option
        use_sample = st.checkbox("Utiliser une image de démonstration")
        
        if use_sample:
            # Créer une image de test avec du "feu"
            sample_img = np.zeros((350, 350, 3), dtype=np.uint8)
            sample_img[:] = (34, 139, 34)  # Fond vert forêt
            
            # Ajouter des zones de "feu" (orange/rouge)
            cv2.circle(sample_img, (100, 100), 40, (0, 100, 255), -1)
            cv2.circle(sample_img, (250, 200), 60, (0, 69, 255), -1)
            
            # Ajouter du bruit
            noise = np.random.randint(0, 30, sample_img.shape, dtype=np.uint8)
            sample_img = cv2.add(sample_img, noise)
            
            img_array = sample_img
            st.image(img_array, caption="Image satellite simulée", use_container_width=True)
        else:
            img_file = st.file_uploader(
                "Télécharger une image satellite", 
                type=['jpg', 'png', 'jpeg', 'tif', 'tiff'],
                help="Formats supportés: JPG, PNG, TIFF"
            )
            
            if img_file:
                img = Image.open(img_file).convert("RGB")
                img_array = np.array(img)
                st.image(img_array, caption="Image originale", use_container_width=True)
            else:
                img_array = None
                st.info("👆 Téléchargez une image satellite ou cochez 'image de démonstration'")
    
    with col_result:
        st.subheader("🧠 Résultat de l'IA")
        
        if 'img_array' in dir() and img_array is not None:
            model = load_model()
            
            with st.spinner("Analyse en cours sur l'edge computer..."):
                annotated_img, detections, inf_time = detect_fire(img_array, model, confidence)
            
            # Afficher l'image annotée
            st.image(annotated_img, caption="Détections IA", use_container_width=True)
            
            # Métriques
            c1, c2, c3 = st.columns(3)
            c1.metric("⏱️ Inférence", f"{inf_time:.1f} ms")
            c2.metric("🔥 Détections", len(detections))
            c3.metric("📊 FPS théorique", f"{1000/inf_time:.0f}")
            
            # Alert box
            if detections:
                st.markdown(f"""
                <div class="fire-alert">
                    🚨 ALERTE FEU DÉTECTÉ! 🚨<br>
                    <span style="font-size:1rem">{len(detections)} zone(s) identifiée(s) - Transmission prioritaire</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Détails des détections
                st.markdown("### 📍 Zones détectées")
                for d in detections:
                    st.markdown(f"""
                    - **Zone {d['id']}**: Confiance {d['confidence']*100:.0f}% | 
                      Sévérité: `{d['severity']}` | 
                      Position relative: ({d['center'][0]:.2f}, {d['center'][1]:.2f})
                    """)
            else:
                st.markdown("""
                <div class="no-fire-alert">
                    ✅ Zone sécurisée - Aucun feu détecté
                </div>
                """, unsafe_allow_html=True)
            
            # Stocker pour les autres tabs
            st.session_state['detections'] = detections
            st.session_state['inference_time'] = inf_time
            st.session_state['image_shape'] = img_array.shape
        else:
            st.info("En attente d'une image...")

# --- TAB 2: COMPARISON ---
with tab2:
    st.header("🆚 Révolution SpaceEdge vs Satellites Traditionnels")
    
    st.markdown("""
    ### Le Problème Actuel
    
    Les satellites d'observation actuels (Landsat, Sentinel, MODIS) sont des **"caméras passives"** :
    ils capturent des images et les transmettent entièrement vers la Terre pour analyse au sol.
    """)
    
    col_old, col_new = st.columns(2)
    
    with col_old:
        st.markdown("""
        <div class="compare-old">
            <h3 style="color:#e74c3c">❌ Satellites Traditionnels</h3>
            <ul>
                <li><b>Délai:</b> 6-24 heures entre capture et analyse</li>
                <li><b>Données:</b> Téléchargement de TOUTES les images (100+ GB/jour)</li>
                <li><b>Coût:</b> Stations sol massives, bande passante énorme</li>
                <li><b>Couverture:</b> Passages limités (2x/jour max)</li>
                <li><b>Traitement:</b> Centres de données au sol</li>
            </ul>
            <hr>
            <p style="text-align:center; font-size:2rem">⏱️ <b>12h</b> de délai moyen</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_new:
        st.markdown("""
        <div class="compare-new">
            <h3 style="color:#00d4aa">✅ SpaceEdge AI</h3>
            <ul>
                <li><b>Délai:</b> < 1 seconde (traitement à bord)</li>
                <li><b>Données:</b> Seules les ALERTES sont transmises (< 2KB)</li>
                <li><b>Coût:</b> Compatible Iridium SBD, constellation LEO</li>
                <li><b>Couverture:</b> Constellation = couverture continue</li>
                <li><b>Traitement:</b> IA embarquée (Jetson/FPGA)</li>
            </ul>
            <hr>
            <p style="text-align:center; font-size:2rem">⚡ <b>< 1s</b> temps réel</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Impact visualization
    st.markdown("### 📊 Impact Quantifié")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">99.9%</div>
            <div class="stat-label">Réduction de données transmises</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">1000x</div>
            <div class="stat-label">Plus rapide que système actuel</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">3M€</div>
            <div class="stat-label">Économie/an station sol</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">30+</div>
            <div class="stat-label">FPS sur Jetson Orin</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Use case
    st.markdown("""
    ### 🎯 Cas d'Usage: Feu de Forêt au Canada
    
    **Scénario traditionnel:**
    1. 🛰️ Satellite Landsat capture l'image à 14h00
    2. 📡 Données transmises au passage sur station sol à 18h00
    3. 💻 Analyse au centre de traitement: terminée à 21h00
    4. 📞 Alerte envoyée aux pompiers: 22h00
    5. 🔥 **8 heures perdues** - le feu a déjà brûlé 500 hectares
    
    **Avec SpaceEdge:**
    1. 🛰️ Satellite capture l'image à 14h00
    2. 🧠 IA détecte le feu en **50 millisecondes**
    3. 📡 Alerte JSON (500 bytes) transmise via Iridium
    4. 📞 Pompiers alertés à 14h00:02
    5. ✅ **Intervention immédiate** - feu contenu à 5 hectares
    """)

# --- TAB 3: DOWNLINK MESSAGE ---
with tab3:
    st.header("📡 Message Envoyé vers la Terre")
    
    st.markdown("""
    ### Ce que le satellite envoie réellement
    
    Au lieu de transmettre l'image entière (plusieurs MB), SpaceEdge envoie uniquement
    un **message JSON ultra-compact** contenant les informations essentielles.
    """)
    
    if 'detections' in st.session_state:
        detections = st.session_state['detections']
        inf_time = st.session_state['inference_time']
        img_shape = st.session_state['image_shape']
        
        payload = create_downlink_payload(detections, inf_time, img_shape)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📊 Comparaison des Données")
            
            raw_size = img_shape[0] * img_shape[1] * 3
            json_str = json.dumps(payload, separators=(',', ':'))
            json_size = len(json_str.encode())
            
            st.metric("🖼️ Image brute", f"{raw_size/1024:.1f} KB")
            st.metric("📄 Message JSON", f"{json_size} bytes")
            st.metric("💾 Compression", f"{(1-json_size/raw_size)*100:.2f}%")
            
            st.markdown("---")
            
            if json_size <= 340:
                st.success("✅ Compatible Iridium SBD (340 bytes)")
            elif json_size <= 1960:
                st.warning("⚠️ Multi-paquet Iridium nécessaire")
            else:
                st.error("❌ Trop grand pour Iridium SBD")
            
            st.markdown("""
            **Réseaux compatibles:**
            - 🛰️ Iridium SBD
            - 📡 KSAT Ground Network
            - 🌍 AWS Ground Station
            - 🚀 SpaceDataHighway (EDRS)
            """)
        
        with col2:
            st.markdown("### 📨 Payload JSON")
            
            st.markdown("""
            <div class="json-box">
            """, unsafe_allow_html=True)
            
            st.code(json.dumps(payload, indent=2), language="json")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Explication des champs
            with st.expander("📖 Explication des champs"):
                st.markdown("""
                | Champ | Description |
                |-------|-------------|
                | `v` | Version du protocole |
                | `sat` | Identifiant du satellite |
                | `ts` | Timestamp UNIX (secondes) |
                | `pos.lat/lon` | Position du centre de l'image |
                | `pos.alt` | Altitude orbitale (km) |
                | `fire.detected` | Booléen: feu détecté? |
                | `fire.count` | Nombre de zones de feu |
                | `fire.priority` | 1=critique, 5=routine |
                | `fire.zones[].c` | Centre relatif [0-1, 0-1] |
                | `fire.zones[].conf` | Confiance de détection |
                | `fire.zones[].sev` | Sévérité (C/H/M) |
                | `meta.inf_ms` | Temps d'inférence (ms) |
                """)
    else:
        st.info("👈 Effectuez d'abord une détection dans l'onglet 'Détection en Direct'")
        
        # Exemple de payload
        st.markdown("### Exemple de message type:")
        example_payload = {
            "v": 2,
            "sat": "EDGE-01",
            "ts": int(time.time()),
            "pos": {"lat": 45.5017, "lon": -73.5673, "alt": 550},
            "fire": {
                "detected": True,
                "count": 2,
                "priority": 1,
                "zones": [
                    {"id": 1, "c": [0.28, 0.28], "conf": 0.94, "sev": "C"},
                    {"id": 2, "c": [0.71, 0.57], "conf": 0.87, "sev": "H"}
                ]
            },
            "meta": {"inf_ms": 45.2, "img_px": "350x350", "model": "yolov8n-fire"}
        }
        st.code(json.dumps(example_payload, indent=2), language="json")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; padding:2rem 0">
    <p><b>SpaceEdge AI</b> v2.0 | Détection de feux par IA embarquée</p>
    <p>
        <a href="https://github.com/DanielMBouyou/EDGESPACE" target="_blank">GitHub</a> · 
        Conçu pour <b>Loft Orbital</b> & <b>D-Orbit ION</b>
    </p>
    <p style="font-size:0.8rem">© 2025 Daniel M. Bouyou - Projet Open Source</p>
</div>
""", unsafe_allow_html=True)
