# EDGESPACE

Fast wildfire detection from satellite imagery with YOLOv8 and a Streamlit demo.

## Features
- Streamlit demo to test a satellite image
- YOLOv8n model trained quickly (file `models/fire_best.pt`)
- Scripts to prepare a TS-SatFire subset and retrain

## Quickstart
```powershell
cd C:\Users\danie\OneDrive\Documents\IA\space-edge-fire-detector
python -m pip install uv
python -m uv sync
python -m uv run streamlit run app.py
```

## Model
- The app loads `models/fire_best.pt` if it exists.
- Otherwise it falls back to `yolov8n.pt`.

## Sample images (local only)
Kaggle dataset files cannot be redistributed, so sample images are generated locally.

```powershell
python -m uv run python scripts\export_sample_images.py
```

## Fast training (optional)
Prereq: a valid Kaggle token in `C:\Users\danie\.kaggle\kaggle.json`.

```powershell
python -m uv run python scripts\prepare_ts_satfire_yolo.py --max-pairs 20 --fire-band 2 --fire-threshold 0.1 --min-area 1
python -m uv run python scripts\train_fire_yolo.py --epochs 3 --batch 4 --imgsz 512
```

## Notes
- `data/` and `runs/` are not versioned.
- The TS-SatFire dataset is large and is not included in the repo.
