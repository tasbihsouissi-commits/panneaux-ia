"""Fine-tuning de YOLO11 sur GTSDB (après prepare_gtsdb.py)."""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="datasets/gtsdb/data.yaml")
ap.add_argument("--model", default="yolo11n.pt", help="yolo11n.pt, yolo11s.pt, ...")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--imgsz", type=int, default=1280, help="panneaux petits : garder une grande résolution")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--name", default="gtsdb_yolo11")
a = ap.parse_args()

model = YOLO(a.model)  # préentraîné COCO
model.train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, name=a.name,
    fliplr=0.0, flipud=0.0,      # un flip change le sens des panneaux (gauche/droite)
    degrees=5, scale=0.5,        # petites rotations, variations d'échelle
    hsv_v=0.5, hsv_s=0.6,        # robustesse luminosité / contre-jour
    patience=30,
)
