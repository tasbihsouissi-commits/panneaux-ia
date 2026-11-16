"""Évaluation détection : précision, rappel, mAP50, mAP50-95, mAP par classe, FPS."""
import argparse, json, time
from pathlib import Path
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--data", default="datasets/gtsdb/data.yaml")
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--split", default="test")
a = ap.parse_args()

model = YOLO(a.weights)
r = model.val(data=a.data, split=a.split, imgsz=a.imgsz)
res = {
    "precision": float(r.box.mp), "recall": float(r.box.mr),
    "mAP50": float(r.box.map50), "mAP50-95": float(r.box.map),
}
per_class = {model.names[int(c)]: float(m) for c, m in zip(r.box.ap_class_index, r.box.maps[r.box.ap_class_index])}
res["worst_classes_mAP50-95"] = dict(sorted(per_class.items(), key=lambda kv: kv[1])[:8])

# Vitesse d'inférence (images du split)
import yaml
root = Path(yaml.safe_load(open(a.data))["path"]) / "images" / a.split
files = sorted(root.glob("*.jpg"))[:50]
model.predict(files[0], imgsz=a.imgsz, verbose=False)  # warm-up
t = time.time()
for f in files:
    model.predict(f, imgsz=a.imgsz, verbose=False)
res["latence_ms"] = (time.time() - t) / len(files) * 1000
res["fps"] = 1000 / res["latence_ms"]
print(json.dumps(res, indent=2, ensure_ascii=False))
Path("results_detection.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
