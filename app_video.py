"""Application : détection + reconnaissance de panneaux sur image, vidéo ou webcam.

Exemples :
  python app_video.py --weights best.pt --source 0                      # webcam
  python app_video.py --weights best.pt --source route.mp4 --save out.mp4 --no-show
  python app_video.py --weights best_1classe.pt --classifier classifier_mobilenetv2.pt --source route.mp4
"""
import argparse, time, unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path
import cv2
from ultralytics import YOLO
from signs import ALERT_IDS, ALERT_MSG, CATEGORY_LABELS, NAMES, category_of

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
COLORS = {"interdiction": (60, 60, 230), "danger": (0, 165, 255),
          "obligation": (230, 120, 30), "autre": (160, 160, 160)}


def txt(s):  # polices OpenCV : pas d'accents
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def beep():
    try:
        import winsound; winsound.Beep(1000, 300)
    except ImportError:
        print("\a", end="", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="0", help="0 = webcam, chemin vidéo ou image")
    ap.add_argument("--classifier", help="poids MobileNetV2 (approche B)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--smooth", type=int, default=7, help="images pour le vote par piste")
    ap.add_argument("--save"); ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()

    model = YOLO(a.weights)
    clf = None
    if a.classifier:
        from classifier import SignClassifier
        clf = SignClassifier(a.classifier)
    use43 = clf is not None or len(model.names) == 43
    hist = defaultdict(lambda: deque(maxlen=a.smooth))
    last_alert, banner_until, banner = {}, 0, ""

    src = Path(a.source)
    is_image = src.suffix.lower() in IMG_EXT
    cap = None if is_image else cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    writer, t_prev = None, time.time()

    while True:
        if is_image:
            frame = cv2.imread(a.source)
        else:
            ok, frame = cap.read()
            if not ok:
                break
        if is_image:
            r = model.predict(frame, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        else:  # suivi ByteTrack pour stabiliser les prédictions
            r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                            conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        visible = {}
        b = r.boxes
        ids = b.id.int().tolist() if (b is not None and b.id is not None) else [None] * (len(b) if b is not None else 0)
        for k in range(len(ids)):
            x1, y1, x2, y2 = map(int, b.xyxy[k].tolist())
            cid, conf = int(b.cls[k]), float(b.conf[k])
            if clf is not None:
                m = int(0.1 * max(x2 - x1, y2 - y1))
                crop = frame[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m]
                if crop.shape[0] >= 8 and crop.shape[1] >= 8:
                    cid, c2 = clf.predict(crop); conf = conf * c2
            if ids[k] is not None:  # vote majoritaire sur la piste
                hist[ids[k]].append(cid)
                cid = Counter(hist[ids[k]]).most_common(1)[0][0]
            if use43:
                label, cat = NAMES[cid], category_of(cid)
            else:
                label = model.names[cid]; cat = label if label in COLORS else "autre"
            col = COLORS[cat]
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"{txt(label)} {conf:.2f}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            visible[label] = (CATEGORY_LABELS.get(cat, cat), conf)
            if use43 and cid in ALERT_IDS and time.time() - last_alert.get(cid, 0) > 4:
                last_alert[cid] = time.time(); banner_until = time.time() + 1.5
                banner = ALERT_MSG[cid]; beep()

        y = 28  # panneau d'information : signification des panneaux visibles
        for label, (cat, conf) in list(visible.items())[:5]:
            cv2.putText(frame, txt(f"{label} [{cat}] {conf:.0%}"), (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, txt(f"{label} [{cat}] {conf:.0%}"), (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 30
        if time.time() < banner_until:
            cv2.rectangle(frame, (0, frame.shape[0] - 70), (frame.shape[1], frame.shape[0]), (0, 0, 200), -1)
            cv2.putText(frame, txt("ALERTE : " + banner), (14, frame.shape[0] - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        now = time.time()
        cv2.putText(frame, f"{1 / max(now - t_prev, 1e-6):.0f} FPS", (frame.shape[1] - 120, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        t_prev = now

        if a.save:
            if is_image:
                cv2.imwrite(a.save, frame)
            else:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                writer.write(frame)
        if not a.no_show:
            cv2.imshow("Panneaux", frame)
            if cv2.waitKey(0 if is_image else 1) & 0xFF == ord("q"):
                break
        if is_image:
            break
    if cap: cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
