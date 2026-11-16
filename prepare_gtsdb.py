"""Convertit GTSDB (FullIJCNN2013) au format YOLO.

Découpage par blocs d'images consécutives (les images d'une même séquence
sont proches) : 0-499 train, 500-599 val, 600-899 test (test officiel).

--mode 43 : 43 classes | 4 : catégories | 1 : une seule classe « panneau »
(mode 1 : à combiner avec le classifieur MobileNetV2, approche B).
"""
import argparse
from collections import defaultdict
from pathlib import Path
from PIL import Image
from signs import NAMES, CATEGORY_KEYS, category_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dossier FullIJCNN2013 (gt.txt + .ppm)")
    ap.add_argument("--out", default="datasets/gtsdb")
    ap.add_argument("--mode", type=int, choices=[43, 4, 1], default=43)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)

    boxes = defaultdict(list)
    for line in (src / "gt.txt").read_text().splitlines():
        if line.strip():
            f, x1, y1, x2, y2, c = line.strip().split(";")
            boxes[f].append((int(x1), int(y1), int(x2), int(y2), int(c)))

    names = {43: NAMES, 4: CATEGORY_KEYS, 1: ["panneau"]}[a.mode]
    counts = {"train": 0, "val": 0, "test": 0}
    for img_path in sorted(src.glob("*.ppm")):
        idx = int(img_path.stem)
        split = "train" if idx < 500 else "val" if idx < 600 else "test"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        im.save(out / "images" / split / f"{img_path.stem}.jpg", quality=95)
        rows = []
        for x1, y1, x2, y2, c in boxes.get(img_path.name, []):
            cls = c if a.mode == 43 else CATEGORY_KEYS.index(category_of(c)) if a.mode == 4 else 0
            rows.append(f"{cls} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        (out / "labels" / split / f"{img_path.stem}.txt").write_text("\n".join(rows))
        counts[split] += 1

    yaml = [f"path: {out.resolve()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")
    print("Images :", counts, "->", out / "data.yaml")


if __name__ == "__main__":
    main()
