"""Fine-tuning de MobileNetV2 sur GTSRB (classification de panneaux recadrés).

--train-dir : GTSRB/Final_Training/Images (dossiers 00000..00042)
--test-dir / --test-csv : Final_Test/Images et GT-final_test.csv (optionnel)
Validation découpée PAR SÉQUENCE (30 images consécutives d'un même panneau)
pour éviter toute fuite entre train et validation.
"""
import argparse, csv
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from classifier import MEAN, STD, SIZE, build, eval_tf

ap = argparse.ArgumentParser()
ap.add_argument("--train-dir", required=True)
ap.add_argument("--test-dir")
ap.add_argument("--test-csv")
ap.add_argument("--out", default="classifier_mobilenetv2.pt")
ap.add_argument("--head-epochs", type=int, default=3)
ap.add_argument("--ft-epochs", type=int, default=10)
ap.add_argument("--batch", type=int, default=128)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

# Pas de flip : il change le sens de certains panneaux.
train_tf = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
tr_ds = datasets.ImageFolder(a.train_dir, train_tf)
va_ds = datasets.ImageFolder(a.train_dir, eval_tf)
assert len(tr_ds.classes) == 43, "43 dossiers de classes attendus"
track = [int(Path(p).stem.split("_")[0]) for p, _ in tr_ds.samples]
val_idx = [i for i, t in enumerate(track) if t % 7 == 0]
tr_idx = [i for i, t in enumerate(track) if t % 7 != 0]
tl = DataLoader(Subset(tr_ds, tr_idx), a.batch, shuffle=True, num_workers=2)
vl = DataLoader(Subset(va_ds, val_idx), a.batch, num_workers=2)

counts = np.bincount([tr_ds.samples[i][1] for i in tr_idx], minlength=43)
w = torch.tensor((counts.mean() / counts) ** 0.5, dtype=torch.float32, device=dev)  # classes déséquilibrées
loss_fn = nn.CrossEntropyLoss(weight=w)
model = build(43).to(dev)


@torch.no_grad()
def predict(loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps += model(x.to(dev)).argmax(1).cpu().tolist(); ys += y.tolist()
    return ys, ps


def fit(params, epochs, lr, freeze_features):
    best = getattr(fit, "best", -1)
    opt = torch.optim.AdamW(params, lr=lr)
    for ep in range(epochs):
        model.train()
        if freeze_features:
            model.features.eval()
        for x, y in tl:
            opt.zero_grad()
            loss_fn(model(x.to(dev)), y.to(dev)).backward()
            opt.step()
        ys, ps = predict(vl)
        f1 = f1_score(ys, ps, average="macro")
        print(f"epoch {ep+1}/{epochs} val acc={accuracy_score(ys, ps):.4f} F1 macro={f1:.4f}")
        if f1 > best:
            best = f1; torch.save(model.state_dict(), a.out)
    fit.best = best


for p in model.features.parameters():
    p.requires_grad = False
fit(model.classifier.parameters(), a.head_epochs, 1e-3, True)   # 1) entraîner la tête
for p in model.features.parameters():
    p.requires_grad = True
fit(model.parameters(), a.ft_epochs, 1e-4, False)               # 2) fine-tuning complet

if a.test_dir and a.test_csv:
    model.load_state_dict(torch.load(a.out, map_location=dev))
    class T(Dataset):
        rows = list(csv.DictReader(open(a.test_csv), delimiter=";"))
        def __len__(s): return len(s.rows)
        def __getitem__(s, i):
            r = s.rows[i]
            return eval_tf(Image.open(Path(a.test_dir) / r["Filename"]).convert("RGB")), int(r["ClassId"])
    ys, ps = predict(DataLoader(T(), a.batch, num_workers=2))
    print(f"TEST accuracy={accuracy_score(ys, ps):.4f} F1 macro={f1_score(ys, ps, average='macro'):.4f}")
    print(classification_report(ys, ps, digits=3))
