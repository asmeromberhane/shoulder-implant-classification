"""
Step 8: 5-Fold Stratified Cross-Validation (ResNet50 fine-tuned)
=================================================================
The literature review committed to "test overfitting through cross-validation".
Single 70/15/15 splits are noisy on 597 images, so this script runs stratified
5-fold CV on the winning architecture (ResNet50 with the same fine-tuning recipe
as step5) and reports mean ± std macro F1 across folds.

For each fold:
- 4 folds → train  (with internal 90/10 split for early stopping)
- 1 fold → test
The final number is mean ± std macro F1 over the 5 held-out fold tests.
"""

import os
# Project root (one level above src/). Paths are built from ROOT, so the script
# works from any directory and never changes the working directory.
# NOT import-safe yet: importing this file creates the output folders and runs
# the whole step (loading data, training or evaluation). Run it as a script
# only. Wrapping it in main() is on the README roadmap.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, "images")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from step1_load_dataset import load_dataset, CLASS_NAMES

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(ROOT, "dataset")
MODEL_DIR    = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 40
LR_HEAD = 5e-4
LR_BACKBONE = 5e-5
EARLY_STOP_PATIENCE = 8
N_FOLDS = 5
SEED = 42

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────
# 2. LOAD ALL DATA (CV uses every image)
# ─────────────────────────────────────────────────────────────
print("\n── Loading dataset for CV ──────────────────────────────\n")
images_raw, labels, _ = load_dataset(DATASET_PATH, IMG_SIZE)
images = images_raw.astype(np.float32) / 255.0
images = np.repeat(images[:, :, :, np.newaxis], 3, axis=3)
images = np.transpose(images, (0, 3, 1, 2))   # (N, 3, 224, 224)
print(f"Total images: {len(images)}")

# ─────────────────────────────────────────────────────────────
# 3. TRANSFORMS
# ─────────────────────────────────────────────────────────────
train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=20),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
eval_transform = T.Compose([T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X, self.y, self.transform = X, y, transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[idx]


# ─────────────────────────────────────────────────────────────
# 4. RESNET50 BUILDER (matches step5 fine-tune head)
# ─────────────────────────────────────────────────────────────
def build_resnet50_finetune(num_classes: int = 4):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in m.parameters():
        p.requires_grad = False
    for p in m.layer4.parameters():
        p.requires_grad = True
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(in_f, 256), nn.ReLU(),
        nn.Dropout(0.4), nn.Linear(256, num_classes),
    )
    param_groups = [
        {"params": m.layer4.parameters(), "lr": LR_BACKBONE},
        {"params": m.fc.parameters(),     "lr": LR_HEAD},
    ]
    return m, param_groups


# ─────────────────────────────────────────────────────────────
# 5. TRAIN ONE FOLD AND RETURN TEST METRICS
# ─────────────────────────────────────────────────────────────
def run_fold(fold_idx, train_idx, test_idx):
    print(f"\n──────────────────────────────────────────────────────")
    print(f"  FOLD {fold_idx + 1}/{N_FOLDS}")
    print(f"──────────────────────────────────────────────────────")

    X_tr_full, y_tr_full = images[train_idx], labels[train_idx]
    X_te,      y_te      = images[test_idx],  labels[test_idx]

    # Internal val split inside the training fold (90/10) for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full,
        test_size=0.10,
        stratify=y_tr_full,
        random_state=SEED + fold_idx,
    )

    cw = compute_class_weight(class_weight="balanced",
                              classes=np.unique(y_tr), y=y_tr)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(
        TransformedDataset(torch.tensor(X_tr, dtype=torch.float32),
                           torch.tensor(y_tr, dtype=torch.long),
                           transform=train_transform),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TransformedDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.long),
                           transform=eval_transform),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    test_loader = DataLoader(
        TransformedDataset(torch.tensor(X_te, dtype=torch.float32),
                           torch.tensor(y_te, dtype=torch.long),
                           transform=eval_transform),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    model, param_groups = build_resnet50_finetune()
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(param_groups)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    best_val_f1 = -1.0
    epochs_since_improve = 0
    best_state = None

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for inputs, ys in train_loader:
            inputs, ys = inputs.to(DEVICE), ys.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, ys)
            loss.backward()
            optimizer.step()

        # validate
        model.eval()
        vp, vy = [], []
        with torch.no_grad():
            for inputs, ys in val_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                vp.extend(outputs.argmax(1).cpu().numpy())
                vy.extend(ys.numpy())
        val_f1 = f1_score(vy, vp, average="macro", zero_division=0)
        scheduler.step(val_f1)

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        print(f"  Fold {fold_idx + 1} ep {epoch + 1:02d} "
              f"val macroF1 {val_f1:.4f}{' ←best' if improved else ''}")
        if epochs_since_improve >= EARLY_STOP_PATIENCE:
            print(f"  ⏹  Early stop at epoch {epoch + 1}")
            break

    # Reload best and evaluate on the held-out fold-test set
    model.load_state_dict(best_state)
    model.eval()
    tp, ty = [], []
    with torch.no_grad():
        for inputs, ys in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            tp.extend(outputs.argmax(1).cpu().numpy())
            ty.extend(ys.numpy())

    acc      = accuracy_score(ty, tp)
    macro_f1 = f1_score(ty, tp, average="macro", zero_division=0)
    wt_f1    = f1_score(ty, tp, average="weighted", zero_division=0)

    elapsed = (time.time() - t0) / 60
    print(f"  Fold {fold_idx + 1} done in {elapsed:.1f} min — "
          f"test acc {acc:.4f} macroF1 {macro_f1:.4f} weightedF1 {wt_f1:.4f}")
    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": wt_f1,
            "best_val_macro_f1": best_val_f1}


# ─────────────────────────────────────────────────────────────
# 6. RUN 5-FOLD CV
# ─────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_results = []
for i, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(images)), labels)):
    fold_results.append(run_fold(i, tr_idx, te_idx))

# ─────────────────────────────────────────────────────────────
# 7. AGGREGATE
# ─────────────────────────────────────────────────────────────
accs   = np.array([r["accuracy"]    for r in fold_results])
mf1s   = np.array([r["macro_f1"]    for r in fold_results])
wf1s   = np.array([r["weighted_f1"] for r in fold_results])

summary = {
    "n_folds": N_FOLDS,
    "per_fold": fold_results,
    "mean_accuracy":    float(accs.mean()),    "std_accuracy":    float(accs.std()),
    "mean_macro_f1":    float(mf1s.mean()),    "std_macro_f1":    float(mf1s.std()),
    "mean_weighted_f1": float(wf1s.mean()),    "std_weighted_f1": float(wf1s.std()),
}
with open(os.path.join(MODEL_DIR, "cv_results_resnet50.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved models/cv_results_resnet50.json")

# Plot per-fold macro F1
fig, ax = plt.subplots(figsize=(8, 5))
xs = np.arange(1, N_FOLDS + 1)
ax.bar(xs, mf1s, color="#4A90D9", edgecolor="white")
ax.axhline(mf1s.mean(), color="#E84855", linestyle="--",
           label=f"mean = {mf1s.mean():.3f}")
for x, v in zip(xs, mf1s):
    ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
ax.set_xticks(xs); ax.set_xlabel("Fold"); ax.set_ylabel("Test Macro F1")
ax.set_ylim(0, 1.0)
ax.set_title(f"5-Fold CV — ResNet50 (fine-tuned)\n"
             f"mean = {mf1s.mean():.3f} ± {mf1s.std():.3f}")
ax.legend(); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "cv_resnet50.png"), dpi=150)
print("Saved cv_resnet50.png")

print("\n" + "=" * 60)
print("  5-FOLD CV SUMMARY — ResNet50 (fine-tuned)")
print("=" * 60)
print(f"  Accuracy    : {accs.mean():.4f} ± {accs.std():.4f}")
print(f"  Macro F1    : {mf1s.mean():.4f} ± {mf1s.std():.4f}")
print(f"  Weighted F1 : {wf1s.mean():.4f} ± {wf1s.std():.4f}")
print("=" * 60)
