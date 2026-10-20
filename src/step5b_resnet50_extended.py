"""
Step 5b: Extended Fine-Tune for ResNet50 only
==============================================
ResNet50 was the strongest model in step5 and was still climbing at epoch 30
when early stopping triggered. This script reruns the same recipe with:
- EARLY_STOP_PATIENCE = 15  (was 8)
- EPOCHS = 80               (was 60)

Saves to a SEPARATE checkpoint (`finetuned_resnet50_extended.pth`) so the
original `finetuned_resnet50.pth` is preserved for comparison. The ensemble
script (step10) automatically picks the extended checkpoint if it exists.
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
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, accuracy_score

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────
SPLITS_DIR = os.path.join(ROOT, "splits")
MODEL_DIR  = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 80
LR_HEAD = 5e-4
LR_BACKBONE = 5e-5
EARLY_STOP_PATIENCE = 15
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n── Loading splits ──────────────────────────────────────\n")
X_train = torch.tensor(np.load(os.path.join(SPLITS_DIR, "X_train.npy")), dtype=torch.float32)
y_train = torch.tensor(np.load(os.path.join(SPLITS_DIR, "y_train.npy")), dtype=torch.long)
X_val   = torch.tensor(np.load(os.path.join(SPLITS_DIR, "X_val.npy")),   dtype=torch.float32)
y_val   = torch.tensor(np.load(os.path.join(SPLITS_DIR, "y_val.npy")),   dtype=torch.long)
X_test  = torch.tensor(np.load(os.path.join(SPLITS_DIR, "X_test.npy")),  dtype=torch.float32)
y_test  = torch.tensor(np.load(os.path.join(SPLITS_DIR, "y_test.npy")),  dtype=torch.long)
class_weights = torch.tensor(
    np.load(os.path.join(SPLITS_DIR, "class_weights.npy")),
    dtype=torch.float32,
).to(DEVICE)
print(f"Train: {tuple(X_train.shape)}  Val: {tuple(X_val.shape)}  Test: {tuple(X_test.shape)}")

# ─────────────────────────────────────────────────────────────
# 3. TRANSFORMS  (same as step5)
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


train_loader = DataLoader(
    TransformedDataset(X_train, y_train, transform=train_transform),
    batch_size=BATCH_SIZE, shuffle=True,
)
val_loader = DataLoader(
    TransformedDataset(X_val, y_val, transform=eval_transform),
    batch_size=BATCH_SIZE, shuffle=False,
)
test_loader = DataLoader(
    TransformedDataset(X_test, y_test, transform=eval_transform),
    batch_size=BATCH_SIZE, shuffle=False,
)

# ─────────────────────────────────────────────────────────────
# 4. BUILD RESNET50  (matches step5 head)
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


model, param_groups = build_resnet50_finetune()
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(param_groups)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=4
)

# ─────────────────────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────────────────────
print(f"\n── Extended fine-tune ResNet50 (patience={EARLY_STOP_PATIENCE}, "
      f"max_epochs={EPOCHS}) ──\n")
history = {"train_loss": [], "train_acc": [],
           "val_loss": [], "val_acc": [], "val_f1": []}
best_f1 = -1.0
best_epoch = -1
epochs_since_improve = 0
save_path = os.path.join(MODEL_DIR, "finetuned_resnet50_extended.pth")

t0 = time.time()
for epoch in range(EPOCHS):
    # ── train ──
    model.train()
    rl, c, n = 0.0, 0, 0
    for inputs, ys in train_loader:
        inputs, ys = inputs.to(DEVICE), ys.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, ys)
        loss.backward()
        optimizer.step()
        rl += loss.item() * inputs.size(0)
        c += (outputs.argmax(1) == ys).sum().item()
        n += ys.size(0)
    train_loss, train_acc = rl / n, c / n

    # ── validate ──
    model.eval()
    rvl, vc, vn = 0.0, 0, 0
    vp, vy = [], []
    with torch.no_grad():
        for inputs, ys in val_loader:
            inputs, ys = inputs.to(DEVICE), ys.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, ys)
            rvl += loss.item() * inputs.size(0)
            preds = outputs.argmax(1)
            vc += (preds == ys).sum().item()
            vn += ys.size(0)
            vp.extend(preds.cpu().numpy()); vy.extend(ys.cpu().numpy())
    val_loss, val_acc = rvl / vn, vc / vn
    val_f1 = f1_score(vy, vp, average="macro", zero_division=0)
    scheduler.step(val_f1)

    history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss);     history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    improved = val_f1 > best_f1
    print(f"Epoch [{epoch + 1:02d}/{EPOCHS}]  "
          f"train_loss {train_loss:.4f} acc {train_acc:.4f}  |  "
          f"val_loss {val_loss:.4f} acc {val_acc:.4f} macroF1 {val_f1:.4f}"
          f"{'  ←best' if improved else ''}")

    if improved:
        best_f1, best_epoch, epochs_since_improve = val_f1, epoch + 1, 0
        torch.save(model.state_dict(), save_path)
    else:
        epochs_since_improve += 1
        if epochs_since_improve >= EARLY_STOP_PATIENCE:
            print(f"  ⏹  Early stop at epoch {epoch + 1}")
            break

elapsed = (time.time() - t0) / 60
print(f"\nDone: best val macroF1 = {best_f1:.4f} (epoch {best_epoch}), "
      f"elapsed {elapsed:.1f} min")

# ─────────────────────────────────────────────────────────────
# 6. EVALUATE ON TEST
# ─────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(save_path, map_location=DEVICE))
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
report = classification_report(ty, tp, target_names=["Cofield", "Depuy",
                                                     "Tornier", "Zimmer"],
                               digits=3, zero_division=0)

print("\n── Test ResNet50 (extended fine-tune) ──────────────────")
print(f"Test accuracy : {acc * 100:.2f}%")
print(f"Macro F1      : {macro_f1:.4f}")
print(f"Weighted F1   : {wt_f1:.4f}\n")
print(report)

with open(os.path.join(RESULTS_DIR, "classification_report_finetuned_resnet50_extended.txt"), "w") as f:
    f.write(f"Test accuracy: {acc * 100:.2f}%\n")
    f.write(f"Macro F1:      {macro_f1:.4f}\n")
    f.write(f"Weighted F1:   {wt_f1:.4f}\n\n")
    f.write(report)

# Plot history
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["val_loss"], label="val")
axes[0].set_title("ResNet50 (extended) — Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history["train_acc"], label="train"); axes[1].plot(history["val_acc"], label="val")
axes[1].set_title("ResNet50 (extended) — Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[2].plot(history["val_f1"], label="val macro F1", color="#E84855")
axes[2].set_title("ResNet50 (extended) — Val Macro F1"); axes[2].legend(); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "finetuned_resnet50_extended_history.png"), dpi=150)
print("Saved finetuned_resnet50_extended_history.png")

with open(os.path.join(MODEL_DIR, "resnet50_extended_results.json"), "w") as f:
    json.dump({
        "test_accuracy":     acc,
        "test_macro_f1":     macro_f1,
        "test_weighted_f1":  wt_f1,
        "best_val_macro_f1": best_f1,
        "best_epoch":        best_epoch,
        "epochs_run":        len(history["val_f1"]),
        "patience":          EARLY_STOP_PATIENCE,
    }, f, indent=2)
print("Saved models/resnet50_extended_results.json")
