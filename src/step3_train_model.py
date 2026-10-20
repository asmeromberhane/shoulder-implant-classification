"""
Step 3: Build & Train Transfer Learning Models (Multi-Backbone)
================================================================
Improvements over the original step3:
- ImageNet mean/std normalisation applied AFTER augmentation
- Tracks macro F1 per epoch and selects best checkpoint by macro F1
  (more appropriate than accuracy under class imbalance)
- Early stopping on macro F1 (patience = 8)
- Trains three backbones for comparison:
    1. MobileNetV2
    2. ResNet50
    3. EfficientNet-B0

For each backbone the convolutional features are frozen; only a custom
classification head is trained (feature-extraction stage). Fine-tuning
of deeper layers is performed in step5.
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
from sklearn.metrics import f1_score

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────
SPLITS_DIR = os.path.join(ROOT, "splits")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 8
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BACKBONES = ["mobilenet_v2", "resnet50", "efficientnet_b0"]

# ─────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n── Loading splits ──────────────────────────────────────\n")
X_train = torch.tensor(np.load(os.path.join(SPLITS_DIR, "X_train.npy")), dtype=torch.float32)
y_train = torch.tensor(np.load(os.path.join(SPLITS_DIR, "y_train.npy")), dtype=torch.long)
X_val   = torch.tensor(np.load(os.path.join(SPLITS_DIR, "X_val.npy")),   dtype=torch.float32)
y_val   = torch.tensor(np.load(os.path.join(SPLITS_DIR, "y_val.npy")),   dtype=torch.long)
class_weights = torch.tensor(
    np.load(os.path.join(SPLITS_DIR, "class_weights.npy")),
    dtype=torch.float32,
).to(DEVICE)

print(f"Train: {tuple(X_train.shape)}  Val: {tuple(X_val.shape)}")

# ─────────────────────────────────────────────────────────────
# 3. TRANSFORMS  (augmentation + ImageNet normalisation)
# ─────────────────────────────────────────────────────────────
train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
eval_transform = T.Compose([
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


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

# ─────────────────────────────────────────────────────────────
# 4. MODEL FACTORY  (MobileNetV2 / ResNet50 / EfficientNet-B0)
# ─────────────────────────────────────────────────────────────
def build_model(name: str, num_classes: int = 4):
    """Return (model, head_parameters_iter) with frozen feature extractor."""
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for p in m.features.parameters():
            p.requires_grad = False
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_f, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        head = m.classifier.parameters()

    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for p in m.parameters():
            p.requires_grad = False
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_f, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        head = m.fc.parameters()

    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for p in m.features.parameters():
            p.requires_grad = False
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_f, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        head = m.classifier.parameters()

    else:
        raise ValueError(f"Unknown backbone: {name}")

    return m, head


# ─────────────────────────────────────────────────────────────
# 5. TRAIN ONE BACKBONE
# ─────────────────────────────────────────────────────────────
def train_one(backbone: str):
    print("\n" + "=" * 60)
    print(f"  TRAINING (head only): {backbone}")
    print("=" * 60)

    model, head_params = build_model(backbone)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(head_params, lr=LEARNING_RATE)

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "val_f1": []}

    best_f1 = -1.0
    best_epoch = -1
    epochs_since_improve = 0
    save_path = os.path.join(MODEL_DIR, f"best_{backbone}.pth")

    t0 = time.time()
    for epoch in range(EPOCHS):
        # ── train ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # ── validate ──
        model.eval()
        running_val_loss, vcorrect, vtotal = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(1)
                vcorrect += (preds == labels).sum().item()
                vtotal += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss = running_val_loss / vtotal
        val_acc = vcorrect / vtotal
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        improved = val_f1 > best_f1
        marker = "  ←best" if improved else ""
        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}]  "
              f"train_loss {train_loss:.4f} acc {train_acc:.4f}  |  "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f} macroF1 {val_f1:.4f}"
              f"{marker}")

        if improved:
            best_f1 = val_f1
            best_epoch = epoch + 1
            epochs_since_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= EARLY_STOP_PATIENCE:
                print(f"  ⏹  Early stopping at epoch {epoch + 1} "
                      f"(no macro-F1 improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    elapsed = (time.time() - t0) / 60
    print(f"\nDone {backbone}: best macro F1 = {best_f1:.4f} "
          f"(epoch {best_epoch}), elapsed {elapsed:.2f} min")
    return history, best_f1, best_epoch


# ─────────────────────────────────────────────────────────────
# 6. RUN ALL BACKBONES + PLOT
# ─────────────────────────────────────────────────────────────
all_histories = {}
results = {}
for bb in BACKBONES:
    hist, best_f1, best_ep = train_one(bb)
    all_histories[bb] = hist
    results[bb] = {"best_val_macro_f1": best_f1, "best_epoch": best_ep,
                   "epochs_run": len(hist["val_f1"])}

with open(os.path.join(MODEL_DIR, "baseline_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved models/baseline_results.json")

# Plot per-backbone curves
fig, axes = plt.subplots(len(BACKBONES), 3, figsize=(16, 4 * len(BACKBONES)))
for row, bb in enumerate(BACKBONES):
    h = all_histories[bb]
    axes[row, 0].plot(h["train_loss"], label="train"); axes[row, 0].plot(h["val_loss"], label="val")
    axes[row, 0].set_title(f"{bb} — Loss"); axes[row, 0].legend(); axes[row, 0].grid(alpha=0.3)
    axes[row, 1].plot(h["train_acc"], label="train"); axes[row, 1].plot(h["val_acc"], label="val")
    axes[row, 1].set_title(f"{bb} — Accuracy"); axes[row, 1].legend(); axes[row, 1].grid(alpha=0.3)
    axes[row, 2].plot(h["val_f1"], label="val macro F1", color="#E84855")
    axes[row, 2].set_title(f"{bb} — Val Macro F1"); axes[row, 2].legend(); axes[row, 2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "training_history.png"), dpi=150)
print("Saved training_history.png")

# Comparison summary
print("\n" + "=" * 60)
print("  BASELINE (head-only) SUMMARY")
print("=" * 60)
print(f"  {'Backbone':<20} {'Best Val Macro F1':>20} {'Best Epoch':>12}")
for bb, r in results.items():
    print(f"  {bb:<20} {r['best_val_macro_f1']:>20.4f} {r['best_epoch']:>12}")
print("=" * 60)
