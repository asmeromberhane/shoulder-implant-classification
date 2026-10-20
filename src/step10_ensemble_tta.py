"""
Step 10: Test-Time Augmentation (TTA) + Ensemble
=================================================
Uses the three fine-tuned checkpoints from step5 with no extra training:

1. For each backbone, apply horizontal-flip TTA at inference: predict on the
   original image AND its horizontally-flipped copy, average the softmax
   probabilities → less variance, typically +1-3 points accuracy.

2. Ensemble: average the TTA softmax probabilities across the three backbones.
   Even when individual models disagree, the average is usually more accurate
   than any single one (model decorrelation effect).

Outputs:
- classification_report_tta_<backbone>.txt   (per-backbone TTA result)
- classification_report_ensemble.txt          (final ensemble result)
- confusion_matrix_ensemble.png
- ensemble_comparison.csv                     (single-model vs TTA vs ensemble)
- ensemble_comparison.png
- models/ensemble_results.json
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
import csv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)

sys.path.insert(0, os.path.dirname(__file__))
from step1_load_dataset import CLASS_NAMES

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────
SPLITS_DIR = os.path.join(ROOT, "splits")
MODEL_DIR  = os.path.join(ROOT, "models")

BATCH_SIZE = 32
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Looks for an extended ResNet50 checkpoint first; falls back to the standard one.
RESNET_EXTENDED = os.path.join(MODEL_DIR, "finetuned_resnet50_extended.pth")
RESNET_STANDARD = os.path.join(MODEL_DIR, "finetuned_resnet50.pth")
RESNET_CKPT = RESNET_EXTENDED if os.path.exists(RESNET_EXTENDED) else RESNET_STANDARD
print(f"Using ResNet50 checkpoint: {os.path.basename(RESNET_CKPT)}")

CHECKPOINTS = {
    "mobilenet_v2":    os.path.join(MODEL_DIR, "finetuned_mobilenet_v2.pth"),
    "resnet50":        RESNET_CKPT,
    "efficientnet_b0": os.path.join(MODEL_DIR, "finetuned_efficientnet_b0.pth"),
}

# ─────────────────────────────────────────────────────────────
# 2. LOAD TEST DATA
# ─────────────────────────────────────────────────────────────
print("\n── Loading test data ───────────────────────────────────\n")
X_test = torch.tensor(np.load(os.path.join(SPLITS_DIR, "X_test.npy")), dtype=torch.float32)
y_test = torch.tensor(np.load(os.path.join(SPLITS_DIR, "y_test.npy")), dtype=torch.long)
print(f"Test set size: {len(X_test)}")

normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


# ─────────────────────────────────────────────────────────────
# 3. SKELETONS (must match step5 fine-tune head)
# ─────────────────────────────────────────────────────────────
def build_skeleton(name: str, num_classes: int = 4):
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(in_f, 256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, num_classes),
        )
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(in_f, 256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, num_classes),
        )
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(in_f, 256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, num_classes),
        )
    else:
        raise ValueError(name)
    return m


# ─────────────────────────────────────────────────────────────
# 4. PREDICT WITH TTA  → returns (N, 4) softmax probabilities
# ─────────────────────────────────────────────────────────────
def predict_tta_probs(model, X):
    """Average softmax over identity + horizontal flip."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            batch = X[start:start + BATCH_SIZE].to(DEVICE)
            batch_norm = torch.stack([normalize(x) for x in batch])

            # original
            logits1 = model(batch_norm)
            p1 = F.softmax(logits1, dim=1)

            # horizontal flip (last dim = width for CHW tensors)
            batch_flip = torch.flip(batch_norm, dims=[3])
            logits2 = model(batch_flip)
            p2 = F.softmax(logits2, dim=1)

            all_probs.append(((p1 + p2) / 2.0).cpu())
    return torch.cat(all_probs, dim=0).numpy()


def report_metrics(name: str, y_true, y_pred, save_report: bool = True):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=3, zero_division=0
    )
    print(f"\n── {name} ──────────────────────────────────────────")
    print(f"Test accuracy : {acc * 100:.2f}%")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Weighted F1   : {weighted_f1:.4f}\n")
    print(report)
    if save_report:
        with open(os.path.join(RESULTS_DIR, f"classification_report_{name}.txt"), "w") as f:
            f.write(f"Test accuracy: {acc * 100:.2f}%\n")
            f.write(f"Macro F1:      {macro_f1:.4f}\n")
            f.write(f"Weighted F1:   {weighted_f1:.4f}\n\n")
            f.write(report)
    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


# ─────────────────────────────────────────────────────────────
# 5. PER-BACKBONE TTA + COLLECT PROBS FOR ENSEMBLE
# ─────────────────────────────────────────────────────────────
y_true = y_test.numpy()
all_probs_per_model = {}     # name → (N, 4) softmax
results = {}

for name, ckpt in CHECKPOINTS.items():
    if not os.path.exists(ckpt):
        print(f"  [skip] {ckpt} not found.")
        continue
    print(f"\nLoading {name} ← {os.path.basename(ckpt)}")
    model = build_skeleton(name).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    probs = predict_tta_probs(model, X_test)
    all_probs_per_model[name] = probs

    preds_tta = probs.argmax(axis=1)
    results[f"tta_{name}"] = report_metrics(f"tta_{name}", y_true, preds_tta)

# ─────────────────────────────────────────────────────────────
# 6. ENSEMBLE: AVERAGE SOFTMAX ACROSS THE BACKBONES
# ─────────────────────────────────────────────────────────────
if len(all_probs_per_model) >= 2:
    stacked = np.stack(list(all_probs_per_model.values()), axis=0)   # (M, N, 4)
    ens_probs = stacked.mean(axis=0)                                  # (N, 4)
    ens_preds = ens_probs.argmax(axis=1)

    results["ensemble"] = report_metrics("ensemble", y_true, ens_preds)

    cm = confusion_matrix(y_true, ens_preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix — Ensemble (TTA, soft-vote)")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrix_ensemble.png"), dpi=150)
    plt.close()
    print("Saved confusion_matrix_ensemble.png")

# ─────────────────────────────────────────────────────────────
# 7. COMPARISON CSV / CHART (single vs TTA vs ensemble)
# ─────────────────────────────────────────────────────────────
single_csv = os.path.join(RESULTS_DIR, "finetuned_comparison.csv")
single = {}
if os.path.exists(single_csv):
    with open(single_csv) as f:
        for row in csv.DictReader(f):
            single[row["backbone"]] = {
                "accuracy":    float(row["accuracy"]),
                "macro_f1":    float(row["macro_f1"]),
                "weighted_f1": float(row["weighted_f1"]),
            }

with open(os.path.join(RESULTS_DIR, "ensemble_comparison.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "accuracy", "macro_f1", "weighted_f1"])
    # single-model fine-tuned (from step6)
    for bb, r in single.items():
        w.writerow([f"finetuned_{bb}", f"{r['accuracy']:.4f}",
                    f"{r['macro_f1']:.4f}", f"{r['weighted_f1']:.4f}"])
    # TTA per backbone + ensemble
    for k, r in results.items():
        w.writerow([k, f"{r['accuracy']:.4f}",
                    f"{r['macro_f1']:.4f}", f"{r['weighted_f1']:.4f}"])
print("Saved ensemble_comparison.csv")

# Bar chart: macro F1 — single vs TTA vs ensemble
labels = []
mf1s   = []
colors = []
for bb in ["mobilenet_v2", "resnet50", "efficientnet_b0"]:
    if bb in single:
        labels.append(f"{bb}\n(single)")
        mf1s.append(single[bb]["macro_f1"])
        colors.append("#9CC6E8")
    if f"tta_{bb}" in results:
        labels.append(f"{bb}\n(TTA)")
        mf1s.append(results[f"tta_{bb}"]["macro_f1"])
        colors.append("#4A90D9")
if "ensemble" in results:
    labels.append("ENSEMBLE\n(3-model TTA)")
    mf1s.append(results["ensemble"]["macro_f1"])
    colors.append("#E84855")

if labels:
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, mf1s, color=colors, edgecolor="white")
    for x, v in zip(xs, mf1s):
        ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Macro F1 (test)")
    ax.set_title("Single fine-tune vs TTA vs Ensemble (macro F1)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "ensemble_comparison.png"), dpi=150)
    print("Saved ensemble_comparison.png")

# Save JSON summary for the report
with open(os.path.join(MODEL_DIR, "ensemble_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Saved models/ensemble_results.json")

# ─────────────────────────────────────────────────────────────
# 8. FINAL TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL — TTA AND ENSEMBLE")
print("=" * 65)
print(f"  {'Model':<24} {'Acc':>8} {'MacroF1':>10} {'WeightedF1':>12}")
for k, r in results.items():
    print(f"  {k:<24} {r['accuracy']:>8.4f} "
          f"{r['macro_f1']:>10.4f} {r['weighted_f1']:>12.4f}")
print("=" * 65)
