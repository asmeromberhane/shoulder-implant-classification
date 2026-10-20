"""
Step 6: Evaluate Fine-Tuned Models on Test Set (Multi-Backbone)
================================================================
Loads each fine-tuned checkpoint produced by step5, evaluates on the
held-out test set, and writes:
- classification_report_finetuned_<backbone>.txt
- confusion_matrix_finetuned_<backbone>.png
- finetuned_comparison.csv / finetuned_comparison.png
- combined_comparison.png   (baseline vs fine-tuned, side by side)

Test inputs are normalised with ImageNet mean/std (must match training).
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

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────
SPLITS_DIR = os.path.join(ROOT, "splits")
MODEL_DIR  = os.path.join(ROOT, "models")

BATCH_SIZE = 32
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BACKBONES = ["mobilenet_v2", "resnet50", "efficientnet_b0"]

sys.path.insert(0, os.path.dirname(__file__))
from step1_load_dataset import CLASS_NAMES

# ─────────────────────────────────────────────────────────────
# 2. LOAD TEST DATA
# ─────────────────────────────────────────────────────────────
print("\n── Loading test data ───────────────────────────────────\n")
X_test = torch.tensor(np.load(os.path.join(SPLITS_DIR, "X_test.npy")), dtype=torch.float32)
y_test = torch.tensor(np.load(os.path.join(SPLITS_DIR, "y_test.npy")), dtype=torch.long)
print(f"Test set size: {len(X_test)}")

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


test_loader = DataLoader(
    TransformedDataset(X_test, y_test, transform=eval_transform),
    batch_size=BATCH_SIZE, shuffle=False,
)


# ─────────────────────────────────────────────────────────────
# 3. MODEL SHAPES (must match step5 fine-tune head)
# ─────────────────────────────────────────────────────────────
def build_skeleton_finetuned(name: str, num_classes: int = 4):
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
# 4. EVALUATE EACH BACKBONE
# ─────────────────────────────────────────────────────────────
def evaluate(name: str):
    ckpt = os.path.join(MODEL_DIR, f"finetuned_{name}.pth")
    if not os.path.exists(ckpt):
        print(f"  [skip] {ckpt} not found — run step5 first.")
        return None

    model = build_skeleton_finetuned(name).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES, digits=3, zero_division=0,
    )
    print(f"\n── fine-tuned {name} ──────────────────────────────")
    print(f"Test accuracy : {acc * 100:.2f}%")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Weighted F1   : {weighted_f1:.4f}\n")
    print(report)

    with open(os.path.join(RESULTS_DIR, f"classification_report_finetuned_{name}.txt"), "w") as f:
        f.write(f"Test accuracy: {acc * 100:.2f}%\n")
        f.write(f"Macro F1:      {macro_f1:.4f}\n")
        f.write(f"Weighted F1:   {weighted_f1:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix — fine-tuned {name} (test)")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"confusion_matrix_finetuned_{name}.png"), dpi=150)
    plt.close()

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


finetuned_results = {}
for bb in BACKBONES:
    r = evaluate(bb)
    if r is not None:
        finetuned_results[bb] = r

# ─────────────────────────────────────────────────────────────
# 5. SAVE COMPARISON CSV + BAR CHART
# ─────────────────────────────────────────────────────────────
if finetuned_results:
    with open(os.path.join(RESULTS_DIR, "finetuned_comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "accuracy", "macro_f1", "weighted_f1"])
        for bb, r in finetuned_results.items():
            w.writerow([bb, f"{r['accuracy']:.4f}",
                        f"{r['macro_f1']:.4f}", f"{r['weighted_f1']:.4f}"])
    print("\nSaved finetuned_comparison.csv")

    names    = list(finetuned_results.keys())
    accs     = [finetuned_results[n]["accuracy"] for n in names]
    macro_f1 = [finetuned_results[n]["macro_f1"] for n in names]
    wt_f1    = [finetuned_results[n]["weighted_f1"] for n in names]

    x = np.arange(len(names)); width = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, accs,     width, label="Accuracy",    color="#4A90D9")
    ax.bar(x,         macro_f1, width, label="Macro F1",    color="#E84855")
    ax.bar(x + width, wt_f1,    width, label="Weighted F1", color="#44BBA4")
    for i, vals in enumerate(zip(accs, macro_f1, wt_f1)):
        for j, v in enumerate(vals):
            ax.text(x[i] + (j - 1) * width, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Fine-Tuned Test Performance")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "finetuned_comparison.png"), dpi=150)
    print("Saved finetuned_comparison.png")

# ─────────────────────────────────────────────────────────────
# 6. COMBINED COMPARISON: BASELINE vs FINE-TUNED  (macro F1)
# ─────────────────────────────────────────────────────────────
baseline_csv = os.path.join(RESULTS_DIR, "baseline_comparison.csv")
if finetuned_results and os.path.exists(baseline_csv):
    baseline = {}
    with open(baseline_csv) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            baseline[row["backbone"]] = {
                "accuracy":    float(row["accuracy"]),
                "macro_f1":    float(row["macro_f1"]),
                "weighted_f1": float(row["weighted_f1"]),
            }

    common = [n for n in BACKBONES if n in baseline and n in finetuned_results]
    if common:
        x = np.arange(len(common)); width = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        b_f1 = [baseline[n]["macro_f1"]        for n in common]
        f_f1 = [finetuned_results[n]["macro_f1"] for n in common]
        ax.bar(x - width / 2, b_f1, width, label="Baseline (head only)",
               color="#4A90D9")
        ax.bar(x + width / 2, f_f1, width, label="Fine-tuned",
               color="#E84855")
        for i, (bv, fv) in enumerate(zip(b_f1, f_f1)):
            ax.text(x[i] - width / 2, bv + 0.01, f"{bv:.2f}",
                    ha="center", fontsize=9)
            ax.text(x[i] + width / 2, fv + 0.01, f"{fv:.2f}",
                    ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(common)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Macro F1 (test)")
        ax.set_title("Baseline vs Fine-Tuned — Macro F1 by Backbone")
        ax.legend(); ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, "combined_comparison.png"), dpi=150)
        print("Saved combined_comparison.png")

# ─────────────────────────────────────────────────────────────
# 7. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
if finetuned_results:
    print("\n" + "=" * 60)
    print("  FINE-TUNED TEST SUMMARY")
    print("=" * 60)
    print(f"  {'Backbone':<20} {'Acc':>8} {'MacroF1':>10} {'WeightedF1':>12}")
    for bb, r in finetuned_results.items():
        print(f"  {bb:<20} {r['accuracy']:>8.4f} "
              f"{r['macro_f1']:>10.4f} {r['weighted_f1']:>12.4f}")
    print("=" * 60)
