"""
Step 7: Traditional ML Baseline (HOG + SVM / Random Forest)
============================================================
Reproduces the "traditional ML" comparison promised in the literature review:
- Extract HOG features from each X-ray (hand-crafted shape/texture descriptor)
- Train SVM (RBF kernel) and Random Forest, both with class_weight='balanced'
- Evaluate on the same test split used by the deep models
- Report accuracy, macro F1, per-class metrics, confusion matrices

This gives the report a like-for-like comparison against MobileNetV2 / ResNet50
/ EfficientNet-B0 instead of citing literature numbers second-hand.
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
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────
SPLITS_DIR = os.path.join(ROOT, "splits")
CLASS_NAMES = ["Cofield", "Depuy", "Tornier", "Zimmer"]

# HOG settings — typical small-object descriptor parameters
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True,
)


def to_grayscale_uint8(X):
    """X is (N, 3, 224, 224) float in [0,1]; collapse to (N, 224, 224) uint8."""
    g = X.mean(axis=1)
    g = (g * 255).clip(0, 255).astype(np.uint8)
    return g


def extract_hog(images):
    feats = []
    for img in tqdm(images, desc="HOG features"):
        feats.append(hog(img, **HOG_PARAMS))
    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# 2. LOAD SPLITS AND CONVERT TO HOG FEATURES
# ─────────────────────────────────────────────────────────────
print("\n── Loading splits ──────────────────────────────────────\n")
X_train = np.load(os.path.join(SPLITS_DIR, "X_train.npy"))
y_train = np.load(os.path.join(SPLITS_DIR, "y_train.npy"))
X_val   = np.load(os.path.join(SPLITS_DIR, "X_val.npy"))
y_val   = np.load(os.path.join(SPLITS_DIR, "y_val.npy"))
X_test  = np.load(os.path.join(SPLITS_DIR, "X_test.npy"))
y_test  = np.load(os.path.join(SPLITS_DIR, "y_test.npy"))

# We don't need val for traditional ML; merge train+val for fitting
X_fit_imgs = np.concatenate([X_train, X_val], axis=0)
y_fit      = np.concatenate([y_train, y_val], axis=0)

print(f"Fit set : {len(X_fit_imgs)} images   Test set: {len(X_test)} images")

print("\n── Extracting HOG features ─────────────────────────────\n")
F_fit  = extract_hog(to_grayscale_uint8(X_fit_imgs))
F_test = extract_hog(to_grayscale_uint8(X_test))
print(f"Feature dim: {F_fit.shape[1]}")

# Standardise — RBF SVM in particular needs scaled inputs
scaler = StandardScaler()
F_fit_s  = scaler.fit_transform(F_fit)
F_test_s = scaler.transform(F_test)


# ─────────────────────────────────────────────────────────────
# 3. TRAIN AND EVALUATE EACH CLASSIFIER
# ─────────────────────────────────────────────────────────────
def evaluate_model(name: str, clf, X_fit, y_fit, X_test, y_test):
    print(f"\n── {name} ──────────────────────────────────────────")
    clf.fit(X_fit, y_fit)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    report = classification_report(
        y_test, preds, target_names=CLASS_NAMES, digits=3, zero_division=0
    )
    print(f"Test accuracy : {acc * 100:.2f}%")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Weighted F1   : {weighted_f1:.4f}\n")
    print(report)

    safe_name = name.lower().replace(" ", "_")
    with open(os.path.join(RESULTS_DIR, f"classification_report_{safe_name}.txt"), "w") as f:
        f.write(f"Test accuracy: {acc * 100:.2f}%\n")
        f.write(f"Macro F1:      {macro_f1:.4f}\n")
        f.write(f"Weighted F1:   {weighted_f1:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix — {name} (test)")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"confusion_matrix_{safe_name}.png"), dpi=150)
    plt.close()

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


results = {}
results["hog_svm_rbf"] = evaluate_model(
    "hog_svm_rbf",
    SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", random_state=42),
    F_fit_s, y_fit, F_test_s, y_test,
)
results["hog_random_forest"] = evaluate_model(
    "hog_random_forest",
    RandomForestClassifier(n_estimators=400, class_weight="balanced",
                           n_jobs=-1, random_state=42),
    F_fit_s, y_fit, F_test_s, y_test,
)

# ─────────────────────────────────────────────────────────────
# 4. SAVE COMPARISON
# ─────────────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "traditional_baseline.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "accuracy", "macro_f1", "weighted_f1"])
    for name, r in results.items():
        w.writerow([name, f"{r['accuracy']:.4f}",
                    f"{r['macro_f1']:.4f}", f"{r['weighted_f1']:.4f}"])
print("\nSaved traditional_baseline.csv")

print("\n" + "=" * 60)
print("  TRADITIONAL ML SUMMARY (HOG features)")
print("=" * 60)
print(f"  {'Model':<22} {'Acc':>8} {'MacroF1':>10} {'WeightedF1':>12}")
for name, r in results.items():
    print(f"  {name:<22} {r['accuracy']:>8.4f} "
          f"{r['macro_f1']:>10.4f} {r['weighted_f1']:>12.4f}")
print("=" * 60)
