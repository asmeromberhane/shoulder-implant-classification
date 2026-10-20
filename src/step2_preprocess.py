"""
Step 2: Preprocessing & Data Splitting
=======================================
- Normalise pixels: uint8 [0,255] → float32 [0.0, 1.0]
- Repeat grayscale channel × 3 → (N, 3, 224, 224)  [PyTorch channel-first format]
- Stratified split: 70% train / 15% val / 15% test
- Compute class weights to handle Depuy imbalance
- Save splits as .npy files for reuse in later steps

Note: ImageNet mean/std normalisation is NOT applied here. It is applied as a
torchvision transform in step3/step5 (after augmentation), which is the standard
order for pretrained models.
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
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves to file, no GUI popup
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Re-use loader from step 1 ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from step1_load_dataset import load_dataset, CLASS_NAMES, LABEL_MAP

DATASET_PATH = os.path.join(ROOT, "dataset")
IMG_SIZE     = (224, 224)
SAVE_DIR     = os.path.join(ROOT, "splits")
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42

# ─────────────────────────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────────────────────────
print("\n── Loading raw dataset ─────────────────────────────────\n")
images_raw, labels, _ = load_dataset(DATASET_PATH, IMG_SIZE)
# images_raw : (597, 224, 224)  uint8
# labels     : (597,)           int64

# ─────────────────────────────────────────────────────────────
# 2. NORMALISE  →  float32 in [0, 1]
# ─────────────────────────────────────────────────────────────
print("\n── Step 2a: Normalising pixels ─────────────────────────\n")
images_norm = images_raw.astype(np.float32) / 255.0
print(f"  Before: dtype={images_raw.dtype}  range=[{images_raw.min()}, {images_raw.max()}]")
print(f"  After : dtype={images_norm.dtype}  range=[{images_norm.min():.3f}, {images_norm.max():.3f}]")

# ─────────────────────────────────────────────────────────────
# 3. EXPAND TO 3-CHANNEL  →  (N, 3, 224, 224)
#    MobileNetV2 expects 3-channel RGB input.
#    We repeat the grayscale channel 3 times (pseudo-RGB).
# ─────────────────────────────────────────────────────────────
print("\n── Step 2b: Expanding to 3-channel (pseudo-RGB) ────────\n")
# (N, H, W) → (N, H, W, 1) → (N, H, W, 3) → (N, 3, H, W)  [PyTorch CHW format]
images_3ch = np.repeat(images_norm[:, :, :, np.newaxis], 3, axis=3)   # (N, 224, 224, 3)
images_3ch = np.transpose(images_3ch, (0, 3, 1, 2))                   # (N, 3, 224, 224) ← PyTorch
print(f"  images_3ch shape : {images_3ch.shape}  (N, C, H, W)")
print(f"  dtype            : {images_3ch.dtype}")

# ─────────────────────────────────────────────────────────────
# 4. STRATIFIED TRAIN / VAL / TEST SPLIT
#    70% train | 15% validation | 15% test
# ─────────────────────────────────────────────────────────────
print("\n── Step 2c: Stratified splitting (70/15/15) ────────────\n")

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    images_3ch, labels,
    test_size=0.30,
    stratify=labels,
    random_state=SEED
)

# Second split of temp (30%): 50% → val (15% overall), 50% → test (15% overall)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=SEED
)

print(f"  Train : {len(X_train):>4} images  ({len(X_train)/len(images_3ch)*100:.1f}%)")
print(f"  Val   : {len(X_val):>4} images  ({len(X_val)/len(images_3ch)*100:.1f}%)")
print(f"  Test  : {len(X_test):>4} images  ({len(X_test)/len(images_3ch)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# 5. CLASS WEIGHTS  (for imbalance handling during training)
# ─────────────────────────────────────────────────────────────
print("\n── Step 2d: Computing class weights ────────────────────\n")
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights_array))

print(f"  {'Class':<10} {'Label':>5} {'Train count':>13} {'Weight':>10}")
print("  " + "-" * 44)
train_counts = Counter(y_train.tolist())
for name, idx in LABEL_MAP.items():
    print(f"  {name:<10} {idx:>5} {train_counts[idx]:>13} {class_weights[idx]:>10.4f}")

print("\n  → Higher weight = model penalised more for misclassifying this class")
print(f"  → Tornier (rarest) gets weight {class_weights[2]:.4f} vs "
      f"Depuy (most common) {class_weights[1]:.4f}")

# ─────────────────────────────────────────────────────────────
# 6. SAVE SPLITS TO DISK  (reusable in steps 3-5)
# ─────────────────────────────────────────────────────────────
print("\n── Step 2e: Saving splits to disk ──────────────────────\n")

np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_DIR, "X_val.npy"),   X_val)
np.save(os.path.join(SAVE_DIR, "X_test.npy"),  X_test)
np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_DIR, "y_val.npy"),   y_val)
np.save(os.path.join(SAVE_DIR, "y_test.npy"),  y_test)
np.save(os.path.join(SAVE_DIR, "class_weights.npy"),
        np.array(list(class_weights.values())))

for fname in os.listdir(SAVE_DIR):
    fpath = os.path.join(SAVE_DIR, fname)
    size_mb = os.path.getsize(fpath) / (1024 ** 2)
    print(f"  {fname:<25}  {size_mb:.2f} MB")

# ─────────────────────────────────────────────────────────────
# 7. VISUALISE: SPLIT DISTRIBUTION PER CLASS
# ─────────────────────────────────────────────────────────────
print("\n── Step 2f: Plotting split distribution ────────────────\n")

def split_counts(y):
    c = Counter(y.tolist())
    return [c[LABEL_MAP[n]] for n in CLASS_NAMES]

train_counts_list = split_counts(y_train)
val_counts_list   = split_counts(y_val)
test_counts_list  = split_counts(y_test)

x = np.arange(len(CLASS_NAMES))
width = 0.25
colors = {"Train": "#4A90D9", "Val": "#44BBA4", "Test": "#E84855"}

fig, ax = plt.subplots(figsize=(9, 5))
bars_tr = ax.bar(x - width,     train_counts_list, width, label="Train", color=colors["Train"],  edgecolor="white")
bars_vl = ax.bar(x,             val_counts_list,   width, label="Val",   color=colors["Val"],    edgecolor="white")
bars_ts = ax.bar(x + width,     test_counts_list,  width, label="Test",  color=colors["Test"],   edgecolor="white")

for bars in [bars_tr, bars_vl, bars_ts]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, str(int(h)),
                ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_ylabel("Number of Images", fontsize=11)
ax.set_title("Class Distribution Across Train / Val / Test Splits", fontsize=13, pad=12)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "split_distribution.png"), dpi=150)
print("  [Saved] split_distribution.png")
plt.show()

# ─────────────────────────────────────────────────────────────
# 8. VISUALISE: CLASS WEIGHTS BAR CHART
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
weight_vals = [class_weights[LABEL_MAP[n]] for n in CLASS_NAMES]
bar_colors  = ["#4A90D9", "#E84855", "#44BBA4", "#F9C74F"]
bars = ax.bar(CLASS_NAMES, weight_vals, color=bar_colors, edgecolor="white", linewidth=1.2)

for bar, val in zip(bars, weight_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="Weight = 1.0 (balanced)")
ax.set_ylabel("Class Weight", fontsize=11)
ax.set_title("Class Weights (inverse frequency — corrects imbalance)", fontsize=12, pad=10)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "class_weights.png"), dpi=150)
print("  [Saved] class_weights.png")
plt.show()

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 2 COMPLETE — PREPROCESSING SUMMARY")
print("="*55)
print(f"  X_train : {X_train.shape}  y_train : {y_train.shape}")
print(f"  X_val   : {X_val.shape}    y_val   : {y_val.shape}")
print(f"  X_test  : {X_test.shape}   y_test  : {y_test.shape}")
print()
print(f"  Normalisation : [0,255] → [0.0, 1.0]  ✓")
print(f"  Channels      : grayscale → 3-ch (CHW) ✓")
print(f"  Split ratio   : 70 / 15 / 15           ✓")
print(f"  Stratified    : yes (class balance kept)✓")
print(f"  Class weights : computed for training   ✓")
print(f"  Saved to      : splits/                 ✓")
print("="*55)
print("\n→ Ready for Step 3: Build & Train Transfer Learning Model\n")
