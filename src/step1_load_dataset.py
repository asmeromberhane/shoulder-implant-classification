"""
Step 1: Load and Explore the Shoulder Implant X-Ray Dataset
============================================================
Dataset path: dataset/
Labels are encoded in filenames: Manufacturer.N.jpg
Classes: Cofield (0), Depuy (1), Tornier (2), Zimmer (3)
"""

import os
# Project root (one level above src/). Paths are built from ROOT, so the script
# works from any directory. Importing this module only defines constants and
# functions; output folders are created only when it is run as a script.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, "images")
RESULTS_DIR = os.path.join(ROOT, "results")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from collections import Counter
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(ROOT, "dataset")
IMG_SIZE = (224, 224)           # Standard size for pretrained CNNs (MobileNetV2, ResNet, etc.)

# Map manufacturer names → integer class labels
LABEL_MAP = {
    "Cofield": 0,
    "Depuy":   1,
    "Tornier": 2,
    "Zimmer":  3,
}
CLASS_NAMES = list(LABEL_MAP.keys())

# ─────────────────────────────────────────────────────────────
# 2. LOAD IMAGES AND LABELS FROM FILENAMES
# ─────────────────────────────────────────────────────────────
def load_dataset(dataset_path: str, img_size: tuple) -> tuple:
    """
    Read all .jpg images from dataset_path.
    Extract label from filename prefix (e.g. 'Cofield.1.jpg' → label 0).
    Returns:
        images : np.ndarray  shape (N, H, W)   — grayscale, uint8
        labels : np.ndarray  shape (N,)         — integer class indices
        fnames : list of str                    — original filenames
    """
    images, labels, fnames = [], [], []

    all_files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".jpg")])
    print(f"Found {len(all_files)} image files in '{dataset_path}'")

    for fname in tqdm(all_files, desc="Loading images"):
        manufacturer = fname.split(".")[0]          # 'Cofield', 'Depuy', etc.

        if manufacturer not in LABEL_MAP:
            print(f"  [SKIP] Unknown manufacturer in filename: {fname}")
            continue

        img_path = os.path.join(dataset_path, fname)
        img = Image.open(img_path).convert("L")     # convert to grayscale
        img = img.resize(img_size, Image.LANCZOS)   # resize to target size

        images.append(np.array(img))
        labels.append(LABEL_MAP[manufacturer])
        fnames.append(fname)

    images = np.array(images, dtype=np.uint8)   # shape: (N, 224, 224)
    labels = np.array(labels, dtype=np.int64)   # shape: (N,)

    return images, labels, fnames


# ─────────────────────────────────────────────────────────────
# 3. PRINT DATASET SUMMARY
# ─────────────────────────────────────────────────────────────
def print_summary(images: np.ndarray, labels: np.ndarray):
    print("\n" + "="*50)
    print("  DATASET SUMMARY")
    print("="*50)
    print(f"  Total images  : {len(images)}")
    print(f"  Image shape   : {images.shape[1:]}  (H x W, grayscale)")
    print(f"  Label shape   : {labels.shape}")
    print(f"  dtype images  : {images.dtype}")
    print(f"  Pixel range   : [{images.min()}, {images.max()}]")
    print()

    counts = Counter(labels.tolist())
    print("  Class distribution:")
    print(f"  {'Class':<10} {'Label':>6} {'Count':>8} {'% of total':>12}")
    print("  " + "-"*40)
    for name, idx in LABEL_MAP.items():
        pct = counts[idx] / len(labels) * 100
        print(f"  {name:<10} {idx:>6} {counts[idx]:>8} {pct:>11.1f}%")
    print("="*50 + "\n")


# ─────────────────────────────────────────────────────────────
# 4. VISUALISE: CLASS DISTRIBUTION BAR CHART
# ─────────────────────────────────────────────────────────────
def plot_class_distribution(labels: np.ndarray, save_path: str = os.path.join(IMAGES_DIR, "class_distribution.png")):
    counts = Counter(labels.tolist())
    values = [counts[LABEL_MAP[n]] for n in CLASS_NAMES]
    colors = ["#4A90D9", "#E84855", "#44BBA4", "#F9C74F"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CLASS_NAMES, values, color=colors, edgecolor="white", linewidth=1.2)

    # Annotate bars with count and percentage
    total = sum(values)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{val}\n({val/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_title("Shoulder Implant X-Ray — Class Distribution", fontsize=14, pad=14)
    ax.set_xlabel("Manufacturer", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_ylim(0, max(values) + 50)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  [Saved] {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# 5. VISUALISE: SAMPLE IMAGES PER CLASS (4 × 4 GRID)
# ─────────────────────────────────────────────────────────────
def plot_sample_images(images: np.ndarray, labels: np.ndarray, n_samples: int = 4,
                       save_path: str = os.path.join(IMAGES_DIR, "sample_images.png")):
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Sample X-Ray Images per Manufacturer", fontsize=14, y=1.02)

    gs = gridspec.GridSpec(n_samples, len(CLASS_NAMES), figure=fig,
                           hspace=0.05, wspace=0.05)

    for col, (name, idx) in enumerate(LABEL_MAP.items()):
        class_indices = np.where(labels == idx)[0]
        sample_indices = class_indices[:n_samples]

        for row, img_idx in enumerate(sample_indices):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(images[img_idx], cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=11, fontweight="bold", pad=6)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  [Saved] {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# 6. VISUALISE: PIXEL INTENSITY HISTOGRAM PER CLASS
# ─────────────────────────────────────────────────────────────
def plot_pixel_histograms(images: np.ndarray, labels: np.ndarray,
                          save_path: str = os.path.join(IMAGES_DIR, "pixel_histograms.png")):
    colors = ["#4A90D9", "#E84855", "#44BBA4", "#F9C74F"]
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(16, 4), sharey=True)
    fig.suptitle("Pixel Intensity Distribution per Class", fontsize=13)

    for ax, (name, idx), color in zip(axes, LABEL_MAP.items(), colors):
        class_imgs = images[labels == idx].flatten()
        ax.hist(class_imgs, bins=64, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Pixel value (0–255)")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  [Saved] {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(IMAGES_DIR, exist_ok=True)
    print("\n── STEP 1: Loading Dataset ─────────────────────────────\n")

    # Load
    images, labels, fnames = load_dataset(DATASET_PATH, IMG_SIZE)

    # Summary
    print_summary(images, labels)

    # Plots
    print("── STEP 2: Visualising Data ────────────────────────────\n")
    plot_class_distribution(labels)
    plot_sample_images(images, labels)
    plot_pixel_histograms(images, labels)

    print("\n✅  Step 1 complete. images and labels ready for preprocessing.\n")
    print(f"   images.shape : {images.shape}")
    print(f"   labels.shape : {labels.shape}")
