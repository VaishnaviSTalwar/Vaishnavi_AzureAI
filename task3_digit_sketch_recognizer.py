"""
Handwritten Digit Sketch Recognizer
====================================
- Generates synthetic digit-like patterns using NumPy
- Applies CNN convolution + pooling from scratch (no ML libraries)
- Classifies using a Decision Tree (sklearn)
- Visualizes feature maps for each filter
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. SYNTHETIC DIGIT GENERATOR
# ─────────────────────────────────────────────

def make_digit(digit: int, size: int = 16, noise: float = 0.05) -> np.ndarray:
    """
    Draw a rough digit pattern on a size×size canvas using simple geometric rules.
    Returns a float32 array in [0, 1].
    """
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    r = size // 3

    def draw_hline(row, c1, c2, v=1.0):
        img[row, c1:c2] = v

    def draw_vline(col, r1, r2, v=1.0):
        img[r1:r2, col] = v

    def draw_circle(cx, cy, r, v=1.0):
        for angle in np.linspace(0, 2 * np.pi, 200):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < size and 0 <= y < size:
                img[x, y] = v

    s = size

    if digit == 0:
        draw_circle(cx, cy, r)
    elif digit == 1:
        draw_vline(cx, 2, s - 2)
        draw_hline(2, cx - 2, cx + 1)          # top serif
    elif digit == 2:
        draw_hline(2, 3, s - 3)                 # top
        draw_vline(s - 3, 2, cx)                # top-right down
        draw_hline(cx, 3, s - 3)                # middle
        draw_vline(3, cx, s - 2)                # bottom-left down
        draw_hline(s - 2, 3, s - 3)             # bottom
    elif digit == 3:
        draw_hline(2, 3, s - 3)
        draw_hline(cx, cx // 2 + 1, s - 3)
        draw_hline(s - 2, 3, s - 3)
        draw_vline(s - 3, 2, s - 2)
    elif digit == 4:
        draw_vline(3, 2, cx + 1)
        draw_hline(cx, 3, s - 3)
        draw_vline(s - 3, 2, s - 2)
    elif digit == 5:
        draw_hline(2, 3, s - 3)                 # top
        draw_vline(3, 2, cx)                     # top-left
        draw_hline(cx, 3, s - 3)                 # middle
        draw_vline(s - 3, cx, s - 2)             # bottom-right
        draw_hline(s - 2, 3, s - 3)             # bottom
    elif digit == 6:
        draw_circle(cx + 2, cy, r - 1)          # bottom loop
        draw_vline(3, 2, cx + 2)                # left stem
        draw_hline(2, 3, s - 3)                 # top bar
    elif digit == 7:
        draw_hline(2, 3, s - 3)
        draw_vline(s - 3, 2, s - 2)
        img[cx, 3:s - 3] = 0.6                  # slight diagonal hint
    elif digit == 8:
        draw_circle(cx - 2, cy, r - 2)
        draw_circle(cx + 3, cy, r - 2)
    elif digit == 9:
        draw_circle(cx - 2, cy, r - 1)
        draw_vline(s - 3, cx - 2, s - 2)

    # Gaussian-blur for smoothness (manual 3×3 kernel)
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16
    from scipy.ndimage import convolve
    img = convolve(img, kernel)

    # Add noise
    img += np.random.normal(0, noise, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    return img


def generate_dataset(samples_per_class: int = 80, size: int = 16):
    X, y = [], []
    for digit in range(10):
        for _ in range(samples_per_class):
            img = make_digit(digit, size=size, noise=0.06)
            X.append(img)
            y.append(digit)
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# 2. CNN BUILDING BLOCKS (from scratch)
# ─────────────────────────────────────────────

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Single-channel 2-D convolution (valid padding, stride=1).
    """
    kH, kW = kernel.shape
    iH, iW = image.shape
    oH, oW = iH - kH + 1, iW - kW + 1
    out = np.zeros((oH, oW), dtype=np.float32)
    for i in range(oH):
        for j in range(oW):
            out[i, j] = np.sum(image[i:i+kH, j:j+kW] * kernel)
    return out


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def max_pool2d(feature_map: np.ndarray, pool_size: int = 2) -> np.ndarray:
    """
    Non-overlapping max pooling.
    """
    H, W = feature_map.shape
    pH, pW = H // pool_size, W // pool_size
    out = np.zeros((pH, pW), dtype=np.float32)
    for i in range(pH):
        for j in range(pW):
            patch = feature_map[i*pool_size:(i+1)*pool_size,
                                j*pool_size:(j+1)*pool_size]
            out[i, j] = patch.max()
    return out


# ─────────────────────────────────────────────
# 3. HAND-CRAFTED FILTERS
# ─────────────────────────────────────────────

FILTERS = {
    "Vertical Edge": np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=np.float32),

    "Horizontal Edge": np.array([[-1, -2, -1],
                                  [ 0,  0,  0],
                                  [ 1,  2,  1]], dtype=np.float32),

    "Diagonal (/)": np.array([[ 0,  1,  2],
                               [-1,  0,  1],
                               [-2, -1,  0]], dtype=np.float32),

    "Diagonal (\\)": np.array([[ 2,  1,  0],
                                [ 1,  0, -1],
                                [ 0, -1, -2]], dtype=np.float32),

    "Blob/Spot": np.array([[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]], dtype=np.float32),

    "Smooth": np.ones((3, 3), dtype=np.float32) / 9,
}


def extract_cnn_features(image: np.ndarray) -> np.ndarray:
    """
    Apply each filter → ReLU → max-pool → flatten → concat into one feature vector.
    """
    features = []
    for kernel in FILTERS.values():
        conv = convolve2d(image, kernel)   # convolution
        act  = relu(conv)                  # activation
        pool = max_pool2d(act, pool_size=2)# pooling
        features.append(pool.flatten())
    return np.concatenate(features)


def build_feature_matrix(X: np.ndarray) -> np.ndarray:
    print("Extracting CNN features …")
    rows = []
    for i, img in enumerate(X):
        if i % 100 == 0:
            print(f"  {i}/{len(X)}")
        rows.append(extract_cnn_features(img))
    return np.array(rows, dtype=np.float32)


# ─────────────────────────────────────────────
# 4. VISUALISE FEATURE MAPS
# ─────────────────────────────────────────────

def visualize_feature_maps(image: np.ndarray, digit: int):
    filter_names = list(FILTERS.keys())
    n = len(filter_names)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"CNN Feature Maps  —  Digit: {digit}", fontsize=16, fontweight="bold")

    # Columns: Original | conv | relu | pool   for each filter → 3 cols × n rows + 1 col for original
    cols = 3
    rows = n
    gs = gridspec.GridSpec(rows, cols + 1, figure=fig, hspace=0.5, wspace=0.35)

    # Original image (spans all rows in col 0)
    ax_orig = fig.add_subplot(gs[:, 0])
    ax_orig.imshow(image, cmap="gray", interpolation="nearest")
    ax_orig.set_title("Input\nImage", fontsize=10, fontweight="bold")
    ax_orig.axis("off")

    for row, (name, kernel) in enumerate(FILTERS.items()):
        conv = convolve2d(image, kernel)
        act  = relu(conv)
        pool = max_pool2d(act, pool_size=2)

        stages = [("Conv\n" + name, conv),
                  ("ReLU", act),
                  ("MaxPool", pool)]

        for col, (stage_title, data) in enumerate(stages):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(data, cmap="viridis", interpolation="nearest")
            if row == 0:
                ax.set_title(stage_title, fontsize=8, fontweight="bold")
            elif col == 0:
                ax.set_title(stage_title, fontsize=7)
            ax.set_ylabel(name if col == 0 else "", fontsize=6, rotation=0,
                          labelpad=60, va="center")
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

    plt.savefig("feature_maps.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Feature map saved → feature_maps.png")


# ─────────────────────────────────────────────
# 5. CONFUSION MATRIX (from scratch)
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, classes):
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Digit Recognizer", fontsize=14, fontweight="bold")

    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    plt.show()
    print("Confusion matrix saved → confusion_matrix.png")


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    np.random.seed(42)

    # ── Generate dataset ──────────────────────
    print("=" * 55)
    print("  HANDWRITTEN DIGIT SKETCH RECOGNIZER")
    print("=" * 55)
    print("\n[1] Generating synthetic digit dataset …")
    X, y = generate_dataset(samples_per_class=80, size=16)
    print(f"    Dataset shape: {X.shape}  |  Classes: {np.unique(y)}")

    # ── Show sample digits ────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Sample Synthetic Digits", fontsize=14, fontweight="bold")
    for digit in range(10):
        idx = np.where(y == digit)[0][0]
        r, c = divmod(digit, 5)
        axes[r, c].imshow(X[idx], cmap="gray", interpolation="nearest")
        axes[r, c].set_title(f"Digit {digit}", fontsize=10)
        axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig("sample_digits.png", dpi=120)
    plt.show()
    print("    Sample grid saved → sample_digits.png")

    # ── Visualise feature maps for one example ─
    print("\n[2] Visualising CNN feature maps …")
    sample_img = make_digit(3, size=16, noise=0.04)
    visualize_feature_maps(sample_img, digit=3)

    # ── Extract CNN features ──────────────────
    print("\n[3] Building CNN feature matrix (this may take ~30 s) …")
    X_feat = build_feature_matrix(X)
    print(f"    Feature vector size per image: {X_feat.shape[1]}")

    # ── Train/Test split ──────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=y)

    # ── Decision Tree ─────────────────────────
    print("\n[4] Training Decision Tree classifier …")
    clf = DecisionTreeClassifier(max_depth=20, min_samples_leaf=2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"\n    Test Accuracy : {acc*100:.2f}%")
    print("\n    Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=[str(d) for d in range(10)]))

    # ── Confusion Matrix ──────────────────────
    print("\n[5] Plotting confusion matrix …")
    plot_confusion_matrix(y_test, y_pred, classes=list(range(10)))

    # ── Feature importance (top filters) ─────
    print("\n[6] Top-10 most important CNN features (by filter):")
    importances = clf.feature_importances_
    filter_names = list(FILTERS.keys())
    feat_per_filter = X_feat.shape[1] // len(filter_names)
    filter_importance = {}
    for i, name in enumerate(filter_names):
        start = i * feat_per_filter
        end   = start + feat_per_filter
        filter_importance[name] = importances[start:end].sum()

    for name, score in sorted(filter_importance.items(),
                               key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 200)
        print(f"  {name:<20s} {score:.4f}  {bar}")

    print("\n✓ All done! Check the saved .png files for visuals.")
    print("=" * 55)


if __name__ == "__main__":
    main()
