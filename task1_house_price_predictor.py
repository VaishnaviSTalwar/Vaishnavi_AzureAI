# ============================================================
# AZURE AIML ASSIGNMENT
# Student : Vaishnavi
# Task 1  : House Price Predictor with Feature Scaling
# Models  : Linear Regression + Logistic Regression (from scratch)
# ============================================================

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ──────────────────────────────────────────────
# HELPER : Min-Max Normalizer (from scratch)
# ──────────────────────────────────────────────

class MinMaxScaler:
    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)

    def transform(self, X):
        return (X - self.min_) / (self.max_ - self.min_ + 1e-8)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ──────────────────────────────────────────────
# LINEAR REGRESSION (from scratch)
# ──────────────────────────────────────────────

class HousePriceLinearRegression:

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations

    def train(self, X, y):
        rows, cols = X.shape
        self.w = np.zeros(cols)
        self.b = 0
        self.loss_history = []

        for _ in range(self.iterations):
            pred = np.dot(X, self.w) + self.b
            error = pred - y

            dw = (2 / rows) * np.dot(X.T, error)
            db = (2 / rows) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            mse = np.mean(error ** 2)
            self.loss_history.append(mse)

    def predict(self, X):
        return np.dot(X, self.w) + self.b


# ──────────────────────────────────────────────
# LOGISTIC REGRESSION (from scratch)
# ──────────────────────────────────────────────

class HouseTrendLogisticRegression:

    def __init__(self, learning_rate=0.1, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def train(self, X, y):
        rows, cols = X.shape
        self.w = np.zeros(cols)
        self.b = 0
        self.loss_history = []

        for _ in range(self.iterations):
            z = np.dot(X, self.w) + self.b
            pred = self.sigmoid(z)

            dw = (1 / rows) * np.dot(X.T, pred - y)
            db = (1 / rows) * np.sum(pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            loss = -np.mean(
                y * np.log(pred + 1e-8) + (1 - y) * np.log(1 - pred + 1e-8)
            )
            self.loss_history.append(loss)

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ──────────────────────────────────────────────
# METRICS (from scratch)
# ──────────────────────────────────────────────

def precision_score(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp + 1e-8)

def recall_score(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn + 1e-8)

def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-8)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return np.array([[tn, fp], [fn, tp]])


# ══════════════════════════════════════════════
# PART 1 : LINEAR REGRESSION — House Price
# ══════════════════════════════════════════════

def main():
    print("\n" + "=" * 50)
    print("  VAISHNAVI — TASK 1 : HOUSE PRICE PREDICTOR")
    print("=" * 50)

    print("\n─── LINEAR REGRESSION (House Price) ───")

    # Synthetic dataset: 3 features → area, rooms, age
    np.random.seed(42)
    n_samples = 300

    area  = np.random.randint(500, 3500, n_samples).astype(float)
    rooms = np.random.randint(1, 6, n_samples).astype(float)
    age   = np.random.randint(1, 40, n_samples).astype(float)

    # Price formula with noise
    price = (
        150 * area +
        20000 * rooms -
        500 * age +
        np.random.randn(n_samples) * 15000
    )

    X = np.column_stack([area, rooms, age])
    y = price

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Scale
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Train WITH scaling
    model_scaled = HousePriceLinearRegression(learning_rate=0.05, iterations=800)
    model_scaled.train(X_train_scaled, y_train)
    pred_scaled = model_scaled.predict(X_test_scaled)

    # Train WITHOUT scaling
    model_raw = HousePriceLinearRegression(learning_rate=0.0000001, iterations=800)
    model_raw.train(X_train, y_train)
    pred_raw = model_raw.predict(X_test)

    mse_scaled = mean_squared_error(y_test, pred_scaled)
    mse_raw    = mean_squared_error(y_test, pred_raw)

    print(f"  MSE  WITH feature scaling : {mse_scaled:,.2f}")
    print(f"  MSE WITHOUT feature scaling: {mse_raw:,.2f}")
    print(f"  Scaling reduces MSE by     : {((mse_raw - mse_scaled)/mse_raw)*100:.1f}%")

    # ── Plot 1: Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Vaishnavi — House Price Predictor", fontsize=14, fontweight="bold")

    axes[0].scatter(y_test, pred_scaled, alpha=0.6, color="steelblue", label="Predicted")
    axes[0].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], 'r--', label="Perfect Fit")
    axes[0].set_title("Actual vs Predicted (With Scaling)")
    axes[0].set_xlabel("Actual Price")
    axes[0].set_ylabel("Predicted Price")
    axes[0].legend()

    # ── Plot 2: Convergence comparison
    axes[1].plot(model_scaled.loss_history, label="With Scaling", color="steelblue")
    axes[1].plot(model_raw.loss_history,    label="Without Scaling", color="tomato")
    axes[1].set_title("Training Loss: Scaling vs No Scaling")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("MSE")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("house_price_regression.png", dpi=120)
    plt.show()


    # ══════════════════════════════════════════════
    # PART 2 : LOGISTIC REGRESSION — Price Trend
    # ══════════════════════════════════════════════

    print("\n─── LOGISTIC REGRESSION (Price Trend: Up / Down) ───")

    X_cls, y_cls = make_classification(
        n_samples=400,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_cls, y_cls, test_size=0.25, random_state=42
    )

    scaler2 = MinMaxScaler()
    X_tr_s = scaler2.fit_transform(X_tr)
    X_te_s  = scaler2.transform(X_te)

    log_model = HouseTrendLogisticRegression(learning_rate=0.1, iterations=1000)
    log_model.train(X_tr_s, y_tr)
    preds = log_model.predict(X_te_s)

    print(f"  Accuracy  : {accuracy(y_te, preds)*100:.2f}%")
    print(f"  Precision : {precision_score(y_te, preds):.4f}")
    print(f"  Recall    : {recall_score(y_te, preds):.4f}")
    print(f"  F1 Score  : {f1_score(y_te, preds):.4f}")

    cm = confusion_matrix(y_te, preds)
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    # ── Plot 3 & 4
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Vaishnavi — House Trend Classifier", fontsize=14, fontweight="bold")

    axes[0].plot(log_model.loss_history, color="purple")
    axes[0].set_title("Logistic Regression Loss Curve")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")

    # Confusion matrix heatmap (from scratch)
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["Pred: Down", "Pred: Up"])
    axes[1].set_yticklabels(["Actual: Down", "Actual: Up"])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]),
                         ha="center", va="center", fontsize=16,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[1].set_title("Confusion Matrix")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig("house_trend_classification.png", dpi=120)
    plt.show()

    print("\n✓ Task 1 Complete — Vaishnavi")


if __name__ == "__main__":
    main()

