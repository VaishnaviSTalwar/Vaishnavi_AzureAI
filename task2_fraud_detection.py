# ============================================================
# AZURE AIML ASSIGNMENT
# Student : Vaishnavi
# Task 2  : Fraud Detection System
# Models  : Decision Tree, Random Forest, SVM, PCA,
#           Q-Learning, LSTM, Q-Network
# ============================================================

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report


# ──────────────────────────────────────────────
# HELPER: Manual Oversampling (from scratch)
# ──────────────────────────────────────────────

def oversample_minority(X, y):
    """Duplicate minority class samples until balanced."""
    classes, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()
    X_balanced, y_balanced = X.copy(), y.copy()

    for cls, count in zip(classes, counts):
        if count < majority_count:
            X_minority = X[y == cls]
            needed = majority_count - count
            idx = np.random.choice(len(X_minority), needed, replace=True)
            X_balanced = np.vstack([X_balanced, X_minority[idx]])
            y_balanced = np.concatenate([y_balanced, np.full(needed, cls)])

    # Shuffle
    perm = np.random.permutation(len(y_balanced))
    return X_balanced[perm], y_balanced[perm]


def f1_manual(y_true, y_pred, cls=1):
    tp = np.sum((y_pred == cls) & (y_true == cls))
    fp = np.sum((y_pred == cls) & (y_true != cls))
    fn = np.sum((y_pred != cls) & (y_true == cls))
    p  = tp / (tp + fp + 1e-8)
    r  = tp / (tp + fn + 1e-8)
    return 2 * p * r / (p + r + 1e-8)


# ══════════════════════════════════════════════
# DATASET: Imbalanced Fraud Data
# ══════════════════════════════════════════════

def main():
    print("\n" + "=" * 55)
    print("  VAISHNAVI — TASK 2 : FRAUD DETECTION SYSTEM")
    print("=" * 55)

    np.random.seed(7)

    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=7,
        weights=[0.92, 0.08],   # 92% legit, 8% fraud → imbalanced
        flip_y=0.01,
        random_state=7
    )

    print(f"\n  Dataset — Legit: {np.sum(y==0)}  |  Fraud: {np.sum(y==1)}  (imbalanced)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    # Balanced training set
    X_bal, y_bal = oversample_minority(X_train, y_train)
    print(f"  After oversampling — Legit: {np.sum(y_bal==0)}  |  Fraud: {np.sum(y_bal==1)}")


    # ══════════════════════════════════════════════
    # SECTION 1 : DECISION TREE
    # ══════════════════════════════════════════════

    print("\n─── DECISION TREE CLASSIFIER ───")

    results = {}

    for label, X_tr, y_tr in [("Imbalanced", X_train, y_train),
                                ("Balanced",   X_bal,   y_bal)]:
        dt = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=7)
        dt.fit(X_tr, y_tr)
        pred = dt.predict(X_test)
        acc  = accuracy_score(y_test, pred)
        f1   = f1_manual(y_test, pred)
        print(f"  [{label}]  Accuracy: {acc:.4f}  |  F1 (Fraud): {f1:.4f}")
        results[f"DT-{label}"] = (acc, f1)


    # ══════════════════════════════════════════════
    # SECTION 2 : RANDOM FOREST
    # ══════════════════════════════════════════════

    print("\n─── RANDOM FOREST CLASSIFIER ───")

    for label, X_tr, y_tr in [("Imbalanced", X_train, y_train),
                                ("Balanced",   X_bal,   y_bal)]:
        rf = RandomForestClassifier(n_estimators=60, random_state=7)
        rf.fit(X_tr, y_tr)
        pred = rf.predict(X_test)
        acc  = accuracy_score(y_test, pred)
        f1   = f1_manual(y_test, pred)
        print(f"  [{label}]  Accuracy: {acc:.4f}  |  F1 (Fraud): {f1:.4f}")
        results[f"RF-{label}"] = (acc, f1)

    # Feature importance
    rf_best = RandomForestClassifier(n_estimators=60, random_state=7)
    rf_best.fit(X_bal, y_bal)

    plt.figure(figsize=(9, 4))
    plt.bar(range(X.shape[1]), rf_best.feature_importances_, color="steelblue")
    plt.title("Vaishnavi — Random Forest: Feature Importance (Fraud Detection)", fontweight="bold")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("fraud_feature_importance.png", dpi=120)
    plt.show()


    # ══════════════════════════════════════════════
    # SECTION 3 : SVM
    # ══════════════════════════════════════════════

    print("\n─── SUPPORT VECTOR MACHINE ───")

    for label, X_tr, y_tr in [("Imbalanced", X_train, y_train),
                                ("Balanced",   X_bal,   y_bal)]:
        svm = SVC(kernel="linear", random_state=7)
        svm.fit(X_tr, y_tr)
        pred = svm.predict(X_test)
        acc  = accuracy_score(y_test, pred)
        f1   = f1_manual(y_test, pred)
        print(f"  [{label}]  Accuracy: {acc:.4f}  |  F1 (Fraud): {f1:.4f}")
        results[f"SVM-{label}"] = (acc, f1)


    # ══════════════════════════════════════════════
    # COMPARISON CHART
    # ══════════════════════════════════════════════

    labels_plot = list(results.keys())
    accs = [v[0] for v in results.values()]
    f1s  = [v[1] for v in results.values()]

    x = np.arange(len(labels_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width/2, accs, width, label="Accuracy", color="steelblue")
    ax.bar(x + width/2, f1s,  width, label="F1 (Fraud)", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, rotation=20, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_title("Vaishnavi — Model Comparison: Imbalanced vs Balanced", fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fraud_model_comparison.png", dpi=120)
    plt.show()


    # ══════════════════════════════════════════════
    # SECTION 4 : PCA VISUALIZATION
    # ══════════════════════════════════════════════

    print("\n─── PCA VISUALIZATION ───")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Vaishnavi — PCA: Fraud vs Legit Transactions", fontweight="bold")

    for ax, (data, title) in zip(axes, [(X_pca, "Original Imbalanced"),
                                         (pca.fit_transform(
                                             np.vstack([X, X[y==1][
                                                 np.random.choice(np.sum(y==1),
                                                 np.sum(y==0)-np.sum(y==1), replace=True)
                                             ]])
                                         ), "After Oversampling")]):
        y_vis = y if title == "Original Imbalanced" else \
                np.concatenate([y, np.ones(np.sum(y==0)-np.sum(y==1))])
        scatter = ax.scatter(data[:, 0], data[:, 1],
                             c=y_vis, cmap="coolwarm", alpha=0.5, s=10)
        ax.set_title(title)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")

    plt.colorbar(scatter, ax=axes[1], label="0=Legit / 1=Fraud")
    plt.tight_layout()
    plt.savefig("fraud_pca_visualization.png", dpi=120)
    plt.show()


    # ══════════════════════════════════════════════
    # SECTION 5 : Q-LEARNING (Fraud Patrol Agent)
    # ══════════════════════════════════════════════

    print("\n─── Q-LEARNING : FRAUD PATROL AGENT ───")

    # 5x5 grid: agent learns to reach the fraud hotspot
    # Some cells are "traps" (penalties), one is the goal
    GRID = 5
    q_table = np.zeros((GRID, GRID, 4))  # 4 actions: up/down/left/right

    traps  = {(1, 3), (3, 1), (2, 4)}
    goal   = (4, 4)
    lr, gamma, eps = 0.2, 0.9, 0.15
    rewards_ep = []

    for episode in range(500):
        pos = [0, 0]
        total_r = 0

        for _ in range(100):
            if np.random.random() < eps:
                action = np.random.randint(4)
            else:
                action = np.argmax(q_table[pos[0], pos[1]])

            nxt = pos.copy()
            if   action == 0 and pos[0] > 0:       nxt[0] -= 1
            elif action == 1 and pos[0] < GRID-1:  nxt[0] += 1
            elif action == 2 and pos[1] > 0:       nxt[1] -= 1
            elif action == 3 and pos[1] < GRID-1:  nxt[1] += 1

            if tuple(nxt) == goal:
                reward = 50
            elif tuple(nxt) in traps:
                reward = -20
            else:
                reward = -1

            best_next = np.max(q_table[nxt[0], nxt[1]])
            q_table[pos[0], pos[1], action] += lr * (
                reward + gamma * best_next - q_table[pos[0], pos[1], action]
            )
            pos = nxt
            total_r += reward

            if tuple(pos) == goal:
                break

        rewards_ep.append(total_r)

    print("  Q-Learning Training Complete (500 episodes)")
    print(f"  Avg reward (last 50 eps): {np.mean(rewards_ep[-50:]):.2f}")

    plt.figure(figsize=(9, 4))
    plt.plot(rewards_ep, alpha=0.4, color="gray", label="Per Episode")
    window = np.convolve(rewards_ep, np.ones(20)/20, mode="valid")
    plt.plot(range(19, len(rewards_ep)), window, color="steelblue", label="20-ep Moving Avg")
    plt.title("Vaishnavi — Q-Learning: Reward over Episodes", fontweight="bold")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fraud_q_learning_rewards.png", dpi=120)
    plt.show()


    # ══════════════════════════════════════════════
    # SECTION 6 : LSTM CELL DEMO
    # ══════════════════════════════════════════════

    print("\n─── LSTM CELL DEMONSTRATION ───")

    input_size  = 6
    hidden_size = 8
    combined    = input_size + hidden_size

    W = {
        "f": np.random.randn(hidden_size, combined) * 0.1,
        "i": np.random.randn(hidden_size, combined) * 0.1,
        "g": np.random.randn(hidden_size, combined) * 0.1,
        "o": np.random.randn(hidden_size, combined) * 0.1,
    }

    sigmoid = lambda z: 1 / (1 + np.exp(-z))

    h_prev = np.zeros(hidden_size)
    c_prev = np.zeros(hidden_size)
    x_t    = np.random.randn(input_size)
    concat = np.concatenate([h_prev, x_t])

    f = sigmoid(W["f"] @ concat)
    i = sigmoid(W["i"] @ concat)
    g = np.tanh(W["g"] @ concat)
    o = sigmoid(W["o"] @ concat)

    c_t = f * c_prev + i * g
    h_t = o * np.tanh(c_t)

    print(f"  Forget Gate  : {np.round(f[:4], 3)}")
    print(f"  Input Gate   : {np.round(i[:4], 3)}")
    print(f"  Cell State   : {np.round(c_t[:4], 3)}")
    print(f"  Output Gate  : {np.round(o[:4], 3)}")
    print(f"  Hidden State : {np.round(h_t[:4], 3)}")


    # ══════════════════════════════════════════════
    # SECTION 7 : Q-NETWORK DEMO
    # ══════════════════════════════════════════════

    print("\n─── Q-NETWORK DEMONSTRATION ───")

    state_dim, hidden_dim, action_dim = 8, 16, 4

    W1 = np.random.randn(state_dim, hidden_dim) * 0.1
    W2 = np.random.randn(hidden_dim, action_dim) * 0.1

    state = np.random.randn(1, state_dim)
    hidden_out = np.maximum(0, state @ W1)        # ReLU
    q_values   = hidden_out @ W2

    print(f"  Input state shape  : {state.shape}")
    print(f"  Q-values (actions) : {np.round(q_values[0], 4)}")
    print(f"  Best action        : {np.argmax(q_values)}")

    print("\n✓ Task 2 Complete — Vaishnavi")


if __name__ == "__main__":
    main()

