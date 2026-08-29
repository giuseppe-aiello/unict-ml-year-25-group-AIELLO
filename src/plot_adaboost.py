import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import confusion_matrix, accuracy_score

FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"
MODEL_DIR = "../results/models_classical/adaboost"
MEDIA_DIR = "../media"


def load_grid_results():
    with open(os.path.join(MODEL_DIR, "grid_search_results.json")) as f:
        return json.load(f)


def plot_depth_n_estimators(results, out_path):
    """Per ogni max_depth, come cresce val_acc con n_estimators (best su lr/class_weight)."""
    depths = [1, 2, 3]
    n_values = [50, 100, 200]

    fig, ax = plt.subplots(figsize=(7, 5))
    for depth in depths:
        accs = []
        for n in n_values:
            candidates = [r['val_acc'] for r in results.values()
                          if r['max_depth'] == depth and r['n_estimators'] == n]
            accs.append(max(candidates) * 100 if candidates else np.nan)
        ax.plot([str(n) for n in n_values], accs, marker='o', label=f"max_depth={depth}")

    ax.set_xlabel('n_estimators')
    ax.set_ylabel('Val Accuracy (%) (migliore su lr/class_weight)')
    ax.set_title('AdaBoost: depth dello stump x numero di round')
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_learning_rate_effect(results, out_path):
    """Effetto di learning_rate per depth=3 (le config che contano), per n_estimators."""
    n_values = [50, 100, 200]
    learning_rates = [0.5, 1.0]

    x = np.arange(len(n_values))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, lr in enumerate(learning_rates):
        accs = []
        for n in n_values:
            candidates = [r['val_acc'] for r in results.values()
                          if r['max_depth'] == 3 and r['n_estimators'] == n and r['learning_rate'] == lr]
            accs.append(max(candidates) * 100 if candidates else np.nan)
        offset = (i - 0.5) * width
        ax.bar(x + offset, accs, width, label=f"learning_rate={lr}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in n_values])
    ax.set_ylabel('Val Accuracy (%) (max_depth=3, migliore su class_weight)')
    ax.set_title('AdaBoost: effetto del learning_rate (depth=3)')
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_class_weight_effect(results, out_path):
    """class_weight=None vs 'balanced': confronto su val_acc E val_f1_macro."""
    groups = {None: [], 'balanced': []}
    for r in results.values():
        groups[r['base_class_weight']].append(r)

    best_none = max(groups[None], key=lambda r: r['val_acc'])
    best_bal = max(groups['balanced'], key=lambda r: r['val_acc'])

    labels = ['class_weight=None', "class_weight='balanced'"]
    val_accs = [best_none['val_acc'] * 100, best_bal['val_acc'] * 100]
    val_f1s = [best_none['val_f1_macro'] * 100, best_bal['val_f1_macro'] * 100]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - width / 2, val_accs, width, label='Val Accuracy')
    ax.bar(x + width / 2, val_f1s, width, label='Val F1-macro')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('%')
    ax.set_title("AdaBoost: class_weight='balanced' - accuracy invariata, F1-macro migliore")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm, cmap='viridis', cbar=True, ax=ax, xticklabels=5, yticklabels=5)
    ax.set_xlabel('Classe predetta')
    ax.set_ylabel('Classe reale')
    ax.set_title('AdaBoost: Confusion Matrix sul test set (43 classi)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_per_class_acc_vs_frequency(y_train, y_true, y_pred, out_path):
    num_classes = 43
    train_counts = np.bincount(y_train, minlength=num_classes)

    per_class_acc = []
    for c in range(num_classes):
        mask = y_true == c
        per_class_acc.append(accuracy_score(y_true[mask], y_pred[mask]) * 100 if mask.sum() else np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(train_counts, per_class_acc)
    for c in range(num_classes):
        ax.annotate(str(c), (train_counts[c], per_class_acc[c]), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('Esempi in train per classe')
    ax.set_ylabel('Test Accuracy per classe (%)')
    ax.set_title('AdaBoost: accuracy per classe vs frequenza in train')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")

    corr = np.corrcoef(train_counts, per_class_acc)[0, 1]
    print(f"Correlazione (frequenza train, test accuracy per classe): {corr:.3f}")


if __name__ == "__main__":
    os.makedirs(MEDIA_DIR, exist_ok=True)

    grid = load_grid_results()
    results = grid['results']
    failed = grid.get('failed_configs', {})
    best_model = load(os.path.join(MODEL_DIR, "best_model.joblib"))

    print("========================================")
    print(f"Config migliori: {grid['best_config']}")
    print(f"  val_acc={results[grid['best_config']]['val_acc']*100:.2f}%")
    print(f"Config completate: {len(results)}  |  fallite: {len(failed)}")
    print("========================================")

    plot_depth_n_estimators(results, os.path.join(MEDIA_DIR, "ada_depth_n_estimators.png"))
    plot_learning_rate_effect(results, os.path.join(MEDIA_DIR, "ada_learning_rate_effect.png"))
    plot_class_weight_effect(results, os.path.join(MEDIA_DIR, "ada_class_weight_effect.png"))

    data = np.load(FEATURES_PATH)
    y_train = data['y_tr']
    X_te, y_te = data['X_te'], data['y_te']
    y_pred = best_model.predict(X_te)

    test_acc = accuracy_score(y_te, y_pred)
    print(f"\n(promemoria) Test Accuracy AdaBoost: {test_acc*100:.2f}%")

    plot_confusion_matrix(y_te, y_pred, os.path.join(MEDIA_DIR, "ada_confusion_matrix.png"))
    plot_per_class_acc_vs_frequency(y_train, y_te, y_pred,
                                     os.path.join(MEDIA_DIR, "ada_acc_vs_frequency.png"))
