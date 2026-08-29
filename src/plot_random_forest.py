import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import confusion_matrix, accuracy_score

FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"
RF_DIR = "../results/models_classical/random_forest"
MEDIA_DIR = "../media"


def load_grid_results(model_dir):
    with open(os.path.join(model_dir, "grid_search_results.json")) as f:
        return json.load(f)


def plot_n_estimators_effect(results, out_path):
    n_values = [100, 300]
    class_weights = [None, 'balanced']

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for cw in class_weights:
        accs = []
        for n in n_values:
            candidates = [r['val_acc'] for r in results.values()
                          if r['n_estimators'] == n and r['class_weight'] == cw]
            accs.append(max(candidates) * 100)
        ax.plot([str(n) for n in n_values], accs, marker='o', label=f"class_weight={cw}")

    ax.set_xlabel('n_estimators')
    ax.set_ylabel('Val Accuracy (%) (migliore su depth/leaf)')
    ax.set_title('Random Forest: effetto del numero di alberi')
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_class_weight_leaf_interaction(results, out_path):
    leaves = [1, 5]
    class_weights = [None, 'balanced']

    x = np.arange(len(leaves))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, cw in enumerate(class_weights):
        accs = []
        for leaf in leaves:
            candidates = [r['val_acc'] for r in results.values()
                          if r['min_samples_leaf'] == leaf and r['class_weight'] == cw]
            accs.append(max(candidates) * 100)
        offset = (i - 0.5) * width
        ax.bar(x + offset, accs, width, label=f"class_weight={cw}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"min_samples_leaf={l}" for l in leaves])
    ax.set_ylabel('Val Accuracy (%) (migliore su n_estimators/depth)')
    ax.set_title('Random Forest: interazione class_weight x min_samples_leaf')
    ax.set_ylim(85, 95)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_feature_importance(model, out_path, top_n=20):
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([f"dim {i}" for i in top_idx][::-1], importances[top_idx][::-1])
    ax.set_xlabel('Importance (media sugli alberi)')
    ax.set_title(f'Random Forest: top {top_n} dimensioni più importanti (su 512)')

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
    ax.set_title('Random Forest: Confusion Matrix sul test set (43 classi)')

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
    ax.set_title('Random Forest: accuracy per classe vs frequenza in train')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")

    corr = np.corrcoef(train_counts, per_class_acc)[0, 1]
    print(f"Correlazione (frequenza train, test accuracy per classe): {corr:.3f}")


if __name__ == "__main__":
    os.makedirs(MEDIA_DIR, exist_ok=True)

    rf_grid = load_grid_results(RF_DIR)
    rf_results = rf_grid['results']
    rf_best = rf_results[rf_grid['best_config']]

    rf_model = load(os.path.join(RF_DIR, "best_model.joblib"))

    data = np.load(FEATURES_PATH)
    y_train = data['y_tr']
    X_te, y_te = data['X_te'], data['y_te']

    rf_pred = rf_model.predict(X_te)
    rf_test_acc = accuracy_score(y_te, rf_pred)

    print("========================================")
    print(f"Random Forest  - val_acc={rf_best['val_acc']*100:.2f}%  test_acc={rf_test_acc*100:.2f}%")
    print("========================================")

    plot_n_estimators_effect(rf_results, os.path.join(MEDIA_DIR, "rf_n_estimators_effect.png"))
    plot_class_weight_leaf_interaction(rf_results, os.path.join(MEDIA_DIR, "rf_class_weight_leaf_interaction.png"))
    plot_feature_importance(rf_model, os.path.join(MEDIA_DIR, "rf_feature_importance.png"))
    plot_confusion_matrix(y_te, rf_pred, os.path.join(MEDIA_DIR, "rf_confusion_matrix.png"))
    plot_per_class_acc_vs_frequency(y_train, y_te, rf_pred, os.path.join(MEDIA_DIR, "rf_acc_vs_frequency.png"))
