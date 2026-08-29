import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import confusion_matrix, accuracy_score

FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"
MODEL_DIR = "../results/models_classical/decision_tree"
MEDIA_DIR = "../media"


def load_grid_results():
    with open(os.path.join(MODEL_DIR, "grid_search_results.json")) as f:
        return json.load(f)


def plot_depth_vs_acc(results, out_path):
    """Curva di bias/variance: quanto conta la profondità massima dell'albero."""
    depths = [10, 20, 30, None]
    depth_labels = ['10', '20', '30', 'None\n(no limit)']
    criteria = ['gini', 'entropy']
    class_weights = [None, 'balanced']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, cw in zip(axes, class_weights):
        for crit in criteria:
            accs = []
            for depth in depths:
                # per ogni depth, prendo il val_acc migliore tra i min_samples_leaf provati
                candidates = [
                    r['val_acc'] for r in results.values()
                    if r['max_depth'] == depth and r['criterion'] == crit and r['class_weight'] == cw
                ]
                accs.append(max(candidates) * 100)
            ax.plot(depth_labels, accs, marker='o', label=crit)
        ax.set_xlabel('max_depth')
        ax.set_title(f"class_weight={cw}")
        ax.legend()

    axes[0].set_ylabel('Val Accuracy (%) (migliore su min_samples_leaf)')
    fig.suptitle('Decision Tree: sensibilità alla profondità massima')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_class_weight_effect(results, out_path):
    """Confronto class_weight=None vs 'balanced' sulla config migliore di ciascun gruppo."""
    groups = {None: [], 'balanced': []}
    for r in results.values():
        groups[r['class_weight']].append(r)

    best_none = max(groups[None], key=lambda r: r['val_acc'])
    best_bal = max(groups['balanced'], key=lambda r: r['val_acc'])

    labels = ['class_weight=None', "class_weight='balanced'"]
    val_accs = [best_none['val_acc'] * 100, best_bal['val_acc'] * 100]
    val_f1s = [best_none['val_f1_macro'] * 100, best_bal['val_f1_macro'] * 100]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - width / 2, val_accs, width, label='Val Accuracy')
    ax.bar(x + width / 2, val_f1s, width, label='Val F1-macro')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('%')
    ax.set_title("Effetto di class_weight='balanced' (miglior config in ciascun gruppo)")
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
    ax.set_xlabel('Importance (Gini/Entropy decrease)')
    ax.set_title(f'Decision Tree: top {top_n} dimensioni più importanti (su 512)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm, cmap='viridis', cbar=True, ax=ax,
                xticklabels=5, yticklabels=5)
    ax.set_xlabel('Classe predetta')
    ax.set_ylabel('Classe reale')
    ax.set_title('Decision Tree: Confusion Matrix sul test set (43 classi)')

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
        if mask.sum() == 0:
            per_class_acc.append(np.nan)
            continue
        per_class_acc.append(accuracy_score(y_true[mask], y_pred[mask]) * 100)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(train_counts, per_class_acc)
    for c in range(num_classes):
        ax.annotate(str(c), (train_counts[c], per_class_acc[c]), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('Esempi in train per classe')
    ax.set_ylabel('Test Accuracy per classe (%)')
    ax.set_title('Decision Tree: accuracy per classe vs frequenza in train')

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
    best_model = load(os.path.join(MODEL_DIR, "best_model.joblib"))

    plot_depth_vs_acc(results, os.path.join(MEDIA_DIR, "dt_val_acc_vs_depth.png"))
    plot_class_weight_effect(results, os.path.join(MEDIA_DIR, "dt_class_weight_effect.png"))
    plot_feature_importance(best_model, os.path.join(MEDIA_DIR, "dt_feature_importance.png"))

    data = np.load(FEATURES_PATH)
    y_train = data['y_tr']
    X_te, y_te = data['X_te'], data['y_te']
    y_pred = best_model.predict(X_te)

    test_acc = accuracy_score(y_te, y_pred)
    print(f"\n(promemoria) Test Accuracy Decision Tree: {test_acc*100:.2f}%")

    plot_confusion_matrix(y_te, y_pred, os.path.join(MEDIA_DIR, "dt_confusion_matrix.png"))
    plot_per_class_acc_vs_frequency(y_train, y_te, y_pred,
                                     os.path.join(MEDIA_DIR, "dt_acc_vs_frequency.png"))
