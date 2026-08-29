import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import confusion_matrix, accuracy_score

FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"
SVM_DIR = "../results/models_classical/svm"
MEDIA_DIR = "../media"


def load_grid(model_dir):
    with open(os.path.join(model_dir, "grid_search_results.json")) as f:
        return json.load(f)


def plot_linear_C_sensitivity(results, out_path):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    class_weights = [None, 'balanced']

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for cw in class_weights:
        accs = [results[f"C_{C}_cw_{cw}"]['val_acc'] * 100 for C in Cs]
        ax.plot([str(c) for c in Cs], accs, marker='o', label=f"class_weight={cw}")

    ax.set_xlabel('C')
    ax.set_ylabel('Val Accuracy (%)')
    ax.set_title('SVM Lineare: sensibilita\' al parametro C')
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_rbf_C_gamma(results, out_path):
    Cs = [1, 10]
    gammas = ['scale', 0.01]

    x = np.arange(len(Cs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, gamma in enumerate(gammas):
        accs = []
        for C in Cs:
            candidates = [r['val_acc'] for k, r in results.items()
                          if r['C'] == C and r['gamma'] == gamma]
            accs.append(max(candidates) * 100)
        offset = (i - 0.5) * width
        ax.bar(x + offset, accs, width, label=f"gamma={gamma}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"C={c}" for c in Cs])
    ax.set_ylabel('Val Accuracy (%) (migliore su class_weight)')
    ax.set_title('SVM RBF: sensibilita\' a C e gamma')
    ax.set_ylim(94, 100)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm, cmap='viridis', cbar=True, ax=ax, xticklabels=5, yticklabels=5)
    ax.set_xlabel('Classe predetta')
    ax.set_ylabel('Classe reale')
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


if __name__ == "__main__":
    os.makedirs(MEDIA_DIR, exist_ok=True)

    svm_grid = load_grid(SVM_DIR)

    linear_results = svm_grid['linear']['results']
    rbf_results = svm_grid['rbf']['results']

    linear_best = linear_results[svm_grid['linear']['best_config']]
    rbf_best = rbf_results[svm_grid['rbf']['best_config']]

    data = np.load(FEATURES_PATH)
    X_te, y_te = data['X_te'], data['y_te']

    linear_model = load(os.path.join(SVM_DIR, "linear_best_model.joblib"))
    rbf_model = load(os.path.join(SVM_DIR, "rbf_best_model.joblib"))

    linear_pred = linear_model.predict(X_te)
    rbf_pred = rbf_model.predict(X_te)

    linear_test_acc = accuracy_score(y_te, linear_pred)
    rbf_test_acc = accuracy_score(y_te, rbf_pred)

    print("========================================")
    print(f"SVM Lineare    - val_acc={linear_best['val_acc']*100:.2f}%  test_acc={linear_test_acc*100:.2f}%")
    print(f"SVM RBF        - val_acc={rbf_best['val_acc']*100:.2f}%  test_acc={rbf_test_acc*100:.2f}%")
    print("========================================")

    plot_linear_C_sensitivity(linear_results, os.path.join(MEDIA_DIR, "svm_linear_C_sensitivity.png"))
    plot_rbf_C_gamma(rbf_results, os.path.join(MEDIA_DIR, "svm_rbf_C_gamma.png"))
    plot_confusion_matrix(
        y_te, rbf_pred, "SVM RBF: Confusion Matrix sul test set (43 classi)",
        os.path.join(MEDIA_DIR, "svm_rbf_confusion_matrix.png")
    )
