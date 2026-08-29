"""
Grafici di confronto TRA modelli (non specifici di un singolo modello).
Da lanciare a parte, quando tutti i modelli sono stati allenati e valutati -
non fa parte dei plot_<nome_modello>.py, che restano ciascuno specifico
del proprio modello.
"""
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import accuracy_score
from models import MLPClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"
DT_DIR = "../results/models_classical/decision_tree"
RF_DIR = "../results/models_classical/random_forest"
SVM_DIR = "../results/models_classical/svm"
ADABOOST_DIR = "../results/models_classical/adaboost"
MLP_CHECKPOINTS_DIR = "../results/models"
MLP_GRID_SEARCH_RESULTS = "../results/models_mlp/grid_search_results.json"
MEDIA_DIR = "../media"

# Presi dall'esecuzione gia' confermata di evaluation.py (vedi docs/log_modifiche.md, 2026-08-26)
SOFTMAX_VAL_ACC, SOFTMAX_TEST_ACC = 0.9527, 0.8663
OVR_VAL_ACC, OVR_TEST_ACC = 0.9867, 0.8539


def load_grid(model_dir):
    with open(os.path.join(model_dir, "grid_search_results.json")) as f:
        return json.load(f)


def best_val_acc_by_class_weight(results, key):
    none_vals = [r['val_acc'] for r in results.values() if r[key] is None]
    bal_vals = [r['val_acc'] for r in results.values() if r[key] == 'balanced']
    return max(none_vals) * 100, max(bal_vals) * 100


def plot_class_weight_across_models(model_results, out_path):
    """model_results: lista di (nome, results_dict, chiave_class_weight)"""
    labels = [m[0] for m in model_results]
    none_vals, bal_vals = [], []
    for _, results, key in model_results:
        n, b = best_val_acc_by_class_weight(results, key)
        none_vals.append(n)
        bal_vals.append(b)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, none_vals, width, label='class_weight=None')
    ax.bar(x + width / 2, bal_vals, width, label="class_weight='balanced'")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Val Accuracy (%) (migliore config in ciascun gruppo)')
    ax.set_title("Effetto di class_weight='balanced' sui modelli classici")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_dt_vs_rf(dt_val_acc, dt_test_acc, rf_val_acc, rf_test_acc, out_path):
    labels = ['Val Accuracy', 'Test Accuracy']
    dt_vals = [dt_val_acc * 100, dt_test_acc * 100]
    rf_vals = [rf_val_acc * 100, rf_test_acc * 100]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - width / 2, dt_vals, width, label='Decision Tree (singolo)')
    ax.bar(x + width / 2, rf_vals, width, label='Random Forest (ensemble)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('%')
    ax.set_title("Effetto dell'ensembling: Decision Tree vs Random Forest")
    ax.legend()

    for i, v in enumerate(dt_vals):
        ax.text(x[i] - width / 2, v + 1, f"{v:.1f}", ha='center')
    for i, v in enumerate(rf_vals):
        ax.text(x[i] + width / 2, v + 1, f"{v:.1f}", ha='center')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_all_models_comparison(model_vals, out_path):
    """model_vals: lista di (nome, val_acc, test_acc)"""
    labels = [m[0] for m in model_vals]
    val_accs = [m[1] * 100 for m in model_vals]
    test_accs = [m[2] * 100 for m in model_vals]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2, val_accs, width, label='Val Accuracy')
    ax.bar(x + width / 2, test_accs, width, label='Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('%')
    ax.set_title('Confronto di tutti i modelli (GTSRB, embedding ResNet18)')
    ax.legend()
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


if __name__ == "__main__":
    os.makedirs(MEDIA_DIR, exist_ok=True)

    dt_grid = load_grid(DT_DIR)
    rf_grid = load_grid(RF_DIR)
    svm_grid = load_grid(SVM_DIR)
    ada_grid = load_grid(ADABOOST_DIR)
    with open(MLP_GRID_SEARCH_RESULTS) as f:
        mlp_grid = json.load(f)

    dt_results = dt_grid['results']
    rf_results = rf_grid['results']
    linear_results = svm_grid['linear']['results']
    rbf_results = svm_grid['rbf']['results']
    ada_results = ada_grid['results']
    mlp_results = mlp_grid['results']

    dt_best = dt_results[dt_grid['best_config']]
    rf_best = rf_results[rf_grid['best_config']]
    linear_best = linear_results[svm_grid['linear']['best_config']]
    rbf_best = rbf_results[svm_grid['rbf']['best_config']]
    ada_best = ada_results[ada_grid['best_config']]
    mlp_best_name = mlp_grid['best_config']
    mlp_best = mlp_results[mlp_best_name]
    mlp_best_val_acc = max(mlp_best['history']['val_acc']) / 100

    data = np.load(FEATURES_PATH)
    X_te, y_te = data['X_te'], data['y_te']

    dt_model = load(os.path.join(DT_DIR, "best_model.joblib"))
    rf_model = load(os.path.join(RF_DIR, "best_model.joblib"))
    linear_model = load(os.path.join(SVM_DIR, "linear_best_model.joblib"))
    rbf_model = load(os.path.join(SVM_DIR, "rbf_best_model.joblib"))
    ada_model = load(os.path.join(ADABOOST_DIR, "best_model.joblib"))

    dt_test_acc = accuracy_score(y_te, dt_model.predict(X_te))
    rf_test_acc = accuracy_score(y_te, rf_model.predict(X_te))
    linear_test_acc = accuracy_score(y_te, linear_model.predict(X_te))
    rbf_test_acc = accuracy_score(y_te, rbf_model.predict(X_te))
    ada_test_acc = accuracy_score(y_te, ada_model.predict(X_te))

    mlp_model = MLPClassifier(512, 43, hidden_size=mlp_best['hidden_size'],
                               dropout=mlp_best['dropout']).to(DEVICE)
    mlp_model.load_state_dict(torch.load(os.path.join(MLP_CHECKPOINTS_DIR, mlp_best_name, "mlp_model.pth"), weights_only=True))
    mlp_model.eval()
    with torch.no_grad():
        X = torch.from_numpy(X_te).float().to(DEVICE)
        _, mlp_preds = torch.max(mlp_model(X), 1)
    mlp_test_acc = accuracy_score(y_te, mlp_preds.cpu().numpy())

    print("========================================")
    print(f"Softmax        - val_acc={SOFTMAX_VAL_ACC*100:.2f}%  test_acc={SOFTMAX_TEST_ACC*100:.2f}%")
    print(f"OvR            - val_acc={OVR_VAL_ACC*100:.2f}%  test_acc={OVR_TEST_ACC*100:.2f}%")
    print(f"MLP            - val_acc={mlp_best_val_acc*100:.2f}%  test_acc={mlp_test_acc*100:.2f}%")
    print(f"Decision Tree  - val_acc={dt_best['val_acc']*100:.2f}%  test_acc={dt_test_acc*100:.2f}%")
    print(f"Random Forest  - val_acc={rf_best['val_acc']*100:.2f}%  test_acc={rf_test_acc*100:.2f}%")
    print(f"SVM Lineare    - val_acc={linear_best['val_acc']*100:.2f}%  test_acc={linear_test_acc*100:.2f}%")
    print(f"SVM RBF        - val_acc={rbf_best['val_acc']*100:.2f}%  test_acc={rbf_test_acc*100:.2f}%")
    print(f"AdaBoost       - val_acc={ada_best['val_acc']*100:.2f}%  test_acc={ada_test_acc*100:.2f}%")
    print("========================================")

    plot_class_weight_across_models(
        [
            ("Decision Tree", dt_results, 'class_weight'),
            ("Random Forest", rf_results, 'class_weight'),
            ("SVM Lineare", linear_results, 'class_weight'),
            ("SVM RBF", rbf_results, 'class_weight'),
            ("AdaBoost", ada_results, 'base_class_weight'),
        ],
        os.path.join(MEDIA_DIR, "class_weight_across_models.png")
    )

    plot_dt_vs_rf(dt_best['val_acc'], dt_test_acc, rf_best['val_acc'], rf_test_acc,
                  os.path.join(MEDIA_DIR, "dt_vs_rf_comparison.png"))

    model_vals = [
        ("Softmax", SOFTMAX_VAL_ACC, SOFTMAX_TEST_ACC),
        ("OvR", OVR_VAL_ACC, OVR_TEST_ACC),
        ("MLP", mlp_best_val_acc, mlp_test_acc),
        ("Decision Tree", dt_best['val_acc'], dt_test_acc),
        ("Random Forest", rf_best['val_acc'], rf_test_acc),
        ("SVM Lineare", linear_best['val_acc'], linear_test_acc),
        ("SVM RBF", rbf_best['val_acc'], rbf_test_acc),
        ("AdaBoost", ada_best['val_acc'], ada_test_acc),
    ]
    plot_all_models_comparison(model_vals, os.path.join(MEDIA_DIR, "all_models_comparison.png"))
