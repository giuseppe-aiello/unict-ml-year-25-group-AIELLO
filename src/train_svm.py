import os
import json
from itertools import product
import numpy as np
from joblib import dump
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score

FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"
OUT_DIR = "../results/models_classical/svm"


def load_features():
    data = np.load(FEATURES_PATH)
    return data['X_tr'], data['y_tr'], data['X_val'], data['y_val']


def eval_config(clf, X_val, y_val):
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1_macro = f1_score(y_val, val_preds, average='macro')
    return val_acc, val_f1_macro


def grid_search_linear(X_tr, y_tr, X_val, y_val):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    class_weights = [None, 'balanced']

    results = {}
    best_val_acc = -1.0
    best_config_name = None
    best_model = None

    for C, class_weight in product(Cs, class_weights):
        config_name = f"C_{C}_cw_{class_weight}"
        print(f"[Linear SVM] Training {config_name}...")

        clf = LinearSVC(C=C, class_weight=class_weight, max_iter=5000, dual='auto')
        clf.fit(X_tr, y_tr)

        val_acc, val_f1_macro = eval_config(clf, X_val, y_val)
        results[config_name] = {
            'C': C, 'class_weight': class_weight,
            'val_acc': val_acc, 'val_f1_macro': val_f1_macro
        }
        print(f"  val_acc={val_acc*100:.2f}%  val_f1_macro={val_f1_macro:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config_name = config_name
            best_model = clf

    return results, best_config_name, best_model


def grid_search_rbf(X_tr, y_tr, X_val, y_val):
    # Griglia volutamente piccola: SVC con kernel RBF (libsvm) scala male
    # oltre qualche migliaio di esempi; su 31k campioni ogni fit puo' essere lento.
    Cs = [1, 10]
    gammas = ['scale', 0.01]
    class_weights = [None, 'balanced']

    results = {}
    best_val_acc = -1.0
    best_config_name = None
    best_model = None

    for C, gamma, class_weight in product(Cs, gammas, class_weights):
        config_name = f"C_{C}_gamma_{gamma}_cw_{class_weight}"
        print(f"[RBF SVM] Training {config_name}... (puo' volerci un po')")

        clf = SVC(kernel='rbf', C=C, gamma=gamma, class_weight=class_weight)
        clf.fit(X_tr, y_tr)

        val_acc, val_f1_macro = eval_config(clf, X_val, y_val)
        results[config_name] = {
            'C': C, 'gamma': gamma, 'class_weight': class_weight,
            'val_acc': val_acc, 'val_f1_macro': val_f1_macro
        }
        print(f"  val_acc={val_acc*100:.2f}%  val_f1_macro={val_f1_macro:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config_name = config_name
            best_model = clf

    return results, best_config_name, best_model


def main():
    X_tr, y_tr, X_val, y_val = load_features()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n=== Grid search SVM lineare (5 C x 2 class_weight = 10 config) ===")
    linear_results, linear_best_name, linear_best_model = grid_search_linear(X_tr, y_tr, X_val, y_val)

    print("\n=== Grid search SVM RBF (2 C x 2 gamma x 2 class_weight = 8 config) ===")
    rbf_results, rbf_best_name, rbf_best_model = grid_search_rbf(X_tr, y_tr, X_val, y_val)

    print("\n========================================")
    print(f"Config migliore Linear SVM: {linear_best_name} (val_acc={linear_results[linear_best_name]['val_acc']*100:.2f}%)")
    print(f"Config migliore RBF SVM:    {rbf_best_name} (val_acc={rbf_results[rbf_best_name]['val_acc']*100:.2f}%)")
    print("========================================")

    dump(linear_best_model, os.path.join(OUT_DIR, "linear_best_model.joblib"))
    dump(rbf_best_model, os.path.join(OUT_DIR, "rbf_best_model.joblib"))

    with open(os.path.join(OUT_DIR, "grid_search_results.json"), 'w') as f:
        json.dump({
            'linear': {'results': linear_results, 'best_config': linear_best_name},
            'rbf': {'results': rbf_results, 'best_config': rbf_best_name}
        }, f, indent=4)

    print(f"Modelli salvati in {OUT_DIR}/")
    print(f"Risultati salvati in {OUT_DIR}/grid_search_results.json")


if __name__ == "__main__":
    main()
