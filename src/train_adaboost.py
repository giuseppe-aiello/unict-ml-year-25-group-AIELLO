import os
import json
from itertools import product
import numpy as np
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score

FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"
OUT_DIR = "../results/models_classical/adaboost"


def load_features():
    data = np.load(FEATURES_PATH)
    return data['X_tr'], data['y_tr'], data['X_val'], data['y_val']


def main():
    X_tr, y_tr, X_val, y_val = load_features()

    # Weak learner = Decision Stump come nel LAB7 (max_depth=1), ma un vero
    # max_depth=1 ha solo 2 foglie -> puo' predire al massimo 2 classi su 43,
    # ed e' strutturalmente incapace di reggere un problema a 43 classi (il
    # lab lo usa su un problema binario, make_moons). Per questo max_depth
    # diventa un iperparametro: 1 (vero stump, probabilmente fallisce), 2, 3.
    max_depths = [1, 2, 3]
    n_estimators_list = [50, 100, 200]
    learning_rates = [0.5, 1.0]
    base_class_weights = [None, 'balanced']

    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {}
    failed_configs = {}
    best_val_acc = -1.0
    best_config_name = None
    best_model = None

    for max_depth, n_estimators, learning_rate, base_cw in product(
        max_depths, n_estimators_list, learning_rates, base_class_weights
    ):
        config_name = f"depth_{max_depth}_n_{n_estimators}_lr_{learning_rate}_stumpcw_{base_cw}"
        print(f"Training {config_name}...")

        stump = DecisionTreeClassifier(max_depth=max_depth, class_weight=base_cw, random_state=42)
        clf = AdaBoostClassifier(
            estimator=stump,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )

        try:
            clf.fit(X_tr, y_tr)
        except ValueError as e:
            print(f"  FALLITO: {e}")
            failed_configs[config_name] = {
                'max_depth': max_depth, 'n_estimators': n_estimators,
                'learning_rate': learning_rate, 'base_class_weight': base_cw,
                'error': str(e)
            }
            continue

        val_preds = clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        val_f1_macro = f1_score(y_val, val_preds, average='macro')

        all_results[config_name] = {
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'base_class_weight': base_cw,
            'val_acc': val_acc,
            'val_f1_macro': val_f1_macro,
        }

        print(f"  val_acc={val_acc*100:.2f}%  val_f1_macro={val_f1_macro:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config_name = config_name
            best_model = clf

    if failed_configs:
        print(f"\n{len(failed_configs)} config fallite (weak learner peggio del caso), escluse dalla selezione:")
        for name in failed_configs:
            print(f"  - {name}")

    print("\n========================================")
    print("RICERCA COMPLETATA. Configurazione ottimale (AdaBoost):")
    print(f"  {best_config_name}")
    print(f"  Val Accuracy: {best_val_acc*100:.2f}%")
    print("========================================")

    dump(best_model, os.path.join(OUT_DIR, "best_model.joblib"))

    with open(os.path.join(OUT_DIR, "grid_search_results.json"), 'w') as f:
        json.dump({
            'results': all_results,
            'best_config': best_config_name,
            'failed_configs': failed_configs
        }, f, indent=4)

    print(f"Modello salvato in {OUT_DIR}/best_model.joblib")
    print(f"Risultati salvati in {OUT_DIR}/grid_search_results.json")


if __name__ == "__main__":
    main()
