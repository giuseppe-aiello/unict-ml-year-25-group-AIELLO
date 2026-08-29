import os
import json
from itertools import product
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"
OUT_DIR = "../results/models_classical/random_forest"


def load_features():
    data = np.load(FEATURES_PATH)
    return data['X_tr'], data['y_tr'], data['X_val'], data['y_val']


def main():
    X_tr, y_tr, X_val, y_val = load_features()

    # max_depth: dal Decision Tree sappiamo che 10 è nettamente peggiore e che
    # 20/30/None si comportano in modo simile -> testiamo solo 20 e None (no limite).
    n_estimators_list = [100, 300]
    max_depths = [20, None]
    min_samples_leaves = [1, 5]
    class_weights = [None, 'balanced']

    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {}
    best_val_acc = -1.0
    best_config_name = None
    best_model = None

    for n_estimators, max_depth, min_samples_leaf, class_weight in product(
        n_estimators_list, max_depths, min_samples_leaves, class_weights
    ):
        config_name = f"n_{n_estimators}_depth_{max_depth}_leaf_{min_samples_leaf}_cw_{class_weight}"
        print(f"Training {config_name}...")

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            criterion='entropy',
            n_jobs=-1,
            random_state=42
        )
        clf.fit(X_tr, y_tr)

        val_preds = clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        val_f1_macro = f1_score(y_val, val_preds, average='macro')

        all_results[config_name] = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'class_weight': class_weight,
            'val_acc': val_acc,
            'val_f1_macro': val_f1_macro,
        }

        print(f"  val_acc={val_acc*100:.2f}%  val_f1_macro={val_f1_macro:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config_name = config_name
            best_model = clf

    print("\n========================================")
    print("RICERCA COMPLETATA. Configurazione ottimale (Random Forest):")
    print(f"  {best_config_name}")
    print(f"  Val Accuracy: {best_val_acc*100:.2f}%")
    print("========================================")

    dump(best_model, os.path.join(OUT_DIR, "best_model.joblib"))

    with open(os.path.join(OUT_DIR, "grid_search_results.json"), 'w') as f:
        json.dump({'results': all_results, 'best_config': best_config_name}, f, indent=4)

    print(f"Modello salvato in {OUT_DIR}/best_model.joblib")
    print(f"Risultati salvati in {OUT_DIR}/grid_search_results.json")


if __name__ == "__main__":
    main()
