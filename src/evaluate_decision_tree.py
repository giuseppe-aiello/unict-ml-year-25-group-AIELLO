import os
import json
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score

FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"
MODEL_DIR = "../results/models_classical/decision_tree"


def main():
    data = np.load(FEATURES_PATH)
    X_te, y_te = data['X_te'], data['y_te']

    with open(os.path.join(MODEL_DIR, "grid_search_results.json")) as f:
        info = json.load(f)
    best_config_name = info['best_config']
    best_config = info['results'][best_config_name]

    clf = load(os.path.join(MODEL_DIR, "best_model.joblib"))

    preds = clf.predict(X_te)
    test_acc = accuracy_score(y_te, preds)

    print("========================================")
    print(f"Decision Tree - config migliore: {best_config_name}")
    print(f"  max_depth={best_config['max_depth']}, min_samples_leaf={best_config['min_samples_leaf']}, "
          f"criterion={best_config['criterion']}, class_weight={best_config['class_weight']}")
    print(f"Val Accuracy:  {best_config['val_acc']*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print("========================================")


if __name__ == "__main__":
    main()
