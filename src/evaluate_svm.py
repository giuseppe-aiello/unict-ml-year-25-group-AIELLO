import os
import json
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score

FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"
MODEL_DIR = "../results/models_classical/svm"


def main():
    data = np.load(FEATURES_PATH)
    X_te, y_te = data['X_te'], data['y_te']

    with open(os.path.join(MODEL_DIR, "grid_search_results.json")) as f:
        info = json.load(f)

    for family, model_file in [('linear', 'linear_best_model.joblib'), ('rbf', 'rbf_best_model.joblib')]:
        best_name = info[family]['best_config']
        best_config = info[family]['results'][best_name]

        clf = load(os.path.join(MODEL_DIR, model_file))
        preds = clf.predict(X_te)
        test_acc = accuracy_score(y_te, preds)

        print("========================================")
        print(f"SVM {family.upper()} - config migliore: {best_name}")
        print(f"  {best_config}")
        print(f"Val Accuracy:  {best_config['val_acc']*100:.2f}%")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        print("========================================")


if __name__ == "__main__":
    main()
