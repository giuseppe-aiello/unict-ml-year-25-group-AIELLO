import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from models import SoftmaxClassifier, LogisticRegression

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"
MODELS_DIR = "../results/models"
GRID_SEARCH_RESULTS = os.path.join(MODELS_DIR, "grid_search_results.json")
MEDIA_DIR = "../media"
NUM_CLASSES = 43


def find_best_config(all_results, key):
    best_name, _ = min(
        all_results.items(),
        key=lambda item: min(item[1][key]['val_loss'])
    )
    return best_name


def predict_softmax(model, X_te):
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(X_te).float().to(DEVICE)
        outputs = model(X)
        _, preds = torch.max(outputs, 1)
    return preds.cpu().numpy()


def predict_ovr(models_dir, X_te, num_classes=NUM_CLASSES):
    X = torch.from_numpy(X_te).float().to(DEVICE)
    scores = torch.zeros(X.shape[0], num_classes).to(DEVICE)
    for i in range(num_classes):
        model = LogisticRegression(512).to(DEVICE)
        path = os.path.join(models_dir, f"logistic_class_{i}.pth")
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        with torch.no_grad():
            logits = model(X)
            scores[:, i] = torch.sigmoid(logits).squeeze()
    _, preds = torch.max(scores, dim=1)
    return preds.cpu().numpy()


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


def plot_per_class_acc_vs_frequency(y_train, y_true, y_pred, title, out_path):
    train_counts = np.bincount(y_train, minlength=NUM_CLASSES)

    per_class_acc = []
    for c in range(NUM_CLASSES):
        mask = y_true == c
        per_class_acc.append(accuracy_score(y_true[mask], y_pred[mask]) * 100 if mask.sum() else np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(train_counts, per_class_acc)
    for c in range(NUM_CLASSES):
        ax.annotate(str(c), (train_counts[c], per_class_acc[c]), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('Esempi in train per classe')
    ax.set_ylabel('Test Accuracy per classe (%)')
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")

    corr = np.corrcoef(train_counts, per_class_acc)[0, 1]
    print(f"Correlazione (frequenza train, test accuracy per classe): {corr:.3f}")


if __name__ == "__main__":
    os.makedirs(MEDIA_DIR, exist_ok=True)

    with open(GRID_SEARCH_RESULTS) as f:
        all_results = json.load(f)

    best_softmax_config = find_best_config(all_results, 'softmax')
    best_ovr_config = find_best_config(all_results, 'ovr_mean')

    data = np.load(FEATURES_PATH)
    y_train = data['y_tr']
    X_te, y_te = data['X_te'], data['y_te']

    softmax_path = os.path.join(MODELS_DIR, best_softmax_config, "softmax", "softmax_model.pth")
    softmax_model = SoftmaxClassifier(512, NUM_CLASSES).to(DEVICE)
    softmax_model.load_state_dict(torch.load(softmax_path, weights_only=True))
    softmax_pred = predict_softmax(softmax_model, X_te)

    ovr_dir = os.path.join(MODELS_DIR, best_ovr_config, "ovr")
    ovr_pred = predict_ovr(ovr_dir, X_te)

    softmax_test_acc = accuracy_score(y_te, softmax_pred)
    ovr_test_acc = accuracy_score(y_te, ovr_pred)

    print("========================================")
    print(f"Softmax ({best_softmax_config}) - test_acc={softmax_test_acc*100:.2f}%")
    print(f"OvR     ({best_ovr_config}) - test_acc={ovr_test_acc*100:.2f}%")
    print("========================================")

    plot_confusion_matrix(
        y_te, softmax_pred, "Softmax: Confusion Matrix sul test set (43 classi)",
        os.path.join(MEDIA_DIR, "softmax_confusion_matrix.png")
    )
    plot_confusion_matrix(
        y_te, ovr_pred, "OvR: Confusion Matrix sul test set (43 classi)",
        os.path.join(MEDIA_DIR, "ovr_confusion_matrix.png")
    )

    plot_per_class_acc_vs_frequency(
        y_train, y_te, softmax_pred, "Softmax: accuracy per classe vs frequenza in train",
        os.path.join(MEDIA_DIR, "softmax_acc_vs_frequency.png")
    )
    plot_per_class_acc_vs_frequency(
        y_train, y_te, ovr_pred, "OvR: accuracy per classe vs frequenza in train",
        os.path.join(MEDIA_DIR, "ovr_acc_vs_frequency.png")
    )
