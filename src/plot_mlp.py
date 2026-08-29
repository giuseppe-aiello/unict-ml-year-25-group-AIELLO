import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from models import MLPClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"
CHECKPOINTS_DIR = "../results/models"
GRID_SEARCH_RESULTS = "../results/models_mlp/grid_search_results.json"
MEDIA_DIR = "../media"
NUM_CLASSES = 43


def load_grid():
    with open(GRID_SEARCH_RESULTS) as f:
        return json.load(f)


def plot_train_val_curve(hist, title, out_path):
    epochs = range(1, len(hist['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, hist['train_loss'], label='Train Loss')
    axes[0].plot(epochs, hist['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoca')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot(epochs, hist['train_acc'], label='Train Acc')
    axes[1].plot(epochs, hist['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoca')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_hidden_size_effect(results, out_path):
    hidden_sizes = [64, 128, 256]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    accs = []
    for h in hidden_sizes:
        candidates = [max(r['history']['val_acc']) for r in results.values() if r['hidden_size'] == h]
        accs.append(max(candidates))
    ax.plot([str(h) for h in hidden_sizes], accs, marker='o')

    ax.set_xlabel('hidden_size')
    ax.set_ylabel('Val Accuracy (%) (migliore su lr/momentum/dropout)')
    ax.set_title('MLP: effetto della dimensione del layer nascosto')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_dropout_effect(results, out_path):
    dropouts = [0.0, 0.3]

    x = np.arange(2)
    labels = ['Train Acc (finale)', 'Val Acc (migliore)']
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, d in enumerate(dropouts):
        candidates = [r for r in results.values() if r['dropout'] == d]
        best = max(candidates, key=lambda r: max(r['history']['val_acc']))
        train_acc_final = best['history']['train_acc'][-1]
        val_acc_best = max(best['history']['val_acc'])
        offset = (i - 0.5) * width
        ax.bar(x + offset, [train_acc_final, val_acc_best], width, label=f"dropout={d}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('MLP: effetto del dropout su train vs val (gap = overfitting)')
    ax.legend()

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
    ax.set_title('MLP: Confusion Matrix sul test set (43 classi)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")


def plot_per_class_acc_vs_frequency(y_train, y_true, y_pred, out_path):
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
    ax.set_title('MLP: accuracy per classe vs frequenza in train')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato {out_path}")

    corr = np.corrcoef(train_counts, per_class_acc)[0, 1]
    print(f"Correlazione (frequenza train, test accuracy per classe): {corr:.3f}")


if __name__ == "__main__":
    os.makedirs(MEDIA_DIR, exist_ok=True)

    grid = load_grid()
    results = grid['results']
    best_config_name = grid['best_config']
    best_config = results[best_config_name]

    print("========================================")
    print(f"Config migliore MLP: {best_config_name}")
    print(f"Val Loss: {min(best_config['history']['val_loss']):.4f}")
    print("========================================")

    plot_train_val_curve(
        best_config['history'], f"MLP - miglior config ({best_config_name})",
        os.path.join(MEDIA_DIR, "mlp_curve_best.png")
    )
    plot_hidden_size_effect(results, os.path.join(MEDIA_DIR, "mlp_hidden_size_effect.png"))
    plot_dropout_effect(results, os.path.join(MEDIA_DIR, "mlp_dropout_effect.png"))

    data = np.load(FEATURES_PATH)
    y_train = data['y_tr']
    X_te, y_te = data['X_te'], data['y_te']

    model_path = os.path.join(CHECKPOINTS_DIR, best_config_name, "mlp_model.pth")
    model = MLPClassifier(512, NUM_CLASSES, hidden_size=best_config['hidden_size'],
                           dropout=best_config['dropout']).to(DEVICE)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        X = torch.from_numpy(X_te).float().to(DEVICE)
        _, preds = torch.max(model(X), 1)
    y_pred = preds.cpu().numpy()

    test_acc = accuracy_score(y_te, y_pred)
    print(f"\n(promemoria) Test Accuracy MLP: {test_acc*100:.2f}%")

    plot_confusion_matrix(y_te, y_pred, os.path.join(MEDIA_DIR, "mlp_confusion_matrix.png"))
    plot_per_class_acc_vs_frequency(y_train, y_te, y_pred,
                                     os.path.join(MEDIA_DIR, "mlp_acc_vs_frequency.png"))
