import os
import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from utils import FeatureDataset
from models import SoftmaxClassifier, LogisticRegression

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"
MODELS_DIR = "../results/models"
GRID_SEARCH_RESULTS = os.path.join(MODELS_DIR, "grid_search_results.json")
NUM_CLASSES = 43


def evaluate_softmax(model, test_loader):
    """
    Valuta il modello Softmax multiclasse sull'intero Test Set.
    """
    model.eval()
    model.to(DEVICE)

    all_preds = []
    all_labels = []

    print("\n>>> Avvio Valutazione Multiclasse (Softmax)...")

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f" -> [TEST RESULT] SOFTMAX Global Accuracy: {acc*100:.2f}%")

    return acc


def evaluate_ovr_global(test_loader, models_dir, num_classes=NUM_CLASSES):
    """
    Carica i 43 modelli binari OvR e per ogni immagine del test set assegna
    la classe che ha ottenuto la probabilita' (sigmoide) piu' alta.
    """
    print("\n>>> Avvio Valutazione OvR Global...")

    models = []
    for i in range(num_classes):
        model = LogisticRegression(512).to(DEVICE)
        path = os.path.join(models_dir, f"logistic_class_{i}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modello per classe {i} non trovato: {path}")
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        models.append(model)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            batch_size = X_batch.size(0)

            scores_matrix = torch.zeros(batch_size, num_classes).to(DEVICE)
            for class_idx, model in enumerate(models):
                logits = model(X_batch)
                scores_matrix[:, class_idx] = torch.sigmoid(logits).squeeze()

            _, preds = torch.max(scores_matrix, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f" -> [TEST RESULT] OVR Global Accuracy: {acc*100:.2f}%")

    return acc


def find_best_config(all_results, key):
    """
    Trova la config con la val_loss minima per la famiglia 'key' ('softmax' o 'ovr_mean').
    """
    best_name, best_hist = min(
        all_results.items(),
        key=lambda item: min(item[1][key]['val_loss'])
    )
    best_val_loss = min(best_hist[key]['val_loss'])
    return best_name, best_val_loss


if __name__ == "__main__":
    with open(GRID_SEARCH_RESULTS) as f:
        all_results = json.load(f)

    best_softmax_config, best_softmax_val_loss = find_best_config(all_results, 'softmax')
    best_ovr_config, best_ovr_val_loss = find_best_config(all_results, 'ovr_mean')

    print("========================================")
    print(f"Miglior config Softmax: {best_softmax_config} (val_loss={best_softmax_val_loss:.4f})")
    print(f"Miglior config OvR:     {best_ovr_config} (val_loss={best_ovr_val_loss:.4f})")
    print("========================================")

    test_ds = FeatureDataset(FEATURES_PATH, split='test')
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    softmax_path = os.path.join(MODELS_DIR, best_softmax_config, "softmax", "softmax_model.pth")
    softmax_model = SoftmaxClassifier(512, NUM_CLASSES).to(DEVICE)
    softmax_model.load_state_dict(torch.load(softmax_path, weights_only=True))
    softmax_test_acc = evaluate_softmax(softmax_model, test_loader)

    ovr_dir = os.path.join(MODELS_DIR, best_ovr_config, "ovr")
    ovr_test_acc = evaluate_ovr_global(test_loader, ovr_dir)

    print("\n======== Confronto Softmax vs OvR sul Test Set ========")
    print(f"Softmax ({best_softmax_config}): {softmax_test_acc*100:.2f}%")
    print(f"OvR     ({best_ovr_config}):     {ovr_test_acc*100:.2f}%")
    print("=========================================================")
