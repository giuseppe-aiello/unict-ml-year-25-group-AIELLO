import os
import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from utils import FeatureDataset
from models import MLPClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"
CHECKPOINTS_DIR = "../results/models"
GRID_SEARCH_RESULTS = "../results/models_mlp/grid_search_results.json"
NUM_CLASSES = 43


def evaluate_mlp(model, test_loader):
    model.eval()
    model.to(DEVICE)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return accuracy_score(all_labels, all_preds)


if __name__ == "__main__":
    with open(GRID_SEARCH_RESULTS) as f:
        grid = json.load(f)

    best_config_name = grid['best_config']
    best_config = grid['results'][best_config_name]
    best_val_loss = min(best_config['history']['val_loss'])
    best_val_acc = max(best_config['history']['val_acc'])

    print("========================================")
    print(f"Config migliore MLP: {best_config_name}")
    print(f"  hidden_size={best_config['hidden_size']}, lr={best_config['lr']}, "
          f"momentum={best_config['momentum']}, dropout={best_config['dropout']}")
    print(f"Val Loss: {best_val_loss:.4f}")
    print(f"Val Accuracy: {best_val_acc:.2f}%")
    print("========================================")

    test_ds = FeatureDataset(FEATURES_PATH, split='test')
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model_path = os.path.join(CHECKPOINTS_DIR, best_config_name, "mlp_model.pth")
    model = MLPClassifier(512, NUM_CLASSES, hidden_size=best_config['hidden_size'],
                           dropout=best_config['dropout']).to(DEVICE)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    test_acc = evaluate_mlp(model, test_loader)
    print(f"\n[TEST RESULT] MLP Test Accuracy: {test_acc*100:.2f}%")
