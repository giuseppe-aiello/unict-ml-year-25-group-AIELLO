import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models import SoftmaxClassifier, LogisticRegression, MLPClassifier
from utils import FeatureDataset

FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"
MODELS_OUT_DIR = "../results/models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_softmax(lr=0.001, momentum=0.9, weight_decay=0.0, batch_size=64, epochs=20, subdir=""):
    print(f"\n=== Training Softmax Classifier (LR={lr}, Epochs={epochs}) ===")
    
    train_ds = FeatureDataset(FEATURES_PATH, split='train')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    val_ds = FeatureDataset(FEATURES_PATH, split='val')
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    in_features = train_ds.X.shape[1]
    num_classes = len(torch.unique(train_ds.y))
    
    model = SoftmaxClassifier(in_features, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    patience = 5
    counter = 0
    best_loss = float('inf')
    
    final_path = os.path.join(MODELS_OUT_DIR, subdir)
    os.makedirs(final_path, exist_ok=True)
    best_model_path = os.path.join(final_path, "softmax_model.pth")

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = 100 * correct / total
        
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
            
        print(f"Epoch {epoch+1}/{epochs} | T_Loss: {avg_train_loss:.4f} T_Acc: {avg_train_acc:.2f}% | V_Loss: {avg_val_loss:.4f} V_Acc: {avg_val_acc:.2f}%")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early Stopping attivato all'epoca {epoch+1}. Nessun miglioramento per {patience} epoche consecutive.")
                break

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    return model, history 

def train_mlp(lr=0.001, momentum=0.9, weight_decay=0.0, batch_size=64, epochs=20,
              hidden_size=128, dropout=0.0, subdir=""):
    print(f"\n=== Training MLP (hidden={hidden_size}, dropout={dropout}, LR={lr}, Epochs={epochs}) ===")

    train_ds = FeatureDataset(FEATURES_PATH, split='train')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = FeatureDataset(FEATURES_PATH, split='val')
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    in_features = train_ds.X.shape[1]
    num_classes = len(torch.unique(train_ds.y))

    model = MLPClassifier(in_features, num_classes, hidden_size=hidden_size, dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    patience = 5
    counter = 0
    best_loss = float('inf')

    final_path = os.path.join(MODELS_OUT_DIR, subdir)
    os.makedirs(final_path, exist_ok=True)
    best_model_path = os.path.join(final_path, "mlp_model.pth")

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = 100 * correct / total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100 * val_correct / val_total

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        print(f"Epoch {epoch+1}/{epochs} | T_Loss: {avg_train_loss:.4f} T_Acc: {avg_train_acc:.2f}% | V_Loss: {avg_val_loss:.4f} V_Acc: {avg_val_acc:.2f}%")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early Stopping attivato all'epoca {epoch+1}. Nessun miglioramento per {patience} epoche consecutive.")
                break

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))

    return model, history


def train_logistic_ovr(target_class_id=14, lr=0.001, momentum=0.9, weight_decay=0.0, batch_size=64, epochs=20, subdir=""):
    print(f"\n=== Training Logistic (Class {target_class_id}) ===")
    
    train_ds = FeatureDataset(FEATURES_PATH, split='train')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    val_ds = FeatureDataset(FEATURES_PATH, split='val')
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = LogisticRegression(train_ds.X.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([40.0]).to(DEVICE))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    patience = 5
    counter = 0
    best_loss = float('inf')
    
    final_path = os.path.join(MODELS_OUT_DIR, subdir)
    os.makedirs(final_path, exist_ok=True)
    best_model_path = os.path.join(final_path, f"logistic_class_{target_class_id}.pth")

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_binary = (y_batch == target_class_id).float().view(-1, 1)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_binary)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (torch.sigmoid(logits) > 0.5).float()
            total += y_binary.size(0)
            correct += (predicted == y_binary).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = 100 * correct / total
        
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                y_binary = (y_batch == target_class_id).float().view(-1, 1)
                
                logits = model(X_batch)
                loss = criterion(logits, y_binary)
                
                val_loss += loss.item()
                predicted = (torch.sigmoid(logits) > 0.5).float()
                val_total += y_binary.size(0)
                val_correct += (predicted == y_binary).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
            
        print(f"Epoch {epoch+1}/{epochs} | T_Loss: {avg_train_loss:.4f} T_Acc: {avg_train_acc:.2f}% | V_Loss: {avg_val_loss:.4f} V_Acc: {avg_val_acc:.2f}%")
    
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early Stopping attivato all'epoca {epoch+1}. Nessun miglioramento per {patience} epoche consecutive.")
                break

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    return model, history