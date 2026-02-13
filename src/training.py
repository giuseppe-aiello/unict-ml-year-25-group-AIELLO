import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models import SoftmaxClassifier, LogisticRegression
from utils import FeatureDataset

# Configurazioni di base
FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"
MODELS_OUT_DIR = "../results/models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_softmax(lr=0.001, momentum=0.9, epochs=20, batch_size=64, subdir=""):
    print(f"\n=== Training Softmax Classifier (LR={lr}, Epochs={epochs}) ===")
    
    train_ds = FeatureDataset(FEATURES_PATH, split='train')
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    in_features = train_ds.X.shape[1]
    num_classes = len(torch.unique(train_ds.y))
    
    model = SoftmaxClassifier(in_features, num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    # Dizionario per salvare l'andamento
    history = {'loss': [], 'acc': []}

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item() #aggiornamento dinamico per l'accuracy
            
        # abbiamo tante loss quanti batch visto che utilizziamo dataloader
        # quindi facciamo la media dopo ogni batch e conserviamo loss e acc  
        avg_loss = total_loss / len(loader)
        avg_acc = 100 * correct / total
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%")

    # Salvataggio su sottocartella di /results/models/ per piu esperimenti
    final_path = os.path.join(MODELS_OUT_DIR, subdir)

    os.makedirs(final_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_path, "softmax_model.pth"))
    
    return model, history 

def train_logistic_ovr(target_class_id=14, lr=0.001, momentum=0.9, epochs=20, batch_size=64, subdir=""):
    print(f"\n=== Training Logistic (Class {target_class_id}) ===")
    
    train_ds = FeatureDataset(FEATURES_PATH, split='train')
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    model = LogisticRegression(train_ds.X.shape[1]).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([40.0]).to(DEVICE))
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Dizionario per salvare l'andamento
    history = {'loss': [], 'acc': []}

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            # Label binaria: 1 se è la target_class, 0 altrimenti
            y_binary = (y_batch == target_class_id).float().view(-1, 1)
            
            optimizer.zero_grad()
            probs = model(X_batch)
            loss = criterion(probs, y_binary)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (torch.sigmoid(probs) > 0.5).float()
            total += y_binary.size(0)
            correct += (predicted == y_binary).sum().item()

        avg_loss = total_loss / len(loader)
        avg_acc = 100 * correct / total
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%")
    
    final_path = os.path.join(MODELS_OUT_DIR, subdir)
    os.makedirs(final_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_path, f"logistic_class_{target_class_id}.pth"))    
    
    return model, history 