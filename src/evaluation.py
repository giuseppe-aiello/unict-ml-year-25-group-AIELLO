import os,torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from utils import FeatureDataset
from models import SoftmaxClassifier, LogisticRegression

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES_PATH = "../results/features/gtsrb_resnet18_feats.npz"

############################ PROTOTIPO DI EVALUATION ################################

def evaluate_softmax(model):
    """
    Valuta il modello Softmax multiclasse sull'intero Test Set.
    """
    # Carichiamo il Test Set (split='test')
    test_ds = FeatureDataset(FEATURES_PATH, split='test')
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    model.eval()
    model.to(DEVICE)
    
    all_preds = []
    all_labels = []
    
    print(f"\n>>> Avvio Valutazione Multiclasse (Softmax)...")
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            # Per il Softmax usiamo le etichette originali (0-42)
            y_batch = y_batch.to(DEVICE)

            # Output del modello (Logits)
            outputs = model(X_batch)

            # Prendi l'indice con il valore di logit (o probabilità) maggiore
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # Calcolo accuratezza finale
    acc = accuracy_score(all_labels, all_preds)
    print(f" -> [TEST RESULT] SOFTMAX Global Accuracy: {acc*100:.2f}%")
    
    return acc

def evaluate_ovr_single(model, target_class):
    """
    Valuta un singolo classificatore binario OvR su una specifica classe.
    Il test set viene trasformato in: 1 (classe target) vs 0 (tutte le altre 42 classi).
    """
    # Carichiamo il Test Set (split='test')
    test_ds = FeatureDataset(FEATURES_PATH, split='test')
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    model.eval()
    model.to(DEVICE)
    
    all_preds = []
    all_labels = []
    
    print(f"\n>>> Avvio Valutazione Binaria OvR (Classe Target: {target_class})...")
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            # Trasformiamo le etichette reali in binarie: 1 se è la classe target, 0 altrimenti
            y_binary = (y_batch == target_class).float().to(DEVICE)

            # Output del modello (Logits)
            logits = model(X_batch)

            # Calcoliamo la probabilità con la sigmoide e applichiamo la soglia 0.5
            #
            probs = torch.sigmoid(logits).view(-1)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_binary.cpu().numpy())

    # Calcolo accuratezza binaria
    acc = accuracy_score(all_labels, all_preds)
    print(f" -> [TEST RESULT] OVR Binary Accuracy (Class {target_class}): {acc*100:.2f}%")
    
    return acc

def evaluate_ovr_global(test_loader, models_dir, num_classes=43):
    """
    Carica i 43 modelli binari e per ogni immagine del test set assegna la classe 
    che ha ottenuto la probabilità (sigmoide) più alta.
    """
    print(">>> Avvio valutazione ovr...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Carichiamo tutti i 43 modelli in una lista (o dizionario)
    models = []
    for i in range(num_classes):
        # Inizializza un modello vuoto
        model = LogisticRegression(input_dim=512).to(device) 
        # Carica i pesi
        path = os.path.join(models_dir, f"logistic_class_{i}.pth")
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
            model.eval()
            models.append(model)
        else:
            print(f"Attenzione: Modello per classe {i} non trovato!")
            return 0.0

    # 2. Loop di Valutazione
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            batch_size = X_batch.size(0)
            
            # Matrice per salvare i punteggi: [Batch_Size, 43]
            # Ogni colonna 'i' conterrà la probabilità data dal modello 'i'
            scores_matrix = torch.zeros(batch_size, num_classes).to(device)
            
            # Chiediamo a ogni modello la sua opinione
            for class_idx, model in enumerate(models):
                logits = model(X_batch)
                probs = torch.sigmoid(logits) # Probabilità [0, 1]
                scores_matrix[:, class_idx] = probs.squeeze()
            
            # 3. Decisione Finale: Chi ha il punteggio più alto vince
            # argmax sulla dimensione 1 (colonne/classi)
            _, predictions = torch.max(scores_matrix, dim=1)
            
            total += y_batch.size(0)
            correct += (predictions == y_batch).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy Totale OvR Ensemble: {accuracy:.2f}%")
    return accuracy