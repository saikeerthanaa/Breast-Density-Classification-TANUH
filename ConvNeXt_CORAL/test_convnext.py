import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from prepare_data import get_dataloaders
from train_experiment_new_archs import CORAL, coral_predict

# Configuration
CHECKPOINT_PATH = 'convnext_coral_best.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model():
    print(f"Using device: {DEVICE}")
    
    # Load Data
    # Note: Using the same prepare_data I modified to accept num_workers
    _, _, test_loader = get_dataloaders(batch_size=64, num_workers=4)
    
    # Initialize Model - must match training initialization
    model = CORAL(num_classes=4, backbone_name='convnext_tiny').to(DEVICE)
    
    # Load Checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Model file not found: {CHECKPOINT_PATH}")
        return

    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    print("Checkpoint loaded.")

    all_preds, all_labels = [], []

    print("Starting testing...")
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            logits = model(imgs)
            preds = coral_predict(logits)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = np.mean(y_true == y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    cm = confusion_matrix(y_true, y_pred, labels=[1,2,3,4])
    report = classification_report(y_true, y_pred, labels=[1,2,3,4], target_names=['Fatty', 'Scattered', 'Heterogeneous', 'Dense'])

    print(f"\n--- Test Results ---")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test Quadratic Kappa: {kappa:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

if __name__ == "__main__":
    test_model()
