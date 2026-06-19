import torch
import torch.optim as optim
import os
from train_experiment_new_archs import CORAL, coral_loss, coral_predict, run_epoch
from prepare_data import get_dataloaders
import numpy as np

CHECKPOINT_PATH = 'convnext_coral_best.pth'

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    batch_size = 32
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size, num_workers=8)
    
    # Initialize Model
    model = CORAL(num_classes=4, backbone_name='convnext_tiny').to(device)
    
    # Load Checkpoint if exists
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print("Checkpoint loaded.")
    else:
        print("No checkpoint found, starting from scratch.")
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training Loop
    epochs = 30
    best_kappa = -1
    
    # Optional: If you resume, you might want to evaluate first to set a proper best_kappa
    
    print("Starting/Continuing training with ConvNeXt + CORAL...")
    
    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, coral_loss, optimizer, 'coral', device, train=True)
        val_metrics   = run_epoch(model, val_loader, coral_loss, None, 'coral', device, train=False)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_metrics['loss']:.4f} | Val Kappa: {val_metrics['quadratic_kappa']:.4f}")
        
        if val_metrics['quadratic_kappa'] > best_kappa:
            best_kappa = val_metrics['quadratic_kappa']
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("  --> Model saved (best kappa).")

if __name__ == "__main__":
    train()
