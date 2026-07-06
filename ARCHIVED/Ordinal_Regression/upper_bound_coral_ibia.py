import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
import numpy as np
import json
import sys

# Add analysis directory to path for imports
sys.path.append('/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/mammography-datasets/analysis')

from prepare_data import BreastDensityDataset
from train_experiment import CORAL, coral_loss, coral_predict, evaluate

def prepare_ibia_splits(metadata_csv, png_root):
    df = pd.read_csv(metadata_csv)
    label_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    df['breast_density'] = df['breast_density'].map(label_map)
    
    def get_png_path(row):
        filename = os.path.basename(row['gcs_url']).replace('.dcm', '.png')
        folder = filename.split('_')[0]
        return os.path.join(png_root, folder, filename)
    
    df['image_path'] = df.apply(get_png_path, axis=1)
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    
    # Stratified split: 20% train, 10% val, 70% test
    train_val_df, test_df = train_test_split(
        df, test_size=0.70, stratify=df['breast_density'], random_state=42
    )
    
    train_df, val_df = train_test_split(
        train_val_df, test_size=1/3, stratify=train_val_df['breast_density'], random_state=42
    )
    
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

def train_full_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True
    
    # Differential Learning Rates: Very slow backbone, standard head
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(),       'lr': 1e-3},
        {'params': [model.bias],                'lr': 1e-3}
    ])
    
    best_val_kappa = -1
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = coral_loss(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = coral_predict(logits)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_metrics = evaluate(np.array(val_labels), np.array(val_preds))
        val_kappa = val_metrics['quadratic_kappa']
        
        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_model_state = model.state_dict().copy()
            # To avoid memory issues with large state dicts in a loop, we could save to disk, 
            # but for ResNet50 it should be fine in memory.
            print(f"Epoch {epoch+1}: New best Val Kappa = {val_kappa:.4f}")
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Kappa: {val_kappa:.4f}")

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    png_root = '/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/mammography-datasets/IBIA_Data_PNG/'
    metadata_csv = '/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/mammography-datasets/database/db_service_account_access/ibia_metadata/ibia_images.csv'

    train_df, val_df, test_df = prepare_ibia_splits(metadata_csv, png_root)
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = BreastDensityDataset(train_df, transform=val_transform)
    val_ds   = BreastDensityDataset(val_df, transform=val_transform)
    test_ds  = BreastDensityDataset(test_df, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    print("Loading pre-trained CORAL model (starting point)...")
    model_path = '/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/best_coral_model.pt'
    model = CORAL(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print("Starting end-to-end full model training (Upper Bound)...")
    model = train_full_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3)

    print("Evaluating Upper Bound on IBIA Test Set (70%)...")
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = coral_predict(logits)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    metrics = evaluate(np.array(test_labels), np.array(test_preds))

    print("\nUpper Bound CORAL Results on IBIA Test Set:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"MAE:      {metrics['MAE']:.4f}")
    print(f"Quadratic Kappa: {metrics['quadratic_kappa']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])

    output = {
        'upper_bound_coral': {
            'accuracy': float(metrics['accuracy']),
            'MAE': float(metrics['MAE']),
            'quadratic_kappa': float(metrics['quadratic_kappa']),
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'baseline_kappa': 0.4844,
            'recalibrated_kappa': 0.5746
        }
    }
    
    output_path = '/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/mammography-datasets/ibia_upper_bound_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
