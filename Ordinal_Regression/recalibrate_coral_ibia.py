import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import os
import json
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

sys.path.append('/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/mammography-datasets/analysis')
from prepare_data import BreastDensityDataset
from train_experiment_new_archs import CORAL, coral_loss, coral_predict, evaluate

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

    # Same split as ResNet: 20% train, 10% val, 70% test
    train_val_df, test_df = train_test_split(
        df, test_size=0.70, stratify=df['breast_density'], random_state=42
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=1/3, stratify=train_val_df['breast_density'], random_state=42
    )
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df


def train_head(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Unfreeze head only
    model.fc.weight.requires_grad = True
    model.bias.requires_grad = True

    optimizer = torch.optim.Adam([
        {'params': model.fc.parameters()},
        {'params': [model.bias]}
    ], lr=lr)

    best_val_kappa = -1
    best_head_state = None

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
            best_head_state = {
                'fc_weight': model.fc.weight.data.clone(),
                'bias': model.bias.data.clone()
            }
            print(f"Epoch {epoch+1}: New best Val Kappa = {val_kappa:.4f}")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Kappa: {val_kappa:.4f}")

    if best_head_state:
        model.fc.weight.data.copy_(best_head_state['fc_weight'])
        model.bias.data.copy_(best_head_state['bias'])

    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    png_root     = '/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/datasets/IBIA_Data_PNG/'
    metadata_csv = '/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/mammography-datasets/database/db_service_account_access/ibia_metadata/ibia_images.csv'
    model_path   = '/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/models/convnext_coral_best.pth'

    train_df, val_df, test_df = prepare_ibia_splits(metadata_csv, png_root)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(BreastDensityDataset(train_df, transform), batch_size=32, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(BreastDensityDataset(val_df,   transform), batch_size=32, shuffle=False, num_workers=4)
    test_loader  = DataLoader(BreastDensityDataset(test_df,  transform), batch_size=32, shuffle=False, num_workers=4)

    print("Loading ConvNeXt + CORAL model...")
    model = CORAL(num_classes=4, backbone_name='convnext_tiny')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    print("Starting head-only recalibration...")
    model = train_head(model, train_loader, val_loader, device, epochs=50, lr=1e-3)

    print("\nEvaluating on IBIA Test Set (70%)...")
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = coral_predict(model(imgs))
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    metrics = evaluate(np.array(test_labels), np.array(test_preds))
    print(f"\nAccuracy:        {metrics['accuracy']:.4f}")
    print(f"MAE:             {metrics['MAE']:.4f}")
    print(f"Quadratic Kappa: {metrics['quadratic_kappa']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"\nBaseline zero-shot Kappa (ConvNeXt): 0.4514")

    output = {
        'recalibrated_convnext_coral_head': {
            'accuracy': float(metrics['accuracy']),
            'MAE': float(metrics['MAE']),
            'quadratic_kappa': float(metrics['quadratic_kappa']),
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'baseline_zeroshot_kappa': 0.4514
        }
    }
    out_path = '/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/mammography-datasets/ibia_convnext_recalibration_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()