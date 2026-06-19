import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys

# Add analysis directory to path for imports
sys.path.append('/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/mammography-datasets/analysis')

from prepare_data import BreastDensityDataset
from train_experiment import CORAL

def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            # Forward pass through backbone only
            # In our CORAL model: features = self.backbone(x)
            feat = model.backbone(imgs)
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load EMBED samples
    embed_csv = 'mammography-datasets/analysis/final_stratified_png_dataset.csv'
    embed_df = pd.read_csv(embed_csv).sample(500, random_state=42)
    # Ensure relative paths are correct
    embed_df['image_path'] = embed_df['image_path'].apply(lambda x: os.path.join('/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets', x))

    # 2. Load IBIA samples
    ibia_metadata = '/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/mammography-datasets/database/db_service_account_access/ibia_metadata/ibia_images.csv'
    ibia_png_root = '/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/mammography-datasets/IBIA_Data_PNG/'
    ibia_df_full = pd.read_csv(ibia_metadata)
    
    def get_ibia_png_path(row):
        filename = os.path.basename(row['gcs_url']).replace('.dcm', '.png')
        folder = filename.split('_')[0]
        return os.path.join(ibia_png_root, folder, filename)
    
    ibia_df_full['image_path'] = ibia_df_full.apply(get_ibia_png_path, axis=1)
    ibia_df_full = ibia_df_full[ibia_df_full['image_path'].apply(os.path.exists)]
    ibia_df = ibia_df_full.sample(500, random_state=42).copy()
    
    # Map labels to 1-4
    label_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    ibia_df['breast_density'] = ibia_df['breast_density'].map(label_map)

    # 3. Setup Dataset/DataLoader
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # BreastDensityDataset renames 'breast_density' to 'density_label'
    embed_ds = BreastDensityDataset(embed_df[['image_path', 'breast_density']], transform=val_transform)
    ibia_ds  = BreastDensityDataset(ibia_df[['image_path', 'breast_density']], transform=val_transform)

    embed_loader = DataLoader(embed_ds, batch_size=32, shuffle=False)
    ibia_loader  = DataLoader(ibia_ds, batch_size=32, shuffle=False)

    # 4. Load Model
    print("Loading model...")
    model_path = '/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/best_coral_model.pt'
    model = CORAL(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 5. Extract Features
    print("Extracting EMBED features...")
    embed_features = extract_features(model, embed_loader, device)
    print("Extracting IBIA features...")
    ibia_features = extract_features(model, ibia_loader, device)

    # 6. Run t-SNE
    print("Running t-SNE...")
    all_features = np.concatenate([embed_features, ibia_features], axis=0)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(all_features)

    # 7. Plot
    print("Plotting results...")
    plt.figure(figsize=(10, 7))
    plt.scatter(tsne_results[:500, 0], tsne_results[:500, 1], c='blue', label='EMBED (US)', alpha=0.6, s=15)
    plt.scatter(tsne_results[500:, 0], tsne_results[500:, 1], c='red', label='IBIA (India)', alpha=0.6, s=15)
    plt.legend()
    plt.title('t-SNE Visualization of Feature Space Domain Shift\n(ResNet50 Features before CORAL head)')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
    plot_path = '/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/tsne_domain_shift.png'
    plt.savefig(plot_path, dpi=300)
    print(f"t-SNE plot saved to {plot_path}")

if __name__ == "__main__":
    main()
