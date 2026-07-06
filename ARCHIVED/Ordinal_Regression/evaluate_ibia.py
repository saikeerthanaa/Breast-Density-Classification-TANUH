import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
import json
import sys

sys.path.append('/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/mammography-datasets/analysis')

from prepare_data import BreastDensityDataset
from train_experiment import ORNN, CORAL, CORN, ornn_predict, coral_predict, corn_predict, evaluate

def prepare_ibia_df(metadata_csv, png_root):
    df = pd.read_csv(metadata_csv)
    label_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    df['density_label'] = df['breast_density'].map(label_map)
    def get_png_path(row):
        filename = os.path.basename(row['gcs_url']).replace('.dcm', '.png')
        folder = filename.split('_')[0]
        return os.path.join(png_root, folder, filename)
    df['image_path'] = df.apply(get_png_path, axis=1)
    df = df[df['image_path'].apply(os.path.exists)]
    return df[['image_path', 'density_label']]

def run_inference(model, loader, method, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            if method == 'corn':
                logits, _ = model(imgs)
            else:
                logits = model(imgs)
            
            if method == 'ornn': preds = ornn_predict(logits)
            elif method == 'coral': preds = coral_predict(logits)
            elif method == 'corn': preds = corn_predict(logits)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
png_root = '/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/mammography-datasets/IBIA_Data_PNG/'
metadata_csv = '/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/mammography-datasets/database/db_service_account_access/ibia_metadata/ibia_images.csv'

ibia_df = prepare_ibia_df(metadata_csv, png_root)
val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
ibia_dataset = BreastDensityDataset(ibia_df, transform=val_transform)
ibia_loader  = DataLoader(ibia_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

ornn_model = ORNN(num_classes=4).to(device)
ornn_model.load_state_dict(torch.load('/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/best_ornn_model.pt', map_location=device))
ornn_model.eval()

coral_model = CORAL(num_classes=4).to(device)
coral_model.load_state_dict(torch.load('/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/best_coral_model.pt', map_location=device))
coral_model.eval()

corn_model = CORN(num_classes=4).to(device)
corn_model.load_state_dict(torch.load('/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/best_corn_model.pt', map_location=device))
corn_model.eval()

y_true_ibia, y_pred_ornn  = run_inference(ornn_model,  ibia_loader, 'ornn',  device)
y_true_ibia, y_pred_coral = run_inference(coral_model, ibia_loader, 'coral', device)
y_true_ibia, y_pred_corn  = run_inference(corn_model,  ibia_loader, 'corn',  device)

results_ibia = {
    'ornn': evaluate(y_true_ibia, y_pred_ornn),
    'coral': evaluate(y_true_ibia, y_pred_coral),
    'corn': evaluate(y_true_ibia, y_pred_corn)
}

output = {}
for method in ['ornn', 'coral', 'corn']:
    m = results_ibia[method]
    output[method] = {'accuracy': float(m['accuracy']), 'MAE': float(m['MAE']), 'quadratic_kappa': float(m['quadratic_kappa']), 'confusion_matrix': m['confusion_matrix'].tolist()}
with open('/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/mammography-datasets/ibia_results.json', 'w') as f: json.dump(output, f, indent=2)
print("Evaluation complete. Results saved.")
