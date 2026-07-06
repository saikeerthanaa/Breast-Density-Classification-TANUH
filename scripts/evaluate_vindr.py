import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import pydicom
import cv2
from PIL import Image
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, confusion_matrix
import json
from datetime import datetime

# Import original get_model and converter helper from training_script
from training_script import get_model, get_loss_and_output_converter

# Constants
class_names = ['A', 'B', 'C', 'D']
VINDR_TO_BIRADS = {
    'DENSITY A': 0,
    'DENSITY B': 1,
    'DENSITY C': 2,
    'DENSITY D': 3
}

# Image transform matching the training script exactly
val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- HIGH-SPEED MULTI-THREADED VINDR DATASET CLASS ---
class VinDrDataset(Dataset):
    def __init__(self, image_mapping, transform=None):
        self.image_ids = list(image_mapping.keys())
        self.image_mapping = image_mapping
        self.transform = transform
        
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        data = self.image_mapping[img_id]
        path = data['path']
        label = data['label']
        
        # 1. Load DICOM pixel array
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)
        
        # 2. Invert pixel values if PHOTOMETRIC INTERPRETATION is MONOCHROME1 (black-on-white)
        if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = img.max() - img
            
        # 3. Min-Max Normalization to [0, 255]
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 4. Otsu Dynamic Canvas Bounding Box Crop
        _, binary_mask = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        nonzero_coords = np.argwhere(binary_mask)
        if nonzero_coords.size > 0:
            ymin, xmin = nonzero_coords.min(axis=0)
            ymax, xmax = nonzero_coords.max(axis=0)
            pad = 15
            h, w = img_norm.shape
            ymin, xmin = max(0, ymin - pad), max(0, xmin - pad)
            ymax, xmax = min(h, ymax + pad), min(w, xmax + pad)
            img_cropped = img_norm[ymin:ymax, xmin:xmax]
        else:
            img_cropped = img_norm
            
        # 5. CLAHE Contrast Enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_cropped)
        
        # 6. Resize to 1024x1024 (matching the model input resolution)
        img_resized = cv2.resize(img_enhanced, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        
        # 7. Convert to RGB (3 identical channels)
        img_rgb = np.stack([img_resized] * 3, axis=-1)
        
        # 8. Convert to PIL and apply standard transform
        img_pil = Image.fromarray(img_rgb)
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)
            
        return img_tensor, label, img_id

def evaluate_on_dataset(models_dict, dataset_mapping, dataset_name, condition, device):
    """
    Evaluates all loaded models on a dataset mapping in a fast batch mode.
    """
    results = {
        'dataset': dataset_name,
        'condition': condition,
        'timestamp': datetime.now().isoformat(),
        'models': {}
    }
    
    # Initialize DataLoader (using 8 workers for parallel CPU preprocessing)
    dataset = VinDrDataset(dataset_mapping, val_test_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    
    print(f"\nEvaluating models on {dataset_name} ({condition}). Total samples: {len(dataset)}")
    
    for model_name, model in models_dict.items():
        print(f" -> Evaluating {model_name}...")
        
        all_preds = []
        all_labels = []
        
        # Determine converter (CE vs CORAL vs CORN)
        loss_name = 'CE' if 'CE' in model_name else ('CORAL' if 'CORAL' in model_name else 'CORN')
        _, converter = get_loss_and_output_converter(loss_name)
        
        model.eval()
        with torch.no_grad():
            for images, labels, _ in dataloader:
                images = images.to(device, non_blocking=True)
                logits = model(images)
                preds = converter(logits).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                
        # Calculate metrics
        kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
        
        per_class_acc = {}
        for cls in range(4):
            cls_mask = np.array(all_labels) == cls
            if cls_mask.sum() > 0:
                per_class_acc[class_names[cls]] = float(cm[cls, cls] / cls_mask.sum())
                
        results['models'][model_name] = {
            'quadratic_weighted_kappa': float(kappa),
            'macro_accuracy': float(acc),
            'macro_f1_score': float(f1),
            'per_class_accuracy': per_class_acc,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(all_labels)
        }
        
        print(f"    - Kappa: {kappa:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Checkpoint paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_paths = {
        'ResNet50_CE': os.path.join(script_dir, '../models/resnet50_ce_balanced.pth'),
        'ConvNeXt_CORAL': os.path.join(script_dir, '../models/convnext_small_coral_balanced.pth'),
        'ConvNeXt_CORN': os.path.join(script_dir, '../models/convnext_small_corn_balanced.pth')
    }
    
    # 2. Load models using training_script get_model helper to avoid key mismatch
    print("Loading models...")
    models_dict = {}
    
    # ResNet50 + CE
    resnet50 = get_model('resnet50', 'CE')
    resnet50.load_state_dict(torch.load(checkpoint_paths['ResNet50_CE'], map_location=device))
    models_dict['ResNet50_CE'] = resnet50.to(device)
    
    # ConvNeXt + CORAL
    convnext_coral = get_model('convnext_small', 'CORAL')
    convnext_coral.load_state_dict(torch.load(checkpoint_paths['ConvNeXt_CORAL'], map_location=device))
    models_dict['ConvNeXt_CORAL'] = convnext_coral.to(device)
    
    # ConvNeXt + CORN
    convnext_corn = get_model('convnext_small', 'CORN')
    convnext_corn.load_state_dict(torch.load(checkpoint_paths['ConvNeXt_CORN'], map_location=device))
    models_dict['ConvNeXt_CORN'] = convnext_corn.to(device)
    
    print("✅ All models successfully loaded")
    
    # 3. Read VinDr annotations
    print("Loading VinDr annotations...")
    annot_df = pd.read_csv("/home/tanuh/Desktop/VINDR/breast-level_annotations.csv")
    images_dir = "/home/tanuh/Desktop/VINDR/images"
    
    # 4. Map images to paths and integer BIRADS labels
    print("Building path mappings...")
    image_mapping = {}
    for _, row in annot_df.iterrows():
        image_id = row['image_id']
        study_id = row['study_id']
        raw_density = row['breast_density']
        
        label = VINDR_TO_BIRADS.get(raw_density, None)
        if label is None:
            continue
            
        path = os.path.join(images_dir, study_id, f"{image_id}.dicom")
        if os.path.exists(path):
            image_mapping[image_id] = {
                'label': label,
                'path': path
            }
            
    print(f"✅ Mapped {len(image_mapping)} valid images")
    
    # 5. Print imbalanced class distribution
    labels_list = [v['label'] for v in image_mapping.values()]
    counts = pd.Series(labels_list).value_counts()
    
    print("\nVinDr Imbalanced Class Distribution:")
    for cls in range(4):
        cnt = counts.get(cls, 0)
        pct = (cnt / len(labels_list)) * 100
        print(f"  Class {class_names[cls]}: {cnt} ({pct:.2f}%)")
        
    # 6. Create Balanced VinDr mapping (reproducible downsampling via fixed seed)
    np.random.seed(42)
    minority_size = counts.min()
    print(f"\nMinority class size: {minority_size} (Class {class_names[counts.idxmin()]})")
    
    balanced_mapping = {}
    for cls in range(4):
        cls_images = [img_id for img_id, d in image_mapping.items() if d['label'] == cls]
        sampled = np.random.choice(cls_images, size=minority_size, replace=False)
        for img_id in sampled:
            balanced_mapping[img_id] = image_mapping[img_id]
            
    print(f"✅ Created Balanced VinDr split with {len(balanced_mapping)} images ({minority_size} per class)")
    
    # 7. Run Evaluations
    all_results = []
    
    # Imbalanced
    results_imb = evaluate_on_dataset(models_dict, image_mapping, 'VinDr', 'Imbalanced', device)
    all_results.append(results_imb)
    
    # Balanced
    results_bal = evaluate_on_dataset(models_dict, balanced_mapping, 'VinDr', 'Balanced', device)
    all_results.append(results_bal)
    
    # Save output to JSON
    output_path = "/home/tanuh/Desktop/VINDR_EVALUATION_RESULTS.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ All results successfully saved to: {output_path}")
    
    # 8. Print comparison table
    print("\n" + "="*100)
    print("COMPREHENSIVE RESULTS COMPARISON SUMMARY")
    print("="*100)
    
    # IBIA reference results
    ibia_results = {
        'Imbalanced': {'ResNet50_CE': 0.3733, 'ConvNeXt_CORAL': 0.5078, 'ConvNeXt_CORN': 0.4774},
        'Balanced': {'ResNet50_CE': 0.5088, 'ConvNeXt_CORAL': 0.7206, 'ConvNeXt_CORN': 0.6939}
    }
    
    print(f"\n{'Dataset':<15} {'Condition':<15} {'ResNet50+CE':<18} {'ConvNeXt+CORAL':<18} {'ConvNeXt+CORN':<18}")
    print("-" * 100)
    
    # IBIA
    for condition in ['Imbalanced', 'Balanced']:
        res = ibia_results[condition]
        print(f"{'IBIA':<15} {condition:<15} {res['ResNet50_CE']:<18.4f} {res['ConvNeXt_CORAL']:<18.4f} {res['ConvNeXt_CORN']:<18.4f}")
        
    # VinDr
    for condition, res_data in zip(['Imbalanced', 'Balanced'], all_results):
        res = {
            m: res_data['models'][m]['quadratic_weighted_kappa']
            for m in ['ResNet50_CE', 'ConvNeXt_CORAL', 'ConvNeXt_CORN']
        }
        print(f"{'VinDr':<15} {condition:<15} {res['ResNet50_CE']:<18.4f} {res['ConvNeXt_CORAL']:<18.4f} {res['ConvNeXt_CORN']:<18.4f}")
        
    print("\n" + "="*100)

if __name__ == '__main__':
    main()
