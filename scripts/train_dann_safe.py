import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2
import pydicom
from PIL import Image
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, confusion_matrix
import json
from datetime import datetime
from tqdm import tqdm

# Import original get_model and converter helper from training_script
from training_script import get_model, get_loss_and_output_converter, MammographyDataset

# Constants
class_names = ['A', 'B', 'C', 'D']
VINDR_TO_BIRADS = {'DENSITY A': 0, 'DENSITY B': 1, 'DENSITY C': 2, 'DENSITY D': 3}
IBIA_TO_BIRADS = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

# Preprocessing transforms
val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- GRADIENT REVERSAL LAYER (GRL) ---
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GRL, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_)

# --- CONVNEXT + CORN + DANN ARCHITECTURE ---
class ConvNeXtCornDann(nn.Module):
    def __init__(self, base_model, feature_dim):
        super(ConvNeXtCornDann, self).__init__()
        # Backbone extracts features from torchvision model (remove final classifier[2])
        self.backbone = nn.Sequential(
            base_model.features,
            base_model.avgpool,
            base_model.classifier[0],  # LayerNorm2d
            base_model.classifier[1]   # Flatten
        )
        
        # CORN head (Linear layer)
        self.corn_head = base_model.classifier[2]
        
        # Domain Discriminator
        self.grl = GRL(lambda_=0.0)  # Will be dynamically updated during training steps
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Binary classification: Source (0) vs Target (1)
        )
        
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        corn_output = self.corn_head(features)
        
        reversed_features = self.grl(features)
        domain_output = self.domain_discriminator(reversed_features)
        
        if return_features:
            return corn_output, domain_output, features
        return corn_output, domain_output

# --- ON-THE-FLY TARGET DATASET LOADER ---
class UnlabeledTargetDataset(Dataset):
    def __init__(self, target_list, transform=None):
        self.target_list = target_list
        self.transform = transform
        
    def __len__(self):
        return len(self.target_list)
        
    def __getitem__(self, idx):
        item = self.target_list[idx]
        path = item['path']
        dtype = item['type']
        
        # 1. Load raw grayscale image
        if dtype == 'vindr':
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)
            if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
                img = img.max() - img
        else:  # ibia
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((1024, 1024), dtype=np.uint8)
                
        # 2. Min-Max Normalization to [0, 255]
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 3. Otsu Bounding Box Crop
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
            
        # 4. CLAHE Enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_cropped)
        
        # 5. Resize to 1024x1024
        img_resized = cv2.resize(img_enhanced, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        
        # 6. Convert to 3-channel RGB
        img_rgb = np.stack([img_resized] * 3, axis=-1)
        
        # 7. Convert to PIL and apply transform
        img_pil = Image.fromarray(img_rgb)
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)
            
        return img_tensor

# --- CORN LOSS FUNCTION ---
def corn_loss(logits, labels, num_classes=4):
    set_labels = []
    for i in range(1, num_classes):
        set_labels.append((labels >= i).float())
    set_labels = torch.stack(set_labels, dim=1)
    
    loss = 0
    for i in range(num_classes - 1):
        loss += F.binary_cross_entropy_with_logits(logits[:, i], set_labels[:, i])
    return loss / (num_classes - 1)

# --- POST-TRAINING EVALUATION HELPER ---
def evaluate_dann_model(model, dataset_mapping, is_png, device):
    all_preds = []
    all_labels = []
    
    # Instantiate dataset & dataloader (safe worker configuration)
    if is_png:
        from evaluate_ibia import IBIADataset
        dataset = IBIADataset(dataset_mapping, val_test_transform)
    else:
        from evaluate_vindr import VinDrDataset
        dataset = VinDrDataset(dataset_mapping, val_test_transform)
        
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    
    # Determine best AMP dtype for the hardware
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model.eval()
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                corn_logits, _ = model(images)
            
            # Predict labels from cumulative CORN probabilities
            probs = torch.sigmoid(corn_logits)
            preds = torch.sum(probs > 0.5, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    # Compute metrics
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return {
        'quadratic_weighted_kappa': float(kappa),
        'accuracy': float(acc),
        'macro_f1': float(f1)
    }

# --- SOURCE VALIDATION EVALUATION HELPER ---
def evaluate_source_val(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    val_loss = 0.0
    
    # Determine best AMP dtype for the hardware
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                corn_logits, _ = model(images)
                loss = corn_loss(corn_logits, labels)
                
            val_loss += loss.item()
            
            # Predict labels from cumulative CORN probabilities
            probs = torch.sigmoid(corn_logits)
            preds = torch.sum(probs > 0.5, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    acc = accuracy_score(all_labels, all_preds)
    
    return val_loss / len(val_loader), float(kappa), float(acc)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} (Safe mode activated)")
    
    # 1. Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, '../models/convnext_small_corn_balanced.pth')
    embed_csv = "/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/balanced_37k_preprocessed.csv"
    embed_img_dir = "/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/preprocessed_1024x1024"
    
    # 2. Load base model using training_script config to prevent shape mismatches
    print("Loading pre-trained ConvNeXt Small + CORN weights...")
    base_model = get_model('convnext_small', 'CORN')
    base_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    feature_dim = base_model.classifier[2].in_features
    print(f"Base model loaded. Feature dimension: {feature_dim}")
    
    # 3. Create DANN model wrapper
    model_dann = ConvNeXtCornDann(base_model, feature_dim).to(device)
    print("✅ ConvNeXt + CORN + DANN model initialized")
    
    # 4. Load EMBED Source Dataset (labeled)
    print("Setting up source domain (EMBED) dataloader...")
    df_embed = pd.read_csv(embed_csv)
    df_train = df_embed[df_embed['split'] == 'train']
    train_paths = df_train['image_path'].tolist()
    train_labels = df_train['breast_density'].tolist()
    
    source_dataset = MammographyDataset(train_paths, train_labels, embed_img_dir, val_test_transform)
    # Capped num_workers=2 to prevent CPU starvation and hardware trips
    source_loader = DataLoader(source_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    print(f"✅ Source Dataloader ready: {len(source_dataset)} images ({len(source_loader)} batches)")
    
    df_val = df_embed[df_embed['split'] == 'val']
    val_paths = df_val['image_path'].tolist()
    val_labels = df_val['breast_density'].tolist()
    val_dataset = MammographyDataset(val_paths, val_labels, embed_img_dir, val_test_transform)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    print(f"✅ Source Validation Dataloader ready: {len(val_dataset)} images ({len(val_loader)} batches)")
    
    # 5. Load Target Domain Dataset (unlabeled: VinDr + IBIA combined)
    print("Setting up target domain image pool...")
    target_images = []
    
    # A. VinDr Target
    df_vindr = pd.read_csv("/home/tanuh/Desktop/VINDR/breast-level_annotations.csv")
    vindr_dir = "/home/tanuh/Desktop/VINDR/images"
    image_mapping_vindr = {}
    for _, row in df_vindr.iterrows():
        img_id = row['image_id']
        study_id = row['study_id']
        density = row['breast_density']
        lbl = VINDR_TO_BIRADS.get(density, None)
        if lbl is not None:
            p = os.path.join(vindr_dir, study_id, f"{img_id}.dicom")
            if os.path.exists(p):
                target_images.append({'type': 'vindr', 'path': p})
                image_mapping_vindr[img_id] = {'label': lbl, 'path': p}
                
    # B. IBIA Target
    df_ibia = pd.read_csv("/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/datasets/IBIA_Data/mamos.csv")
    ibia_dir = "/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/datasets/IBIA_Data_PNG"
    image_mapping_ibia = {}
    for _, row in df_ibia.iterrows():
        raw_path = row['Image File Name (with path)']
        basename = os.path.basename(raw_path)
        name_no_ext = os.path.splitext(basename)[0]
        folder = name_no_ext.split('_')[0]
        p = os.path.join(ibia_dir, folder, f"{name_no_ext}.png")
        density = row['Breast density category']
        lbl = IBIA_TO_BIRADS.get(density, None)
        if lbl is not None and os.path.exists(p):
            target_images.append({'type': 'ibia', 'path': p})
            image_mapping_ibia[name_no_ext] = {'label': lbl, 'path': p}
            
    print(f"Total target domain images loaded: {len(target_images)} (VinDr + IBIA)")
    
    target_dataset = UnlabeledTargetDataset(target_images, val_test_transform)
    # Capped num_workers=2 to prevent CPU starvation and hardware trips
    target_loader = DataLoader(target_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    print(f"✅ Target Dataloader ready: {len(target_dataset)} images ({len(target_loader)} batches)")
    
    # 6. Set up Optimizer
    optimizer = torch.optim.Adam([
        {'params': model_dann.backbone.parameters(), 'lr': 1e-5},
        {'params': model_dann.corn_head.parameters(), 'lr': 1e-5},
        {'params': model_dann.domain_discriminator.parameters(), 'lr': 1e-4}
    ])
    
    # 7. Training Config
    num_epochs = 15
    save_interval = 5
    accumulation_steps = 4  # Gradient accumulation to keep VRAM footprint and peak GPU load smooth
    total_steps = num_epochs * len(source_loader)
    
    # Determine best AMP dtype for the hardware (Blackwell GPUs natively support BF16)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda') if amp_dtype == torch.float16 else None
    print(f"Using AMP datatype: {amp_dtype}")
    
    # Validation early stopping variables
    best_val_kappa = -1.0
    patience, patience_counter = 3, 0
    start_epoch = 0
    
    # Smooth Resuming from Checkpoint
    checkpoint_path = "/home/tanuh/Desktop/convnext_corn_dann_checkpoint.pth"
    best_weights_path = "/home/tanuh/Desktop/convnext_corn_dann_best.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"🔄 Found checkpoint at {checkpoint_path}. Resuming training smoothly...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_dann.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_kappa = checkpoint['best_val_kappa']
        patience_counter = checkpoint['patience_counter']
        print(f"   Resumed from Epoch {start_epoch + 1} (Best Val Kappa: {best_val_kappa:.4f}, Patience Counter: {patience_counter})")
    elif os.path.exists(best_weights_path):
        print(f"🔄 Found best weights at {best_weights_path}. Initializing with best validation state...")
        model_dann.load_state_dict(torch.load(best_weights_path, map_location=device))
        # Since it was saved at the end of epoch 2 (1-indexed), resume from Epoch 3 (2 in 0-indexed)
        start_epoch = 2
        best_val_kappa = 0.9200
        patience_counter = 0
        print(f"   Resuming from Epoch {start_epoch + 1} (Best Val Kappa: {best_val_kappa:.4f}, Patience Counter: {patience_counter})")
    
    print(f"\n==================================================")
    print(f"Starting Domain Adversarial Fine-Tuning (DANN)")
    print(f"==================================================\n")
    
    target_iter = iter(target_loader)
    
    for epoch in range(start_epoch, num_epochs):
        model_dann.train()
        epoch_corn_loss = 0
        epoch_domain_loss = 0
        epoch_domain_acc = 0
        
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(source_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (src_images, src_labels) in enumerate(pbar):
            # Dynamic GRL lambda parameter scheduling (scales from 0.0 to 1.0)
            step = epoch * len(source_loader) + batch_idx
            p = float(step) / total_steps
            lambda_p = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
            model_dann.grl.lambda_ = lambda_p
            
            # Get target domain batch
            try:
                tgt_images = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_images = next(target_iter)
                
            src_images = src_images.to(device, non_blocking=True)
            src_labels = src_labels.to(device, non_blocking=True)
            tgt_images = tgt_images.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                # Forward Source (predict classes + domain)
                src_corn_logits, src_domain_logits = model_dann(src_images)
                # Forward Target (predict domain)
                _, tgt_domain_logits = model_dann(tgt_images)
                
                # Task Classification Loss (CORN)
                loss_task = corn_loss(src_corn_logits, src_labels)
                
                # Domain Classification Loss (Source=0, Target=1)
                src_domain_labels = torch.zeros(src_images.size(0), dtype=torch.long, device=device)
                tgt_domain_labels = torch.ones(tgt_images.size(0), dtype=torch.long, device=device)
                loss_domain_src = F.cross_entropy(src_domain_logits, src_domain_labels)
                loss_domain_tgt = F.cross_entropy(tgt_domain_logits, tgt_domain_labels)
                loss_domain = 0.5 * (loss_domain_src + loss_domain_tgt)
                
                # Combined loss (scaled down by accumulation steps)
                total_loss = (loss_task + 0.1 * loss_domain) / accumulation_steps
                
            if scaler is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            # Gradient Step execution
            if (batch_idx + 1) % accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model_dann.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model_dann.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Compute domain classification accuracy
            with torch.no_grad():
                src_preds = torch.argmax(src_domain_logits, dim=1)
                tgt_preds = torch.argmax(tgt_domain_logits, dim=1)
                acc_src = (src_preds == src_domain_labels).float().mean().item()
                acc_tgt = (tgt_preds == tgt_domain_labels).float().mean().item()
                domain_acc = 0.5 * (acc_src + acc_tgt)
                
            epoch_corn_loss += loss_task.item()
            epoch_domain_loss += loss_domain.item()
            epoch_domain_acc += domain_acc
            
            pbar.set_postfix({
                'task_loss': f"{loss_task.item():.4f}",
                'domain_loss': f"{loss_domain.item():.4f}",
                'domain_acc': f"{domain_acc:.2f}",
                'grl_lambda': f"{lambda_p:.3f}"
            })
            
        avg_task_loss = epoch_corn_loss / len(source_loader)
        avg_domain_loss = epoch_domain_loss / len(source_loader)
        avg_domain_acc = epoch_domain_acc / len(source_loader)
        
        print(f"\nEpoch {epoch+1} Complete:")
        print(f"  Avg CORN Task Loss: {avg_task_loss:.4f}")
        print(f"  Avg Domain Loss:    {avg_domain_loss:.4f}")
        print(f"  Avg Domain Acc:     {avg_domain_acc:.4f}")
        
        # Evaluate validation set
        val_loss, val_kappa, val_acc = evaluate_source_val(model_dann, val_loader, device)
        print(f"  Source Val Loss: {val_loss:.4f} | Val Kappa: {val_kappa:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            patience_counter = 0
            torch.save(model_dann.state_dict(), "/home/tanuh/Desktop/convnext_corn_dann_best.pth")
            print(f"  🏆 New best validation Kappa! Saved model to /home/tanuh/Desktop/convnext_corn_dann_best.pth")
        else:
            patience_counter += 1
            print(f"  Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"  Early stopping triggered at Epoch {epoch+1}")
                break
                
        # Save checkpoints
        chk_path = "/home/tanuh/Desktop/convnext_corn_dann_checkpoint.pth"
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model_dann.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'best_val_kappa': best_val_kappa,
            'patience_counter': patience_counter
        }
        torch.save(checkpoint_data, chk_path)
        print(f"  💾 Saved training checkpoint to {chk_path}")
        
        if (epoch + 1) % save_interval == 0:
            chk_path_fixed = f"/home/tanuh/Desktop/convnext_corn_dann_epoch_{epoch+1}.pth"
            torch.save(model_dann.state_dict(), chk_path_fixed)
            print(f"  ✅ Saved epoch checkpoint: {chk_path_fixed}")
            
    # Load best validation model for final evaluation
    if os.path.exists("/home/tanuh/Desktop/convnext_corn_dann_best.pth"):
        model_dann.load_state_dict(torch.load("/home/tanuh/Desktop/convnext_corn_dann_best.pth", map_location=device))
        print("✅ Loaded best model weights for post-training zero-shot evaluation.")
    else:
        # Save final model if no validation improvement occurred
        final_path = "/home/tanuh/Desktop/convnext_corn_dann_final.pth"
        torch.save(model_dann.state_dict(), final_path)
        print(f"\n✅ Fine-tuning complete. Saved final model to: {final_path}")
    
    # 8. Post-Training Evaluation
    print("\n" + "="*80)
    print("RUNNING ZERO-SHOT EVALUATION ON ADAPTED DANN MODEL")
    print("="*80)
    
    # Create Balanced datasets (downsampled to minority class size)
    # A. VinDr Balanced
    np.random.seed(42)
    vindr_lbls = [v['label'] for v in image_mapping_vindr.values()]
    counts_vindr = pd.Series(vindr_lbls).value_counts()
    min_size_vindr = counts_vindr.min()
    balanced_vindr = {}
    for cls in range(4):
        cls_ids = [k for k, v in image_mapping_vindr.items() if v['label'] == cls]
        sampled = np.random.choice(cls_ids, size=min_size_vindr, replace=False)
        for i in sampled:
            balanced_vindr[i] = image_mapping_vindr[i]
            
    # B. IBIA Balanced
    ibia_lbls = [v['label'] for v in image_mapping_ibia.values()]
    counts_ibia = pd.Series(ibia_lbls).value_counts()
    min_size_ibia = counts_ibia.min()
    balanced_ibia = {}
    for cls in range(4):
        cls_ids = [k for k, v in image_mapping_ibia.items() if v['label'] == cls]
        sampled = np.random.choice(cls_ids, size=min_size_ibia, replace=False)
        for i in sampled:
            balanced_ibia[i] = image_mapping_ibia[i]
            
    # Run evaluation runs
    print("Evaluating on VinDr Imbalanced...")
    eval_vindr_imb = evaluate_dann_model(model_dann, image_mapping_vindr, is_png=False, device=device)
    
    print("Evaluating on VinDr Balanced...")
    eval_vindr_bal = evaluate_dann_model(model_dann, balanced_vindr, is_png=False, device=device)
    
    print("Evaluating on IBIA Imbalanced...")
    eval_ibia_imb = evaluate_dann_model(model_dann, image_mapping_ibia, is_png=True, device=device)
    
    print("Evaluating on IBIA Balanced...")
    eval_ibia_bal = evaluate_dann_model(model_dann, balanced_ibia, is_png=True, device=device)
    
    # Save JSON results
    output_results = [
        {'dataset': 'VinDr', 'condition': 'Imbalanced', 'quadratic_weighted_kappa': eval_vindr_imb['quadratic_weighted_kappa'], 'accuracy': eval_vindr_imb['accuracy'], 'macro_f1': eval_vindr_imb['macro_f1']},
        {'dataset': 'VinDr', 'condition': 'Balanced', 'quadratic_weighted_kappa': eval_vindr_bal['quadratic_weighted_kappa'], 'accuracy': eval_vindr_bal['accuracy'], 'macro_f1': eval_vindr_bal['macro_f1']},
        {'dataset': 'IBIA', 'condition': 'Imbalanced', 'quadratic_weighted_kappa': eval_ibia_imb['quadratic_weighted_kappa'], 'accuracy': eval_ibia_imb['accuracy'], 'macro_f1': eval_ibia_imb['macro_f1']},
        {'dataset': 'IBIA', 'condition': 'Balanced', 'quadratic_weighted_kappa': eval_ibia_bal['quadratic_weighted_kappa'], 'accuracy': eval_ibia_bal['accuracy'], 'macro_f1': eval_ibia_bal['macro_f1']}
    ]
    
    results_path = "/home/tanuh/Desktop/CONVNEXT_CORN_DANN_RESULTS.json"
    with open(results_path, 'w') as f:
        json.dump(output_results, f, indent=2)
    print(f"✅ Evaluation results successfully saved to: {results_path}")
    
    # 9. Comparison table output
    print("\n" + "="*100)
    print("COMPARISON SUMMARY: ORIGINAL CORN VS ADAPTED CORN + DANN")
    print("="*100)
    
    # Standard CORN baselines (from previous runs)
    baselines = {
        'IBIA_Imbalanced': 0.5303,
        'IBIA_Balanced': 0.7165,
        'VinDr_Imbalanced': 0.4580,
        'VinDr_Balanced': 0.7629
    }
    
    dann_kappas = {
        'IBIA_Imbalanced': eval_ibia_imb['quadratic_weighted_kappa'],
        'IBIA_Balanced': eval_ibia_bal['quadratic_weighted_kappa'],
        'VinDr_Imbalanced': eval_vindr_imb['quadratic_weighted_kappa'],
        'VinDr_Balanced': eval_vindr_bal['quadratic_weighted_kappa']
    }
    
    print(f"\n{'Dataset':<20} {'Condition':<15} {'Original CORN':<18} {'Adapted DANN+CORN':<20} {'Improvement':<15}")
    print("-" * 90)
    
    for key in baselines.keys():
        dataset, condition = key.split('_')
        b_kappa = baselines[key]
        d_kappa = dann_kappas[key]
        diff = d_kappa - b_kappa
        pct = (diff / b_kappa) * 100 if b_kappa > 0 else 0.0
        print(f"{dataset:<20} {condition:<15} {b_kappa:<18.4f} {d_kappa:<20.4f} {diff:+.4f} ({pct:+.1f}%)")
        
    print("\n" + "="*100)

if __name__ == '__main__':
    main()
