import os
import sys
import time
import logging
import warnings
import gc
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from tqdm import tqdm

# Setup clean production logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
warnings.filterwarnings("ignore")

# --- HIGH-SPEED DATASET CLASS ---
class MammographyDataset(Dataset):
    def __init__(self, image_paths, breast_densities, img_dir, transform=None):
        self.image_paths = list(image_paths)
        self.breast_densities = list(breast_densities)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        raw_density = self.breast_densities[idx]
        label = int(float(raw_density)) - 1       
        
        img_path = os.path.join(self.img_dir, img_name)
        
        # Open and instantly force-load into RAM to release file descriptors immediately
        with Image.open(img_path) as img:
            image = img.convert("RGB")
            image.load() 
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
# --- Loss Function Formulations ---
class CORALLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, target):
        sets = []
        for i in range(self.num_classes - 1):
            label = (target > i).float()
            sets.append(label)
        num_tasks = self.num_classes - 1
        
        loss = 0.0
        for i in range(num_tasks):
            pred = logits[:, i]
            loss += F.binary_cross_entropy_with_logits(pred, sets[i], reduction='none')
        return loss.mean()

class CORNLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, target):
        loss = 0.0
        for i in range(self.num_classes - 1):
            mask = target >= i
            if mask.sum() == 0:
                continue
            sub_logits = logits[mask, i]
            sub_target = (target[mask] > i).float()
            loss += F.binary_cross_entropy_with_logits(sub_logits, sub_target)
        return loss / (self.num_classes - 1)

def get_loss_and_output_converter(loss_name, num_classes=4):
    if loss_name == 'CE':
        return nn.CrossEntropyLoss(), lambda logits: torch.argmax(logits, dim=1)
    elif loss_name == 'CORAL':
        def coral_converter(logits):
            probas = torch.sigmoid(logits)
            return torch.sum(probas > 0.5, dim=1)
        return CORALLoss(num_classes), coral_converter
    elif loss_name == 'CORN':
        def corn_converter(logits):
            probas = torch.sigmoid(logits)
            probas = torch.cumprod(probas, dim=1)
            return torch.sum(probas > 0.5, dim=1)
        return CORNLoss(num_classes), corn_converter

def get_model(model_name, loss_name, num_classes=4):
    out_features = num_classes if loss_name == 'CE' else num_classes - 1
    if model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, out_features)
    elif model_name == 'convnext_small':
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, out_features)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, out_features)
    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, out_features)
    return model

@torch.no_grad()
def evaluate(model, dataloader, device, converter):
    model.eval()
    all_preds, all_targets = [], []
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = converter(logits).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
    return cohen_kappa_score(np.array(all_targets), np.array(all_preds), weights='quadratic'), accuracy_score(all_targets, all_preds), f1_score(all_targets, all_preds, average='macro')

# --- Main Engine Runner ---
if __name__ == '__main__':
    # Enable TensorFloat-32 (TF32) for massive matrix multiplication speedup on GPU Tensor Cores
    torch.set_float32_matmul_precision('high')
    # Enable cuDNN autotuning to choose fastest kernel algorithms for our fixed 1024x1024 resolution
    torch.backends.cudnn.benchmark = True
    
    csv_path = "/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/balanced_37k_preprocessed.csv"
    img_dir = "/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/preprocessed_1024x1024"
    os.makedirs("results", exist_ok=True)
    metrics_csv_path = "results/EMBED_metrics.csv"
    
    # Load dataset splits
    df = pd.read_csv(csv_path)
    df_train = df[df['split'] == 'train']
    df_val = df[df['split'] == 'val']
    df_test = df[df['split'] == 'test']
    
    train_paths, train_labels = df_train['image_path'].tolist(), df_train['breast_density'].tolist()
    val_paths, val_labels = df_val['image_path'].tolist(), df_val['breast_density'].tolist()
    test_paths, test_labels = df_test['image_path'].tolist(), df_test['breast_density'].tolist()
    
    model_configs = [
        {'id': 'resnet50_ce', 'model_name': 'resnet50', 'loss_name': 'CE', 'device': 'cuda:0'},
        {'id': 'convnext_small_coral', 'model_name': 'convnext_small', 'loss_name': 'CORAL', 'device': 'cuda:0'},
        {'id': 'convnext_small_corn', 'model_name': 'convnext_small', 'loss_name': 'CORN', 'device': 'cuda:0'},
        {'id': 'convnext_tiny_coral', 'model_name': 'convnext_tiny', 'loss_name': 'CORAL', 'device': 'cuda:0'},
        {'id': 'convnext_tiny_corn', 'model_name': 'convnext_tiny', 'loss_name': 'CORN', 'device': 'cuda:0'},
        {'id': 'efficientnet_b2_coral', 'model_name': 'efficientnet_b2', 'loss_name': 'CORAL', 'device': 'cuda:0'}
    ]
    
    if len(sys.argv) > 1:
        target_id = sys.argv[1]
        model_configs = [cfg for cfg in model_configs if cfg['id'] == target_id]
        if not model_configs:
            logging.error(f"Configuration '{target_id}' not found in model_configs!")
            sys.exit(1)
        logging.info(f"Targeting single model configuration: {target_id}")
    
    # Initialize or load historical metrics
    if os.path.exists(metrics_csv_path):
        results_df = pd.read_csv(metrics_csv_path)
        final_results = results_df.to_dict('records')
    else:
        final_results = []

    completed_models = [r['Model'] for r in final_results]

    # Run each model one after the other to completely prevent multi-process locks
    for config in model_configs:
        model_id = config['id']
        if model_id in completed_models:
            logging.info(f"🏆 Model Configuration {model_id} already completed previously. Skipping.")
            continue
            
        model_name = config['model_name']
        loss_name = config['loss_name']
        device = torch.device(config['device'])
        
        logging.info(f"🚀 Starting Training Sequence for Model-ID: {model_id}")
        
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_ds = MammographyDataset(train_paths, train_labels, img_dir, train_transform)
        val_ds = MammographyDataset(val_paths, val_labels, img_dir, val_test_transform)
        test_ds = MammographyDataset(test_paths, test_labels, img_dir, val_test_transform)
        
        # Optimized DataLoader settings to prevent CPU RAM OOM when running multiple models in parallel.
        # train_loader uses 2 workers to balance data loading speed and memory overhead.
        # val_loader and test_loader run in the main process (num_workers=0) to consume 0 extra processes and RAM.
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
        
        model = get_model(model_name, loss_name).to(device)
        criterion, converter = get_loss_and_output_converter(loss_name)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        
        # Determine best AMP dtype for the hardware (Blackwell GPUs natively support BF16)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logging.info(f"Using AMP datatype: {amp_dtype}")
        
        # GradScaler is ONLY needed and supported for Float16. BF16 has same dynamic range as FP32 and doesn't require scaling.
        scaler = torch.amp.GradScaler('cuda') if amp_dtype == torch.float16 else None
        
        best_val_kappa = -1.0
        patience, patience_counter = 5, 0
        best_metrics = {}
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "../models")
        os.makedirs(models_dir, exist_ok=True)
        chk_path = os.path.join(models_dir, f"{model_id}_latest_checkpoint.pth")
        start_epoch = 1

        if os.path.exists(chk_path):
            checkpoint = torch.load(chk_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scaler is not None and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_kappa = checkpoint['best_val_kappa']
            best_metrics = checkpoint.get('best_metrics', {})
            logging.info(f"🔄 Resuming {model_id} smoothly from Epoch {start_epoch}")
        
        for epoch in range(start_epoch, 31):
            model.train()
            running_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"[{model_id}] Epoch {epoch}/30", leave=False)
            for images, targets in pbar:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                    logits = model(images)
                    loss = criterion(logits, targets)
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                
            # Run Evaluation instantly
            val_kappa, val_acc, val_f1_m = evaluate(model, val_loader, device, converter)
            logging.info(f"[{model_id}] Epoch {epoch}/30 | Loss: {running_loss/len(train_loader):.4f} | Val Kappa: {val_kappa:.4f}")
            
            if val_kappa > best_val_kappa:
                best_val_kappa = val_kappa
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(models_dir, f"{model_id}_balanced.pth"))
                
                # Run quick test set benchmark for final metrics
                test_kappa, test_acc, test_f1_m = evaluate(model, test_loader, device, converter)
                best_metrics = {
                    'Model': model_id, 'Val_Kappa': val_kappa, 'Val_Acc': val_acc, 'Val_F1_Macro': val_f1_m,
                    'Test_Kappa': test_kappa, 'Test_Acc': test_acc, 'Test_F1_Macro': test_f1_m
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"[{model_id}] Early stopping hit at epoch {epoch}.")
                    break
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_kappa': best_val_kappa,
                'best_metrics': best_metrics
            }
            if scaler is not None:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            torch.save(save_dict, chk_path)
            
            torch.cuda.empty_cache()
            gc.collect()
                    
        if best_metrics:
            final_results.append(best_metrics)
            pd.DataFrame(final_results).to_csv(metrics_csv_path, index=False)
            
        if os.path.exists(chk_path):
            os.remove(chk_path)
            
        logging.info(f"✅ Configuration {model_id} completely finalized.")
        
        # Fully free up VRAM for the next architecture in line
        del model, optimizer, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    logging.info("🎉 TRAINING SEQUENCE SUCCESSFULLY COMPLETE! Summary written cleanly inside results/EMBED_metrics.csv")