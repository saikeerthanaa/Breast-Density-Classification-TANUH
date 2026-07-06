import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torchvision import models
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from prepare_data import get_dataloaders
from torch.cuda.amp import GradScaler, autocast

# --- 1. Backbone ---
def get_backbone(freeze_until_layer=6):
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    layers = list(backbone.children())
    for layer in layers[:freeze_until_layer]:
        for param in layer.parameters():
            param.requires_grad = False
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim

# --- 2. Methods (OR-NN, CORAL, CORN) ---
class ORNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone, feature_dim = get_backbone()
        K = num_classes - 1
        self.binary_classifiers = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(K)])

    def forward(self, x):
        features = self.backbone(x)
        return torch.cat([clf(features) for clf in self.binary_classifiers], dim=1)

class CORAL(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone, feature_dim = get_backbone()
        K = num_classes - 1
        self.fc = nn.Linear(feature_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.tensor([0.5, 0.0, -0.5], dtype=torch.float32))

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features) + self.bias

class CORN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone, feature_dim = get_backbone()
        K = num_classes - 1
        self.classifiers = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(K)])

    def forward(self, x):
        features = self.backbone(x)
        return torch.cat([clf(features) for clf in self.classifiers], dim=1), features

# --- 3. Losses ---
def ornn_loss(logits, labels):
    K = logits.shape[1]
    batch_size = logits.shape[0]
    targets = torch.zeros(batch_size, K, device=logits.device)
    for k in range(K):
        targets[:, k] = (labels > (k + 1)).float()
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='mean')

def coral_loss(logits, labels):
    return ornn_loss(logits, labels)

def corn_loss(logits, labels, num_classes=4):
    K = num_classes - 1
    total_loss = torch.zeros(1, device=logits.device)
    for k in range(K):
        mask = labels > k
        if mask.sum() == 0: continue
        subset_logits = logits[mask, k]
        subset_labels = labels[mask]
        binary_targets = (subset_labels > (k + 1)).float()
        total_loss += nn.functional.binary_cross_entropy_with_logits(subset_logits, binary_targets, reduction='mean')
    return total_loss / K

# --- 4. Prediction ---
def ornn_predict(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1) + 1

def coral_predict(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1) + 1

def corn_predict(logits):
    probs = torch.sigmoid(logits)
    unconditional_probs = torch.cumprod(probs, dim=1)
    return (unconditional_probs > 0.5).sum(dim=1) + 1

# --- 5. Training Runner ---
def evaluate(y_true, y_pred):
    acc = np.mean(y_true == y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    cm = confusion_matrix(y_true, y_pred, labels=[1,2,3,4])
    return {'accuracy': acc, 'MAE': mae, 'quadratic_kappa': kappa, 'confusion_matrix': cm}

def run_epoch(model, loader, loss_fn, optimizer, method, device, train=True, scaler=None):
    model.train() if train else model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train):
        for i, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)

            with autocast(enabled=(device.type == 'cuda')):
                if method == 'corn':
                    logits, _ = model(imgs)
                    loss = loss_fn(logits, labels)
                else:
                    logits = model(imgs)
                    loss = loss_fn(logits, labels)

            if train:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
            
                if (i + 1) % 100 == 0:
                    print(f"Batch {i+1}/{len(loader)} processed | Loss: {loss.item():.4f}", flush=True)

            if method == 'ornn': preds = ornn_predict(logits)
            elif method == 'coral': preds = coral_predict(logits)
            elif method == 'corn': preds = corn_predict(logits)
            else: raise ValueError(f"Unknown method: {method}")
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = evaluate(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = total_loss / len(loader) if train else 0
    return metrics

def train_method(method, epochs=30, lr=1e-3, batch_size=128, patience=5):
    device = torch.device('cuda')
    print(f"\n{'='*20}\nStarting Experiment: {method.upper()}\n{'='*20}")
    print(f"Using device: {device}, batch_size: {batch_size}, AMP: True")
    
    if torch.cuda.is_available():
        pass # Placeholder for actual implementation if it was truncated earlier
