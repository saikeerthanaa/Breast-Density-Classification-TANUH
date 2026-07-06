import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# Setup verbose execution logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

CSV_PATH = "/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/balanced_37k_preprocessed.csv"
IMG_DIR = "/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/preprocessed_1024x1024"

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
        
        # Open and instantly force-load into RAM
        with Image.open(img_path) as img:
            image = img.convert("RGB")
            image.load() 
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def run_debug():
    logging.info("Step 1: Checking GPU availability...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Targeting Device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logging.info(f"Current VRAM Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB")
        logging.info(f"Current VRAM Reserved: {torch.cuda.memory_reserved(0)/(1024**2):.2f} MB")

    logging.info("Step 2: Loading CSV file...")
    if not os.path.exists(CSV_PATH):
        logging.error(f"CSV file not found at: {CSV_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(CSV_PATH)
    logging.info(f"Successfully loaded CSV. Total rows: {len(df)}")
    logging.info(f"Columns present: {df.columns.tolist()}")
    logging.info(f"Splits distribution:\n{df['split'].value_counts().to_string()}")
    logging.info(f"Breast density distribution:\n{df['breast_density'].value_counts(dropna=False).to_string()}")

    logging.info("Step 3: Checking image file existence on disk...")
    df_train = df[df['split'] == 'train']
    train_paths = df_train['image_path'].tolist()
    train_labels = df_train['breast_density'].tolist()
    
    missing_count = 0
    checked_limit = min(len(train_paths), 1000)  # Check first 1000 or all
    logging.info(f"Checking first {checked_limit} image paths...")
    
    for i in range(checked_limit):
        img_path = os.path.join(IMG_DIR, train_paths[i])
        if not os.path.exists(img_path):
            logging.error(f"Missing file: {img_path}")
            missing_count += 1
            if missing_count >= 5:
                logging.error("Found 5 missing files. Aborting existence check.")
                break
                
    if missing_count > 0:
        logging.error(f"Validation failed: some image files are missing!")
        sys.exit(1)
    else:
        logging.info("Image existence check passed successfully.")

    # Setup transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    logging.info("Step 4: Initializing Dataset class...")
    dataset = MammographyDataset(train_paths, train_labels, IMG_DIR, train_transform)
    logging.info(f"Dataset length: {len(dataset)}")

    logging.info("Step 5: Testing Single-Process DataLoader (num_workers=0)...")
    loader_single = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    start_time = time.time()
    try:
        iterator = iter(loader_single)
        for i in range(3):
            logging.info(f"  Fetching batch {i+1}/3 (num_workers=0)...")
            images, labels = next(iterator)
            logging.info(f"  Batch {i+1} loaded. Images shape: {images.shape}, Labels shape: {labels.shape}")
        logging.info(f"Single-Process DataLoader passed. Time taken: {time.time() - start_time:.2f}s")
    except Exception as e:
        logging.error(f"Single-Process DataLoader failed: {e}", exc_info=True)
        sys.exit(1)

    logging.info("Step 6: Testing Multi-Process DataLoader (num_workers=4, persistent_workers=True)...")
    loader_multi = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    
    start_time = time.time()
    try:
        iterator = iter(loader_multi)
        for i in range(3):
            logging.info(f"  Fetching batch {i+1}/3 (num_workers=4)...")
            images, labels = next(iterator)
            logging.info(f"  Batch {i+1} loaded. Images shape: {images.shape}, Labels shape: {labels.shape}")
        logging.info(f"Multi-Process DataLoader passed. Time taken: {time.time() - start_time:.2f}s")
    except Exception as e:
        logging.error(f"Multi-Process DataLoader failed/hung: {e}", exc_info=True)
        sys.exit(1)

    logging.info("Step 7: Testing GPU Model Forward Pass...")
    try:
        logging.info("Initializing ResNet50 model...")
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 4)
        model = model.to(device)
        model.eval()
        
        logging.info("Moving dummy batch to device...")
        # Use a real batch from single process loader
        images = images.to(device, non_blocking=True)
        
        logging.info("Running forward pass...")
        with torch.no_grad():
            outputs = model(images)
            
        logging.info(f"Forward pass completed. Outputs shape: {outputs.shape}")
        logging.info("GPU Model Forward Pass passed successfully.")
    except Exception as e:
        logging.error(f"GPU Model Forward Pass failed: {e}", exc_info=True)
        sys.exit(1)

    logging.info("🎉 ALL TESTS PASSED! No issues found in data loading, multi-processing, or forward pass.")

if __name__ == '__main__':
    run_debug()
