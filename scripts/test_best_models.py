import os
import sys
import torch
import pandas as pd
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Import dataset class and architecture utilities from training_script
from training_script import (
    MammographyDataset, 
    get_model, 
    get_loss_and_output_converter, 
    evaluate
)

# Standard evaluation transform
val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    parser = argparse.ArgumentParser(description="Evaluate best saved checkpoints on the test set.")
    parser.add_argument("--gpu", action="store_true", help="Run evaluation on GPU (safe, but uses some VRAM).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation.")
    args = parser.parse_args()

    # Select device
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using device: GPU (cuda:0) - Safe mode")
    else:
        device = torch.device("cpu")
        print("Using device: CPU - 100% Risk-Free Mode")

    csv_path = "/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/balanced_37k_preprocessed.csv"
    img_dir = "/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/preprocessed_1024x1024"
    
    # Load test split
    print("Loading test split metadata...")
    df = pd.read_csv(csv_path)
    df_test = df[df['split'] == 'test']
    test_paths = df_test['image_path'].tolist()
    test_labels = df_test['breast_density'].tolist()
    
    print(f"Total test images: {len(test_paths)}")
    
    # Create DataLoader
    test_ds = MammographyDataset(test_paths, test_labels, img_dir, val_test_transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Models configuration
    model_configs = [
        {'id': 'resnet50_ce', 'model_name': 'resnet50', 'loss_name': 'CE'},
        {'id': 'convnext_small_coral', 'model_name': 'convnext_small', 'loss_name': 'CORAL'},
        {'id': 'convnext_small_corn', 'model_name': 'convnext_small', 'loss_name': 'CORN'},
    ]
    
    for cfg in model_configs:
        model_id = cfg['id']
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_file = os.path.join(script_dir, f"../models/{model_id}_balanced.pth")
        
        if not os.path.exists(weights_file):
            print(f"\n[INFO] Checkpoint {weights_file} not found yet (model is still in early epochs). Skipping.")
            continue
            
        print(f"\n==================================================")
        print(f"Evaluating {model_id}...")
        print(f"==================================================")
        
        # Create a temporary copy to prevent read/write conflicts with the running training process
        temp_weights = os.path.join(script_dir, f"temp_eval_{model_id}_balanced.pth")
        import shutil
        shutil.copy(weights_file, temp_weights)
        
        try:
            # 1. Initialize model architecture
            model = get_model(cfg['model_name'], cfg['loss_name'])
            
            # 2. Load weights
            state_dict = torch.load(temp_weights, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            
            # 3. Get output converter
            _, converter = get_loss_and_output_converter(cfg['loss_name'])
            
            # 4. Run evaluation
            kappa, acc, f1 = evaluate(model, test_loader, device, converter)
            
            print(f"\nTest Results for {model_id}:")
            print(f"  - Quadratic Weighted Kappa: {kappa:.4f}")
            print(f"  - Accuracy:                  {acc:.4f}")
            print(f"  - Macro F1-Score:            {f1:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {model_id}: {e}")
        finally:
            if os.path.exists(temp_weights):
                os.remove(temp_weights)

if __name__ == "__main__":
    main()
