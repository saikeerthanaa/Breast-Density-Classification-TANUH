import os
import sys
import cv2
import time
import logging
import pandas as pd
import numpy as np
from PIL import Image
import torch.multiprocessing as mp
from functools import partial

# Setup simple execution logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def process_single_image(row_tuple, img_dirs, output_dir):
    """
    Worker function executed by each CPU core to process an individual mammogram.
    """
    idx, row = row_tuple
    img_name = row['image_path']
    
    # 1. Resolve original image location
    img_path = None
    for d in img_dirs:
        base_name = os.path.basename(img_name)
        potential_path = os.path.join(d, base_name)
        if os.path.exists(potential_path):
            img_path = potential_path
            break
        elif os.path.exists(img_name):
            img_path = img_name
            break

    if img_path is None:
        return idx, None, False  # Mark as failed to avoid breaking the loop

    try:
        # 2. Load grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return idx, None, False
            
        # 3. Min-Max Normalization [0, 255]
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 4. Otsu Dynamic Canvas Bounding Box
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
            
        # 5. Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_cropped)
        
        # 6. High-Res Downsampling to 1024x1024 using high-quality cubic interpolation
        img_resized = cv2.resize(img_enhanced, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        
        # 7. Save out clean output file using unique naming convention
        out_filename = f"proc_{idx}_{os.path.basename(img_path)}"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, img_resized)
        
        return idx, out_filename, True
        
    except Exception as e:
        return idx, None, False

if __name__ == '__main__':
    # Define paths
    csv_path = "/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/balanced_37k_dataset.csv"
    img_dirs = [
        "/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/EMBED_new_PNG",
        "/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/datasets/EMBED_Data_PNG"
    ]
    output_dir = "/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/preprocessed_1024x1024"
    new_csv_path = "/home/tanuh/Desktop/TANUH_Keerthana/Mammography_Datasets/balanced_37k_preprocessed.csv"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load DataFrame
    df = pd.read_csv(csv_path)
    total_images = len(df)
    logging.info(f"Loaded CSV containing {total_images} target row references.")
    
    start_time = time.time()
    
    # Use 18 out of 20 CPU cores to leave a tiny system buffer overhead
    num_cores = 18
    logging.info(f"Spawning parallel pool across {num_cores} CPU cores...")
    
    # Create iterable structure for pool mapping
    row_items = list(df.iterrows())
    
    worker_func = partial(process_single_image, img_dirs=img_dirs, output_dir=output_dir)
    
    processed_count = 0
    failed_count = 0
    
    # Execute batch mapping
    with mp.Pool(processes=num_cores) as pool:
        for idx, out_filename, success in pool.imap_unordered(worker_func, row_items):
            if success:
                df.at[idx, 'image_path'] = out_filename
                processed_count += 1
            else:
                failed_count += 1
                
            # Log progress ticks every 1,000 images
            if (processed_count + failed_count) % 1000 == 0:
                elapsed = time.time() - start_time
                logging.info(f"Progress: {processed_count + failed_count}/{total_images} completed | Failed: {failed_count} | Elapsed: {elapsed:.1f}s")
                
    # Filter out failed entries if any image paths were completely unresolvable
    if failed_count > 0:
        logging.warning(f"Dropping {failed_count} unresolvable rows from final manifest index.")
        # Failed ones didn't update to a filename string, so we clear them
        df = df[df['image_path'].str.startswith('proc_')]

    # Save out modified dataset manifest tracking file
    df.to_csv(new_csv_path, index=False)
    
    total_time = time.time() - start_time
    logging.info(f"🎉 OFFLINE PROCESSING COMPLETE!")
    logging.info(f"Saved {len(df)} images to: {output_dir}")
    logging.info(f"Generated new manifest dataset file at: {new_csv_path}")
    logging.info(f"Total processing runtime: {total_time/60:.2f} minutes.")