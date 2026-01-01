import cv2
import os
import numpy as np

# Define your dataset path
DATASET_DIR = "./dataset"

def create_colored_variations():
    print("--- GENERATING RED & BLUE VARIATIONS ---")
    
    # Loop through all classes (0, 1, 2, 3)
    for class_name in os.listdir(DATASET_DIR):
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_path): continue
        
        print(f"Processing {class_name}...")
        
        # Loop through all images in the folder
        for filename in os.listdir(class_path):
            # Only process original images (skip already colored ones)
            if "red" in filename or "blue" in filename or "augmented" in filename:
                continue
                
            img_path = os.path.join(class_path, filename)
            image = cv2.imread(img_path)
            
            if image is None: continue
            
            # --- CREATE RED VARIATION ---
            # Keep Red channel, darken Green/Blue channels significantly
            red_img = image.copy()
            red_img[:, :, 0] = red_img[:, :, 0] * 0.2 # Blue Channel -> Low
            red_img[:, :, 1] = red_img[:, :, 1] * 0.2 # Green Channel -> Low
            # Red Channel stays high (Native Red)
            
            # Save Red Version
            red_filename = f"aug_red_{filename}"
            cv2.imwrite(os.path.join(class_path, red_filename), red_img)
            
            # --- CREATE BLUE VARIATION ---
            # Keep Blue channel, darken Red/Green
            blue_img = image.copy()
            blue_img[:, :, 1] = blue_img[:, :, 1] * 0.2 # Green -> Low
            blue_img[:, :, 2] = blue_img[:, :, 2] * 0.2 # Red -> Low
            # Blue Channel stays high
            
            # Save Blue Version
            blue_filename = f"aug_blue_{filename}"
            cv2.imwrite(os.path.join(class_path, blue_filename), blue_img)

    print("--- DONE! Dataset Tripled. ---")

if __name__ == "__main__":
    create_colored_variations()