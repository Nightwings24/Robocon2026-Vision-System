import cv2
import os

# --- CONFIGURATION ---
DATASET_DIR = "./dataset"
CLASSES = ["class_0_logo", "class_1_oracle", "class_2_random", "class_3_background"]

def generate_cube_rotations():
    print("--- GENERATING 90/180/270 ROTATIONS ---")
    
    for class_name in CLASSES:
        folder_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(folder_path):
            continue
            
        print(f"Processing {class_name}...")
        
        # Get list of existing images
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        count = 0
        for filename in images:
            # Skip images that are already rotated to avoid duplicates
            if "rot_" in filename:
                continue

            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            
            if image is None: continue

            # --- 1. Rotate 90 Degrees Clockwise ---
            rot_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(folder_path, f"rot_90_{filename}"), rot_90)

            # --- 2. Rotate 180 Degrees ---
            rot_180 = cv2.rotate(image, cv2.ROTATE_180)
            cv2.imwrite(os.path.join(folder_path, f"rot_180_{filename}"), rot_180)

            # --- 3. Rotate 270 Degrees (90 Counter-Clockwise) ---
            rot_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(folder_path, f"rot_270_{filename}"), rot_270)
            
            count += 3

        print(f"  -> Created {count} new rotated images.")

    print("\n--- DONE! Dataset Quadrupled. ---")

if __name__ == "__main__":
    generate_cube_rotations()