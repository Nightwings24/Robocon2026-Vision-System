import cv2
import numpy as np
import os

def extract_and_save_symbols(image_path, output_dir='extracted_symbols', target_size=(200, 200)):
    # 1. Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # 3. Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get white symbols on black background
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # --- NEW STEP: DILATION ---
    # We dilate (thicken) the image to connect disjoint parts of the same symbol.
    # The kernel size (20, 20) determines how much to thicken. 
    # It needs to be big enough to bridge gaps within a symbol, 
    # but small enough not to merge separate symbols.
    kernel = np.ones((20, 20), np.uint8) 
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    # --------------------------

    # 4. Find contours on the DILATED image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Filter and Sort
    min_area = 500 # Increased min_area slightly to ignore tiny artifacts
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    if not valid_contours:
        print("No symbols found. Check image path or thresholding.")
        return

    # Get bounding boxes
    boundingBoxes = [cv2.boundingRect(c) for c in valid_contours]

    # Sort: Top-to-bottom, then left-to-right
    # We use a tolerance factor (h_tolerance) to group items in the same 'row'
    def sort_key(box):
        x, y, w, h = box
        h_tolerance = 50 # Pixels of tolerance to consider items in the same row
        row_id = y // h_tolerance
        return (row_id, x)

    # Sort the contours and boxes together
    combined = sorted(zip(valid_contours, boundingBoxes), key=lambda b: sort_key(b[1]))
    cnts, boxes = zip(*combined)

    print(f"Found {len(cnts)} symbols (Target should be around 15).")

    # 6. Process and Save
    for i, (cnt, box) in enumerate(zip(cnts, boxes)):
        x, y, w, h = box
        
        # KEY POINT: We use the box from the dilated image to crop the ORIGINAL image
        # We add a small padding so we don't cut off the edges
        pad = 5
        y1 = max(0, y - pad)
        y2 = min(img.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img.shape[1], x + w + pad)
        
        symbol = img[y1:y2, x1:x2]

        # Create blank square canvas
        final_img = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
        
        # Center the symbol
        h_sym, w_sym, _ = symbol.shape
        x_offset = (target_size[0] - w_sym) // 2
        y_offset = (target_size[1] - h_sym) // 2
        
        # Ensure offsets are positive (handle cases where symbol > target)
        x_offset = max(0, x_offset)
        y_offset = max(0, y_offset)

        # Place symbol (handling edge cases if symbol is larger than canvas)
        h_place = min(h_sym, target_size[1])
        w_place = min(w_sym, target_size[0])
        
        final_img[y_offset:y_offset+h_place, x_offset:x_offset+w_place] = symbol[:h_place, :w_place]

        # Save
        output_path = os.path.join(output_dir, f'symbol_{i+1:02d}.png')
        cv2.imwrite(output_path, final_img)
        print(f"Saved {output_path}")

    print("Extraction complete!")

# --- Configuration ---
image_file = 'img.png' 
output_folder = 'extracted_symbols' 
final_dimension = 200 
# ---------------------

extract_and_save_symbols(image_file, output_folder, (final_dimension, final_dimension))