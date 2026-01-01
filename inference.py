import onnxruntime as ort
import numpy as np
import cv2
import time
import os
from collections import deque

class RoboconVision:
    def __init__(self, model_path="robocon_vision.onnx"):
        # Load ONNX model
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
        except Exception:
            print("Warning: Model not found. Make sure you trained the 4-class model!")
            self.session = None

        # 4 Classes: 0:Logo, 1:Oracle, 2:Fake, 3:Background
        self.classes = ["Robocon Logo", "Oracle Bone", "Random/Fake", "Background"]
        
        # Rolling buffer for smoothing (Last 5 frames)
        self.prediction_buffer = deque(maxlen=5)

    def preprocess(self, image):
        # 1. Digital Zoom (Center Crop 60%) - Solves the distance issue
        h, w = image.shape[:2]
        crop_factor = 0.6 
        crop_h, crop_w = int(h * crop_factor), int(w * crop_factor)
        start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
        
        cropped = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
        
        # 2. Process (Grayscale -> Resize -> Normalize)
        if len(cropped.shape) == 3:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped
        
        resized = cv2.resize(gray, (224, 224))
        input_data = resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.expand_dims(input_data, axis=0)
        return input_data

    def predict(self, image):
        if self.session is None: return "Error", 0, 0, 0

        start_time = time.time()
        input_tensor = self.preprocess(image)
        
        # Run Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Softmax
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Temporal Smoothing
        self.prediction_buffer.append(probs)
        avg_probs = np.mean(self.prediction_buffer, axis=0)
        
        # Result
        conf = np.max(avg_probs)
        class_id = np.argmax(avg_probs)
        inference_time = (time.time() - start_time) * 1000

        return self.classes[class_id], conf, inference_time, class_id

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Ensure dataset folders exist
    dataset_paths = {
        0: "dataset/class_0_logo",
        1: "dataset/class_1_oracle",
        2: "dataset/class_2_random",
        3: "dataset/class_3_background"
    }
    for path in dataset_paths.values():
        if not os.path.exists(path): os.makedirs(path)

    classifier = RoboconVision()
    cap = cv2.VideoCapture(0)
    
    save_timer = 0
    save_message = ""

    print("--- ROBOCON VISION SYSTEM ---")
    print("Press 0-3 to capture training data.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Run Prediction
        label, conf, latency, class_id = classifier.predict(frame)
        display_frame = frame.copy()

        # --- DRAW UI (Standard Size) ---
        
        # 1. Draw "Zoom Box" (So you know where to hold the symbol)
        h, w = frame.shape[:2]
        cf = 0.6 
        ch, cw = int(h*cf), int(w*cf)
        y1, x1 = (h-ch)//2, (w-cw)//2
        cv2.rectangle(display_frame, (x1, y1), (x1+cw, y1+ch), (255, 255, 0), 1)

        # 2. Display Result
        if class_id == 3: # Background
            cv2.putText(display_frame, "SCANNING...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        else: # Symbol Detected
            color = (0, 255, 0) if conf > 0.7 else (0, 0, 255)
            cv2.putText(display_frame, f"{label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(display_frame, f"{conf:.2f} ({latency:.0f}ms)", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # 3. Save Message
        if save_timer > 0:
            cv2.putText(display_frame, save_message, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            save_timer -= 1
        
        # 4. Tiny Instructions at bottom
        cv2.putText(display_frame, "0:Logo 1:Oracle 2:Fake 3:BG", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Robocon Vision", display_frame)

        # --- CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        
        # Capture Data
        elif key in [ord('0'), ord('1'), ord('2'), ord('3')]:
            target_class = int(chr(key))
            filename = f"{dataset_paths[target_class]}/capture_{int(time.time()*1000)}.jpg"
            cv2.imwrite(filename, frame) # Save the full frame
            
            names = ["LOGO", "ORACLE", "FAKE", "BG"]
            save_message = f"SAVED: {names[target_class]}"
            save_timer = 20

    cap.release()
    cv2.destroyAllWindows()