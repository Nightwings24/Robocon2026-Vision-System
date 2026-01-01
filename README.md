# Robust Vision-Based Symbol Classification for Robocon 2026

---

# Quick links
- Report: `Robocon_ML_Report.pdf` 
- Trained model: `robocon_model.pth`  
- ONNX model: `robocon_vision.onnx`  
- Main scripts: `train.py`, `inference.py`, `export_onnx.py`, `extract_symbols.py`, `augment_rotation.py`, `augment_colors.py`

---

# Project summary
A lightweight, real-time vision system to classify Robocon symbols (Robocon Logo, Oracle-like symbol, random/fake patterns, and background) under strict size and latency constraints. The pipeline extracts symbol crops from camera frames, trains a small MobileNetV3-based classifier on grayscale inputs, exports the model to ONNX for efficient CPU inference, and runs a live demo with temporal smoothing and manual logging for reproducible manual tests. See the included technical report for full details.

---

# Repository layout

```text
Robocon_ML/
├─ dataset/
│ ├─ class_0_logo/
│ ├─ class_1_oracle/
│ ├─ class_2_random/
│ └─ class_3_background/
├─ augment_colors.py
├─ augment_rotation.py
├─ export_onnx.py
├─ extract_symbols.py
├─ inference.py
├─ train.py
├─ requirements.txt
├─ robocon_model.pth
├─ robocon_vision.onnx
└─ Robocon_ML_Report.pdf
```

---

# Prerequisites / installation

Recommended Python: **3.8+**

```
git clone [https://github.com/Nightwings24/Robocon2026-Vision-System.git](https://github.com/Nightwings24/Robocon2026-Vision-System.git)
cd Robocon2026-Vision-System

# Create virtual environment
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate
# Activate (Windows)
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## File Descriptions

Quick start to run the real-time demo:
python inference.py

Show the desired symbol class in the rectangular box on the screen.
Press q to quit.
---

Step 1: Training the Model
To train the model from scratch using the provided dataset:
- What happens: The script loads images, applies geometric augmentations (Zoom, Perspective), and trains for 20 epochs.
- Output: Saves the trained weights to robocon_model.pth.

```
python train.py
```

--- 

Step 2: Export to ONNX
For deployment on the robot, convert the model to the optimized ONNX format:

```
python export_onnx.py
```

Output: Generates robocon_vision.onnx.
---

Step 3: Run Inference (Live Demo)
Connect a webcam and run the real-time detection system:

Inference Controls (Teacher Mode) The system includes a human-in-the-loop feature to correct mistakes live.
- 0: Save current frame as Robocon Logo
- 1: Save current frame as Oracle Bone
- 2: Save current frame as Fake/Random
- 3: Save current frame as Background
- q: Quit the application

```
python inference.py
```


---

Additional Files:
- extract_symbols.py - used to extract all the symbols in one go from single source image
- augment_rotation.py - used to rotate the images by 90/180/270 degrees    
- augment_colors.py - used to add blue and red color filters on image to train them better for game conditions
