<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.15-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Flask-Web%20UI-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/MediaPipe-Face%20Landmarks-blueviolet.svg" alt="MediaPipe">
  
  <h1>👁️ Eye Tracking OS Control 👁️</h1>
  <p><b>A Machine Learning project that lets you control your computer mouse using only your eyes!</b></p>
</div>

---

## 🌟 Overview
Welcome to the Eye Tracking OS Control project! This tool uses your standard webcam to track your eye movements, processes the data using a custom TensorFlow Neural Network, and translates your gaze into real-time on-screen mouse movements.

**Features:**
- 🎨 **Web-Based Data Collector:** A beautiful, responsive Flask app to comfortably collect your calibration data.
- 🧠 **Deep Learning:** Uses Google MediaPipe for sub-millimeter facial landmarks and a custom TensorFlow model to predict screen coordinates.
- 🖱️ **Live OS Control:** Take over your Windows mouse pointer hands-free.

---

## 📁 Repository Structure

```text
Eye-Tracking-OS-Control/
│
├── src/                      # 💻 Core Source Code
│   ├── web_collector.py      # Flask Web UI for data collection
│   ├── train_gaze_model.py   # AI Training Script (TensorFlow)
│   ├── test_tracking.py      # HUD Visualizer for model testing
│   ├── gaze_mouse.py         # Real-time OS mouse controller
│   └── templates/            # HTML/CSS for Web UI
│
├── models/                   # 🧠 ML Models & Weights
│   ├── face_landmarker.task  # MediaPipe landmark model
│   ├── gaze_model.keras      # Saved Keras Neural Network
│   ├── eye_model.tflite      # Optimized TFLite model for speed
│   ├── feature_scaler.pkl    # Data scaler
│   └── training_meta.pkl     # Training metadata
│
├── docs/                     # 📚 Documentation
│   └── eye_tracking_roadmap.pdf
│
└── dataset/                  # 📊 CSV Training Data (Git Ignored)
```

---

## 🚀 How to Use

### 1. Collect Data
Start the web UI server to collect your personal calibration data:
```bash
python src/web_collector.py
```
*Open your browser to `http://localhost:5000` and follow the on-screen target dot.*

### 2. Train the AI Model
Once you've collected enough data, train the neural network:
```bash
python src/train_gaze_model.py
```
*This will generate a highly optimized `eye_model.tflite` in the `models/` folder.*

### 3. Test and Calibrate
Use the HUD Visualizer to test the model's accuracy in real-time:
```bash
python src/test_tracking.py
```

### 4. Control Your Mouse!
Run the live tracker to take control of your OS:
```bash
python src/gaze_mouse.py
```

---

## 🛠️ Architecture

1. **Webcam Input** ➔ **MediaPipe** extracts 478 3D facial landmarks (including precise iris coordinates).
2. **Feature Engineering** ➔ Head pose (Pitch, Yaw, Roll) and normalized eye centers are calculated.
3. **Neural Network** ➔ The features are fed into our custom TensorFlow Dense network.
4. **Output** ➔ The network outputs predicted (X, Y) screen coordinates.
5. **Action** ➔ `pyautogui` moves your system cursor.

<div align="center">
  <i>Built with ❤️ by Aadarsh</i>
</div>
