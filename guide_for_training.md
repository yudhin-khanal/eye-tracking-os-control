# Guide for Training Eye Tracking Model

Hello! Your friend has shared this project with you so you can train the eye-tracking model on your PC. Here are the steps you need to follow:

## Prerequisites

1. Ensure you have **Python 3.11+** installed.
2. Install the necessary dependencies (you can create a virtual environment first if you prefer):
   ```bash
   pip install tensorflow==2.15 mediapipe flask pyautogui pandas numpy scikit-learn
   ```
   *(Note: The exact dependencies can be inferred from the `README.md`, but these are the core ones used in the project.)*

## Step 1: Data Collection (Optional but Recommended)
If you want to train the model on your own eye movements (which usually yields better results for your specific setup):
1. Run the web collector:
   ```bash
   python src/web_collector.py
   ```
2. Open your browser to `http://localhost:5000` and follow the on-screen target dot to collect calibration data. The data will be saved as CSV files in the `dataset/` folder.

## Step 2: Train the Model
Once you have the data (either the provided one or your newly collected one), you can train the AI model:
1. Run the training script:
   ```bash
   python src/train_gaze_model.py
   ```
2. This will process the CSV files in the `dataset/` folder and train a custom TensorFlow Neural Network.
3. The training process will output an optimized `eye_model.tflite` model in the `models/` folder.

## Step 3: Test and Calibrate
Use the HUD Visualizer to test the model's accuracy in real-time:
```bash
python src/test_tracking.py
```

## Step 4: Control Your Mouse!
Run the live tracker to take control of your OS cursor:
```bash
python src/gaze_mouse.py
```

Happy training! Let your friend know if you run into any issues.
