import argparse
import csv
import math
import os
import random
import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Paths
LANDMARKER_PATH = "face_landmarker.task"
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")

# Eye / Iris landmark indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# UI State globals
click_x, click_y = -1, -1

def mouse_callback(event, x, y, flags, param):
    global click_x, click_y
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x, click_y = x, y

def setup_landmarker():
    base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True, # To get head pose
        num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

def extract_head_pose(transformation_matrix):
    """Extract pitch, yaw, roll from the 4x4 transformation matrix."""
    # The upper 3x3 is the rotation matrix
    r_mat = transformation_matrix[:3, :3]
    # Decompose to Euler angles (in degrees)
    euler_angles = cv2.RQDecomp3x3(r_mat)[0]
    pitch, yaw, roll = euler_angles
    return pitch, yaw, roll

def get_center(landmarks, indices, img_w, img_h):
    """Get the 2D average pixel coordinates of a set of landmarks."""
    x = sum([landmarks[i].x for i in indices]) / len(indices)
    y = sum([landmarks[i].y for i in indices]) / len(indices)
    return x * img_w, y * img_h

def get_normalized_center(landmarks, indices):
    """Get the 3D average normalized coordinates."""
    x = sum([landmarks[i].x for i in indices]) / len(indices)
    y = sum([landmarks[i].y for i in indices]) / len(indices)
    z = sum([landmarks[i].z for i in indices]) / len(indices)
    return x, y, z

def main(args):
    global click_x, click_y
    if not os.path.exists(LANDMARKER_PATH):
        print(f"Error: {LANDMARKER_PATH} not found in current directory.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    csv_file = os.path.join(DATA_DIR, f"gaze_data_{args.user}_{int(time.time())}.csv")

    landmarker = setup_landmarker()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    # Get webcam resolution
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam initialized at {cam_w}x{cam_h}")

    # Setup Fullscreen Window
    window_name = "Gaze Data Collector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # We need to know the screen resolution to accurately log target coordinates.
    # We will show a quick screen to prompt the user to hit space to start, and we 
    # can measure the window size.
    cv2.imshow(window_name, np.zeros((400, 400, 3), dtype=np.uint8))
    cv2.waitKey(100) # Give it a moment to enter fullscreen
    _, _, screen_w, screen_h = cv2.getWindowImageRect(window_name)
    print(f"Detected screen resolution: {screen_w}x{screen_h}")

    csv_header = [
        "timestamp", "user", "lighting", "screen_w", "screen_h", "cam_w", "cam_h",
        "target_x", "target_y", 
        "head_pitch", "head_yaw", "head_roll",
        "l_iris_x", "l_iris_y", "l_iris_z", 
        "r_iris_x", "r_iris_y", "r_iris_z",
        "inter_ocular_dist"
    ]

    print("Starting data collection...")
    print(f"User: {args.user}, Lighting: {args.lighting}")
    print(f"Targets: {args.targets}, Frames per target: {args.frames_per_target}")
    
    start_time = time.perf_counter()

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

        # START SCREEN UI
        start_btn_x1, start_btn_y1 = screen_w // 2 - 150, screen_h - 200
        start_btn_x2, start_btn_y2 = screen_w // 2 + 150, screen_h - 100
        
        while True:
            ready_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            
            # Title
            cv2.putText(ready_frame, "GAZE DATA COLLECTION", (screen_w//2 - 250, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(ready_frame, f"User: {args.user}", (screen_w//2 - 100, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 255), 2)
            
            # Guidelines
            cv2.putText(ready_frame, "Guidelines:", (screen_w//2 - 300, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            guidelines = [
                "1. Sit straight and keep your head relatively still.",
                "2. Ensure your face is well-lit and clearly visible.",
                "3. A red dot will appear at random locations on the screen.",
                "4. Follow the dot ONLY with your eyes. Do not move your head.",
                "5. Keep looking at the dot until it moves to a new location."
            ]
            for i, text in enumerate(guidelines):
                cv2.putText(ready_frame, text, (screen_w//2 - 300, 300 + i*40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Draw Start Button
            cv2.rectangle(ready_frame, (start_btn_x1, start_btn_y1), (start_btn_x2, start_btn_y2), (0, 180, 0), -1)
            cv2.putText(ready_frame, "START", (screen_w//2 - 65, screen_h - 135), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            cv2.imshow(window_name, ready_frame)
            k = cv2.waitKey(1)
            
            # Check for button click
            if click_x != -1 and click_y != -1:
                if start_btn_x1 <= click_x <= start_btn_x2 and start_btn_y1 <= click_y <= start_btn_y2:
                    click_x, click_y = -1, -1 # reset
                    break
                click_x, click_y = -1, -1 # reset if clicked elsewhere
                
            if k == 27: # ESC
                cap.release()
                cv2.destroyAllWindows()
                return

        # DATA COLLECTION UI
        finish_btn_x1, finish_btn_y1 = screen_w - 180, 20
        finish_btn_x2, finish_btn_y2 = screen_w - 20, 80
        abort_collection = False

        for target_idx in range(args.targets):
            if abort_collection:
                break
                
            # Generate random target point (padding to avoid edge clipping)
            pad = 50
            tx = random.randint(pad, screen_w - pad)
            ty = random.randint(pad, screen_h - pad)
            
            # Avoid placing target right under the finish button
            if tx > finish_btn_x1 - 50 and ty < finish_btn_y2 + 50:
                tx = tx - 200
                
            frames_captured = 0
            
            while frames_captured < args.frames_per_target:
                if abort_collection:
                    break

                # Check for FINISH button click
                if click_x != -1 and click_y != -1:
                    if finish_btn_x1 <= click_x <= finish_btn_x2 and finish_btn_y1 <= click_y <= finish_btn_y2:
                        print("User clicked FINISH.")
                        abort_collection = True
                        click_x, click_y = -1, -1
                        break
                    click_x, click_y = -1, -1

                success, frame = cap.read()
                if not success:
                    print("Camera read failed!")
                    break
                
                frame = cv2.flip(frame, 1) # Mirror
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                timestamp_ms = int((time.perf_counter() - start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                # Draw UI
                display_img = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                
                # Draw the target
                # Change color slightly to show it's recording
                color = (0, 0, 255) if frames_captured % 10 < 5 else (0, 100, 255) 
                cv2.circle(display_img, (tx, ty), 15, color, -1)
                cv2.circle(display_img, (tx, ty), 2, (255, 255, 255), -1)
                
                # Info text
                cv2.putText(display_img, f"Target {target_idx+1}/{args.targets}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                # Draw Finish Button
                cv2.rectangle(display_img, (finish_btn_x1, finish_btn_y1), (finish_btn_x2, finish_btn_y2), (0, 0, 180), -1)
                cv2.putText(display_img, "FINISH", (screen_w - 150, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                if result.face_landmarks and result.facial_transformation_matrixes:
                    landmarks = result.face_landmarks[0]
                    matrix = result.facial_transformation_matrixes[0]
                    
                    pitch, yaw, roll = extract_head_pose(matrix)
                    
                    lx, ly, lz = get_normalized_center(landmarks, LEFT_IRIS)
                    rx, ry, rz = get_normalized_center(landmarks, RIGHT_IRIS)
                    
                    # Approximate inter-ocular distance in normalized 3D space
                    iod = math.sqrt((lx - rx)**2 + (ly - ry)**2 + (lz - rz)**2)
                    
                    # Log the data
                    writer.writerow([
                        timestamp_ms, args.user, args.lighting, screen_w, screen_h, cam_w, cam_h,
                        tx, ty,
                        pitch, yaw, roll,
                        lx, ly, lz,
                        rx, ry, rz,
                        iod
                    ])
                    frames_captured += 1
                    cv2.putText(display_img, "Tracking: OK", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(display_img, "Face Not Found!", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow(window_name, display_img)
                k = cv2.waitKey(1)
                if k == 27: # ESC
                    print("Aborted by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data collection complete! Saved to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect gaze data mapped to screen coordinates.")
    parser.add_argument("--user", type=str, required=True, help="Name of the participant (e.g., aadarsha, yudhin)")
    parser.add_argument("--lighting", type=str, default="normal", help="Lighting condition (e.g., normal, dim, bright, side-lit)")
    parser.add_argument("--targets", type=int, default=60, help="Number of random screen targets to display")
    parser.add_argument("--frames-per-target", type=int, default=20, help="Number of valid frames to record per target")
    
    args = parser.parse_args()
    main(args)
