import cv2
import mediapipe as mp
import numpy as np
import os
from pose_utils import extract_angles

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def extract_right_knee_angle(landmarks):
    """Extract right knee angle (hip-knee-ankle)"""
    right_hip = landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_HIP]
    right_knee = landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_KNEE]
    right_ankle = landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_ANKLE]
    return calculate_angle(right_hip, right_knee, right_ankle)

def check_idle_angles():
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    good_dir = "data/exercises/right-knee-to-90-degrees/good"
    first_frame_angles = []
    
    # Check first frame of first 10 good videos
    for i, filename in enumerate(sorted(os.listdir(good_dir))[:10]):
        if not filename.endswith('.mp4'):
            continue
            
        video_path = os.path.join(good_dir, filename)
        print(f"Processing {filename}...")
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            
            if results.pose_landmarks:
                angle = extract_right_knee_angle(results.pose_landmarks)
                first_frame_angles.append(angle)
                print(f"  First frame angle: {angle:.1f}°")
        
        cap.release()
    
    holistic.close()
    
    if first_frame_angles:
        avg_angle = np.mean(first_frame_angles)
        min_angle = np.min(first_frame_angles)
        max_angle = np.max(first_frame_angles)
        std_angle = np.std(first_frame_angles)
        
        print(f"\n=== IDLE ANGLE ANALYSIS (First Frames Only) ===")
        print(f"Average angle: {avg_angle:.1f}°")
        print(f"Min angle: {min_angle:.1f}°")
        print(f"Max angle: {max_angle:.1f}°")
        print(f"Std deviation: {std_angle:.1f}°")
        print(f"Recommended IDLE_ANGLE: {avg_angle:.0f}")
        print(f"Recommended IDLE_TOLERANCE: {std_angle * 2:.0f}")
        
        # Filter out angles that are clearly not standing (too bent)
        standing_angles = [a for a in first_frame_angles if a > 160]
        if standing_angles:
            standing_avg = np.mean(standing_angles)
            standing_std = np.std(standing_angles)
            print(f"\n=== STANDING ANGLE ANALYSIS (Filtered >160°) ===")
            print(f"Standing angles: {standing_angles}")
            print(f"Average standing angle: {standing_avg:.1f}°")
            print(f"Standing std deviation: {standing_std:.1f}°")
            print(f"Recommended IDLE_ANGLE: {standing_avg:.0f}")
            print(f"Recommended IDLE_TOLERANCE: {standing_std * 2:.0f}")
    else:
        print("No angles found!")

if __name__ == "__main__":
    check_idle_angles() 