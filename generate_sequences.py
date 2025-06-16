# generate_sequences.py
import os
import cv2
import numpy as np
import math
import mediapipe as mp

VIDEO_BASE_DIR = "./data"

# === MediaPipe init ===
mp_holistic = mp.solutions.holistic

# === Utils ===
def calculate_angle(a, b, c):
    def vector(p1, p2):
        return [p2[0] - p1[0], p2[1] - p1[1]]
    ba = vector(b, a)
    bc = vector(b, c)
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    norm_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    norm_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    cosine_angle = dot / (norm_ba * norm_bc + 1e-8)
    angle = math.degrees(math.acos(min(1, max(-1, cosine_angle))))
    return angle

def extract_angles(image):
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return {}
        lm = results.pose_landmarks.landmark
        points = [(p.x, p.y) for p in lm]
        try:
            angles = {
                'knee': calculate_angle(points[24], points[26], points[28]),
                'hip': calculate_angle(points[12], points[24], points[26]),
                'elbow': calculate_angle(points[12], points[14], points[16]),
                'shoulder': calculate_angle(points[14], points[12], points[24]),
                'ankle': calculate_angle(points[26], points[28], points[32]),
            }
            return angles
        except:
            return {}

def process_video_to_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    sequence = []
    timestamp = 0.0
    while timestamp < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        angles = extract_angles(frame)
        if angles and all(k in angles for k in ['knee', 'hip', 'elbow', 'shoulder', 'ankle']):
            vector = [angles['knee'], angles['hip'], angles['elbow'], angles['shoulder'], angles['ankle']]
            sequence.append(vector)
            print(f"✅ פריים {timestamp:.2f}s - זוויות זוהו")
        else:
            print(f"❌ פריים {timestamp:.2f}s - לא זוהו זוויות")
        timestamp += 0.1
    cap.release()
    return sequence

def generate_all():
    for label_dir in os.listdir(VIDEO_BASE_DIR):
        full_dir = os.path.join(VIDEO_BASE_DIR, label_dir)
        if not os.path.isdir(full_dir):
            continue
        for fname in os.listdir(full_dir):
            if not fname.endswith(".mp4"):
                continue
            full_path = os.path.join(full_dir, fname)
            print(f"🚀 מעבד את {full_path}")
            sequence = process_video_to_sequence(full_path)
            if sequence:
                save_path = full_path.replace(".mp4", ".npy")
                np.save(save_path, sequence)
                print(f"💾 נשמר: {save_path}")
            else:
                print(f"⚠️ לא נוצר רצף עבור: {full_path}")

if __name__ == "__main__":
    generate_all()
