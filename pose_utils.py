import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

mp_holistic = mp.solutions.holistic

# מיפוי שמות איברים לאינדקסים במערך הזוויות של extract_angles
ANGLE_PARTS_MAP = {
    "right_leg": [2, 8, 10, 12, 14],
    "left_leg": [3, 9, 11, 13, 15],
    "right_knee": [2, 8, 10, 12, 14],
    "left_knee": [3, 9, 11, 13, 15],
    "right_arm": [0, 4, 20, 22],
    "left_arm": [1, 5, 21, 23],
    "torso": [6, 7, 16, 17, 18, 19, 24, 25, 26, 27],
    "full_body": list(range(40)),
}

def get_angle_indices_by_parts(focus_parts):
    indices = set()
    if not focus_parts:
        return list(range(40))
    for part in focus_parts:
        part = part.lower()
        if part in ANGLE_PARTS_MAP:
            indices.update(ANGLE_PARTS_MAP[part])
        else:
            raise ValueError(f"Unknown focus part: {part}. Available: {list(ANGLE_PARTS_MAP.keys())}")
    return sorted(indices)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, num_classes=2, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, lengths):
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed_x)
        out = h_n[-1]
        out = self.fc(out)
        return out

def extract_angles(landmarks):
    angles = []
    points = {
        'right_shoulder': landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER],
        'right_elbow': landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW],
        'right_wrist': landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST],
        'left_shoulder': landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER],
        'left_elbow': landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW],
        'left_wrist': landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST],
        'right_hip': landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP],
        'right_knee': landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE],
        'right_ankle': landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE],
        'left_hip': landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP],
        'left_knee': landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE],
        'left_ankle': landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE]
    }

    def calculate_angle(a, b, c):
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])
        c = np.array([c.x, c.y, c.z])
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    # 40 זוויות — לא השתנה
    angles.append(calculate_angle(points['right_shoulder'], points['right_elbow'], points['right_wrist']))
    angles.append(calculate_angle(points['left_shoulder'], points['left_elbow'], points['left_wrist']))
    angles.append(calculate_angle(points['right_hip'], points['right_knee'], points['right_ankle']))
    angles.append(calculate_angle(points['left_hip'], points['left_knee'], points['left_ankle']))
    angles.append(calculate_angle(points['right_elbow'], points['right_shoulder'], points['right_hip']))
    angles.append(calculate_angle(points['left_elbow'], points['left_shoulder'], points['left_hip']))
    angles.append(calculate_angle(points['right_knee'], points['right_hip'], points['right_shoulder']))
    angles.append(calculate_angle(points['left_knee'], points['left_hip'], points['left_shoulder']))
    angles.append(calculate_angle(points['right_ankle'], points['right_knee'], points['right_hip']))
    angles.append(calculate_angle(points['left_ankle'], points['left_knee'], points['left_hip']))
    angles.append(calculate_angle(points['right_shoulder'], points['right_hip'], points['right_knee']))
    angles.append(calculate_angle(points['left_shoulder'], points['left_hip'], points['left_knee']))
    angles.append(calculate_angle(points['right_hip'], points['right_knee'], points['right_ankle']))
    angles.append(calculate_angle(points['left_hip'], points['left_knee'], points['left_ankle']))
    angles.append(calculate_angle(points['right_shoulder'], points['right_hip'], points['right_ankle']))
    angles.append(calculate_angle(points['left_shoulder'], points['left_hip'], points['left_ankle']))
    angles.append(calculate_angle(points['right_shoulder'], points['left_shoulder'], points['left_hip']))
    angles.append(calculate_angle(points['left_shoulder'], points['right_shoulder'], points['right_hip']))
    angles.append(calculate_angle(points['right_shoulder'], points['left_hip'], points['left_ankle']))
    angles.append(calculate_angle(points['left_shoulder'], points['right_hip'], points['right_ankle']))
    angles.append(calculate_angle(points['right_elbow'], points['right_shoulder'], points['left_shoulder']))
    angles.append(calculate_angle(points['left_elbow'], points['left_shoulder'], points['right_shoulder']))
    angles.append(calculate_angle(points['right_knee'], points['right_hip'], points['left_hip']))
    angles.append(calculate_angle(points['left_knee'], points['left_hip'], points['right_hip']))
    angles.append(calculate_angle(points['right_ankle'], points['right_knee'], points['left_knee']))
    angles.append(calculate_angle(points['left_ankle'], points['left_knee'], points['right_knee']))
    angles.append(calculate_angle(points['right_shoulder'], points['right_hip'], points['left_hip']))
    angles.append(calculate_angle(points['left_shoulder'], points['left_hip'], points['right_hip']))
    angles.append(calculate_angle(points['right_shoulder'], points['right_ankle'], points['left_ankle']))
    angles.append(calculate_angle(points['left_shoulder'], points['left_ankle'], points['right_ankle']))
    angles.append(calculate_angle(points['right_hip'], points['right_knee'], points['left_knee']))
    angles.append(calculate_angle(points['left_hip'], points['left_knee'], points['right_knee']))
    while len(angles) < 40:
        angles.append(0.0)
    return angles

def extract_sequence_from_video(video_path, focus_indices=None):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            if results.pose_landmarks:
                angles = extract_angles(results.pose_landmarks)
                if focus_indices is not None:
                    angles = [angles[i] for i in focus_indices]
                sequence.append(angles)
    cap.release()
    return sequence
