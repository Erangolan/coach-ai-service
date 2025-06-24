from mediapipe_init import mp
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

mp_holistic = mp.solutions.holistic

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
    def __init__(self, input_size=40, hidden_size=256, num_layers=3, num_classes=2, dropout=0.5, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        lstm_out = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, num_classes)
        )

    def forward(self, x, lengths):
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed_x)
        if self.bidirectional:
            out = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            out = h_n[-1]
        out = self.fc(out)
        return out

class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, input_size=40, cnn_out=64, kernel_size=3, hidden_size=128, num_layers=2, num_classes=2, dropout=0.5, bidirectional=True):
        super(CNN_LSTM_Classifier, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.cnn = nn.Conv1d(in_channels=input_size, out_channels=cnn_out, kernel_size=kernel_size, padding=1)
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        lstm_out = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, num_classes)
        )
        self.bidirectional = bidirectional

    def forward(self, x, lengths):
        x = x.transpose(1, 2)             # (batch, input_size, seq_len)
        x = self.batch_norm(x)
        x = self.cnn(x)                   # (batch, cnn_out, seq_len)
        x = x.transpose(1, 2)             # (batch, seq_len, cnn_out)
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed_x)
        if self.bidirectional:
            out = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            out = h_n[-1]
        out = self.fc(out)
        return out

# ---- DATA AUGMENTATION ----
def augment_angles(sequence, noise_std=2.0, probability=0.5):
    if np.random.rand() < probability:
        noise = np.random.normal(0, noise_std, sequence.shape)
        sequence = sequence + noise
    return sequence

def speed_up(sequence, factor=2):
    return sequence[::factor]

def slow_down(sequence, factor=2):
    return np.repeat(sequence, factor, axis=0)

def apply_all_augmentations(sequence, debug_path=None):
    aug_sequences = [sequence]  # original
    aug_sequences.append(augment_angles(sequence, noise_std=2.0, probability=1.0))  # gentle noise only
    if len(sequence) > 3:
        aug_sequences.append(speed_up(sequence, factor=2))   # speed up x2 only
    aug_sequences.append(slow_down(sequence, factor=2))      # slow down x2 only
    # Do not add strong noise, no reverse, no cropping, no aggressive speed changes
    # Keep realistic and not exaggerated diversity
    if debug_path:
        print(f"{debug_path}: generated {len(aug_sequences)} augmented samples (orig len: {len(sequence)})")
    return [seq for seq in aug_sequences if len(seq) > 0]



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

def extract_keypoints_xyz(landmarks):
    indices = [
        mp_holistic.PoseLandmark.RIGHT_SHOULDER,
        mp_holistic.PoseLandmark.RIGHT_ELBOW,
        mp_holistic.PoseLandmark.RIGHT_WRIST,
        mp_holistic.PoseLandmark.RIGHT_HIP,
        mp_holistic.PoseLandmark.RIGHT_KNEE,
        mp_holistic.PoseLandmark.RIGHT_ANKLE,
        mp_holistic.PoseLandmark.LEFT_SHOULDER,
        mp_holistic.PoseLandmark.LEFT_ELBOW,
        mp_holistic.PoseLandmark.LEFT_WRIST,
        mp_holistic.PoseLandmark.LEFT_HIP,
        mp_holistic.PoseLandmark.LEFT_KNEE,
        mp_holistic.PoseLandmark.LEFT_ANKLE,
    ]
    coords = []
    for idx in indices:
        pt = landmarks.landmark[idx]
        coords.extend([pt.x, pt.y, pt.z])
    return coords

def extract_sequence_from_video(video_path, focus_indices=None, use_keypoints=False):
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
                features = [angles[i] for i in focus_indices] if focus_indices else angles
                if use_keypoints:
                    coords = extract_keypoints_xyz(results.pose_landmarks)
                    features = features + coords
                sequence.append(features)
    cap.release()
    return np.array(sequence)

class LSTM_Transformer_Classifier(nn.Module):
    def __init__(self, input_size=40, hidden_size=256, num_layers=3, num_classes=2, 
                 dropout=0.5, bidirectional=True, nhead=8, num_transformer_layers=2):
        super(LSTM_Transformer_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.nhead = nhead
        
        # Batch normalization for input
        self.batch_norm = nn.BatchNorm1d(input_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size * 2 if bidirectional else hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Output layers
        lstm_out = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, lstm_out // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out // 2, num_classes)
        )

    def forward(self, x, lengths):
        # Input preprocessing
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        
        # LSTM processing
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed_x)
        
        # Unpack the sequence for transformer
        unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        # Transformer processing (without padding mask for MPS compatibility)
        transformer_out = self.transformer(unpacked_out)
        
        # Global average pooling over sequence dimension
        # Use lengths to exclude padded positions
        lengths_device = lengths.to(x.device)
        mask_expanded = torch.arange(transformer_out.size(1), device=x.device).unsqueeze(0) < lengths_device.unsqueeze(1)
        mask_expanded = mask_expanded.unsqueeze(-1).expand_as(transformer_out)
        transformer_out = transformer_out.masked_fill(~mask_expanded, 0)
        pooled_out = transformer_out.sum(dim=1) / lengths_device.unsqueeze(1).float()
        
        # Final classification
        out = self.fc(pooled_out)
        return out
