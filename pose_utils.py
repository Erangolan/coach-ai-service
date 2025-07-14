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

def subtle_bad_augmentation(sequence, noise_std=6, drop_frame_prob=0.10):
    """יצירת דגימת BAD מתנועה טובה ע"י שיבוש זוויות/השמטת פריימים."""
    seq = sequence.copy()
    seq = augment_angles(seq, noise_std=noise_std, probability=1.0)
    seq = [frame for frame in seq if np.random.rand() > drop_frame_prob]
    return np.array(seq)

def speed_up(sequence, factor=2):
    return sequence[::factor]

def slow_down(sequence, factor=2):
    return np.repeat(sequence, factor, axis=0)

def random_crop(sequence, min_ratio=0.7, max_ratio=1.0):
    """
    Randomly crop a sequence to a length between min_ratio and max_ratio of the original length.
    This helps with temporal robustness by simulating different video lengths.
    """
    if len(sequence) < 5:  # Don't crop very short sequences
        return sequence
    
    # Calculate crop length
    min_length = max(3, int(len(sequence) * min_ratio))
    max_length = min(len(sequence), int(len(sequence) * max_ratio))
    
    if min_length >= max_length:
        return sequence
    
    crop_length = np.random.randint(min_length, max_length + 1)
    
    # Randomly choose start position
    max_start = len(sequence) - crop_length
    start_pos = np.random.randint(0, max_start + 1)
    
    return sequence[start_pos:start_pos + crop_length]

def frame_dropout(sequence, dropout_prob=0.1, max_consecutive_drops=2):
    """
    Randomly drop frames from a sequence to simulate missing or occluded frames.
    This helps the model become robust to frame losses in real-world scenarios.
    
    Args:
        sequence: Input sequence of frames
        dropout_prob: Probability of dropping each frame
        max_consecutive_drops: Maximum number of consecutive frames that can be dropped
    """
    if len(sequence) < 5:  # Don't drop frames from very short sequences
        return sequence
    
    # Create a mask for which frames to keep
    keep_mask = np.ones(len(sequence), dtype=bool)
    
    # Apply dropout with consecutive frame constraint
    consecutive_drops = 0
    for i in range(len(sequence)):
        if np.random.random() < dropout_prob and consecutive_drops < max_consecutive_drops:
            keep_mask[i] = False
            consecutive_drops += 1
        else:
            consecutive_drops = 0
    
    # Ensure we keep at least 3 frames
    if np.sum(keep_mask) < 3:
        # If too many frames were dropped, keep at least 3 frames
        keep_indices = np.random.choice(len(sequence), min(3, len(sequence)), replace=False)
        keep_mask = np.zeros(len(sequence), dtype=bool)
        keep_mask[keep_indices] = True
    
    return sequence[keep_mask]

def time_shift(sequence, max_frames=3):
    """
    Apply a small random time shift by starting the sequence a few frames earlier or later.
    This simulates slight temporal misalignments in video start times.
    
    Args:
        sequence: Input sequence of frames
        max_frames: Maximum number of frames to shift (default: 3)
    """
    if len(sequence) < max_frames + 3:  # Need at least max_frames + 3 to shift
        return sequence
    
    # Generate a random shift (can be negative or positive)
    shift = np.random.randint(-max_frames, max_frames + 1)
    
    if shift == 0:  # No shift
        return sequence
    
    # Apply the shift
    if shift > 0:
        # Start later (skip first few frames)
        shifted_sequence = sequence[shift:]
    else:
        # Start earlier (add padding at the beginning)
        # Repeat the first frame for the missing frames
        padding = np.tile(sequence[0], (abs(shift), 1))
        shifted_sequence = np.vstack([padding, sequence])
    
    return shifted_sequence

def add_velocity_features(sequence):
    """
    Add velocity and acceleration features to a sequence.
    Args:
        sequence: numpy array of shape (frames, features)
    Returns:
        numpy array of shape (frames, features * 3)
    """
    num_frames, num_features = sequence.shape

    # Velocity: difference between consecutive frames, pad first row
    if num_frames < 2:
        velocity_features = np.zeros_like(sequence)
    else:
        velocity = np.diff(sequence, axis=0)
        velocity_features = np.vstack([velocity[0], velocity])

    # Acceleration: difference between consecutive velocities, pad first two rows
    if num_frames < 3:
        acceleration_features = np.zeros_like(sequence)
    else:
        # Calculate acceleration from the original velocity calculation
        velocity_for_accel = np.diff(sequence, axis=0)
        acceleration = np.diff(velocity_for_accel, axis=0)
        # Pad first two frames with the first acceleration value
        acceleration_features = np.vstack([acceleration[0:1], acceleration[0:1], acceleration])

    enhanced_sequence = np.hstack([sequence, velocity_features, acceleration_features])
    return enhanced_sequence

def add_statistical_features(sequence):
    """
    Add statistical features (mean, median, std, max, min, range) for each feature across the sequence.
    Args:
        sequence: numpy array of shape (frames, features)
    Returns:
        numpy array with statistical features appended to each frame
    """
    num_frames, num_features = sequence.shape
    zero_values = np.zeros_like(sequence[0]) if num_frames > 0 else np.zeros(num_features)

    if num_frames < 2:
        # Single frame - use the frame values as statistics
        if num_frames == 1:
            frame_values = sequence[0]
            stats_features = np.tile(
                np.hstack([
                    frame_values,    # mean
                    frame_values,    # median
                    zero_values,     # std
                    frame_values,    # max
                    frame_values,    # min
                    zero_values      # range
                ]),
                (num_frames, 1)
            )
        else:
            # No frames - return zeros
            stats_features = np.zeros((num_frames, num_features * 6))
        return np.hstack([sequence, stats_features])

    # Calculate statistics for each feature across all frames
    mean_features = np.mean(sequence, axis=0)
    median_features = np.median(sequence, axis=0)
    std_features = np.std(sequence, axis=0)
    max_features = np.max(sequence, axis=0)
    min_features = np.min(sequence, axis=0)
    range_features = max_features - min_features

    # Create statistical features for each frame (same values for all frames)
    stats_features = np.tile(
        np.hstack([
            mean_features,
            median_features,
            std_features,
            max_features,
            min_features,
            range_features
        ]),
        (num_frames, 1)
    )
    enhanced_sequence = np.hstack([sequence, stats_features])
    return enhanced_sequence


def add_ratio_features(sequence, focus_indices=None):
    """
    Add ratio features between relevant angles.
    For right knee exercise, we calculate ratio between primary knee angle and primary torso angle.
    
    Args:
        sequence: numpy array of shape (num_frames, num_features)
        focus_indices: indices of focus angles (if None, assumes first 15 are angles)
    
    Returns:
        numpy array with ratio features added
    """
    num_frames, num_features = sequence.shape
    
    if num_frames == 0:
        return sequence
    
    # Determine which features are angles (not keypoints or other features)
    if focus_indices is not None:
        # Use focus_indices to identify angle features
        angle_features = sequence[:, :len(focus_indices)]
        other_features = sequence[:, len(focus_indices):]
    else:
        # Assume first 15 features are angles (for 15 focus angles)
        angle_features = sequence[:, :15]
        other_features = sequence[:, 15:]
    
    # For right knee exercise, we focus on the primary knee angle and primary torso angle
    # Based on focus_parts=['right_knee', 'torso'], we have 15 angles
    # Let's use the first angle from each group as the primary angles
    
    # Primary knee angle (first angle from right_knee group)
    primary_knee_angle = angle_features[:, 0]  # First angle
    
    # Primary torso angle (first angle from torso group)
    primary_torso_angle = angle_features[:, 8]  # 9th angle (first of torso group)
    
    # Calculate the ratio: knee / torso
    # Avoid division by zero
    ratio = primary_knee_angle / (primary_torso_angle + 1e-5)
    
    # Add the ratio as a single feature column
    ratio_features = ratio.reshape(-1, 1)  # Reshape to (num_frames, 1)
    
    enhanced_sequence = np.hstack([sequence, ratio_features])
    return enhanced_sequence

def apply_all_augmentations(sequence, debug_path=None, is_idle=False):
    aug_sequences = [sequence]  # original
    aug_sequences.append(augment_angles(sequence, noise_std=2.0, probability=1.0))
    
    # Add time shifting augmentation
    aug_sequences.append(time_shift(sequence, max_frames=3))
    
    # Don't apply speed_up/slow_down for idle sequences
    if not is_idle:
        if len(sequence) > 3:
            aug_sequences.append(speed_up(sequence, factor=2))
        aug_sequences.append(slow_down(sequence, factor=2))
    
    # Add random cropping augmentations
    if len(sequence) > 5:
        # Add 2 different random crops
        aug_sequences.append(random_crop(sequence, min_ratio=0.7, max_ratio=0.9))
        aug_sequences.append(random_crop(sequence, min_ratio=0.8, max_ratio=1.0))
    
    # Add frame dropout augmentations
    if len(sequence) > 5:
        # Add 2 different frame dropout variations
        aug_sequences.append(frame_dropout(sequence, dropout_prob=0.1, max_consecutive_drops=2))
        aug_sequences.append(frame_dropout(sequence, dropout_prob=0.15, max_consecutive_drops=1))
    
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

def extract_sequence_from_video(video_path, focus_indices=None, use_keypoints=False, use_velocity=False, use_statistics=False, use_ratios=False):
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
    
    sequence = np.array(sequence)
    
    # Add velocity features if requested
    if use_velocity and len(sequence) > 1:
        sequence = add_velocity_features(sequence)
    
    # Add statistical features if requested (now includes range features)
    if use_statistics and len(sequence) > 1:
        sequence = add_statistical_features(sequence)
    
    # Add ratio features if requested
    if use_ratios and len(sequence) > 1:
        sequence = add_ratio_features(sequence, focus_indices)
    
    return sequence

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
