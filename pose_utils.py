from mediapipe_init import mp
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

mp_holistic = mp.solutions.holistic

ANGLE_PARTS_MAP = {
    # Specific joint angles
    "right_arm": [0, 4, 20, 21],  # Right arm angles (shoulder-elbow-wrist, elbow-shoulder-hip, elbow alignments)
    "left_arm": [1, 5, 20, 21],   # Left arm angles (shoulder-elbow-wrist, elbow-shoulder-hip, elbow alignments)
    "right_leg": [2, 8, 12, 14],  # Right leg angles (hip-knee-ankle, ankle-knee-hip, hip-knee-ankle, shoulder-hip-ankle)
    "left_leg": [3, 9, 13, 15],   # Left leg angles (hip-knee-ankle, ankle-knee-hip, hip-knee-ankle, shoulder-hip-ankle)
    
    # Specific joint focus
    "right_knee": [2, 6, 10, 12, 22, 24, 30],  # Right knee related angles
    "left_knee": [3, 7, 11, 13, 23, 25, 31],   # Left knee related angles
    "right_hip": [2, 6, 8, 10, 12, 14, 16, 18, 22, 24, 26, 28, 30],  # Right hip related angles
    "left_hip": [3, 7, 9, 11, 13, 15, 17, 19, 23, 25, 27, 29, 31],   # Left hip related angles
    "right_ankle": [2, 8, 12, 14, 18, 24, 28],  # Right ankle related angles
    "left_ankle": [3, 9, 13, 15, 19, 25, 29],   # Left ankle related angles
    "right_shoulder": [0, 4, 10, 14, 16, 18, 20, 26, 28],  # Right shoulder related angles
    "left_shoulder": [1, 5, 11, 15, 17, 19, 21, 27, 29],   # Left shoulder related angles
    "right_elbow": [0, 4, 20],  # Right elbow related angles
    "left_elbow": [1, 5, 21],   # Left elbow related angles
    "right_wrist": [0],  # Right wrist related angles
    "left_wrist": [1],   # Left wrist related angles
    
    # Body sections
    "shoulders": [4, 5, 10, 11, 16, 17, 26, 27, 28, 29],  # Shoulder-related angles
    "elbows": [0, 1, 4, 5, 20, 21],     # Elbow-related angles
    "wrists": [0, 1],     # Wrist-related angles (arm angles)
    "hips": [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 30, 31],  # Hip-related angles
    "knees": [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 22, 23, 24, 25, 30, 31],  # Knee-related angles
    "ankles": [2, 3, 8, 9, 12, 13, 14, 15, 18, 19, 24, 25, 28, 29],  # Ankle-related angles
    "feet": [2, 3, 8, 9, 12, 13, 14, 15, 18, 19, 24, 25, 28, 29],  # Same as ankles for now
    "hands": [0, 1, 4, 5, 20, 21],  # Hand-related angles (same as elbows)
    
    # Body sections
    "upper_body": [0, 1, 4, 5, 10, 11, 16, 17, 20, 21, 26, 27, 28, 29],  # Arms, shoulders, torso upper
    "lower_body": [2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 22, 23, 24, 25, 30, 31],  # Legs, hips, torso lower
    "right_side": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],  # Right side angles
    "left_side": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],   # Left side angles
    
    # Torso (alignment and cross-body angles)
    "torso": [6, 7, 16, 17, 18, 19, 24, 25, 26, 27],
    
    # Face parts (for future use) - these don't have angles yet
    "face": [],  # No face angles in current implementation
    
    # All angles
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

def print_available_parts():
    """Print all available body parts and their angle indices"""
    print("=== Available Body Parts ===")
    for part, indices in ANGLE_PARTS_MAP.items():
        print(f"{part}: {indices}")
    
    print("\n=== Usage Examples ===")
    print("For shoulders only: focus_parts=['shoulders']")
    print("For upper body: focus_parts=['upper_body']")
    print("For right side: focus_parts=['right_side']")
    print("For multiple parts: focus_parts=['shoulders', 'elbows', 'wrists']")
    print("For legs: focus_parts=['hips', 'knees', 'ankles']")

def get_angle_description(angle_index):
    """Get a description of what each angle represents"""
    angle_descriptions = {
        0: "Right arm angle (shoulder-elbow-wrist)",
        1: "Left arm angle (shoulder-elbow-wrist)",
        2: "Right leg angle (hip-knee-ankle)",
        3: "Left leg angle (hip-knee-ankle)",
        4: "Right shoulder angle (elbow-shoulder-hip)",
        5: "Left shoulder angle (elbow-shoulder-hip)",
        6: "Right knee angle (knee-hip-shoulder)",
        7: "Left knee angle (knee-hip-shoulder)",
        8: "Right ankle angle (ankle-knee-hip)",
        9: "Left ankle angle (ankle-knee-hip)",
        10: "Right shoulder-hip-knee angle",
        11: "Left shoulder-hip-knee angle",
        12: "Right hip-knee-ankle angle",
        13: "Left hip-knee-ankle angle",
        14: "Right shoulder-hip-ankle angle",
        15: "Left shoulder-hip-ankle angle",
        16: "Shoulder alignment (right-left-hip)",
        17: "Shoulder alignment (left-right-hip)",
        18: "Cross body angles (right shoulder-left hip-left ankle)",
        19: "Cross body angles (left shoulder-right hip-right ankle)",
        20: "Elbow alignment (right-left shoulder)",
        21: "Elbow alignment (left-right shoulder)",
        22: "Knee alignment (right-left hip)",
        23: "Knee alignment (left-right hip)",
        24: "Ankle alignment (right-left knee)",
        25: "Ankle alignment (left-right knee)",
        26: "Shoulder-hip alignment (right-left)",
        27: "Shoulder-hip alignment (left-right)",
        28: "Shoulder-ankle alignment (right-left)",
        29: "Shoulder-ankle alignment (left-right)",
        30: "Hip-knee alignment (right-left)",
        31: "Hip-knee alignment (left-right)",
    }
    return angle_descriptions.get(angle_index, f"Angle {angle_index} (no description available)")

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


class GraphConvolution(nn.Module):
    """Simple Graph Convolution Layer"""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        # x: (batch_size, num_nodes, in_features)
        # adj: (batch_size, num_nodes, num_nodes) or (num_nodes, num_nodes)
        support = torch.matmul(x, self.weight)
        if len(adj.shape) == 3:
            output = torch.bmm(adj, support)
        else:
            output = torch.matmul(adj, support)
        if self.bias is not None:
            output += self.bias
        return output


class LSTM_GNN_Classifier(nn.Module):
    def __init__(self, spatial_input_size=67, temporal_input_size=200, hidden_size=256, num_layers=3, num_classes=2, 
                 dropout=0.5, bidirectional=True, gnn_hidden=128, num_gnn_layers=2, 
                 num_joints=12, use_learned_adj=True):
        super(LSTM_GNN_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_joints = num_joints
        self.use_learned_adj = use_learned_adj
        self.spatial_input_size = spatial_input_size
        self.temporal_input_size = temporal_input_size
        
        # Batch normalization for spatial and temporal inputs
        self.spatial_batch_norm = nn.BatchNorm1d(spatial_input_size)
        self.temporal_batch_norm = nn.BatchNorm1d(temporal_input_size)
        
        # GNN layers for spatial modeling (only spatial features)
        self.gnn_layers = nn.ModuleList()
        
        for i in range(num_gnn_layers):
            if i == 0:
                in_features = spatial_input_size // num_joints  # Spatial features per joint
            else:
                in_features = gnn_hidden
            self.gnn_layers.append(GraphConvolution(in_features, gnn_hidden))
        
        # Learnable adjacency matrix
        if use_learned_adj:
            self.adj_matrix = nn.Parameter(torch.randn(num_joints, num_joints))
            self.adj_activation = nn.Sigmoid()
        
        # LSTM layer for temporal modeling (GNN output + temporal features)
        gnn_output_size = gnn_hidden * num_joints
        lstm_input_size = gnn_output_size + temporal_input_size
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Output layers
        lstm_out = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, lstm_out // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out // 2, num_classes)
        )
        
        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def create_body_graph_adjacency(self, batch_size, device):
        """Create adjacency matrix for body joints graph"""
        if self.use_learned_adj:
            # Use learned adjacency matrix
            adj = self.adj_activation(self.adj_matrix)
            # Make it symmetric and add self-loops
            adj = (adj + adj.t()) / 2
            adj = adj + torch.eye(self.num_joints, device=device)
            # Normalize
            adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
            return adj.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # Create fixed adjacency matrix based on body structure
            # This defines connections between joints (shoulder-elbow-wrist, hip-knee-ankle, etc.)
            adj = torch.zeros(self.num_joints, self.num_joints, device=device)
            
            # Define body connections (assuming 12 joints: 6 right + 6 left)
            connections = [
                (0, 1), (1, 2),  # Right arm: shoulder-elbow-wrist
                (3, 4), (4, 5),  # Left arm: shoulder-elbow-wrist
                (6, 7), (7, 8),  # Right leg: hip-knee-ankle
                (9, 10), (10, 11),  # Left leg: hip-knee-ankle
                (0, 3), (1, 4), (2, 5),  # Cross-body connections
                (6, 9), (7, 10), (8, 11),  # Cross-body connections
            ]
            
            for i, j in connections:
                adj[i, j] = 1
                adj[j, i] = 1  # Make symmetric
            
            # Add self-loops
            adj = adj + torch.eye(self.num_joints, device=device)
            
            # Normalize
            adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
            return adj.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, spatial_features, temporal_features, lengths):
        """
        Forward pass with separate spatial and temporal features
        
        Args:
            spatial_features: (batch_size, seq_len, spatial_input_size) - for GNN
            temporal_features: (batch_size, seq_len, temporal_input_size) - for LSTM
            lengths: sequence lengths
        """
        batch_size, seq_len, spatial_feat_size = spatial_features.shape
        _, _, temporal_feat_size = temporal_features.shape
        
        # Process spatial features with GNN
        # Reshape spatial features for GNN: (batch_size * seq_len, num_joints, features_per_joint)
        features_per_joint = spatial_feat_size // self.num_joints
        if spatial_feat_size % self.num_joints != 0:
            # If not divisible, pad to make it divisible
            target_size = features_per_joint * self.num_joints
            if spatial_feat_size < target_size:
                # Pad with zeros
                padding = torch.zeros(batch_size, seq_len, target_size - spatial_feat_size, device=spatial_features.device)
                spatial_features = torch.cat([spatial_features, padding], dim=2)
            else:
                # Truncate
                spatial_features = spatial_features[:, :, :target_size]
            features_per_joint = target_size // self.num_joints
        
        spatial_features = spatial_features.reshape(batch_size * seq_len, self.num_joints, features_per_joint)
        
        # Create adjacency matrix
        adj = self.create_body_graph_adjacency(batch_size * seq_len, spatial_features.device)
        
        # Apply GNN layers to spatial features
        gnn_output = spatial_features
        for gnn_layer in self.gnn_layers:
            gnn_output = gnn_layer(gnn_output, adj)
            gnn_output = self.relu(gnn_output)
            gnn_output = self.dropout(gnn_output)
        
        # Reshape GNN output back to (batch_size, seq_len, gnn_output_size)
        gnn_output_size = self.gnn_layers[-1].out_features
        gnn_output = gnn_output.reshape(batch_size, seq_len, self.num_joints * gnn_output_size)
        
        # Combine GNN output with temporal features for LSTM
        combined_features = torch.cat([gnn_output, temporal_features], dim=2)
        
        # Process with LSTM
        packed_x = pack_padded_sequence(combined_features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed_x)
        
        # Get final hidden state
        if self.bidirectional:
            out = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            out = h_n[-1]
        
        # Final classification
        out = self.fc(out)
        return out

def extract_spatial_features(landmarks):
    """
    Extract spatial features for GNN: keypoints, angles, and distances
    """
    # Extract keypoints (xyz coordinates)
    keypoints = extract_keypoints_xyz(landmarks)
    
    # Extract angles
    angles = extract_angles(landmarks)
    
    # Calculate distances between key joints
    distances = calculate_joint_distances(landmarks)
    
    # Combine spatial features
    spatial_features = keypoints + angles + distances
    return spatial_features

def extract_temporal_features(sequence):
    """
    Extract temporal features for LSTM: velocity, acceleration, std, angle changes, ratios
    """
    if len(sequence) < 2:
        return []
    
    temporal_features = []
    
    # Velocity features
    velocity = np.diff(sequence, axis=0)
    temporal_features.extend(velocity.flatten())
    
    # Acceleration features
    if len(sequence) >= 3:
        acceleration = np.diff(velocity, axis=0)
        temporal_features.extend(acceleration.flatten())
    else:
        temporal_features.extend(np.zeros_like(velocity.flatten()))
    
    # Standard deviation features
    std_features = np.std(sequence, axis=0)
    temporal_features.extend(std_features)
    
    # Angle change features (rate of change)
    angle_changes = calculate_angle_changes(sequence)
    temporal_features.extend(angle_changes)
    
    # Ratio features
    ratios = calculate_ratios(sequence)
    temporal_features.extend(ratios)
    
    return temporal_features

def calculate_joint_distances(landmarks):
    """
    Calculate distances between key body joints
    """
    distances = []
    
    # Define key joint pairs for distance calculation
    joint_pairs = [
        (mp_holistic.PoseLandmark.RIGHT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_ELBOW),
        (mp_holistic.PoseLandmark.RIGHT_ELBOW, mp_holistic.PoseLandmark.RIGHT_WRIST),
        (mp_holistic.PoseLandmark.RIGHT_HIP, mp_holistic.PoseLandmark.RIGHT_KNEE),
        (mp_holistic.PoseLandmark.RIGHT_KNEE, mp_holistic.PoseLandmark.RIGHT_ANKLE),
        (mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.LEFT_ELBOW),
        (mp_holistic.PoseLandmark.LEFT_ELBOW, mp_holistic.PoseLandmark.LEFT_WRIST),
        (mp_holistic.PoseLandmark.LEFT_HIP, mp_holistic.PoseLandmark.LEFT_KNEE),
        (mp_holistic.PoseLandmark.LEFT_KNEE, mp_holistic.PoseLandmark.LEFT_ANKLE),
        (mp_holistic.PoseLandmark.RIGHT_SHOULDER, mp_holistic.PoseLandmark.LEFT_SHOULDER),
        (mp_holistic.PoseLandmark.RIGHT_HIP, mp_holistic.PoseLandmark.LEFT_HIP),
    ]
    
    for joint1_idx, joint2_idx in joint_pairs:
        pt1 = landmarks.landmark[joint1_idx]
        pt2 = landmarks.landmark[joint2_idx]
        
        # Calculate Euclidean distance
        distance = np.sqrt((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2 + (pt1.z - pt2.z)**2)
        distances.append(distance)
    
    return distances

def calculate_angle_changes(sequence):
    """
    Calculate rate of change in angles over time
    """
    if len(sequence) < 2:
        return []
    
    # Assume first 15 features are angles (adjust based on your angle count)
    angle_features = sequence[:, :15] if sequence.shape[1] >= 15 else sequence
    
    # Calculate rate of change (first derivative)
    angle_changes = np.diff(angle_features, axis=0)
    
    # Calculate average rate of change per angle
    avg_changes = np.mean(angle_changes, axis=0)
    
    return avg_changes.tolist()

def calculate_ratios(sequence):
    """
    Calculate ratios between key angles
    """
    if len(sequence) < 1:
        return []
    
    # Assume first 15 features are angles
    angle_features = sequence[:, :15] if sequence.shape[1] >= 15 else sequence
    
    ratios = []
    
    # Calculate ratios between key angles
    if angle_features.shape[1] >= 8:
        # Knee angle / Torso angle ratio
        knee_angle = angle_features[:, 2]  # Right knee angle
        torso_angle = angle_features[:, 8]  # Torso angle
        knee_torso_ratio = knee_angle / (torso_angle + 1e-5)
        ratios.append(np.mean(knee_torso_ratio))
        
        # Shoulder angle / Hip angle ratio
        shoulder_angle = angle_features[:, 4]  # Right shoulder angle
        hip_angle = angle_features[:, 6]  # Right hip angle
        shoulder_hip_ratio = shoulder_angle / (hip_angle + 1e-5)
        ratios.append(np.mean(shoulder_hip_ratio))
    
    return ratios

def extract_sequence_with_separate_features(video_path, focus_indices=None, use_keypoints=True):
    """
    Extract sequence with separate spatial and temporal features
    """
    cap = cv2.VideoCapture(video_path)
    spatial_sequences = []
    temporal_sequences = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            
            if results.pose_landmarks:
                # Extract spatial features for GNN
                spatial_features = extract_spatial_features(results.pose_landmarks)
                spatial_sequences.append(spatial_features)
    
    cap.release()
    
    if len(spatial_sequences) == 0:
        return None, None
    
    spatial_sequences = np.array(spatial_sequences)
    
    # Extract temporal features from spatial sequences
    temporal_features = extract_temporal_features(spatial_sequences)
    
    return spatial_sequences, temporal_features
