from mediapipe_init import mp
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

mp_holistic = mp.solutions.holistic

# Default normalization method for the entire pipeline
DEFAULT_NORMALIZE_METHOD = "minmax"

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

def gaussian_noise_augmentation(sequence, noise_std_range=(0.5, 3.0), probability=0.7):
    """
    Apply Gaussian noise to the sequence with varying noise levels.
    
    Args:
        sequence: Input sequence of shape (frames, features)
        noise_std_range: Range for noise standard deviation (min, max)
        probability: Probability of applying noise
    """
    if np.random.rand() < probability:
        # Randomly sample noise standard deviation from the range
        noise_std = np.random.uniform(noise_std_range[0], noise_std_range[1])
        noise = np.random.normal(0, noise_std, sequence.shape)
        sequence = sequence + noise
    return sequence

def scale_augmentation(sequence, scale_range=(0.8, 1.2), probability=0.6):
    """
    Apply scale changes to the sequence by multiplying features by a random scale factor.
    
    Args:
        sequence: Input sequence of shape (frames, features)
        scale_range: Range for scale factor (min, max)
        probability: Probability of applying scaling
    """
    if np.random.rand() < probability:
        # Randomly sample scale factor from the range
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        sequence = sequence * scale_factor
    return sequence

def time_warp_augmentation(sequence, warp_factor_range=(0.8, 1.2), probability=0.5):
    """
    Apply time warping to the sequence by stretching or compressing the temporal dimension.
    
    Args:
        sequence: Input sequence of shape (frames, features)
        warp_factor_range: Range for warp factor (min, max) - <1 compresses, >1 stretches
        probability: Probability of applying time warping
    """
    if np.random.rand() < probability and len(sequence) > 5:
        # Randomly sample warp factor from the range
        warp_factor = np.random.uniform(warp_factor_range[0], warp_factor_range[1])
        
        # Calculate new sequence length
        new_length = int(len(sequence) * warp_factor)
        new_length = max(3, min(new_length, len(sequence) * 2))  # Keep reasonable bounds
        
        if new_length != len(sequence):
            # Use linear interpolation to warp the sequence
            old_indices = np.arange(len(sequence))
            new_indices = np.linspace(0, len(sequence) - 1, new_length)
            
            # Interpolate each feature dimension
            warped_sequence = np.zeros((new_length, sequence.shape[1]))
            for feature_idx in range(sequence.shape[1]):
                warped_sequence[:, feature_idx] = np.interp(
                    new_indices, old_indices, sequence[:, feature_idx]
                )
            
            sequence = warped_sequence
    
    return sequence

def targeted_augmentation(sequence, angle_noise_range=(0.5, 2.0), keypoint_noise_range=(0.01, 0.05), 
                         scale_range=(0.8, 1.2), warp_factor_range=(0.8, 1.2), probability=0.8):
    """
    Apply targeted augmentation with different noise levels for angles vs keypoints.
    
    Args:
        sequence: Input sequence of shape (frames, features)
        angle_noise_range: Noise range for angles (degrees)
        keypoint_noise_range: Noise range for keypoint coordinates (normalized 0-1)
        scale_range: Range for scale factor (applied to keypoints only)
        warp_factor_range: Range for time warp factor
        probability: Probability of applying each augmentation
    """
    if np.random.rand() < probability:
        # Apply targeted Gaussian noise
        sequence = targeted_gaussian_noise(sequence, angle_noise_range, keypoint_noise_range, probability=0.8)
        
        # Apply scale changes only to keypoints
        sequence = targeted_scale_augmentation(sequence, scale_range, probability=0.7)
        
        # Apply time warping to all features
        sequence = time_warp_augmentation(sequence, warp_factor_range, probability=0.6)
    
    return sequence

def targeted_gaussian_noise(sequence, angle_noise_range=(0.5, 2.0), keypoint_noise_range=(0.01, 0.05), probability=0.7):
    """
    Apply different noise levels to angles vs keypoints.
    
    Args:
        sequence: Input sequence of shape (frames, features)
        angle_noise_range: Noise range for angles (degrees)
        keypoint_noise_range: Noise range for keypoint coordinates (normalized 0-1)
        probability: Probability of applying noise
    """
    if np.random.rand() < probability:
        # Create a copy to avoid modifying original
        noisy_sequence = sequence.copy()
        
        for frame_idx, frame in enumerate(noisy_sequence):
            # Apply noise to angles (first 40 features)
            if len(frame) >= 40:
                angle_noise_std = np.random.uniform(angle_noise_range[0], angle_noise_range[1])
                angle_noise = np.random.normal(0, angle_noise_std, 40)
                noisy_sequence[frame_idx, :40] += angle_noise
            
            # Apply noise to keypoints (features after 40)
            if len(frame) > 40:
                keypoint_noise_std = np.random.uniform(keypoint_noise_range[0], keypoint_noise_range[1])
                keypoint_noise = np.random.normal(0, keypoint_noise_std, len(frame) - 40)
                noisy_sequence[frame_idx, 40:] += keypoint_noise
        
        return noisy_sequence
    
    return sequence

def targeted_scale_augmentation(sequence, scale_range=(0.8, 1.2), probability=0.6):
    """
    Apply scale changes only to keypoint coordinates, not angles.
    
    Args:
        sequence: Input sequence of shape (frames, features)
        scale_range: Range for scale factor
        probability: Probability of applying scaling
    """
    if np.random.rand() < probability:
        # Create a copy to avoid modifying original
        scaled_sequence = sequence.copy()
        
        for frame_idx, frame in enumerate(scaled_sequence):
            # Apply scaling only to keypoints (features after 40)
            if len(frame) > 40:
                scale_factor = np.random.uniform(scale_range[0], scale_range[1])
                scaled_sequence[frame_idx, 40:] *= scale_factor
        
        return scaled_sequence
    
    return sequence

def aggressive_augmentation(sequence, noise_std_range=(1.0, 4.0), scale_range=(0.7, 1.3), 
                          warp_factor_range=(0.7, 1.3), probability=0.8):
    """
    Apply all aggressive augmentation techniques together.
    
    Args:
        sequence: Input sequence of shape (frames, features)
        noise_std_range: Range for Gaussian noise standard deviation
        scale_range: Range for scale factor
        warp_factor_range: Range for time warp factor
        probability: Probability of applying each augmentation
    """
    if np.random.rand() < probability:
        # Apply Gaussian noise
        sequence = gaussian_noise_augmentation(sequence, noise_std_range, probability=0.8)
        
        # Apply scale changes
        sequence = scale_augmentation(sequence, scale_range, probability=0.7)
        
        # Apply time warping
        sequence = time_warp_augmentation(sequence, warp_factor_range, probability=0.6)
    
    return sequence

def subtle_bad_augmentation(sequence, noise_std=6, drop_frame_prob=0.10):
    """יצירת דגימת BAD מתנועה טובה ע"י שיבוש זוויות/השמטת פריימים."""
    seq = sequence.copy()
    seq = augment_angles(seq, noise_std=noise_std, probability=1.0)
    seq = [frame for frame in seq if np.random.rand() > drop_frame_prob]
    return np.array(seq)

def speed_up(sequence, factor=2):
    """
    Speed up a sequence by taking every nth frame.
    For float factors, interpolate to get the desired speed.
    
    Args:
        sequence: Input sequence of shape (frames, features)
        factor: Speed factor (float or int). >1 speeds up, <1 slows down
    """
    if factor <= 0:
        return sequence
    
    if isinstance(factor, int) or factor.is_integer():
        # Integer factor - use simple slicing
        return sequence[::int(factor)]
    else:
        # Float factor - use interpolation
        if factor >= 1:
            # Speed up: take every nth frame with interpolation
            step = 1.0 / factor
            indices = np.arange(0, len(sequence), step)
            indices = np.clip(indices, 0, len(sequence) - 1).astype(int)
            return sequence[indices]
        else:
            # Slow down: interpolate between frames
            return slow_down(sequence, 1.0 / factor)

def slow_down(sequence, factor=2):
    """
    Slow down a sequence by repeating frames or interpolating.
    
    Args:
        sequence: Input sequence of shape (frames, features)
        factor: Slow factor (float or int). >1 slows down
    """
    if factor <= 0:
        return sequence
    
    if isinstance(factor, int) or factor.is_integer():
        # Integer factor - use simple repetition
        return np.repeat(sequence, int(factor), axis=0)
    else:
        # Float factor - use interpolation
        if factor >= 1:
            # Create more frames by interpolating
            new_length = int(len(sequence) * factor)
            if new_length <= len(sequence):
                return sequence
            
            # Use linear interpolation to create more frames
            old_indices = np.arange(len(sequence))
            new_indices = np.linspace(0, len(sequence) - 1, new_length)
            
            # Interpolate each feature dimension
            slowed_sequence = np.zeros((new_length, sequence.shape[1]))
            for feature_idx in range(sequence.shape[1]):
                slowed_sequence[:, feature_idx] = np.interp(
                    new_indices, old_indices, sequence[:, feature_idx]
                )
            
            return slowed_sequence
        else:
            # Speed up instead
            return speed_up(sequence, 1.0 / factor)

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


def rotate_sequence(sequence, angle_degrees):
    """
    Rotate a sequence of (x, y) keypoints by a given angle in degrees.
    Rotation is applied frame-by-frame, preserving the number of points.
    
    Args:
        sequence: Input sequence of shape (frames, features)
        angle_degrees: Rotation angle in degrees
    """
    if len(sequence) == 0:
        return sequence
    
    angle_radians = np.radians(angle_degrees)
    cos_val, sin_val = np.cos(angle_radians), np.sin(angle_radians)
    
    rotated_seq = []
    for frame in sequence:
        # Create a copy of the frame
        rotated_frame = frame.copy()
        
        # Only rotate if we have keypoint coordinates (x,y,z triplets)
        # Keypoints are typically 36 values (12 joints * 3 coordinates) after 40 angles
        if len(frame) > 40:
            # Assume first 40 are angles, rest are keypoints
            angles = frame[:40]
            keypoints = frame[40:]
            
            # Rotate only x,y coordinates (skip z coordinates)
            rotated_keypoints = []
            for i in range(0, len(keypoints), 3):  # Process x,y,z triplets
                if i + 1 < len(keypoints):  # Make sure we have at least x,y
                    x, y = keypoints[i], keypoints[i+1]
                    new_x = x * cos_val - y * sin_val
                    new_y = x * sin_val + y * cos_val
                    rotated_keypoints.extend([new_x, new_y])
                    
                    # Add z coordinate if it exists
                    if i + 2 < len(keypoints):
                        rotated_keypoints.append(keypoints[i+2])
                else:
                    # Handle case where we have incomplete triplet
                    rotated_keypoints.extend(keypoints[i:])
            
            # Combine angles and rotated keypoints
            rotated_frame = np.concatenate([angles, rotated_keypoints])
        
        rotated_seq.append(rotated_frame)
    
    return np.array(rotated_seq)


def apply_all_augmentations(sequence, debug_path=None, is_idle=False):
    aug_sequences = [sequence]  # Keep the original sequence as the first element
    
    # Gaussian noise augmentation with noise_std_range=(1.5, 4.0) and probability=1.0
    aug_sequences.append(gaussian_noise_augmentation(sequence, noise_std_range=(1.5, 4.0), probability=1.0))
    
    # Scale augmentation with scale_range=(0.75, 1.25) and probability=1.0
    aug_sequences.append(scale_augmentation(sequence, scale_range=(0.75, 1.25), probability=1.0))
    
    # Rotation augmentation by calling rotate_sequence with a random angle between -5 and 5 degrees
    aug_sequences.append(rotate_sequence(sequence, angle_degrees=np.random.uniform(-5, 5)))
    
    # Temporal stretch/compression using time_warp_augmentation with warp_factor_range=(0.85, 1.15)
    aug_sequences.append(time_warp_augmentation(sequence, warp_factor_range=(0.85, 1.15), probability=1.0))
    
    # Keep time shifting, but increase max_frames to 4
    aug_sequences.append(time_shift(sequence, max_frames=4))
    
    # For non-idle sequences
    if not is_idle:
        # If sequence length > 3, apply speed_up with a random factor between 1.1 and 1.3
        if len(sequence) > 3:
            aug_sequences.append(speed_up(sequence, factor=np.random.uniform(1.1, 1.3)))
        # Apply slow_down with a random factor between 1.1 and 1.3
        aug_sequences.append(slow_down(sequence, factor=np.random.uniform(1.1, 1.3)))
    
    # Increase random cropping aggressiveness
    if len(sequence) > 5:
        # Add crop with min_ratio=0.65, max_ratio=0.9
        aug_sequences.append(random_crop(sequence, min_ratio=0.65, max_ratio=0.9))
        # Add crop with min_ratio=0.7, max_ratio=0.95
        aug_sequences.append(random_crop(sequence, min_ratio=0.7, max_ratio=0.95))
    
    # Increase frame dropout aggressiveness
    if len(sequence) > 5:
        # Dropout with dropout_prob=0.15, max_consecutive_drops=3
        aug_sequences.append(frame_dropout(sequence, dropout_prob=0.15, max_consecutive_drops=3))
        # Dropout with dropout_prob=0.2, max_consecutive_drops=2
        aug_sequences.append(frame_dropout(sequence, dropout_prob=0.2, max_consecutive_drops=2))
    
    # Add combined aggressive augmentation with:
    # noise_std_range=(1.5, 4.0)
    # scale_range=(0.75, 1.25)
    # warp_factor_range=(0.85, 1.15)
    # probability=1.0
    if len(sequence) > 3:
        aug_sequences.append(aggressive_augmentation(sequence, 
                                                      noise_std_range=(1.5, 4.0), 
                                                      scale_range=(0.75, 1.25), 
                                                      warp_factor_range=(0.85, 1.15), 
                                                      probability=1.0))
    
    # Return only sequences with len(seq) > 0
    filtered_sequences = [seq for seq in aug_sequences if len(seq) > 0]
    
    # If debug_path is provided, print how many augmented samples were generated
    if debug_path:
        print(f"{debug_path}: generated {len(filtered_sequences)} augmented samples (orig len: {len(sequence)})")
    
    return filtered_sequences

def apply_aggressive_augmentations(sequence, num_augmentations=5, debug_path=None, is_idle=False):
    """
    Apply multiple aggressive augmentations to create a larger augmented dataset.
    This is useful for training scenarios where you need more data.
    
    Args:
        sequence: Input sequence of shape (frames, features)
        num_augmentations: Number of aggressive augmentations to generate
        debug_path: Optional debug path for logging
        is_idle: Whether this is an idle sequence (affects which augmentations are applied)
    
    Returns:
        List of augmented sequences including the original
    """
    aug_sequences = [sequence]  # original
    
    for i in range(num_augmentations):
        # Create a copy for this augmentation
        aug_seq = sequence.copy()
        
        # Apply random combinations of aggressive augmentations
        if np.random.rand() < 0.8:
            aug_seq = gaussian_noise_augmentation(aug_seq, 
                                                noise_std_range=(0.3, 4.0), 
                                                probability=1.0)
        
        if np.random.rand() < 0.7:
            aug_seq = scale_augmentation(aug_seq, 
                                       scale_range=(0.7, 1.3), 
                                       probability=1.0)
        
        if np.random.rand() < 0.6 and len(aug_seq) > 5:
            aug_seq = time_warp_augmentation(aug_seq, 
                                           warp_factor_range=(0.7, 1.3), 
                                           probability=1.0)
        
        if np.random.rand() < 0.5:
            aug_seq = time_shift(aug_seq, max_frames=5)
        
        if np.random.rand() < 0.4 and len(aug_seq) > 5:
            aug_seq = random_crop(aug_seq, 
                                min_ratio=0.6, 
                                max_ratio=0.95)
        
        if np.random.rand() < 0.3 and len(aug_seq) > 5:
            aug_seq = frame_dropout(aug_seq, 
                                  dropout_prob=0.15, 
                                  max_consecutive_drops=3)
        
        # Don't apply speed changes to idle sequences
        if not is_idle and np.random.rand() < 0.3:
            if len(aug_seq) > 3:
                if np.random.rand() < 0.5:
                    aug_seq = speed_up(aug_seq, factor=np.random.randint(2, 4))
                else:
                    aug_seq = slow_down(aug_seq, factor=np.random.randint(2, 4))
        
        aug_sequences.append(aug_seq)
    
    if debug_path:
        print(f"{debug_path}: generated {len(aug_sequences)} aggressive augmented samples (orig len: {len(sequence)})")
    
    return [seq for seq in aug_sequences if len(seq) > 0]



def normalize_per_video(features, method="zscore", normalize_indices=None, focus_indices=None, use_keypoints=False, use_velocity=False, use_statistics=False, use_ratios=False):
    """
    Normalize features per video using z-score or min-max normalization.
    Only keypoint features are normalized by default.
    
    Args:
        features: numpy array of shape (frames, features) - features for a single video
        method: normalization method - "zscore" or "minmax"
        normalize_indices: list of column indices to normalize (if None, auto-determine keypoint indices)
        focus_indices: list of angle indices to use (for auto-determination)
        use_keypoints: whether keypoints are included (for auto-determination)
        use_velocity: whether velocity features are included (for auto-determination)
        use_statistics: whether statistical features are included (for auto-determination)
        use_ratios: whether ratio features are included (for auto-determination)
    
    Returns:
        normalized_features: numpy array of same shape as input
    """
    if len(features) == 0:
        return features
    
    # If no specific indices provided, auto-determine keypoint indices
    if normalize_indices is None:
        normalize_indices = get_normalize_indices(
            focus_indices=focus_indices,
            use_keypoints=use_keypoints,
            use_velocity=use_velocity,
            use_statistics=use_statistics,
            use_ratios=use_ratios
        )
    
    # Create a copy to avoid modifying original, ensure float type for normalization
    normalized_features = features.astype(np.float64).copy()
    
    # Apply normalization only to specified indices
    for col_idx in normalize_indices:
        if col_idx >= features.shape[1]:
            continue  # Skip if index is out of bounds
            
        column_data = features[:, col_idx]  # Use original data for calculations
        
        if method.lower() == "zscore":
            # Z-score normalization: (x - mean) / std
            mean_val = np.mean(column_data)
            std_val = np.std(column_data)
            # Avoid division by zero
            if std_val == 0:
                normalized_features[:, col_idx] = 0
            else:
                normalized_features[:, col_idx] = (column_data - mean_val) / std_val
                
        elif method.lower() == "minmax":
            # Min-max normalization: (x - min) / (max - min)
            min_val = np.min(column_data)
            max_val = np.max(column_data)
            range_val = max_val - min_val
            # Avoid division by zero
            if range_val == 0 or np.isclose(range_val, 0, atol=1e-10):
                normalized_features[:, col_idx] = 0
            else:
                normalized_values = (column_data - min_val) / range_val
                normalized_features[:, col_idx] = normalized_values
                
        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'zscore' or 'minmax'")
    
    return normalized_features

def get_normalize_indices(focus_indices=None, use_keypoints=False, use_velocity=False, use_statistics=False, use_ratios=False):
    """
    Determine which feature indices should be normalized based on feature types.
    Only keypoint features (XYZ positions) are normalized.
    
    Args:
        focus_indices: list of angle indices to use (if None, use all 40 angles)
        use_keypoints: whether keypoints are included
        use_velocity: whether velocity features are included
        use_statistics: whether statistical features are included
        use_ratios: whether ratio features are included
    
    Returns:
        normalize_indices: list of column indices that should be normalized (only keypoints)
    """
    normalize_indices = []
    current_idx = 0
    
    # Angles (0-180 degrees) - DO NOT normalize
    num_angles = len(focus_indices) if focus_indices is not None else 40
    current_idx += num_angles
    
    # Keypoints (XYZ positions) - ONLY NORMALIZE THESE
    if use_keypoints:
        keypoint_indices = list(range(current_idx, current_idx + 12 * 3))  # 12 keypoints * 3 coordinates
        normalize_indices.extend(keypoint_indices)
        current_idx += 12 * 3
    
    # Velocity features - DO NOT normalize
    if use_velocity:
        base_features_count = current_idx
        current_idx += base_features_count
    
    # Statistical features - DO NOT normalize
    if use_statistics:
        base_features_count = len(focus_indices) if focus_indices is not None else 40
        if use_keypoints:
            base_features_count += 12 * 3
        current_idx += base_features_count * 6
    
    # Ratio features - DO NOT normalize
    if use_ratios:
        current_idx += 1  # Skip ratio features
    
    return normalize_indices

def normalize_features_batch(features_list, method="zscore", normalize_indices=None, focus_indices=None, use_keypoints=False, use_velocity=False, use_statistics=False, use_ratios=False):
    """
    Normalize a list of feature sequences, each representing a video.
    Only keypoint features are normalized by default.
    
    Args:
        features_list: list of numpy arrays, each of shape (frames, features)
        method: normalization method - "zscore" or "minmax"
        normalize_indices: list of column indices to normalize (if None, auto-determine keypoint indices)
        focus_indices: list of angle indices to use (for auto-determination)
        use_keypoints: whether keypoints are included (for auto-determination)
        use_velocity: whether velocity features are included (for auto-determination)
        use_statistics: whether statistical features are included (for auto-determination)
        use_ratios: whether ratio features are included (for auto-determination)
    
    Returns:
        normalized_features_list: list of normalized numpy arrays
    """
    normalized_features_list = []
    
    for features in features_list:
        if len(features) > 0:
            normalized_features = normalize_per_video(
                features, 
                method=method, 
                normalize_indices=normalize_indices,
                focus_indices=focus_indices,
                use_keypoints=use_keypoints,
                use_velocity=use_velocity,
                use_statistics=use_statistics,
                use_ratios=use_ratios
            )
            normalized_features_list.append(normalized_features)
        else:
            normalized_features_list.append(features)
    
    return normalized_features_list

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

def extract_sequence_from_video(video_path, focus_indices=None, use_keypoints=False, use_velocity=False, use_statistics=False, use_ratios=False, normalize_method=DEFAULT_NORMALIZE_METHOD):
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
    
    # Apply selective normalization per video (only keypoints)
    if len(sequence) > 0:
        normalize_indices = get_normalize_indices(
            focus_indices=focus_indices,
            use_keypoints=use_keypoints,
            use_velocity=use_velocity,
            use_statistics=use_statistics,
            use_ratios=use_ratios
        )
        if normalize_indices:
            print(f"Normalizing only keypoint features using {normalize_method}: {normalize_indices}")
            sequence = normalize_per_video(
                sequence, 
                method=normalize_method, 
                normalize_indices=normalize_indices,
                focus_indices=focus_indices,
                use_keypoints=use_keypoints,
                use_velocity=use_velocity,
                use_statistics=use_statistics,
                use_ratios=use_ratios
            )
        else:
            print("No keypoint features found for normalization")
    
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
    def __init__(self, input_size=40, hidden_size=256, num_layers=3, num_classes=2, 
                 dropout=0.5, bidirectional=True, gnn_hidden=128, num_gnn_layers=2, 
                 num_joints=12, use_learned_adj=True):
        super(LSTM_GNN_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_joints = num_joints
        self.use_learned_adj = use_learned_adj
        
        # Batch normalization for input
        self.batch_norm = nn.BatchNorm1d(input_size)
        
        # GNN layers for spatial modeling
        self.gnn_layers = nn.ModuleList()
        
        for i in range(num_gnn_layers):
            if i == 0:
                in_features = input_size  # Use full input size for first layer
            else:
                in_features = gnn_hidden
            self.gnn_layers.append(GraphConvolution(in_features, gnn_hidden))
        
        # Learnable adjacency matrix
        if use_learned_adj:
            self.adj_matrix = nn.Parameter(torch.randn(num_joints, num_joints))
            self.adj_activation = nn.Sigmoid()
        
        # LSTM layer for temporal modeling
        lstm_input_size = gnn_hidden * num_joints
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

    def forward(self, x, lengths):
        batch_size, seq_len, input_features = x.shape
        
        # Input preprocessing
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        
        # For GNN, we'll treat each joint as having the full feature set
        # This is more flexible and doesn't require divisibility
        # Reshape to (batch_size * seq_len, num_joints, input_features)
        x = x.unsqueeze(2).expand(-1, -1, self.num_joints, -1)
        x = x.reshape(batch_size * seq_len, self.num_joints, input_features)
        
        # Create adjacency matrix
        adj = self.create_body_graph_adjacency(batch_size * seq_len, x.device)
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, adj)
            x = self.relu(x)
            x = self.dropout(x)
        
        # Reshape back to (batch_size, seq_len, num_joints * gnn_hidden)
        gnn_output_size = self.gnn_layers[-1].out_features
        x = x.reshape(batch_size, seq_len, self.num_joints * gnn_output_size)
        
        # LSTM processing for temporal modeling
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed_x)
        
        # Get final hidden state
        if self.bidirectional:
            out = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            out = h_n[-1]
        
        # Final classification
        out = self.fc(out)
        return out