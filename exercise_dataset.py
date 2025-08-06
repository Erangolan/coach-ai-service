# exercise_dataset.py

from pose_utils import extract_sequence_from_video, apply_all_augmentations, subtle_bad_augmentation, DEFAULT_NORMALIZE_METHOD
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

LABELS_MAP = {
    "good": 0,
    "bad-left-angle": 1,
    "bad-lower-knee": 2,
    "bad-right-angle": 3,
    "idle": 4,
}
LABELS = list(LABELS_MAP.keys())

class ExerciseDataset(Dataset):
    def __init__(self, data_dir, exercise_name, focus_indices=None, transform=None, use_keypoints=False, use_velocity=False, use_statistics=False, use_ratios=False, normalize_method=DEFAULT_NORMALIZE_METHOD, print_both=None, apply_gaussian_smoothing=False, gaussian_window_size=5, gaussian_sigma=1.0):
        self.data_dir = data_dir
        self.exercise_name = exercise_name
        self.transform = transform
        self.focus_indices = focus_indices
        self.use_keypoints = use_keypoints
        self.use_velocity = use_velocity
        self.use_statistics = use_statistics
        self.use_ratios = use_ratios
        self.normalize_method = normalize_method
        self.print_both = print_both
        self.apply_gaussian_smoothing = apply_gaussian_smoothing
        self.gaussian_window_size = gaussian_window_size
        self.gaussian_sigma = gaussian_sigma

        self.samples = []
        for label_name in LABELS:
            label_dir = os.path.join(data_dir, exercise_name, label_name)
            if not os.path.exists(label_dir):
                continue
            files = sorted([f for f in os.listdir(label_dir) if f.endswith('.mp4') or f.endswith('.mov')])
            for f in files:
                video_path = os.path.join(label_dir, f)
                label = LABELS_MAP[label_name]
                sequence = extract_sequence_from_video(video_path, focus_indices=self.focus_indices, use_keypoints=self.use_keypoints, use_velocity=self.use_velocity, use_statistics=self.use_statistics, use_ratios=self.use_ratios, normalize_method=self.normalize_method, apply_gaussian_smoothing=self.apply_gaussian_smoothing, gaussian_window_size=self.gaussian_window_size, gaussian_sigma=self.gaussian_sigma)
                if len(sequence) == 0:
                    if self.print_both:
                        self.print_both(f"EMPTY SEQUENCE: {video_path}")
                    else:
                        print(f"EMPTY SEQUENCE: {video_path}")
                    continue
                # Check if this is an idle sample
                is_idle = (label_name == "idle")
                aug_sequences = apply_all_augmentations(sequence, debug_path=video_path, is_idle=is_idle)
                for aug_seq in aug_sequences:
                    if len(aug_seq) > 0:
                        self.samples.append((aug_seq, label, video_path))
        print(f"Total samples in dataset: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, label, filename = self.samples[idx]
        return torch.FloatTensor(sequence.copy()), label, filename

num_classes = 3
