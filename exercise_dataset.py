from pose_utils import extract_sequence_from_video, apply_all_augmentations
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ExerciseDataset(Dataset):
    def __init__(self, data_dir, exercise_name, focus_indices=None, transform=None, use_keypoints=False):
        self.data_dir = data_dir
        self.exercise_name = exercise_name
        self.transform = transform
        self.focus_indices = focus_indices
        self.use_keypoints = use_keypoints

        self.good_dir = os.path.join(data_dir, exercise_name, 'good')
        self.bad_dir = os.path.join(data_dir, exercise_name, 'bad')
        self.good_files = sorted([f for f in os.listdir(self.good_dir) if f.endswith('.mp4')])
        self.bad_files = sorted([f for f in os.listdir(self.bad_dir) if f.endswith('.mp4')])
        self.all_files = [(os.path.join(self.good_dir, f), 1) for f in self.good_files] + \
                         [(os.path.join(self.bad_dir, f), 0) for f in self.bad_files]

        # לכל דגימה תשמור גם את שם הקובץ המקורי
        self.samples = []
        for video_path, label in self.all_files:
            sequence = extract_sequence_from_video(video_path, focus_indices=self.focus_indices, use_keypoints=self.use_keypoints)
            for aug_seq in apply_all_augmentations(sequence):
                if len(aug_seq) > 0:
                    self.samples.append((aug_seq, label, video_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, label, filename = self.samples[idx]
        return torch.FloatTensor(sequence), label, filename
