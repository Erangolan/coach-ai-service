from pose_utils import extract_sequence_from_video

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ExerciseDataset(Dataset):
    def __init__(self, data_dir, exercise_name, focus_indices=None, transform=None):
        self.data_dir = data_dir
        self.exercise_name = exercise_name
        self.transform = transform
        self.focus_indices = focus_indices

        self.good_dir = os.path.join(data_dir, exercise_name, 'good')
        self.bad_dir = os.path.join(data_dir, exercise_name, 'bad')
        self.good_files = sorted([f for f in os.listdir(self.good_dir) if f.endswith('.mp4')])
        self.bad_files = sorted([f for f in os.listdir(self.bad_dir) if f.endswith('.mp4')])
        self.all_files = [(os.path.join(self.good_dir, f), 1) for f in self.good_files] + \
                         [(os.path.join(self.bad_dir, f), 0) for f in self.bad_files]

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        video_path, label = self.all_files[idx]
        sequence = extract_sequence_from_video(video_path, focus_indices=self.focus_indices)
        if len(sequence) == 0:
            raise ValueError(f"Empty sequence extracted from {video_path}")
        sequence = np.array(sequence)
        if self.transform:
            sequence = self.transform(sequence)
        return torch.FloatTensor(sequence), label
