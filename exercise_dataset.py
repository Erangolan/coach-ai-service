# exercise_dataset.py

from pose_utils import extract_sequence_from_video, apply_all_augmentations, subtle_bad_augmentation
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class ExerciseDataset(Dataset):
    def __init__(self, data_dir, exercise_name, focus_indices=None, transform=None, use_keypoints=False, print_both=None):
        self.data_dir = data_dir
        self.exercise_name = exercise_name
        self.transform = transform
        self.focus_indices = focus_indices
        self.use_keypoints = use_keypoints
        self.print_both = print_both

        self.good_dir = os.path.join(data_dir, exercise_name, 'good')
        self.bad_dir = os.path.join(data_dir, exercise_name, 'bad')
        self.good_files = sorted([f for f in os.listdir(self.good_dir) if f.endswith('.mp4')])
        self.bad_files = sorted([f for f in os.listdir(self.bad_dir) if f.endswith('.mp4')])
        self.all_files = [(os.path.join(self.good_dir, f), 1) for f in self.good_files] + \
                         [(os.path.join(self.bad_dir, f), 0) for f in self.bad_files]

        self.samples = []
        good_aug, bad_aug = 0, 0
        good_augmented_samples = []
        bad_augmented_samples = []

        max_aug_per_video = 4  # Limit to 4 augmentations per video

        for video_path, label in self.all_files:
            sequence = extract_sequence_from_video(video_path, focus_indices=self.focus_indices, use_keypoints=self.use_keypoints)
            if len(sequence) == 0:
                if self.print_both:
                    self.print_both(f"EMPTY SEQUENCE: {video_path}")
                else:
                    print(f"EMPTY SEQUENCE: {video_path}")
                continue
            aug_sequences = apply_all_augmentations(sequence, debug_path=video_path)
            aug_sequences = aug_sequences[:max_aug_per_video]
            for aug_seq in aug_sequences:
                if len(aug_seq) > 0:
                    self.samples.append((aug_seq, label, video_path))
                    if label == 1:
                        good_aug += 1
                        good_augmented_samples.append((aug_seq, label, video_path))
                    else:
                        bad_aug += 1
                        bad_augmented_samples.append((aug_seq, label, video_path))

        # איזון דאטה: הזרקת דגימות BAD מתוך good עם Augmentation (לא רק שכפול)
        if bad_aug < good_aug and good_augmented_samples:
            needed = good_aug - bad_aug
            extra_bad_samples = []
            for _ in range(needed):
                # תיצור bad מדגימת good אקראית (simulate mistake)
                good_seq, _, video_path = random.choice(good_augmented_samples)
                fake_bad = subtle_bad_augmentation(good_seq)
                if len(fake_bad) > 0:
                    self.samples.append((fake_bad, 0, video_path + "_bad_from_good"))
                    bad_aug += 1
                    extra_bad_samples.append((fake_bad, 0, video_path + "_bad_from_good"))
            if self.print_both:
                self.print_both(f"Added {len(extra_bad_samples)} synthetic bad samples from good (augmentation) to balance dataset.")
            else:
                print(f"Added {len(extra_bad_samples)} synthetic bad samples from good (augmentation) to balance dataset.")

        msg1 = f"\nTotal videos: Good={len(self.good_files)}, Bad={len(self.bad_files)}"
        msg2 = f"Total samples after augmentation: Good={good_aug}, Bad={bad_aug}"
        msg3 = f"Total samples in dataset: {len(self.samples)}\n"
        if self.print_both:
            self.print_both(msg1)
            self.print_both(msg2)
            self.print_both(msg3)
        else:
            print(msg1)
            print(msg2)
            print(msg3)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, label, filename = self.samples[idx]
        return torch.FloatTensor(sequence.copy()), label, filename
