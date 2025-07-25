import os
import sys
import time
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Create a log file with a timestamp
log_file = f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_file_handle = open(log_file, 'w')

# Save the original stdout/stderr
original_stdout = sys.stdout
original_stderr = sys.stderr

# Redirect all output to the log file
sys.stdout = log_file_handle
sys.stderr = log_file_handle

from suppress_logs import suppress_logs
suppress_logs()

import argparse
from pose_utils import get_angle_indices_by_parts, LSTMClassifier, CNN_LSTM_Classifier, LSTM_Transformer_Classifier
from exercise_dataset import ExerciseDataset
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

LABELS_MAP = {
    "good": 0,
    "bad-left-angle": 1,
    "bad-lower-knee": 2,
    "bad-right-angle": 3,
    "idle": 4,
}
LABELS = list(LABELS_MAP.keys())
label_to_idx = LABELS_MAP

def print_both(message):
    print(message, file=log_file_handle, flush=True)
    print(message, file=original_stdout, flush=True)

def collate_fn(batch):
    sequences, labels, filenames = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded_sequences, torch.tensor(labels), torch.tensor(lengths), filenames

def save_misclassified(dataloader, model, device="cpu"):
    model.eval()
    misclassified = []
    for sequences, labels, lengths, filenames in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        outputs = model(sequences, lengths)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(labels)):
            if predicted[i].item() != labels[i].item():
                misclassified.append({
                    "file": filenames[i],
                    "true_label": int(labels[i].item()),
                    "predicted": int(predicted[i].item())
                })
    pd.DataFrame(misclassified).to_csv("misclassified.csv", index=False)
    print_both(f"Saved misclassified samples to misclassified.csv")

def save_confusion_matrix(model, dataloader, device="cpu", filename="confusion_matrix.png"):
    y_true, y_pred = [], []
    model.eval()
    for sequences, labels, lengths, _ in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        outputs = model(sequences, lengths)
        _, predicted = torch.max(outputs.data, 1)
        y_true += labels.cpu().numpy().tolist()
        y_pred += predicted.cpu().numpy().tolist()
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig(filename)
    plt.close()
    print_both(f"Confusion matrix saved to {filename}")

def train_model(data_dir, exercise_name, focus_parts, num_epochs=50, batch_size=8, learning_rate=0.001,
                use_keypoints=False, use_velocity=False, use_statistics=False, use_ratios=False, augment=False, bidirectional=True, model_type='lstm'):
    print_both(f"Starting training for exercise: {exercise_name}")
    print_both(f"Data directory: {data_dir}")
    print_both(f"Model type: {model_type.upper()}")

    focus_indices = get_angle_indices_by_parts(focus_parts)
    base_features = len(focus_indices) if focus_indices is not None else 40
    if use_keypoints:
        base_features += 12 * 3  # 12 keypoints * (x,y,z)
    input_size = base_features
    if use_velocity:
        input_size *= 3  # Triple the input size for velocity + acceleration features
    if use_statistics:
        input_size += input_size * 6  # Add 6 statistical features per feature (mean, median, std, max, min, range)
    if use_ratios:
        # Add 1 ratio feature: primary knee angle / primary torso angle
        input_size += 1
        print_both(f"Ratio features: 1 (primary knee / primary torso)")
    
    print_both(f"Using input_size={input_size} (focus_parts={focus_parts}, use_keypoints={use_keypoints}, use_velocity={use_velocity}, use_statistics={use_statistics}, use_ratios={use_ratios})")
    print_both(f"focus_indices length: {len(focus_indices) if focus_indices is not None else 40}")

    full_dataset = ExerciseDataset(data_dir, exercise_name, focus_indices=focus_indices, use_keypoints=use_keypoints, use_velocity=use_velocity, use_statistics=use_statistics, use_ratios=use_ratios, print_both=print_both)
    
    # Check if we have enough data
    if len(full_dataset) == 0:
        raise ValueError(f"No training data found for exercise '{exercise_name}'. Please check your data directory.")
    
    # Count samples per class
    labels = [label for _, label, _ in full_dataset.samples]
    label_counts = Counter(labels)
    print_both(f"Data distribution: {label_counts}")
    
    if len(label_counts) < 2:
        raise ValueError(f"Need at least 2 classes (good/bad), but found only {len(label_counts)} class(es).")
    
    min_samples = min(label_counts.values())
    if min_samples < 2:
        print_both(f"Warning: Class with label {min(label_counts, key=label_counts.get)} has only {min_samples} samples.")
        print_both("Consider adding more training data for better results.")
    
    # Collect original video names
    video_to_label = {}  # {filename: label}
    for _, label, filename in full_dataset.samples:
        video_to_label[filename] = label

    # Debug: Print video distribution
    print_both(f"Total samples in dataset: {len(full_dataset.samples)}")
    print_both(f"Unique videos found: {len(set(video_to_label.keys()))}")
    
    # Count videos per class
    video_class_counts = Counter(video_to_label.values())
    print_both(f"Videos per class: {video_class_counts}")
    
    # Show some example filenames
    # print_both("Sample filenames:")
    # for i, (_, label, filename) in enumerate(full_dataset.samples[:5]):
    #     print_both(f"  {i}: {filename} -> label {label}")
    
    # Split by video (not by sample)
    all_videos = list(set(video_to_label.keys()))
    # print_both(f"All videos: {all_videos}")
    
    # Check if we have enough samples for stratified splitting
    labels_for_videos = [video_to_label[v] for v in all_videos]
    label_counts = Counter(labels_for_videos)
    min_samples_per_class = min(label_counts.values())
    
    # print_both(f"Labels for videos: {labels_for_videos}")
    print_both(f"Label counts: {label_counts}")
    print_both(f"Min samples per class: {min_samples_per_class}")
    
    if min_samples_per_class >= 2:
        # Use stratified splitting if we have enough samples
        train_videos, test_videos = train_test_split(all_videos, test_size=0.2, random_state=42, stratify=labels_for_videos)
        train_videos, val_videos = train_test_split(train_videos, test_size=0.25, random_state=42, stratify=[video_to_label[v] for v in train_videos])
    else:
        # Use regular splitting if we don't have enough samples
        print_both(f"Warning: Insufficient samples for stratified splitting. Using regular splitting.")
        print_both(f"Label counts: {label_counts}")
        train_videos, test_videos = train_test_split(all_videos, test_size=0.2, random_state=42)
        train_videos, val_videos = train_test_split(train_videos, test_size=0.25, random_state=42)
    
    print_both(f"Train videos: {len(train_videos)}")
    print_both(f"Val videos: {len(val_videos)}")
    print_both(f"Test videos: {len(test_videos)}")

    def split_by_videos(samples, videos):
        return [i for i, sample in enumerate(samples) if sample[2] in videos]

    train_idx = split_by_videos(full_dataset.samples, train_videos)
    val_idx = split_by_videos(full_dataset.samples, val_videos)
    test_idx = split_by_videos(full_dataset.samples, test_videos)

    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Device selection: CUDA (NVIDIA), MPS (Apple Silicon), else CPU
    if torch.cuda.is_available():
        device = "cuda"
        print_both("Training on CUDA GPU: " + torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print_both("Training on Apple MPS (GPU)")
    else:
        device = "cpu"
        print_both("Training on CPU")

    # Count class samples for weights
    labels = [label for _, label, _ in full_dataset.samples]
    label_counts = Counter(labels)
    num_classes = 5  # עדכן ל-5 קטגוריות (good, bad-left-angle, bad-lower-knee, bad-right-angle, idle)
    class_weights = [0] * num_classes
    total = sum(label_counts.values())
    for i in range(num_classes):
        class_weights[i] = total / (num_classes * label_counts.get(i, 1))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print_both(f"Class weights: {class_weights}")

    # Choose model
    if model_type == 'cnn_lstm':
        model = CNN_LSTM_Classifier(input_size=input_size, num_classes=5, bidirectional=bidirectional)
    elif model_type == 'lstm_transformer':
        model = LSTM_Transformer_Classifier(input_size=input_size, num_classes=5, bidirectional=bidirectional)
    else:
        model = LSTMClassifier(input_size=input_size, num_classes=5, bidirectional=bidirectional)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for sequences, labels, lengths, _ in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Ensure criterion weights are on the same device as labels
            if criterion.weight.device != labels.device:
                criterion.weight = criterion.weight.to(labels.device)
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print_both(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Validation evaluation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for sequences, labels, lengths, _ in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences, lengths)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        epoch_time = time.time() - epoch_start_time
        print_both(f'Validation Accuracy: {val_acc:.2f}%')
        print_both(f'Epoch {epoch+1} completed in {epoch_time:.2f} seconds ({epoch_time/60:.2f} minutes)')

    # Save model and check test set
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{exercise_name}_model.pt"
    torch.save(model.state_dict(), model_path)
    print_both(f"Model saved as {model_path}")

    # Save misclassified from test set
    save_misclassified(test_loader, model, device=device)
    save_confusion_matrix(model, test_loader, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM/CNN-LSTM/LSTM-Transformer model for exercise classification')
    parser.add_argument('--data_dir', type=str, default='data/exercises', help='Directory containing training videos')
    parser.add_argument('--exercise', type=str, required=True, help='Name of the exercise to train')
    parser.add_argument('--focus', nargs='*', default=[], help='Which body parts to focus on (e.g. right_leg right_knee)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--use_keypoints', action='store_true', help='Include xyz keypoints as features')
    parser.add_argument('--use_velocity', action='store_true', help='Add velocity features (change between consecutive frames)')
    parser.add_argument('--use_statistics', action='store_true', help='Add statistical features (mean, median, std, max, min, range)')
    parser.add_argument('--use_ratios', action='store_true', help='Add ratio features between relevant angles')

    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    parser.add_argument('--no_bidirectional', action='store_true', help='Disable BiLSTM')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'cnn_lstm', 'lstm_transformer'], help='Which model to use')
    args = parser.parse_args()
    
    try:
        train_model(
            args.data_dir, args.exercise, args.focus, args.epochs, args.batch_size, args.lr,
            use_keypoints=args.use_keypoints,
            use_velocity=args.use_velocity,
            use_statistics=args.use_statistics,
            use_ratios=args.use_ratios,
            augment=args.augment,
            bidirectional=not args.no_bidirectional,
            model_type=args.model_type,
        )
    finally:
        # Close log file
        log_file_handle.close()
        # Restore original streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"\nTraining complete! Full logs saved to: {log_file}")
