import os
import sys
from datetime import datetime

# יצירת תיקיית לוגים אם לא קיימת
os.makedirs('logs', exist_ok=True)

# יצירת קובץ לוג עם timestamp
log_file = f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_file_handle = open(log_file, 'w')

# שמירת stdout/stderr המקוריים
original_stdout = sys.stdout
original_stderr = sys.stderr

# הפניית כל הפלט לקובץ
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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["bad", "good"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig(filename)
    plt.close()
    print_both(f"Confusion matrix saved to {filename}")

def train_model(data_dir, exercise_name, focus_parts, num_epochs=50, batch_size=8, learning_rate=0.001,
                use_keypoints=False, augment=False, bidirectional=True, model_type='lstm'):
    print_both(f"Starting training for exercise: {exercise_name}")
    print_both(f"Data directory: {data_dir}")
    print_both(f"Model type: {model_type.upper()}")

    focus_indices = get_angle_indices_by_parts(focus_parts)
    input_size = len(focus_indices) if focus_indices is not None else 40
    if use_keypoints:
        input_size += 12 * 3  # 12 keypoints * (x,y,z)
    print_both(f"Using input_size={input_size} (focus_parts={focus_parts})")

    full_dataset = ExerciseDataset(data_dir, exercise_name, focus_indices=focus_indices, use_keypoints=use_keypoints, print_both=print_both)
    # אסוף שמות סרטונים מקוריים
    video_to_label = {}  # {filename: label}
    for _, label, filename in full_dataset.samples:
        orig_file = filename.split('_extra_aug')[0]  # עוזר גם לדגימות מאוגמנטציה
        video_to_label[orig_file] = label

    all_videos = list(set(video_to_label.keys()))
    train_videos, test_videos = train_test_split(all_videos, test_size=0.2, random_state=42, stratify=[video_to_label[v] for v in all_videos])
    train_videos, val_videos = train_test_split(train_videos, test_size=0.25, random_state=42, stratify=[video_to_label[v] for v in train_videos])

    def split_by_videos(samples, videos):
        return [i for i, sample in enumerate(samples) if sample[2].split('_extra_aug')[0] in videos]

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
    num_good = label_counts.get(1, 0)
    num_bad = label_counts.get(0, 0)
    total = num_good + num_bad
    class_weights = [0, 0]
    if num_bad > 0 and num_good > 0:
        class_weights[0] = total / (2 * num_bad)
        class_weights[1] = total / (2 * num_good)
    else:
        class_weights = [1.0, 1.0]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print_both(f"Class weights: {class_weights}")

    # Choose model
    if model_type == 'cnn_lstm':
        model = CNN_LSTM_Classifier(input_size=input_size, num_classes=2, bidirectional=bidirectional)
    elif model_type == 'lstm_transformer':
        model = LSTM_Transformer_Classifier(input_size=input_size, num_classes=2, bidirectional=bidirectional)
    else:
        model = LSTMClassifier(input_size=input_size, num_classes=2, bidirectional=bidirectional)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
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
        print_both(f'Validation Accuracy: {val_acc:.2f}%')

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
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    parser.add_argument('--no_bidirectional', action='store_true', help='Disable BiLSTM')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'cnn_lstm', 'lstm_transformer'], help='Which model to use')
    args = parser.parse_args()
    
    try:
        train_model(
            args.data_dir, args.exercise, args.focus, args.epochs, args.batch_size, args.lr,
            use_keypoints=args.use_keypoints,
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
