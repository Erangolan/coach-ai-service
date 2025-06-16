import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from main import LSTMClassifier, extract_angles, extract_sequence_from_video

# הגדרת התרגיל והתוויות
EXERCISE_NAME = "right-knee-to-90-degrees"
LABELS = ["good", "bad"]

class ExerciseDataset(Dataset):
    def __init__(self, data_dir, exercise_name):
        self.data = []
        self.labels = []
        
        # מבנה תיקיות צפוי:
        # data_dir/
        #   └── exercise_name/
        #       ├── good/
        #       │   ├── video1.mp4
        #       │   ├── video2.mp4
        #       └── bad/
        #           ├── video1.mp4
        #           ├── video2.mp4
        
        exercise_dir = os.path.join(data_dir, exercise_name)
        if not os.path.exists(exercise_dir):
            raise ValueError(f"Exercise directory not found: {exercise_dir}")
        
        for label_idx, label in enumerate(LABELS):
            label_dir = os.path.join(exercise_dir, label)
            if not os.path.exists(label_dir):
                print(f"Warning: {label_dir} not found")
                continue
                
            for video_file in os.listdir(label_dir):
                if not video_file.endswith('.mp4'):
                    continue
                    
                video_path = os.path.join(label_dir, video_file)
                print(f"Processing {video_path}...")
                sequence = extract_sequence_from_video(video_path)
                
                if len(sequence) > 0:
                    self.data.append(sequence)
                    self.labels.append(label_idx)
                    print(f"Added sequence of length {len(sequence)}")
                else:
                    print(f"Warning: No valid sequence extracted from {video_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded_sequences, torch.tensor(labels), torch.tensor(lengths)

def train_model(data_dir, exercise_name, num_epochs=50, batch_size=8, learning_rate=0.001):
    print(f"Starting training for exercise: {exercise_name}")
    print(f"Data directory: {data_dir}")
    
    # Create dataset and dataloader
    dataset = ExerciseDataset(data_dir, exercise_name)
    if len(dataset) == 0:
        raise ValueError("No valid training data found!")
    
    print(f"Found {len(dataset)} sequences for training")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Initialize model
    model = LSTMClassifier(input_size=40, num_classes=len(LABELS))  # 40 angles
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for sequences, labels, lengths in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Print epoch statistics
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Save the trained model
    model_path = f"models/{exercise_name}_model.pt"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train LSTM model for exercise classification')
    parser.add_argument('--data_dir', type=str, default='data/exercises', help='Directory containing training videos')
    parser.add_argument('--exercise', type=str, default=EXERCISE_NAME, help='Name of the exercise to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    train_model(args.data_dir, args.exercise, args.epochs, args.batch_size, args.lr) 