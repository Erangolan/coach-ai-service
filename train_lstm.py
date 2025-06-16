import argparse
from pose_utils import get_angle_indices_by_parts, LSTMClassifier
from exercise_dataset import ExerciseDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded_sequences, torch.tensor(labels), torch.tensor(lengths)

def train_model(data_dir, exercise_name, focus_parts, num_epochs=50, batch_size=8, learning_rate=0.001):
    print(f"Starting training for exercise: {exercise_name}")
    print(f"Data directory: {data_dir}")

    focus_indices = get_angle_indices_by_parts(focus_parts)
    input_size = len(focus_indices) if focus_indices is not None else 40
    print(f"Using input_size={input_size} (focus_parts={focus_parts})")

    dataset = ExerciseDataset(data_dir, exercise_name, focus_indices=focus_indices)
    if len(dataset) == 0:
        raise ValueError("No valid training data found!")

    print(f"Found {len(dataset)} sequences for training")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = LSTMClassifier(input_size=input_size, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for sequences, labels, lengths in dataloader:
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    model_path = f"models/{exercise_name}_model.pt"
    import os; os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM model for exercise classification')
    parser.add_argument('--data_dir', type=str, default='data/exercises', help='Directory containing training videos')
    parser.add_argument('--exercise', type=str, required=True, help='Name of the exercise to train')
    parser.add_argument('--focus', nargs='*', default=[], help='Which body parts to focus on (e.g. right_leg right_knee)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    train_model(args.data_dir, args.exercise, args.focus, args.epochs, args.batch_size, args.lr)
