# LSTM + GNN Classifier Documentation

## Overview

The `LSTM_GNN_Classifier` is a hybrid neural network architecture that combines **Graph Neural Networks (GNN)** with **Long Short-Term Memory (LSTM)** networks for exercise recognition. This model is specifically designed to capture both **spatial relationships** between body joints and **temporal dynamics** of movements.

## Architecture

### Components

1. **Graph Neural Network (GNN)**: Models body joints as a graph structure
2. **LSTM Network**: Processes temporal sequences
3. **Fully Connected Layers**: Final classification

### Data Flow

```
Input Sequence → GNN (Spatial) → LSTM (Temporal) → Classification
```

## Key Features

### 1. Spatial Modeling with GNN
- **Body Joint Graph**: Treats 12 body joints as nodes in a graph
- **Learnable Adjacency Matrix**: Automatically learns relationships between joints
- **Fixed Adjacency Matrix**: Pre-defined body structure connections
- **Graph Convolution**: Captures spatial dependencies between joints

### 2. Temporal Modeling with LSTM
- **Bidirectional LSTM**: Processes sequences in both directions
- **Variable Length Sequences**: Handles different video lengths
- **Temporal Dependencies**: Captures movement patterns over time

### 3. Hybrid Approach
- **Spatial-Temporal Fusion**: Combines spatial and temporal information
- **Flexible Input**: Works with any number of features
- **Robust Architecture**: Handles missing or noisy data

## Model Parameters

### Core Parameters
- `input_size`: Number of input features (e.g., 51 for angles + keypoints)
- `hidden_size`: LSTM hidden layer size (default: 256)
- `num_classes`: Number of output classes (default: 5)
- `num_joints`: Number of body joints (default: 12)

### GNN Parameters
- `gnn_hidden`: GNN hidden layer size (default: 128)
- `num_gnn_layers`: Number of GNN layers (default: 2)
- `use_learned_adj`: Whether to use learnable adjacency matrix (default: True)

### LSTM Parameters
- `bidirectional`: Use bidirectional LSTM (default: True)
- `num_layers`: Number of LSTM layers (default: 3)
- `dropout`: Dropout rate (default: 0.5)

## Body Joint Structure

The model uses 12 body joints organized as follows:

### Right Side (Joints 0-5)
- **Joint 0**: Right Shoulder
- **Joint 1**: Right Elbow
- **Joint 2**: Right Wrist
- **Joint 3**: Right Hip
- **Joint 4**: Right Knee
- **Joint 5**: Right Ankle

### Left Side (Joints 6-11)
- **Joint 6**: Left Shoulder
- **Joint 7**: Left Elbow
- **Joint 8**: Left Wrist
- **Joint 9**: Left Hip
- **Joint 10**: Left Knee
- **Joint 11**: Left Ankle

## Adjacency Matrix Options

### 1. Learnable Adjacency Matrix
```python
use_learned_adj = True
```
- Automatically learns relationships between joints
- Adapts to specific exercise patterns
- More flexible but requires more training data

### 2. Fixed Adjacency Matrix
```python
use_learned_adj = False
```
- Pre-defined connections based on body anatomy
- More stable and interpretable
- Works well with limited training data

## Usage Examples

### Basic Training
```bash
python train_lstm.py \
  --exercise "right-knee-to-90-degrees" \
  --model_type "lstm_gnn" \
  --use_keypoints \
  --use_velocity \
  --use_statistics \
  --focus right_knee right_shoulder torso
```

### Advanced Configuration
```python
model = LSTM_GNN_Classifier(
    input_size=51,           # 15 angles + 36 keypoints
    hidden_size=256,         # LSTM hidden size
    num_classes=5,           # 5 exercise classes
    num_joints=12,           # 12 body joints
    gnn_hidden=128,          # GNN hidden size
    num_gnn_layers=2,        # 2 GNN layers
    use_learned_adj=True,    # Learnable adjacency
    bidirectional=True,       # Bidirectional LSTM
    dropout=0.5              # Dropout rate
)
```

### Real-time Inference
```python
from pose_utils import LSTM_GNN_Classifier

# Load trained model
model = LSTM_GNN_Classifier(input_size=51, num_classes=5)
model.load_state_dict(torch.load('models/exercise_model.pt'))
model.eval()

# Process sequence
sequence = torch.randn(1, 20, 51)  # 1 batch, 20 frames, 51 features
lengths = torch.tensor([20])
output = model(sequence, lengths)
prediction = torch.argmax(output, dim=1)
```

## Advantages

### 1. Spatial Awareness
- **Joint Relationships**: Understands how joints relate to each other
- **Body Structure**: Respects anatomical constraints
- **Spatial Patterns**: Captures complex body postures

### 2. Temporal Dynamics
- **Movement Sequences**: Processes complete movement patterns
- **Temporal Dependencies**: Understands movement progression
- **Variable Length**: Handles different video durations

### 3. Robustness
- **Noise Tolerance**: Handles noisy pose detection
- **Missing Data**: Works with incomplete joint data
- **Generalization**: Better generalization to new subjects

## Performance Considerations

### Computational Requirements
- **Memory**: Higher memory usage due to GNN operations
- **Training Time**: Longer training time compared to simple LSTM
- **GPU Usage**: Benefits from GPU acceleration

### Optimization Tips
1. **Batch Size**: Use smaller batch sizes (4-8) for memory efficiency
2. **GNN Layers**: Start with 1-2 GNN layers
3. **Hidden Size**: Reduce hidden sizes for faster training
4. **Mixed Precision**: Use mixed precision training for speed

## Comparison with Other Models

| Model | Spatial Modeling | Temporal Modeling | Complexity | Performance |
|-------|-----------------|-------------------|------------|-------------|
| LSTM | ❌ | ✅ | Low | Good |
| CNN-LSTM | ✅ (Local) | ✅ | Medium | Better |
| LSTM-Transformer | ❌ | ✅ (Advanced) | High | Excellent |
| **LSTM-GNN** | ✅ (Global) | ✅ | High | **Best** |

## Best Practices

### 1. Data Preparation
- Use consistent joint ordering
- Normalize input features
- Include keypoints for better spatial modeling

### 2. Model Configuration
- Start with `use_learned_adj=True` for complex exercises
- Use `use_learned_adj=False` for simple exercises or limited data
- Adjust `gnn_hidden` based on input complexity

### 3. Training Strategy
- Use learning rate scheduling
- Monitor both training and validation loss
- Use early stopping to prevent overfitting

### 4. Feature Engineering
- Combine angles with keypoints
- Use velocity and statistical features
- Include ratio features for better relationships

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size
   - Reduce hidden sizes
   - Use gradient accumulation

2. **Slow Training**
   - Use GPU acceleration
   - Reduce model complexity
   - Use mixed precision training

3. **Poor Performance**
   - Check data quality
   - Adjust model parameters
   - Try different adjacency matrix types

### Debugging Tips

1. **Check Input Shapes**
   ```python
   print(f"Input shape: {x.shape}")
   print(f"Expected: (batch_size, seq_len, input_size)")
   ```

2. **Monitor GNN Output**
   ```python
   # Add debug prints in forward method
   print(f"GNN output shape: {x.shape}")
   ```

3. **Verify Adjacency Matrix**
   ```python
   adj = model.create_body_graph_adjacency(batch_size, device)
   print(f"Adjacency matrix shape: {adj.shape}")
   ```

## Future Enhancements

### Potential Improvements
1. **Attention Mechanisms**: Add attention to GNN layers
2. **Multi-Scale Processing**: Process different temporal scales
3. **Hierarchical GNN**: Multi-level graph structure
4. **Dynamic Graphs**: Adaptive graph structure
5. **Multi-Modal Fusion**: Combine with other sensors

### Research Directions
1. **Interpretability**: Understanding learned joint relationships
2. **Transfer Learning**: Pre-trained models for new exercises
3. **Real-time Optimization**: Faster inference for live applications
4. **Multi-Person Support**: Handle multiple people in video

## Conclusion

The `LSTM_GNN_Classifier` represents a significant advancement in exercise recognition by combining the strengths of both spatial and temporal modeling. Its ability to capture complex body joint relationships while processing temporal sequences makes it particularly well-suited for exercise form analysis and real-time feedback systems.

For optimal results, ensure you have:
- High-quality pose detection data
- Sufficient training examples
- Appropriate computational resources
- Proper hyperparameter tuning

The model is now ready for use in your exercise recognition system! 