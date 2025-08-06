# Motion Smoothing for Exercise Movement Analysis

This document describes the motion smoothing functionality added to the exercise movement analysis project. The system now supports two types of smoothing:

1. **Gaussian Smoothing** - For offline preprocessing during training
2. **One Euro Filter** - For real-time smoothing during inference

## Overview

Motion smoothing helps reduce noise in pose data from MediaPipe, improving both training data quality and real-time inference accuracy. The implementation provides:

- **Training phase**: Gaussian smoothing applied once per video during dataset loading
- **Inference phase**: One Euro Filter applied frame-by-frame with minimal latency
- **Configurable parameters**: Adjustable smoothing intensity for different use cases
- **Feature compatibility**: Works with angles, keypoints, velocity, statistics, and ratio features

## 1. Gaussian Smoothing (Training)

### Purpose
Gaussian smoothing is applied during the training phase to reduce MediaPipe noise in the training dataset. It's applied once per video during dataset loading, not during model inference.

### Implementation
- **Location**: `pose_utils.py` - `apply_gaussian_smoothing()` function
- **Method**: Temporal Gaussian filtering with configurable window size and sigma
- **Usage**: Applied during dataset loading in `exercise_dataset.py`

### Parameters
- `window_size`: Size of the Gaussian kernel window (must be odd, default: 5)
- `sigma`: Standard deviation of the Gaussian kernel (default: 1.0)

### Usage in Training

#### Command Line
```bash
python train_lstm.py \
    --exercise right-knee-to-90-degrees \
    --focus right_knee torso \
    --apply_gaussian_smoothing \
    --gaussian_window_size 5 \
    --gaussian_sigma 1.0
```

#### Code Example
```python
from exercise_dataset import ExerciseDataset
from pose_utils import get_angle_indices_by_parts

# Get focus indices
focus_indices = get_angle_indices_by_parts(['right_knee', 'torso'])

# Create dataset with Gaussian smoothing
dataset = ExerciseDataset(
    data_dir='data/exercises',
    exercise_name='right-knee-to-90-degrees',
    focus_indices=focus_indices,
    apply_gaussian_smoothing=True,
    gaussian_window_size=5,
    gaussian_sigma=1.0
)
```

### When to Use
- **Use**: When you want to reduce noise in training data
- **Don't use**: For real-time inference (use One Euro Filter instead)
- **Recommended**: Start with `window_size=5, sigma=1.0` and adjust based on results

## 2. One Euro Filter (Real-time Inference)

### Purpose
The One Euro Filter provides adaptive smoothing for real-time pose data. It automatically adjusts smoothing intensity based on movement velocity - more smoothing for slow movements, less smoothing for fast movements.

### Implementation
- **Location**: `pose_utils.py` - `OneEuroFilter` and `PoseSmoother` classes
- **Method**: Adaptive low-pass filtering based on the "1€ Filter" paper
- **Usage**: Applied frame-by-frame during real-time inference

### Parameters
- `min_cutoff`: Minimum cutoff frequency in Hz (default: 1.0)
  - Higher values = less smoothing for slow movements
  - Lower values = more smoothing for slow movements
- `beta`: Speed coefficient (default: 0.007)
  - Higher values = more adaptive to velocity changes
  - Lower values = less adaptive to velocity changes
- `d_cutoff`: Derivative cutoff frequency in Hz (default: 1.0)
  - Controls smoothing of velocity calculations

### Usage in Real-time Inference

#### Basic Usage
```python
from pose_utils import create_pose_smoother_for_features

# Create smoother for your feature configuration
smoother = create_pose_smoother_for_features(
    focus_indices=focus_indices,
    use_keypoints=use_keypoints,
    use_velocity=use_velocity,
    use_statistics=use_statistics,
    use_ratios=use_ratios,
    min_cutoff=1.0,
    beta=0.007,
    d_cutoff=1.0
)

# For each incoming pose frame
def process_frame(pose_data):
    smoothed_pose = smoother.update(pose_data)
    # Use smoothed_pose for model inference
    return model.predict(smoothed_pose)

# Reset when starting a new exercise sequence
def start_new_exercise():
    smoother.reset()
```

#### WebSocket Integration Example
```python
import asyncio
from pose_utils import create_pose_smoother_for_features

class PoseProcessor:
    def __init__(self):
        # Initialize smoother once
        self.smoother = create_pose_smoother_for_features(
            focus_indices=list(range(15)),
            use_keypoints=False,
            min_cutoff=1.0,
            beta=0.007,
            d_cutoff=1.0
        )
    
    async def handle_pose_frame(self, pose_data):
        # Apply real-time smoothing
        smoothed_pose = self.smoother.update(pose_data)
        
        # Process with your model
        result = await self.model.inference(smoothed_pose)
        return result
    
    def reset_for_new_exercise(self):
        self.smoother.reset()
```

### Performance Characteristics
- **Latency**: < 1ms per frame
- **Memory**: Minimal (stores only previous frame state)
- **CPU**: Very low overhead
- **Adaptive**: Automatically adjusts smoothing based on movement speed

## 3. Parameter Tuning Guide

### Gaussian Smoothing Parameters

| Parameter | Low Smoothing | Medium Smoothing | High Smoothing |
|-----------|---------------|------------------|----------------|
| `window_size` | 3 | 5 | 7 |
| `sigma` | 0.5 | 1.0 | 2.0 |

**Recommendations:**
- Start with `window_size=5, sigma=1.0`
- Increase `window_size` for more smoothing
- Increase `sigma` for wider smoothing kernel
- Keep `window_size` odd for symmetric filtering

### One Euro Filter Parameters

| Parameter | Responsive | Balanced | Smooth |
|-----------|------------|----------|--------|
| `min_cutoff` | 2.0 | 1.0 | 0.5 |
| `beta` | 0.01 | 0.007 | 0.003 |
| `d_cutoff` | 2.0 | 1.0 | 0.5 |

**Recommendations:**
- **Responsive**: For fast-paced exercises where you need quick response
- **Balanced**: Good default for most exercises
- **Smooth**: For exercises with slow, controlled movements

### Exercise-Specific Recommendations

#### Fast Movements (e.g., jumping, quick transitions)
```python
# One Euro Filter settings
min_cutoff = 2.0
beta = 0.01
d_cutoff = 2.0
```

#### Slow Movements (e.g., yoga, stretching)
```python
# One Euro Filter settings
min_cutoff = 0.5
beta = 0.003
d_cutoff = 0.5
```

#### Mixed Movements (e.g., squats, lunges)
```python
# One Euro Filter settings
min_cutoff = 1.0
beta = 0.007
d_cutoff = 1.0
```

## 4. Integration Examples

### Training Pipeline with Gaussian Smoothing
```python
# train_lstm.py with smoothing
python train_lstm.py \
    --exercise right-knee-to-90-degrees \
    --focus right_knee torso \
    --apply_gaussian_smoothing \
    --gaussian_window_size 5 \
    --gaussian_sigma 1.0 \
    --epochs 50 \
    --batch_size 8
```

### Real-time Inference with One Euro Filter
```python
# Example inference script
from pose_utils import create_pose_smoother_for_features
import torch

class RealTimeInference:
    def __init__(self, model_path, focus_indices):
        # Load model
        self.model = torch.load(model_path)
        self.model.eval()
        
        # Create pose smoother
        self.smoother = create_pose_smoother_for_features(
            focus_indices=focus_indices,
            use_keypoints=False,
            min_cutoff=1.0,
            beta=0.007,
            d_cutoff=1.0
        )
    
    def process_frame(self, pose_data):
        # Apply smoothing
        smoothed_pose = self.smoother.update(pose_data)
        
        # Model inference
        with torch.no_grad():
            prediction = self.model(smoothed_pose)
        
        return prediction
    
    def reset(self):
        self.smoother.reset()
```

## 5. Testing and Validation

### Test Gaussian Smoothing
```python
# Test script for Gaussian smoothing
import numpy as np
from pose_utils import apply_gaussian_smoothing

# Create noisy test data
noisy_data = np.random.normal(0, 2, (100, 40))
smoothed_data = apply_gaussian_smoothing(noisy_data, window_size=5, sigma=1.0)

print(f"Noise reduction: {np.std(noisy_data - smoothed_data):.2f}")
```

### Test One Euro Filter
```python
# Run the real-time smoothing example
python real_time_smoothing_example.py
```

## 6. Troubleshooting

### Common Issues

#### Gaussian Smoothing
- **Issue**: `window_size` must be odd
- **Solution**: Ensure `window_size` is odd (3, 5, 7, etc.)

#### One Euro Filter
- **Issue**: High latency
- **Solution**: Reduce `min_cutoff` and `d_cutoff` values
- **Issue**: Too much smoothing
- **Solution**: Increase `min_cutoff` and `beta` values

### Performance Optimization
- **Memory**: One Euro Filter uses minimal memory
- **CPU**: Both methods are computationally efficient
- **GPU**: Not required for smoothing operations

## 7. Best Practices

### Training Phase
1. Start with moderate Gaussian smoothing (`window_size=5, sigma=1.0`)
2. Compare model performance with and without smoothing
3. Adjust parameters based on validation results
4. Document the smoothing parameters used for training

### Inference Phase
1. Initialize PoseSmoother once per session
2. Call `update()` for each frame
3. Call `reset()` when starting a new exercise sequence
4. Monitor processing time to ensure real-time performance
5. Adjust parameters based on exercise type and user feedback

### Parameter Selection
1. **Conservative approach**: Start with default parameters
2. **Iterative tuning**: Adjust one parameter at a time
3. **Exercise-specific**: Different exercises may need different settings
4. **User feedback**: Consider user preferences for responsiveness vs. smoothness

## 8. Future Enhancements

Potential improvements for the motion smoothing system:

1. **Adaptive parameters**: Automatically adjust smoothing based on exercise type
2. **Multi-scale smoothing**: Different smoothing levels for different body parts
3. **Kalman filtering**: Alternative smoothing method for specific use cases
4. **Learning-based smoothing**: Train smoothing parameters from data
5. **Real-time parameter adjustment**: Allow users to adjust smoothing in real-time

## 9. References

- **One Euro Filter**: Géry Casiez et al. "1€ Filter" - https://hal.inria.fr/hal-00670496/document
- **Gaussian Smoothing**: Standard temporal filtering technique
- **MediaPipe**: Google's pose estimation framework

---

For questions or issues with motion smoothing, please refer to the main project documentation or create an issue in the project repository. 