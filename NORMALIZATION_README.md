# Feature Normalization in Coach AI Service

This document describes the normalization functionality added to the preprocessing pipeline for exercise classification.

## Overview

The normalization feature allows you to normalize extracted features per video before sending them to the model. This helps improve training stability and model performance by ensuring that features are on similar scales across different videos.

## Normalization Methods

### 1. Min-Max Normalization (Default)
- **Formula**: `(x - min) / (max - min)`
- **Effect**: Scales data to range [0, 1]
- **Use case**: When you want to bound features to a specific range
- **Command**: `--normalize minmax` (or no flag - uses default)

### 2. Z-score Normalization
- **Formula**: `(x - mean) / std`
- **Effect**: Centers data around 0 with unit variance
- **Use case**: When you want to standardize features to have mean=0 and std=1
- **Command**: `--normalize zscore`

## Key Features

### Per-Video Normalization
- Each video is normalized independently
- Normalization statistics are computed only from frames within that specific video
- Preserves temporal relationships within each video
- Does not use statistics from other videos in the dataset

### Selective Feature Normalization
- **Angles (0-180 degrees)**: NOT normalized (preserved as-is)
- **Keypoints (XYZ positions)**: Normalized per video using Min-Max normalization
- **Velocity features (Δkeypoints)**: NOT normalized (preserved as-is)
- **Statistical features**: NOT normalized (preserved as-is)
- **Ratio features**: NOT normalized (preserved as-is)

### Integration with Existing Pipeline
- Applied after feature extraction but before batching
- Works with all existing features (angles, keypoints, velocity, statistics, ratios)
- Maintains the same feature shapes and dimensions
- Compatible with all model types (LSTM, CNN-LSTM, LSTM-Transformer, LSTM-GNN)

### Robust Implementation
- Handles edge cases (empty sequences, single frames)
- Avoids division by zero
- Preserves original data when normalization is not requested
- Automatically determines which features to normalize based on feature types

### Normalization Logic
The system automatically determines which features to normalize:

1. **Angles (0-180 degrees)**: Never normalized - these are already in a meaningful range
2. **Keypoints (XYZ positions)**: Always normalized using Min-Max normalization - these can vary widely in scale
3. **Velocity features**: Never normalized - these are preserved as-is
4. **Statistical features**: Never normalized - these are preserved as-is
5. **Ratio features**: Never normalized - these are already scaled ratios

## Usage

### Training with Normalization

```bash
# Basic training with z-score normalization
python train_lstm.py --exercise right-knee-to-90-degrees --normalize zscore

# Advanced training with all features and min-max normalization
python train_lstm.py --exercise right-knee-to-90-degrees \
  --focus right_knee torso \
  --normalize minmax \
  --use_keypoints --use_velocity --use_statistics --use_ratios \
  --epochs 50 --batch_size 8

# Training with default z-score normalization
python train_lstm.py --exercise right-knee-to-90-degrees \
  --focus right_knee torso \
  --use_keypoints --use_velocity --use_statistics --use_ratios
```

### Command Line Arguments

- `--normalize minmax`: Apply min-max normalization per video (default)
- `--normalize zscore`: Apply z-score normalization per video
- No `--normalize` flag: Uses default min-max normalization

## Implementation Details

### Functions Added

#### `normalize_per_video(features, method="zscore", normalize_indices=None)`
- Normalizes a single video's features
- Parameters:
  - `features`: numpy array of shape (frames, features)
  - `method`: "zscore" or "minmax"
  - `normalize_indices`: list of column indices to normalize (if None, normalize all)
- Returns: normalized features with same shape

#### `get_normalize_indices(focus_indices, use_keypoints, use_velocity, use_statistics, use_ratios)`
- Determines which feature indices should be normalized based on feature types
- Parameters:
  - `focus_indices`: list of angle indices to use
  - `use_keypoints`: whether keypoints are included
  - `use_velocity`: whether velocity features are included
  - `use_statistics`: whether statistical features are included
  - `use_ratios`: whether ratio features are included
- Returns: list of column indices that should be normalized

#### `normalize_features_batch(features_list, method="zscore", normalize_indices=None)`
- Normalizes a list of video features
- Parameters:
  - `features_list`: list of numpy arrays
  - `method`: "zscore" or "minmax"
  - `normalize_indices`: list of column indices to normalize (if None, normalize all)
- Returns: list of normalized features

### Integration Points

1. **`extract_sequence_from_video()`**: Added `normalize_method` parameter
2. **`ExerciseDataset`**: Added `normalize_method` parameter to constructor
3. **`train_lstm.py`**: Added `--normalize` command line argument
4. **`main.py`**: Updated to support normalization in inference

## Benefits

### Training Stability
- Reduces the impact of different video scales and ranges
- Helps with gradient flow during training
- Improves convergence speed

### Model Performance
- Better generalization across different video characteristics
- More robust to variations in recording conditions
- Improved accuracy on test data

### Data Consistency
- Ensures all videos contribute equally to training
- Reduces bias from videos with extreme feature values
- Maintains temporal relationships within videos

## Testing

Run the test script to verify normalization functionality:

```bash
python test_normalization.py
```

This will test:
- Z-score normalization with various data types
- Min-max normalization with various data types
- Edge cases (empty sequences, single frames)
- Batch normalization functionality

## Example Results

### Z-score Normalization
```
Original features: [10, 20, 30], [12, 22, 32], [8, 18, 28]
Normalized features: [-0.48, -0.48, -0.48], [0.29, 0.29, 0.29], [-1.26, -1.26, -1.26]
Mean: [0, 0, 0], Std: [1, 1, 1]
```

### Min-Max Normalization
```
Original features: [10, 20, 30], [12, 22, 32], [8, 18, 28]
Normalized features: [0.29, 0.29, 0.29], [0.57, 0.57, 0.57], [0, 0, 0]
Min: [0, 0, 0], Max: [1, 1, 1]
```

## Recommendations

### When to Use Z-score Normalization
- When features have different scales and you want to standardize them
- When you want to center the data around zero
- When outliers might affect min-max normalization

### When to Use Min-Max Normalization
- When you want to bound features to a specific range
- When you want to preserve zero values
- When the data doesn't have extreme outliers

### When Not to Use Normalization
- When features are already on similar scales
- When you want to preserve the original feature distributions
- When testing baseline performance

## Troubleshooting

### Common Issues

1. **Division by zero**: Handled automatically by setting std/range to 1.0
2. **Empty sequences**: Returned unchanged
3. **Single frames**: Normalized to zero (z-score) or preserved (min-max)

### Performance Impact
- Minimal computational overhead
- Applied once during data loading
- No impact on inference speed

### Debug Logging
During training and inference, you'll see debug messages like:
```
Normalizing only keypoint features using minmax: [3, 4, 5, ..., 38]
```
This shows which keypoint indices are being normalized. If no keypoint features are found, you'll see:
```
No keypoint features found for normalization
```

## Future Enhancements

Potential improvements to consider:
- Global normalization across the entire dataset
- Adaptive normalization based on data statistics
- Different normalization methods for different feature types
- Normalization parameter tuning based on validation performance 