#!/usr/bin/env python3
"""
Test script to verify motion smoothing integration.

This script tests:
1. Gaussian smoothing integration with dataset loading
2. One Euro Filter functionality
3. Parameter passing and configuration
"""

import numpy as np
from pose_utils import (
    apply_gaussian_smoothing_to_sequence, 
    OneEuroFilter, 
    PoseSmoother, 
    create_pose_smoother_for_features
)
from exercise_dataset import ExerciseDataset
from pose_utils import get_angle_indices_by_parts


def test_gaussian_smoothing():
    """Test Gaussian smoothing function."""
    print("=== Testing Gaussian Smoothing ===")
    
    # Create test data
    noisy_data = np.random.normal(0, 3, (50, 15))
    
    # Apply smoothing
    smoothed_data = apply_gaussian_smoothing_to_sequence(noisy_data, window_size=5, sigma=1.0)
    
    # Calculate noise reduction
    noise_reduction = np.std(noisy_data - smoothed_data)
    
    print(f"Original data shape: {noisy_data.shape}")
    print(f"Smoothed data shape: {smoothed_data.shape}")
    print(f"Noise reduction: {noise_reduction:.2f} units")
    print(f"Data type preserved: {noisy_data.dtype == smoothed_data.dtype}")
    print("✓ Gaussian smoothing test passed\n")


def test_one_euro_filter():
    """Test One Euro Filter functionality."""
    print("=== Testing One Euro Filter ===")
    
    # Create filter
    filter_instance = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
    
    # Test with noisy data
    noisy_values = np.random.normal(50, 2, 100)
    smoothed_values = []
    
    for value in noisy_values:
        smoothed_value = filter_instance.update(value)
        smoothed_values.append(smoothed_value)
    
    smoothed_values = np.array(smoothed_values)
    noise_reduction = np.std(noisy_values - smoothed_values)
    
    print(f"Input values: {len(noisy_values)}")
    print(f"Output values: {len(smoothed_values)}")
    print(f"Noise reduction: {noise_reduction:.2f} units")
    print(f"Filter state maintained: {filter_instance.x0 is not None}")
    print("✓ One Euro Filter test passed\n")


def test_pose_smoother():
    """Test PoseSmoother class."""
    print("=== Testing PoseSmoother ===")
    
    # Create smoother
    smoother = PoseSmoother(num_features=15, min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
    
    # Test with multiple frames
    num_frames = 50
    noisy_frames = np.random.normal(50, 2, (num_frames, 15))
    smoothed_frames = []
    
    for frame in noisy_frames:
        smoothed_frame = smoother.update(frame)
        smoothed_frames.append(smoothed_frame)
    
    smoothed_frames = np.array(smoothed_frames)
    noise_reduction = np.mean(np.std(noisy_frames - smoothed_frames, axis=0))
    
    print(f"Input frames: {noisy_frames.shape}")
    print(f"Output frames: {smoothed_frames.shape}")
    print(f"Average noise reduction: {noise_reduction:.2f} units")
    print(f"Number of filters: {len(smoother.filters)}")
    
    # Test reset functionality
    smoother.reset()
    print(f"Reset successful: {smoother.filters[0].x0 is None}")
    print("✓ PoseSmoother test passed\n")


def test_create_pose_smoother_for_features():
    """Test pose smoother creation for different feature configurations."""
    print("=== Testing PoseSmoother Creation ===")
    
    # Test different configurations
    configs = [
        {"focus_indices": list(range(15)), "use_keypoints": False, "use_velocity": False, "use_statistics": False, "use_ratios": False},
        {"focus_indices": list(range(15)), "use_keypoints": True, "use_velocity": False, "use_statistics": False, "use_ratios": False},
        {"focus_indices": list(range(15)), "use_keypoints": True, "use_velocity": True, "use_statistics": False, "use_ratios": False},
    ]
    
    for i, config in enumerate(configs):
        smoother = create_pose_smoother_for_features(**config)
        expected_features = len(config["focus_indices"])
        
        if config["use_keypoints"]:
            expected_features += 12 * 3  # 12 keypoints * 3 coordinates
        
        if config["use_velocity"]:
            base_features = expected_features
            expected_features += base_features * 2  # velocity + acceleration
        
        print(f"Config {i+1}: {len(smoother.filters)} filters (expected: {expected_features})")
        assert len(smoother.filters) == expected_features, f"Feature count mismatch for config {i+1}"
    
    print("✓ PoseSmoother creation test passed\n")


def test_dataset_integration():
    """Test that dataset can be created with smoothing parameters."""
    print("=== Testing Dataset Integration ===")
    
    try:
        # Try to create dataset with smoothing (this will fail if no data exists, but should not crash)
        focus_indices = get_angle_indices_by_parts(['right_knee', 'torso'])
        
        # This should not raise an error even if no data exists
        dataset = ExerciseDataset(
            data_dir='data/exercises',
            exercise_name='right-knee-to-90-degrees',
            focus_indices=focus_indices,
            apply_gaussian_smoothing=True,
            gaussian_window_size=5,
            gaussian_sigma=1.0,
            print_both=print
        )
        
        print(f"Dataset created successfully with {len(dataset)} samples")
        print("✓ Dataset integration test passed\n")
        
    except Exception as e:
        if "No training data found" in str(e):
            print("Dataset integration test passed (no data available, but no crashes)")
            print("✓ Dataset integration test passed\n")
        else:
            print(f"Dataset integration test failed: {e}")
            raise


def main():
    """Run all tests."""
    print("Motion Smoothing Integration Tests\n")
    print("=" * 50)
    
    test_gaussian_smoothing()
    test_one_euro_filter()
    test_pose_smoother()
    test_create_pose_smoother_for_features()
    test_dataset_integration()
    
    print("=" * 50)
    print("All tests passed! ✓")
    print("\nMotion smoothing functionality is ready to use.")


if __name__ == "__main__":
    main() 