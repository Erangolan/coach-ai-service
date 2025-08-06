#!/usr/bin/env python3
"""
Example script showing how to use normalization in training
"""

import subprocess
import sys

def run_training_with_normalization():
    """Run training with different normalization methods"""
    
    print("=== Training with Z-score Normalization ===")
    print("This will normalize features per video using z-score normalization")
    print("Command: python train_lstm.py --exercise right-knee-to-90-degrees --focus right_knee torso --normalize zscore")
    print()
    
    # Example command with z-score normalization
    cmd_zscore = [
        "python", "train_lstm.py",
        "--exercise", "right-knee-to-90-degrees",
        "--focus", "right_knee", "torso",
        "--normalize", "zscore",
        "--use_keypoints",
        "--use_velocity",
        "--use_statistics",
        "--use_ratios",
        "--epochs", "10",  # Reduced for example
        "--batch_size", "4"
    ]
    
    print("=== Training with Min-Max Normalization ===")
    print("This will normalize features per video using min-max normalization")
    print("Command: python train_lstm.py --exercise right-knee-to-90-degrees --focus right_knee torso --normalize minmax")
    print()
    
    # Example command with min-max normalization
    cmd_minmax = [
        "python", "train_lstm.py",
        "--exercise", "right-knee-to-90-degrees",
        "--focus", "right_knee", "torso",
        "--normalize", "minmax",
        "--use_keypoints",
        "--use_velocity",
        "--use_statistics",
        "--use_ratios",
        "--epochs", "10",  # Reduced for example
        "--batch_size", "4"
    ]
    
    print("=== Training with Default Normalization ===")
    print("This will train with default z-score normalization (per video)")
    print("Command: python train_lstm.py --exercise right-knee-to-90-degrees --focus right_knee torso")
    print()
    
    # Example command with default normalization
    cmd_default_norm = [
        "python", "train_lstm.py",
        "--exercise", "right-knee-to-90-degrees",
        "--focus", "right_knee", "torso",
        "--use_keypoints",
        "--use_velocity",
        "--use_statistics",
        "--use_ratios",
        "--epochs", "10",  # Reduced for example
        "--batch_size", "4"
    ]
    
    print("=== Available Normalization Options ===")
    print("1. --normalize zscore  : Z-score normalization (mean=0, std=1) [DEFAULT]")
    print("2. --normalize minmax  : Min-max normalization (range [0,1])")
    print("3. No --normalize flag : Uses default z-score normalization")
    print()
    
    print("=== Key Benefits of Normalization ===")
    print("• Z-score normalization: Centers data around 0, scales to unit variance")
    print("• Min-max normalization: Scales data to [0,1] range")
    print("• Applied per video: Each video is normalized independently")
    print("• Preserves temporal relationships within each video")
    print("• Helps with training stability and convergence")
    print("• Reduces the impact of different video scales and ranges")
    print()
    
    print("=== Usage Examples ===")
    print("Basic training with z-score normalization:")
    print("  python train_lstm.py --exercise right-knee-to-90-degrees --normalize zscore")
    print()
    print("Advanced training with all features and min-max normalization:")
    print("  python train_lstm.py --exercise right-knee-to-90-degrees \\")
    print("    --focus right_knee torso \\")
    print("    --normalize minmax \\")
    print("    --use_keypoints --use_velocity --use_statistics --use_ratios \\")
    print("    --epochs 50 --batch_size 8")
    print()
    
    # Ask user if they want to run a training example
    try:
        choice = input("Would you like to run a training example? (y/n): ").lower().strip()
        if choice == 'y':
            norm_choice = input("Choose normalization method (zscore/minmax/none): ").lower().strip()
            
            if norm_choice == 'zscore':
                cmd = cmd_zscore
                print("Running training with z-score normalization...")
            elif norm_choice == 'minmax':
                cmd = cmd_minmax
                print("Running training with min-max normalization...")
            elif norm_choice == 'none':
                cmd = cmd_default_norm
                print("Running training with default normalization...")
            else:
                print("Invalid choice. Running with default normalization...")
                cmd = cmd_default_norm
            
            print(f"Command: {' '.join(cmd)}")
            print("Starting training...")
            
            # Run the training command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("Training completed successfully!")
                print("Output:")
                print(result.stdout)
            else:
                print("Training failed!")
                print("Error:")
                print(result.stderr)
        else:
            print("No training executed. You can run the commands manually.")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_training_with_normalization() 