#!/usr/bin/env python3
"""
Real-time Pose Smoothing Example

This script demonstrates how to use the One Euro Filter for real-time pose smoothing
during live inference. It shows how to:

1. Initialize a PoseSmoother for your feature configuration
2. Process incoming pose frames in real-time
3. Apply smoothing with minimal latency

Usage:
    python real_time_smoothing_example.py

This is for demonstration purposes. In a real application, you would integrate
this into your websocket handler or video processing pipeline.
"""

import numpy as np
import time
from pose_utils import create_pose_smoother_for_features, PoseSmoother


def simulate_pose_data(num_frames=100, num_features=40, noise_level=2.0):
    """
    Simulate noisy pose data for demonstration.
    
    Args:
        num_frames: number of frames to simulate
        num_features: number of pose features
        noise_level: amount of noise to add
    
    Returns:
        List of noisy pose frames
    """
    # Create a smooth underlying motion pattern
    t = np.linspace(0, 4*np.pi, num_frames)
    
    # Generate smooth sine waves for different features
    smooth_data = []
    for i in range(num_features):
        # Different frequencies for different features
        freq = 0.5 + (i % 3) * 0.3
        phase = (i % 5) * 0.5
        smooth_feature = 50 + 30 * np.sin(freq * t + phase)
        smooth_data.append(smooth_feature)
    
    smooth_data = np.array(smooth_data).T  # Shape: (frames, features)
    
    # Add noise to simulate MediaPipe jitter
    noisy_data = []
    for frame in smooth_data:
        noise = np.random.normal(0, noise_level, len(frame))
        noisy_frame = frame + noise
        noisy_data.append(noisy_frame)
    
    return noisy_data


def demonstrate_real_time_smoothing():
    """
    Demonstrate real-time pose smoothing using One Euro Filter.
    """
    print("=== Real-time Pose Smoothing Demonstration ===\n")
    
    # Configuration for the example
    focus_indices = list(range(15))  # First 15 angles
    use_keypoints = False
    use_velocity = False
    use_statistics = False
    use_ratios = False
    
    # One Euro Filter parameters
    min_cutoff = 1.0      # Minimum cutoff frequency (Hz)
    beta = 0.007          # Speed coefficient
    d_cutoff = 1.0        # Derivative cutoff frequency (Hz)
    
    print(f"Feature configuration:")
    print(f"  - Focus indices: {focus_indices}")
    print(f"  - Use keypoints: {use_keypoints}")
    print(f"  - Use velocity: {use_velocity}")
    print(f"  - Use statistics: {use_statistics}")
    print(f"  - Use ratios: {use_ratios}")
    print(f"  - Total features: {len(focus_indices)}")
    print()
    
    print(f"One Euro Filter parameters:")
    print(f"  - min_cutoff: {min_cutoff} Hz")
    print(f"  - beta: {beta}")
    print(f"  - d_cutoff: {d_cutoff} Hz")
    print()
    
    # Create pose smoother
    smoother = create_pose_smoother_for_features(
        focus_indices=focus_indices,
        use_keypoints=use_keypoints,
        use_velocity=use_velocity,
        use_statistics=use_statistics,
        use_ratios=use_ratios,
        min_cutoff=min_cutoff,
        beta=beta,
        d_cutoff=d_cutoff
    )
    
    print(f"Created PoseSmoother with {len(smoother.filters)} filters")
    print()
    
    # Simulate noisy pose data
    print("Simulating noisy pose data...")
    noisy_frames = simulate_pose_data(num_frames=100, num_features=len(focus_indices), noise_level=3.0)
    print(f"Generated {len(noisy_frames)} noisy frames")
    print()
    
    # Process frames in real-time simulation
    print("Processing frames with real-time smoothing...")
    smoothed_frames = []
    processing_times = []
    
    for i, noisy_frame in enumerate(noisy_frames):
        start_time = time.time()
        
        # Apply real-time smoothing (this is what you'd do in your websocket handler)
        smoothed_frame = smoother.update(noisy_frame)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        processing_times.append(processing_time)
        smoothed_frames.append(smoothed_frame)
        
        # Print progress every 20 frames
        if (i + 1) % 20 == 0:
            avg_time = np.mean(processing_times[-20:])
            print(f"  Processed frame {i+1}/{len(noisy_frames)} (avg processing time: {avg_time:.2f} ms)")
    
    print(f"Completed processing {len(smoothed_frames)} frames")
    print(f"Average processing time: {np.mean(processing_times):.2f} ms")
    print(f"Max processing time: {np.max(processing_times):.2f} ms")
    print()
    
    # Calculate smoothing statistics
    noisy_frames = np.array(noisy_frames)
    smoothed_frames = np.array(smoothed_frames)
    
    # Calculate noise reduction
    noise_reduction = np.std(noisy_frames - smoothed_frames, axis=0)
    avg_noise_reduction = np.mean(noise_reduction)
    
    print(f"Smoothing statistics:")
    print(f"  - Average noise reduction: {avg_noise_reduction:.2f} units")
    print(f"  - Noise reduction range: {np.min(noise_reduction):.2f} - {np.max(noise_reduction):.2f} units")
    print()
    
    # Demonstrate reset functionality
    print("Demonstrating filter reset...")
    smoother.reset()
    print("  - All filters have been reset to initial state")
    print("  - Ready to process a new sequence")
    print()
    
    print("=== Demonstration Complete ===")
    print()
    print("Integration notes:")
    print("1. Initialize PoseSmoother once at the start of your session")
    print("2. Call smoother.update(pose_frame) for each incoming frame")
    print("3. Call smoother.reset() when starting a new exercise sequence")
    print("4. Adjust min_cutoff, beta, and d_cutoff parameters as needed")
    print("5. Processing time is typically < 1ms per frame")


def demonstrate_websocket_integration():
    """
    Show how to integrate the pose smoother into a websocket handler.
    This is a pseudo-code example.
    """
    print("=== WebSocket Integration Example ===\n")
    
    print("Here's how you would integrate the pose smoother into a websocket handler:")
    print()
    
    integration_code = '''
# In your websocket handler or main processing loop:

# 1. Initialize the smoother once (at the start of the session)
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

# 2. For each incoming pose frame:
def process_pose_frame(pose_data):
    # Apply real-time smoothing
    smoothed_pose = smoother.update(pose_data)
    
    # Now use smoothed_pose for your model inference
    prediction = model.predict(smoothed_pose)
    
    return prediction

# 3. Reset when starting a new exercise:
def start_new_exercise():
    smoother.reset()
    # Continue processing...

# 4. Example websocket message handler:
def handle_websocket_message(message):
    pose_data = extract_pose_from_message(message)
    smoothed_pose = smoother.update(pose_data)
    
    # Process with your model
    result = model.inference(smoothed_pose)
    
    # Send result back
    send_response(result)
'''
    
    print(integration_code)


if __name__ == "__main__":
    demonstrate_real_time_smoothing()
    print("\n" + "="*60 + "\n")
    demonstrate_websocket_integration() 