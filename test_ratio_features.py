from pose_utils import extract_sequence_from_video, get_angle_indices_by_parts, add_ratio_features
import os
import numpy as np

# Test parameters
focus_parts = ['right_knee', 'torso']
use_keypoints = True
use_velocity = True
use_statistics = True
use_ratios = True

focus_indices = get_angle_indices_by_parts(focus_parts)
print(f"focus_indices: {focus_indices}")
print(f"len: {len(focus_indices)}")

# Find a test video
video_path = None
for root, dirs, files in os.walk('data/exercises/right-knee-to-90-degrees'):
    for file in files:
        if file.endswith('.mp4'):
            video_path = os.path.join(root, file)
            break
    if video_path:
        break

if video_path:
    print(f"Testing with video: {video_path}")
    
    # Test without ratio features
    print("\n=== Without ratio features ===")
    seq1 = extract_sequence_from_video(video_path, focus_indices=focus_indices, 
                                     use_keypoints=use_keypoints, use_velocity=use_velocity, 
                                     use_statistics=use_statistics, use_ratios=False)
    print(f"Shape: {seq1.shape}")
    print(f"Features per frame: {seq1.shape[1] if len(seq1) > 0 else 'N/A'}")
    
    # Test with ratio features
    print("\n=== With ratio features ===")
    seq2 = extract_sequence_from_video(video_path, focus_indices=focus_indices, 
                                     use_keypoints=use_keypoints, use_velocity=use_velocity, 
                                     use_statistics=use_statistics, use_ratios=True)
    print(f"Shape: {seq2.shape}")
    print(f"Features per frame: {seq2.shape[1] if len(seq2) > 0 else 'N/A'}")
    
    # Calculate expected sizes
    base_features = len(focus_indices) + (12 * 3 if use_keypoints else 0)  # 15 + 36 = 51
    if use_velocity:
        base_features *= 3  # 51 * 3 = 153
    if use_statistics:
        base_features += base_features * 6  # 153 + 918 = 1071
    
    # Calculate ratio features
    total_ratio_features = 1  # Only 1 ratio: primary knee / primary torso
    
    if use_ratios:
        base_features += total_ratio_features  # 1071 + 1 = 1072
    
    print(f"\n=== Expected sizes ===")
    print(f"Base features (angles + keypoints): {len(focus_indices) + (12 * 3 if use_keypoints else 0)}")
    print(f"After velocity: {base_features // (3 if use_velocity else 1) * 3 if use_velocity else base_features // (3 if use_velocity else 1)}")
    print(f"After statistics: {base_features // (3 if use_velocity else 1) * 3 if use_velocity else base_features // (3 if use_velocity else 1) + (base_features // (3 if use_velocity else 1) * 6 if use_statistics else 0)}")
    print(f"After ratios: {base_features}")
    print(f"Ratio features: {total_ratio_features} (primary knee / primary torso)")
    
    print(f"\n=== Actual vs Expected ===")
    print(f"Without ratios: {seq1.shape[1] if len(seq1) > 0 else 'N/A'} (expected: {1071})")
    print(f"With ratios: {seq2.shape[1] if len(seq2) > 0 else 'N/A'} (expected: {1072})")
    
    # Show some ratio features
    if len(seq2) > 0:
        print(f"\n=== Sample ratio features (last 10) ===")
        ratio_start = seq1.shape[1] if len(seq1) > 0 else 0
        ratio_end = seq2.shape[1]
        print(f"Ratio features indices: {ratio_start} to {ratio_end}")
        print(f"Sample ratio values (first frame): {seq2[0, ratio_start:ratio_start+10]}")
        
        # Test the add_ratio_features function directly
        print(f"\n=== Testing add_ratio_features directly ===")
        if len(seq1) > 0:
            # Get the base sequence (without statistics)
            base_seq = extract_sequence_from_video(video_path, focus_indices=focus_indices, 
                                                 use_keypoints=use_keypoints, use_velocity=use_velocity, 
                                                 use_statistics=False, use_ratios=False)
            ratio_seq = add_ratio_features(base_seq, focus_indices)
            print(f"Base sequence shape: {base_seq.shape}")
            print(f"After adding ratios: {ratio_seq.shape}")
            print(f"Added {ratio_seq.shape[1] - base_seq.shape[1]} ratio features")
else:
    print("No test video found!") 