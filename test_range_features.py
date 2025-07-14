from pose_utils import extract_sequence_from_video, get_angle_indices_by_parts
import os

# Test parameters
focus_parts = ['right_knee', 'torso']
use_keypoints = True
use_velocity = True
use_statistics = True
use_range = True

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
    
    # Test without range features
    print("\n=== Without range features ===")
    seq1 = extract_sequence_from_video(video_path, focus_indices=focus_indices, 
                                     use_keypoints=use_keypoints, use_velocity=use_velocity, 
                                     use_statistics=use_statistics, use_range=False)
    print(f"Shape: {seq1.shape}")
    print(f"Features per frame: {seq1.shape[1] if len(seq1) > 0 else 'N/A'}")
    
    # Test with range features
    print("\n=== With range features ===")
    seq2 = extract_sequence_from_video(video_path, focus_indices=focus_indices, 
                                     use_keypoints=use_keypoints, use_velocity=use_velocity, 
                                     use_statistics=use_statistics, use_range=True)
    print(f"Shape: {seq2.shape}")
    print(f"Features per frame: {seq2.shape[1] if len(seq2) > 0 else 'N/A'}")
    
    # Calculate expected sizes
    base_features = len(focus_indices) + (12 * 3 if use_keypoints else 0)  # 15 + 36 = 51
    if use_velocity:
        base_features *= 3  # 51 * 3 = 153
    if use_statistics:
        base_features += base_features * 5  # 153 + 765 = 918
    if use_range:
        range_features = base_features // (3 if use_velocity else 1)  # 918 // 3 = 306 (original features)
        base_features += range_features  # 918 + 306 = 1224
    
    print(f"\n=== Expected sizes ===")
    print(f"Base features (angles + keypoints): {len(focus_indices) + (12 * 3 if use_keypoints else 0)}")
    print(f"After velocity: {base_features // (3 if use_velocity else 1) * 3 if use_velocity else base_features // (3 if use_velocity else 1)}")
    print(f"After statistics: {base_features // (3 if use_velocity else 1) * 3 if use_velocity else base_features // (3 if use_velocity else 1) + (base_features // (3 if use_velocity else 1) * 5 if use_statistics else 0)}")
    print(f"After range: {base_features}")
    
    print(f"\n=== Actual vs Expected ===")
    print(f"Without range: {seq1.shape[1] if len(seq1) > 0 else 'N/A'} (expected: {918})")
    print(f"With range: {seq2.shape[1] if len(seq2) > 0 else 'N/A'} (expected: {1224})")
    
    # Show some range features
    if len(seq2) > 0:
        print(f"\n=== Sample range features (last 10) ===")
        range_start = seq1.shape[1] if len(seq1) > 0 else 0
        range_end = seq2.shape[1]
        print(f"Range features indices: {range_start} to {range_end}")
        print(f"Sample range values (first frame): {seq2[0, range_start:range_start+10]}")
else:
    print("No test video found!") 