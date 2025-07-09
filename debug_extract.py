from pose_utils import extract_sequence_from_video, get_angle_indices_by_parts
import os

focus_indices = get_angle_indices_by_parts(['right_knee', 'torso'])
print('focus_indices:', focus_indices)
print('len:', len(focus_indices))

# Find a video file
video_path = None
for root, dirs, files in os.walk('data/exercises/right-knee-to-90-degrees'):
    for file in files:
        if file.endswith('.mp4'):
            video_path = os.path.join(root, file)
            break
    if video_path:
        break

if video_path:
    print('Testing with video:', video_path)
    
    # Test step by step
    print('\n=== Step by step testing ===')
    
    # 1. Basic features (angles only)
    seq1 = extract_sequence_from_video(video_path, focus_indices=focus_indices, use_keypoints=False, use_velocity=False, use_statistics=False)
    print('1. Angles only:', seq1.shape if len(seq1) > 0 else 'empty')
    
    # 2. With keypoints
    seq2 = extract_sequence_from_video(video_path, focus_indices=focus_indices, use_keypoints=True, use_velocity=False, use_statistics=False)
    print('2. + Keypoints:', seq2.shape if len(seq2) > 0 else 'empty')
    
    # 3. With velocity
    seq3 = extract_sequence_from_video(video_path, focus_indices=focus_indices, use_keypoints=True, use_velocity=True, use_statistics=False)
    print('3. + Velocity:', seq3.shape if len(seq3) > 0 else 'empty')
    
    # 4. With statistics
    seq4 = extract_sequence_from_video(video_path, focus_indices=focus_indices, use_keypoints=True, use_velocity=True, use_statistics=True)
    print('4. + Statistics:', seq4.shape if len(seq4) > 0 else 'empty')
    
    # Expected calculation
    print('\n=== Expected calculation ===')
    base_features = len(focus_indices)  # 15
    print(f'Base features (angles): {base_features}')
    
    if True:  # use_keypoints
        keypoint_features = 12 * 3  # 36
        print(f'Keypoint features: {keypoint_features}')
        base_features += keypoint_features  # 51
        print(f'Total after keypoints: {base_features}')
    
    if True:  # use_velocity
        velocity_features = base_features * 3  # 153
        print(f'Velocity features (x3): {velocity_features}')
        base_features = velocity_features
    
    if True:  # use_statistics
        original_features = 51  # Original features before velocity
        stats_features = original_features * 5  # 255
        print(f'Statistics features: {stats_features}')
        total_features = velocity_features + stats_features  # 153 + 255 = 408
        print(f'Total expected: {total_features}')
    
    print(f'Actual result: {seq4.shape[1] if len(seq4) > 0 else "N/A"}')
    
else:
    print('No video found') 