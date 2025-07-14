import cv2
import os
import random
from pathlib import Path
import glob

def get_video_duration(video_path):
    """Get video duration in seconds"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps > 0:
        return frame_count / fps
    return 0

def split_video_random_segments(input_path, output_dir, min_duration=3.5, max_duration=4.7, num_segments=3):
    """
    Split a video into random segments of specified duration range
    
    Args:
        input_path: Path to input video
        output_dir: Directory to save segments
        min_duration: Minimum segment duration in seconds
        max_duration: Maximum segment duration in seconds
        num_segments: Number of segments to create
    """
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_duration = total_frames / fps
    
    print(f"Video duration: {total_duration:.2f}s, FPS: {fps}, Total frames: {total_frames}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    base_name = Path(input_path).stem
    
    segments_created = []
    
    for segment_idx in range(num_segments):
        # Generate random segment duration
        segment_duration = random.uniform(min_duration, max_duration)
        frames_per_segment = int(segment_duration * fps)
        
        # Calculate maximum possible start frame
        max_start_frame = total_frames - frames_per_segment
        
        if max_start_frame <= 0:
            print(f"Warning: Video too short for {segment_duration:.2f}s segment")
            continue
        
        # Generate random start frame
        start_frame = random.randint(0, max_start_frame)
        end_frame = start_frame + frames_per_segment
        
        # Create output filename
        output_filename = f"{base_name}_segment_{segment_idx:03d}_{segment_duration:.1f}s.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames for this segment
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(start_frame, end_frame):
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break
        
        out.release()
        segments_created.append(output_path)
        print(f"Created segment {segment_idx}: {output_filename} ({segment_duration:.2f}s, frames {start_frame}-{end_frame})")
    
    cap.release()
    return segments_created

def main():
    # Input and output directories
    input_dir = "data/exercises/right-knee-to-90-degrees/idle"
    output_dir = "data/exercises/right-knee-to-90-degrees/idle_segments_last16"
    
    # Get all video files and sort them
    video_files = []
    for ext in ['*.mp4', '*.mov']:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    video_files.sort()  # Sort to get consistent last 16
    print(f"Found {len(video_files)} video files")
    
    # Take last 16 videos
    last_16_videos = video_files[-16:]
    print(f"Processing last 16 videos:")
    for i, video in enumerate(last_16_videos):
        duration = get_video_duration(video)
        print(f"{i+1:2d}. {Path(video).name} ({duration:.2f}s)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    total_segments = 0
    
    for i, video_file in enumerate(last_16_videos):
        print(f"\n{'='*50}")
        print(f"Processing video {i+1}/16: {Path(video_file).name}")
        print(f"{'='*50}")
        
        # Create subdirectory for this video's segments
        video_name = Path(video_file).stem
        video_output_dir = os.path.join(output_dir, video_name)
        
        # Split video into 3 random segments of 3.5-4.7 seconds
        segments = split_video_random_segments(
            video_file, 
            video_output_dir, 
            min_duration=3.5, 
            max_duration=4.7, 
            num_segments=3
        )
        
        total_segments += len(segments)
        print(f"Created {len(segments)} segments for this video")
    
    print(f"\n{'='*50}")
    print(f"COMPLETED!")
    print(f"Total segments created: {total_segments}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 