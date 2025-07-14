import cv2
import os
import argparse
from pathlib import Path

def split_video(input_path, output_dir, segment_duration=1.0, overlap=0.0):
    """
    Split a video into shorter segments
    
    Args:
        input_path: Path to input video
        output_dir: Directory to save segments
        segment_duration: Duration of each segment in seconds
        overlap: Overlap between segments in seconds
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
    
    # Calculate frames per segment
    frames_per_segment = int(segment_duration * fps)
    overlap_frames = int(overlap * fps)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    base_name = Path(input_path).stem
    
    segments_created = []
    frame_count = 0
    segment_count = 0
    
    while frame_count < total_frames:
        # Calculate segment boundaries
        start_frame = frame_count
        end_frame = min(start_frame + frames_per_segment, total_frames)
        
        # Skip if segment is too short
        if end_frame - start_frame < frames_per_segment * 0.5:  # At least 50% of desired length
            break
        
        # Create output filename
        output_filename = f"{base_name}_segment_{segment_count:03d}.mp4"
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
        print(f"Created segment {segment_count}: {output_filename} ({end_frame - start_frame} frames)")
        
        # Move to next segment (with overlap)
        frame_count = end_frame - overlap_frames
        segment_count += 1
    
    cap.release()
    return segments_created

def split_videos_in_directory(input_dir, output_base_dir, segment_duration=1.0, overlap=0.0):
    """
    Split all videos in a directory and its subdirectories
    """
    input_path = Path(input_dir)
    output_base_path = Path(output_base_dir)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(input_path.rglob(f'*{ext}'))
    
    print(f"Found {len(video_files)} video files")
    
    total_segments = 0
    
    for video_file in video_files:
        # Calculate relative path to maintain directory structure
        relative_path = video_file.relative_to(input_path)
        output_dir = output_base_path / relative_path.parent / f"{relative_path.stem}_segments"
        
        print(f"\nProcessing: {video_file}")
        segments = split_video(str(video_file), str(output_dir), segment_duration, overlap)
        total_segments += len(segments)
    
    print(f"\nTotal segments created: {total_segments}")
    return total_segments

def main():
    parser = argparse.ArgumentParser(description='Split videos into shorter segments')
    parser.add_argument('--input', required=True, help='Input directory or video file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--duration', type=float, default=1.0, help='Segment duration in seconds')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap between segments in seconds')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # Single file
        segments = split_video(args.input, args.output, args.duration, args.overlap)
        print(f"Created {len(segments)} segments")
    elif os.path.isdir(args.input):
        # Directory
        total_segments = split_videos_in_directory(args.input, args.output, args.duration, args.overlap)
        print(f"Created {total_segments} segments total")
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main() 