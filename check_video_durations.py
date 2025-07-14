import cv2
import os
from collections import defaultdict

def get_video_duration(video_path):
    """Get video duration in seconds"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    if fps > 0:
        duration = frame_count / fps
        return duration
    return 0

def analyze_video_durations(base_dir, exercise_name):
    """Analyze video durations for each category"""
    categories = ['good', 'bad-knee-angle', 'bad-lower-knee']
    results = {}
    
    for category in categories:
        category_dir = os.path.join(base_dir, exercise_name, category)
        if not os.path.exists(category_dir):
            print(f"Directory not found: {category_dir}")
            continue
        
        durations = []
        video_files = [f for f in os.listdir(category_dir) if f.endswith('.mp4')]
        
        print(f"\n=== {category.upper()} ===")
        print(f"Found {len(video_files)} video files")
        
        for video_file in video_files:
            video_path = os.path.join(category_dir, video_file)
            duration = get_video_duration(video_path)
            if duration > 0:
                durations.append(duration)
                print(f"  {video_file}: {duration:.2f}s")
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            
            results[category] = {
                'count': len(durations),
                'avg': avg_duration,
                'min': min_duration,
                'max': max_duration,
                'total': sum(durations)
            }
            
            print(f"\nSummary for {category}:")
            print(f"  Count: {len(durations)}")
            print(f"  Average: {avg_duration:.2f}s")
            print(f"  Min: {min_duration:.2f}s")
            print(f"  Max: {max_duration:.2f}s")
            print(f"  Total: {sum(durations):.2f}s")
        else:
            print(f"No valid videos found in {category}")
    
    return results

if __name__ == "__main__":
    base_dir = "data/exercises"
    exercise_name = "right-knee-to-90-degrees"
    
    print(f"Analyzing video durations for exercise: {exercise_name}")
    results = analyze_video_durations(base_dir, exercise_name)
    
    # Overall summary
    print("\n" + "="*50)
    print("OVERALL SUMMARY")
    print("="*50)
    
    for category, data in results.items():
        print(f"{category}: {data['avg']:.2f}s average ({data['count']} videos)")
    
    # Calculate overall average
    all_durations = []
    for data in results.values():
        all_durations.extend([data['avg']] * data['count'])
    
    if all_durations:
        overall_avg = sum(all_durations) / len(all_durations)
        print(f"\nOverall average: {overall_avg:.2f}s") 