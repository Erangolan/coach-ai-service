import os
import shutil
import argparse
from pathlib import Path

def setup_training_directories(base_dir):
    """Create the required directory structure for training data."""
    # Create main directories
    for label in ['squat_good', 'squat_bad']:
        os.makedirs(os.path.join(base_dir, label), exist_ok=True)
    
    print(f"Created directory structure in {base_dir}")
    print("Directory structure:")
    print(f"{base_dir}/")
    print("├── squat_good/")
    print("└── squat_bad/")

def organize_videos(source_dir, target_dir):
    """Organize videos from source directory into the training structure."""
    # Create the directory structure
    setup_training_directories(target_dir)
    
    # Get all video files from source directory
    video_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
    
    if not video_files:
        print(f"No video files found in {source_dir}")
        return
    
    print("\nFound the following video files:")
    for i, video in enumerate(video_files, 1):
        print(f"{i}. {video}")
    
    print("\nFor each video, please specify if it's a good (g) or bad (b) squat example.")
    print("Enter 'q' to quit organizing.")
    
    for video in video_files:
        while True:
            choice = input(f"\nIs '{video}' a good (g) or bad (b) squat example? (g/b/q): ").lower()
            
            if choice == 'q':
                return
            
            if choice in ['g', 'b']:
                # Determine target directory
                target_subdir = 'squat_good' if choice == 'g' else 'squat_bad'
                
                # Copy the video to the appropriate directory
                source_path = os.path.join(source_dir, video)
                target_path = os.path.join(target_dir, target_subdir, video)
                
                shutil.copy2(source_path, target_path)
                print(f"Copied {video} to {target_subdir}/")
                break
            
            print("Invalid choice. Please enter 'g' for good, 'b' for bad, or 'q' to quit.")

def main():
    parser = argparse.ArgumentParser(description='Organize training videos for exercise classification')
    parser.add_argument('--source_dir', type=str, required=True, help='Directory containing your video files')
    parser.add_argument('--target_dir', type=str, required=True, help='Directory where organized videos will be stored')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    source_dir = os.path.abspath(args.source_dir)
    target_dir = os.path.abspath(args.target_dir)
    
    # Verify source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    organize_videos(source_dir, target_dir)
    
    print("\nOrganization complete!")
    print(f"\nTo train the model, run:")
    print(f"python train_model.py --data_dir {target_dir}")

if __name__ == "__main__":
    main() 