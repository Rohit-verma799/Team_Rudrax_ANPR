"""
Batch process all videos in the 'newvideos' directory, using a higher plate detection zone,
and save results to 'result2' directory.
"""

import os
import sys
import subprocess
import glob

def main():
    # Make sure output folder 'result2' exists
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result2")
    os.makedirs(result_dir, exist_ok=True)
    
    # Input folder 'newvideos'
    videos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "newvideos")
    if not os.path.exists(videos_dir):
        print(f"Input directory not found: {videos_dir}")
        return
    
    # Supported video formats
    video_extensions = ["*.mp4", "*.webm", "*.avi", "*.mkv", "*.mov"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(videos_dir, ext)))
        
    # Sort files naturally/alphabetically
    video_files.sort()
        
    if not video_files:
        print(f"No video files found in {videos_dir}")
        return
        
    print(f"Found {len(video_files)} video files in 'newvideos' to process.")
    print(f"Outputs will be saved in: {result_dir}")
    
    python_exe = sys.executable
    main_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
    
    for idx, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        name_without_ext, _ = os.path.splitext(video_name)
        output_path = os.path.join(result_dir, f"{name_without_ext}.mp4")
        
        print(f"\n" + "=" * 60)
        print(f"[{idx}/{len(video_files)}] Processing: {video_name}")
        print(f"Output destination: {output_path}")
        print("=" * 60)
        
        # Build command with optimized accuracy, higher plate zone (0.3), and GPU activation
        cmd = [
            python_exe, main_py,
            "--source", video_path,
            "--output", output_path,
            "--confidence", "0.5",
            "--plate-confidence", "0.5",
            "--show", "true",
            "--gpu", "true",
            "--zone-threshold", "0.3"  # Set plate detection zone line higher up (0.3 of frame height)
        ]
        
        try:
            # Run the command and stream output in real-time
            subprocess.run(cmd, check=True)
            print(f"\n[SUCCESS] Finished processing: {video_name}")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Failed to process {video_name} (exit code: {e.returncode})")
        except KeyboardInterrupt:
            print("\nBatch processing interrupted by user.")
            sys.exit(0)

if __name__ == "__main__":
    main()
