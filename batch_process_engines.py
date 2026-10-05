"""
Batch process all videos in 'CeeriVideos' using the TensorRT engines,
and save results to 'FinalTesting' directory with the same names.
"""

import os
import sys
import subprocess
import glob

def main():
    # Make sure output folder 'FinalTesting' exists
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FinalTesting")
    os.makedirs(result_dir, exist_ok=True)
    
    # Input folder 'CeeriVideos'
    videos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CeeriVideos")
    if not os.path.exists(videos_dir):
        print(f"Input directory not found: {videos_dir}")
        return
    
    # Supported video formats
    video_extensions = ["*.mp4", "*.webm", "*.avi", "*.mkv", "*.mov"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(videos_dir, ext)))
        
    # Sort files alphabetically
    video_files.sort()
        
    if not video_files:
        print(f"No video files found in {videos_dir}")
        return
        
    print(f"Found {len(video_files)} video files in 'CeeriVideos' to process.")
    print(f"Outputs will be saved in: {result_dir}")
    
    # Set KMP_DUPLICATE_LIB_OK to avoid OpenMP clashes between PyTorch and PaddlePaddle
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Use the active environment python runner
    python_exe = sys.executable
    main_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
    
    for idx, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        output_path = os.path.join(result_dir, video_name)
        
        print(f"\n" + "=" * 60)
        print(f"[{idx}/{len(video_files)}] Processing: {video_name}")
        print(f"Output destination: {output_path}")
        print("=" * 60)
        
        cmd = [
            python_exe, "-s", main_py,
            "--source", video_path,
            "--output", output_path,
            "--confidence", "0.5",
            "--plate-confidence", "0.5",
            "--show", "false",
            "--gpu", "true",
            "--model", "D:/FinalModels/Detection.engine",
            "--plate-model", "D:/FinalModels/ANPR_best.engine"
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
