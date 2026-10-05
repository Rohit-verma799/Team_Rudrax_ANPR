"""
Batch process all DAV videos in 'CameraVideos' using PyTorch .pt models
inside the 'anpr' environment, and save outputs to 'CameraOuput' as MP4.
"""

import os
import sys
import subprocess
import glob

def main():
    # Make sure output folder 'CameraOuput' exists
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CameraOuput")
    os.makedirs(result_dir, exist_ok=True)
    
    # Input folder 'CameraVideos'
    videos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CameraVideos")
    if not os.path.exists(videos_dir):
        print(f"Input directory not found: {videos_dir}")
        return
    
    # Supported video formats (specifically .dav / .DAV for real camera feeds)
    video_extensions = ["*.dav", "*.DAV"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(videos_dir, ext)))
        
    # Deduplicate files (case-insensitive globbing on Windows)
    video_files = list(set(video_files))
    
    # Exclude already processed videos
    exclude_files = [
        "13.20.10-13.30.13[R][0@0][0].dav",
        "13.29.35-13.39.35[R][0@0][0].dav"
    ]
    video_files = [f for f in video_files if os.path.basename(f) not in exclude_files]
    
    # Sort files alphabetically
    video_files.sort()
        
    if not video_files:
        print(f"No DAV video files found in {videos_dir}")
        return
        
    print(f"Found {len(video_files)} DAV video files in 'CameraVideos' to process.")
    print(f"Outputs will be saved in: {result_dir}")
    
    # Set KMP_DUPLICATE_LIB_OK to avoid OpenMP clashes
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Python executable inside 'anpr' environment
    python_exe = "C:\\Users\\Rohit\\miniconda3\\envs\\anpr\\python.exe"
    main_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
    
    for idx, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        # Convert output extension to .mp4 for standard player compatibility
        output_name = os.path.splitext(video_name)[0] + ".mp4"
        output_path = os.path.join(result_dir, output_name)
        
        print(f"\n" + "=" * 60)
        print(f"[{idx}/{len(video_files)}] Processing: {video_name}")
        print(f"Output destination: {output_path}")
        print("=" * 60)
        
        cmd = [
            python_exe, main_py,
            "--source", video_path,
            "--output", output_path,
            "--confidence", "0.5",
            "--plate-confidence", "0.4",
            "--zone-threshold", "0.15",
            "--show", "false",
            "--gpu", "true",
            "--model", "D:/FinalModels/kaggle.pt",
            "--plate-model", "D:/FinalModels/ANPR_best.pt"
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
