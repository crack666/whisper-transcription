#!/usr/bin/env python3
"""
Convenience script to process all study videos in the studies directory
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Process all videos in the studies directory."""
    
    # Set up paths
    script_dir = Path(__file__).parent
    studies_dir = script_dir / "studies"
    output_dir = script_dir / "output"
    
    if not studies_dir.exists():
        print(f"Error: Studies directory not found: {studies_dir}")
        sys.exit(1)
    
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(studies_dir.glob(f"*{ext}"))
    
    if not video_files:
        print("No video files found in studies directory")
        sys.exit(1)
    
    print(f"Found {len(video_files)} video files to process:")
    for video in video_files:
        print(f"  - {video.name}")
    
    # Confirm processing
    response = input("\nProceed with processing? (y/N): ")
    if response.lower() != 'y':
        print("Processing cancelled.")
        sys.exit(0)
    
    # Run the study material processor
    cmd = [
        sys.executable, "study_material_processor.py",
        "--input", str(studies_dir),
        "--output_dir", str(output_dir),
        "--studies_dir", str(studies_dir),
        "--batch_process",
        "--extract_screenshots",
        "--cleanup_audio",
        "--verbose"
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        print("\n✅ Processing completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        # Open index file if it exists
        index_file = output_dir / "index.html"
        if index_file.exists():
            print(f"📄 Open the index file to view results: {index_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Processing failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()