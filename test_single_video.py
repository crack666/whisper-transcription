#!/usr/bin/env python3
"""
Test script to process a single video from the studies directory
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Test processing of a single video."""
    
    script_dir = Path(__file__).parent
    studies_dir = script_dir / "studies"
    output_dir = script_dir / "test_output"
    
    # Find first video file
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_file = None
    
    for ext in video_extensions:
        videos = list(studies_dir.glob(f"*{ext}"))
        if videos:
            video_file = videos[0]
            break
    
    if not video_file:
        print("No video files found in studies directory")
        sys.exit(1)
    
    print(f"Testing with video: {video_file.name}")
    
    # Create test output directory
    output_dir.mkdir(exist_ok=True)
    
    # Run with minimal settings for testing
    cmd = [
        sys.executable, "study_material_processor.py",
        "--input", str(video_file),
        "--output_dir", str(output_dir),
        "--studies_dir", str(studies_dir),
        "--extract_screenshots",
        "--model", "base",  # Faster model for testing
        "--similarity_threshold", "0.80",
        "--min_time_between_shots", "5.0",
        "--verbose"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("Note: Using 'base' model for faster testing. Use 'large-v3' for production.")
    
    try:
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        print(f"\n✅ Test completed successfully!")
        
        # Show output structure
        video_name = video_file.stem
        video_output_dir = output_dir / video_name
        
        if video_output_dir.exists():
            print(f"\nGenerated files:")
            for file in video_output_dir.rglob("*"):
                if file.is_file():
                    relative_path = file.relative_to(output_dir)
                    print(f"  📄 {relative_path}")
            
            html_report = video_output_dir / f"{video_name}_report.html"
            if html_report.exists():
                print(f"\n🌐 Open the HTML report: {html_report}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()