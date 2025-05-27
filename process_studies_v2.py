#!/usr/bin/env python3
"""
Convenience script for processing study videos using the modular v2.0 system
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Process all videos in the studies directory using v2.0 system."""
    
    # Set up paths
    script_dir = Path(__file__).parent
    studies_dir = script_dir / "studies"
    output_dir = script_dir / "output_v2"
    
    if not studies_dir.exists():
        print(f"❌ Error: Studies directory not found: {studies_dir}")
        sys.exit(1)
    
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(studies_dir.glob(f"*{ext}"))
    
    if not video_files:
        print("❌ No video files found in studies directory")
        sys.exit(1)
    
    print(f"📹 Found {len(video_files)} video files to process:")
    for video in video_files:
        print(f"   - {video.name}")
    
    print(f"\n📁 Output will be saved to: {output_dir}")
    print(f"📚 Using PDFs from: {studies_dir}")
    
    # Confirm processing
    response = input("\n🚀 Proceed with processing? (y/N): ")
    if response.lower() != 'y':
        print("⏹️  Processing cancelled.")
        sys.exit(0)
    
    # Run the modular v2.0 processor
    cmd = [
        sys.executable, "study_processor_v2.py",
        "--input", str(studies_dir),
        "--output", str(output_dir),
        "--studies", str(studies_dir),
        "--batch",
        "--cleanup-audio",
        "--verbose"
    ]
    
    print(f"\n🔧 Running command:")
    print(f"   {' '.join(cmd)}")
    print(f"\n⏱️  This may take several minutes depending on video length and number of files...")
    
    try:
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        print("\n✅ Processing completed successfully!")
        print(f"📄 Results saved to: {output_dir}")
        
        # Check for index file
        index_file = output_dir / "index.html"
        if index_file.exists():
            print(f"🌐 Open the index file to view all results: {index_file}")
        
        # Show individual video reports
        print(f"\n📊 Individual video reports:")
        for video in video_files:
            video_name = video.stem
            report_path = output_dir / video_name / f"{video_name}_report.html"
            if report_path.exists():
                print(f"   📄 {video_name}: {report_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Processing failed with error code {e.returncode}")
        print("💡 Try running with --debug flag for more information:")
        debug_cmd = cmd + ["--debug"]
        print(f"   {' '.join(debug_cmd)}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()