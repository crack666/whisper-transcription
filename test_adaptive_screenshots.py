\
import logging
from pathlib import Path
from src.video_processor import VideoScreenshotExtractor
from src.utils import ensure_directory

# Setup basic logging to see output from the extractor
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

def test_screenshot_extraction():
    video_file_name = "studies/Aufzeichnung - 20.05.2025.mp4" # Changed video file
    video_path = Path(__file__).parent / video_file_name
    
    # Check if the video file exists
    if not video_path.exists():
        logging.error(f"Video file not found: {video_path}")
        print(f"Error: Video file {video_file_name} not found in the workspace root. Please ensure it's there.")
        return

    output_dir_name = f"test_adaptive_screenshots_{video_path.stem}"
    output_dir = Path(__file__).parent / "output" / output_dir_name
    ensure_directory(str(output_dir))

    # Define sample speech segments (start and end times in seconds)
    # These are hypothetical segments for testing purposes.
    speech_segments = [
        {"start": 1.0, "end": 8.5, "text": "This is the first speech segment."},
        {"start": 12.0, "end": 22.3, "text": "Another segment discussing something important."},
        {"start": 28.0, "end": 35.0, "text": "A shorter segment here."},
        {"start": 40.0, "end": 55.0, "text": "This is a much longer speech segment, we might expect more screenshots if there are visual changes."}
    ]

    logging.info(f"Attempting to process video: {video_path}")
    logging.info(f"Output directory for screenshots: {output_dir}")
    logging.info(f"Using speech segments: {speech_segments}")

    # Initialize the extractor with speech segments
    # Using default similarity_threshold (0.85) and min_time_between_shots (2.0)
    # You can adjust these:
    # extractor = VideoScreenshotExtractor(speech_segments=speech_segments, similarity_threshold=0.80, min_time_between_shots=1.5)
    extractor = VideoScreenshotExtractor(speech_segments=speech_segments)

    # Extract screenshots
    screenshots_metadata = extractor.extract_screenshots(str(video_path), str(output_dir))

    if screenshots_metadata:
        logging.info(f"Successfully extracted {len(screenshots_metadata)} screenshots.")
        print(f"\\n--- Extracted Screenshot Metadata ({len(screenshots_metadata)} total) ---")
        for i, meta in enumerate(screenshots_metadata):
            print(f"Screenshot {i+1}:")
            print(f"  Timestamp: {meta['timestamp']:.2f}s ({meta['timestamp_formatted']})")
            print(f"  Filename: {meta['filename']}")
            print(f"  Filepath: {meta['filepath']}")
            print(f"  Original Index/ID: {meta['index']}")
            print(f"  Similarity Score (if change-detected): {meta.get('similarity_score', 'N/A')}")
        print(f"\\nScreenshots saved to: {output_dir}")
    else:
        logging.warning("No screenshots were extracted.")
        print("\\nNo screenshots were extracted. Check logs for details.")

if __name__ == "__main__":
    test_screenshot_extraction()
