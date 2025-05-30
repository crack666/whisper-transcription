\
import argparse
import json
import os
from pathlib import Path
import logging
import sys
from typing import Optional

# Add src to Python path
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from video_processor import VideoScreenshotExtractor
from html_generator import HTMLReportGenerator
from config import DEFAULT_CONFIG
from utils import sanitize_filename, ensure_directory

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def regenerate_screenshots_and_report(analysis_json_path: str, similarity_threshold: Optional[float] = None, min_time_between_shots: Optional[float] = None):
    analysis_json_path_obj = Path(analysis_json_path)
    if not analysis_json_path_obj.exists():
        logger.error(f"Analysis JSON file not found: {analysis_json_path}")
        return

    try:
        with open(analysis_json_path_obj, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {analysis_json_path}: {e}")
        return

    video_path_str = analysis_data.get("video_path")
    if not video_path_str:
        # Fallback for older JSONs or audio-focused processing that might use audio_file_path
        video_path_str = analysis_data.get("audio_file_path") 

    if not video_path_str:
        logger.error(f"No 'video_path' or 'audio_file_path' found in JSON: {analysis_json_path}")
        return
    
    video_path = Path(video_path_str)
    if not video_path.exists():
        logger.error(f"Video file specified in JSON does not exist: {video_path_str}")
        return

    transcription_data = analysis_data.get("transcription", {})
    # Handle nested transcription structure
    if "transcription" in transcription_data:
        inner_transcription = transcription_data["transcription"]
        speech_segments = inner_transcription.get("segments", [])
    else:
        speech_segments = transcription_data.get("segments", [])

    if not speech_segments:
        logger.warning(f"No speech segments found in {analysis_json_path}. Screenshot generation might be limited if relying on segment starts.")

    video_name_from_json = sanitize_filename(video_path.stem)
    
    if "output_directory" not in analysis_data:
        logger.error(f"'output_directory' not found in {analysis_json_path}. Cannot determine where to save screenshots.")
        return
        
    video_specific_output_dir = Path(analysis_data["output_directory"])
    screenshots_output_dir = ensure_directory(video_specific_output_dir / "screenshots")
    
    logger.info(f"Regenerating screenshots for video: {video_path_str}")
    logger.info(f"Using {len(speech_segments)} speech segments.")
    logger.info(f"Saving new screenshots to: {screenshots_output_dir}")

    # Get screenshot config from the original run, or use defaults
    original_config_used = analysis_data.get("config_used", {})
    current_screenshot_config = original_config_used.get("screenshots", DEFAULT_CONFIG['screenshots'].copy())
    
    # Override with command-line parameters if provided
    if similarity_threshold is not None:
        current_screenshot_config['similarity_threshold'] = similarity_threshold
        logger.info(f"Overriding similarity_threshold to: {similarity_threshold}")
    if min_time_between_shots is not None:
        current_screenshot_config['min_time_between_shots'] = min_time_between_shots
        logger.info(f"Overriding min_time_between_shots to: {min_time_between_shots}")

    # Ensure speech_segments are passed to the extractor
    screenshot_extractor = VideoScreenshotExtractor(
        similarity_threshold=current_screenshot_config['similarity_threshold'],
        min_time_between_shots=current_screenshot_config['min_time_between_shots'],
        config=current_screenshot_config,
        speech_segments=speech_segments # Crucial for segment-based screenshot logic
    )

    new_screenshots = screenshot_extractor.extract_screenshots(str(video_path), str(screenshots_output_dir))

    logger.info(f"Generated {len(new_screenshots)} new screenshots.")

    analysis_data["screenshots"] = new_screenshots
    
    # TODO: Re-run screenshot_transcript_mapping if necessary.
    # The original StudyMaterialProcessor._map_screenshots_to_transcript could be used here
    # if an instance of the processor is created or the method is made static/utility.
    # For now, the mapping will not be updated by this script.
    logger.warning("Screenshot-transcript mapping has not been updated by this script. Run full processing if accurate mapping is critical.")


    try:
        with open(analysis_json_path_obj, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Updated analysis JSON: {analysis_json_path}")
    except Exception as e:
        logger.error(f"Error saving updated JSON file {analysis_json_path}: {e}")
        return

    # Regenerate HTML report
    html_report_filename = f"{video_name_from_json}_report.html"
    html_report_path_str = str(video_specific_output_dir / html_report_filename)
    
    logger.info(f"Attempting to regenerate HTML report: {html_report_path_str}")
    try:
        html_generator = HTMLReportGenerator() # Assuming default template dir is fine
        html_generator.generate_report(analysis_data, html_report_path_str)
        logger.info(f"HTML report regenerated successfully at: {html_report_path_str}")
    except Exception as e:
        logger.error(f"Failed to regenerate HTML report for {html_report_path_str}: {e}")
        logger.error("Please check if the HTMLReportGenerator is correctly implemented and all its dependencies are available.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate screenshots and optionally the HTML report from an existing analysis JSON file.")
    parser.add_argument("analysis_json_path", type=str, help="Path to the _analysis.json file (e.g., results/VideoName/VideoName_analysis.json).")
    parser.add_argument("--similarity_threshold", type=float, help="Override similarity threshold for screenshot extraction (0.0-1.0). Higher is more sensitive to changes.")
    parser.add_argument("--min_time_between_shots", type=float, help="Override minimum time between screenshots in seconds.")
    
    args = parser.parse_args()
    
    regenerate_screenshots_and_report(args.analysis_json_path, args.similarity_threshold, args.min_time_between_shots)
