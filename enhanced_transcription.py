#!/usr/bin/env python3
"""
Enhanced Transcription Tool with Video Analysis and Screenshot Extraction
"""

import whisper
import time
import argparse
import json
import logging
import concurrent.futures
from datetime import datetime, timedelta
import tqdm
from pydub import AudioSegment, silence
import os
from glob import glob
import re
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import PyPDF2
from pathlib import Path

# Suppress specific warnings
warnings.filterwarnings("ignore", message="The MPEG_LAYER_III subtype is unknown to TorchAudio")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Language mapping and models from original script
LANGUAGE_MAP = {
    "english": "en", "german": "de", "french": "fr", "spanish": "es", 
    "italian": "it", "japanese": "ja", "chinese": "zh", "portuguese": "pt",
    "russian": "ru", "arabic": "ar", "korean": "ko", "turkish": "tr",
    "hindi": "hi", "dutch": "nl", "swedish": "sv", "indonesian": "id",
    "polish": "pl", "finnish": "fi", "czech": "cs", "danish": "da",
    "auto": None
}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

class VideoAnalyzer:
    """Handles video analysis and screenshot extraction."""
    
    def __init__(self, similarity_threshold=0.85, min_time_between_shots=2.0):
        self.similarity_threshold = similarity_threshold
        self.min_time_between_shots = min_time_between_shots
        
    def extract_frames_at_intervals(self, video_path: str, interval_seconds: float = 1.0) -> List[Tuple[float, np.ndarray]]:
        """Extract frames at regular intervals."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_seconds)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                frames.append((timestamp, frame))
                
            frame_count += 1
            
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames
    
    def detect_scene_changes(self, frames: List[Tuple[float, np.ndarray]]) -> List[Tuple[float, np.ndarray]]:
        """Detect significant scene changes based on image similarity."""
        if not frames:
            return []
        
        scene_changes = [frames[0]]  # Always include first frame
        
        for i in range(1, len(frames)):
            timestamp, current_frame = frames[i]
            _, previous_frame = frames[i-1]
            
            # Convert to grayscale for comparison
            gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate structural similarity
            similarity = ssim(gray_current, gray_previous)
            
            # Check if enough time has passed and similarity is below threshold
            last_timestamp = scene_changes[-1][0]
            if (similarity < self.similarity_threshold and 
                timestamp - last_timestamp >= self.min_time_between_shots):
                scene_changes.append((timestamp, current_frame))
                logger.debug(f"Scene change detected at {timestamp:.2f}s (similarity: {similarity:.3f})")
        
        logger.info(f"Detected {len(scene_changes)} scene changes")
        return scene_changes
    
    def save_screenshots(self, scene_changes: List[Tuple[float, np.ndarray]], output_dir: str, video_name: str) -> List[Dict]:
        """Save screenshots and return metadata."""
        os.makedirs(output_dir, exist_ok=True)
        screenshots = []
        
        for i, (timestamp, frame) in enumerate(scene_changes):
            # Generate filename
            timestamp_str = f"{int(timestamp//3600):02d}-{int((timestamp%3600)//60):02d}-{int(timestamp%60):02d}"
            filename = f"{video_name}_screenshot_{i:03d}_{timestamp_str}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, frame)
            
            screenshot_info = {
                "index": i,
                "timestamp": timestamp,
                "timestamp_formatted": self.format_timestamp(timestamp),
                "filename": filename,
                "filepath": filepath
            }
            screenshots.append(screenshot_info)
            
        logger.info(f"Saved {len(screenshots)} screenshots to {output_dir}")
        return screenshots
    
    def format_timestamp(self, seconds: float) -> str:
        """Format timestamp as HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"

class PDFMatcher:
    """Handles PDF file matching and content extraction."""
    
    def __init__(self, studies_dir: str):
        self.studies_dir = studies_dir
        
    def find_related_pdfs(self, video_filename: str) -> List[Dict]:
        """Find PDFs related to the video by date or name similarity."""
        video_path = Path(video_filename)
        video_name = video_path.stem
        
        # Extract date from video filename
        date_match = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', video_name)
        video_date = None
        if date_match:
            day, month, year = date_match.groups()
            video_date = f"{year}-{month}-{day}"
        
        # Find all PDFs in studies directory
        pdf_files = glob(os.path.join(self.studies_dir, "*.pdf"))
        related_pdfs = []
        
        for pdf_path in pdf_files:
            pdf_name = Path(pdf_path).stem
            relevance_score = 0
            
            # Check for date matches
            if video_date:
                pdf_date_match = re.search(r'(\d{4})(\d{2})(\d{2})', pdf_name)
                if pdf_date_match:
                    year, month, day = pdf_date_match.groups()
                    pdf_date = f"{year}-{month}-{day}"
                    if pdf_date == video_date:
                        relevance_score += 10
                
                # Check for partial date matches
                if video_date.replace('-', '') in pdf_name:
                    relevance_score += 5
            
            # Check for keyword matches
            video_keywords = set(re.findall(r'\w+', video_name.lower()))
            pdf_keywords = set(re.findall(r'\w+', pdf_name.lower()))
            common_keywords = video_keywords.intersection(pdf_keywords)
            relevance_score += len(common_keywords)
            
            if relevance_score > 0:
                pdf_info = {
                    "filepath": pdf_path,
                    "filename": os.path.basename(pdf_path),
                    "relevance_score": relevance_score,
                    "content_preview": self.extract_pdf_preview(pdf_path)
                }
                related_pdfs.append(pdf_info)
        
        # Sort by relevance score
        related_pdfs.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        logger.info(f"Found {len(related_pdfs)} related PDFs for {video_name}")
        return related_pdfs
    
    def extract_pdf_preview(self, pdf_path: str, max_chars: int = 500) -> str:
        """Extract a preview of PDF content."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from first few pages
                for page_num in range(min(3, len(pdf_reader.pages))):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                    
                    if len(text) > max_chars:
                        text = text[:max_chars] + "..."
                        break
                
                return text.strip()
        except Exception as e:
            logger.warning(f"Could not extract text from {pdf_path}: {e}")
            return "Preview not available"

class EnhancedTranscriptionProcessor:
    """Main processor that combines transcription, video analysis, and PDF matching."""
    
    def __init__(self, args):
        self.args = args
        self.video_analyzer = VideoAnalyzer()
        self.pdf_matcher = PDFMatcher(args.studies_dir) if hasattr(args, 'studies_dir') else None
        
    def process_video_file(self, video_path: str) -> Dict:
        """Process a single video file with enhanced features."""
        logger.info(f"Processing video: {video_path}")
        
        # Extract audio from video
        audio_path = self.extract_audio_from_video(video_path)
        
        # Perform standard transcription
        transcription_result = self.transcribe_audio(audio_path)
        
        # Extract screenshots at scene changes
        screenshots = []
        if self.args.extract_screenshots:
            logger.info("Extracting screenshots from video...")
            frames = self.video_analyzer.extract_frames_at_intervals(video_path, 1.0)
            scene_changes = self.video_analyzer.detect_scene_changes(frames)
            
            output_dir = os.path.join(os.path.dirname(video_path), "screenshots")
            video_name = Path(video_path).stem
            screenshots = self.video_analyzer.save_screenshots(scene_changes, output_dir, video_name)
        
        # Find related PDFs
        related_pdfs = []
        if self.pdf_matcher:
            related_pdfs = self.pdf_matcher.find_related_pdfs(video_path)
        
        # Combine all results
        result = {
            "video_path": video_path,
            "audio_path": audio_path,
            "transcription": transcription_result,
            "screenshots": screenshots,
            "related_pdfs": related_pdfs,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Map screenshots to transcription segments
        if screenshots and transcription_result:
            result["screenshot_transcript_mapping"] = self.map_screenshots_to_transcript(
                screenshots, transcription_result
            )
        
        return result
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video file."""
        audio_path = f"{video_path}.wav"
        
        if os.path.exists(audio_path):
            logger.info(f"Audio file already exists: {audio_path}")
            return audio_path
        
        logger.info(f"Extracting audio from {video_path}")
        
        # Use ffmpeg through opencv to extract audio
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        cap.release()
        
        # Use ffmpeg directly for audio extraction
        import subprocess
        cmd = [
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
            "-ar", "44100", "-ac", "2", audio_path, "-y"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Audio extracted to: {audio_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio: {e}")
            raise
        
        return audio_path
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio using the existing transcription logic."""
        # This would integrate with the existing transcription system
        # For now, return a placeholder
        return {
            "segments": [],
            "full_text": "",
            "language": "de",
            "duration": 0
        }
    
    def map_screenshots_to_transcript(self, screenshots: List[Dict], transcription: Dict) -> List[Dict]:
        """Map screenshots to corresponding transcript segments."""
        mappings = []
        
        for screenshot in screenshots:
            screenshot_time = screenshot["timestamp"]
            
            # Find the closest transcript segment
            closest_segment = None
            min_distance = float('inf')
            
            for segment in transcription.get("segments", []):
                if "start_time" in segment:
                    segment_time = segment["start_time"] / 1000.0  # Convert to seconds
                    distance = abs(screenshot_time - segment_time)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_segment = segment
            
            if closest_segment:
                mapping = {
                    "screenshot": screenshot,
                    "transcript_segment": closest_segment,
                    "time_difference": min_distance
                }
                mappings.append(mapping)
        
        return mappings

def setup_enhanced_argparse():
    """Set up argument parser with enhanced options."""
    parser = argparse.ArgumentParser(description="Enhanced transcription with video analysis")
    
    # Basic options
    parser.add_argument("--input_path", type=str, required=True, 
                       help="Path to video file or directory containing videos")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Directory for output files")
    parser.add_argument("--studies_dir", type=str, default="./studies",
                       help="Directory containing study materials and PDFs")
    
    # Video analysis options
    parser.add_argument("--extract_screenshots", action="store_true",
                       help="Extract screenshots at scene changes")
    parser.add_argument("--similarity_threshold", type=float, default=0.85,
                       help="Similarity threshold for scene change detection")
    parser.add_argument("--min_time_between_shots", type=float, default=2.0,
                       help="Minimum time between screenshots (seconds)")
    
    # Transcription options (from original script)
    parser.add_argument("--language", type=str, default="german",
                       help="Language for transcription")
    parser.add_argument("--model", type=str, default="large-v3", choices=WHISPER_MODELS,
                       help="Whisper model to use")
    parser.add_argument("--include_timestamps", action="store_true",
                       help="Include timestamps in transcription")
    
    # Processing options
    parser.add_argument("--batch_process", action="store_true",
                       help="Process all videos in input directory")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    return parser

def main():
    """Main entry point for enhanced transcription."""
    parser = setup_enhanced_argparse()
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    processor = EnhancedTranscriptionProcessor(args)
    
    # Determine input files
    if os.path.isfile(args.input_path):
        video_files = [args.input_path]
    elif os.path.isdir(args.input_path) and args.batch_process:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob(os.path.join(args.input_path, f"*{ext}")))
    else:
        logger.error("Input path must be a file or directory with --batch_process")
        return
    
    if not video_files:
        logger.error("No video files found")
        return
    
    logger.info(f"Processing {len(video_files)} video file(s)")
    
    # Process each video
    all_results = []
    for video_path in tqdm.tqdm(video_files, desc="Processing videos"):
        try:
            result = processor.process_video_file(video_path)
            all_results.append(result)
            
            # Save individual result
            video_name = Path(video_path).stem
            output_file = os.path.join(args.output_dir, f"{video_name}_enhanced.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            continue
    
    # Save combined results
    combined_output = os.path.join(args.output_dir, "combined_results.json")
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Combined results saved to: {combined_output}")
    logger.info("Enhanced transcription processing completed!")

if __name__ == "__main__":
    main()