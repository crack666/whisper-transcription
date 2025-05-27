#!/usr/bin/env python3
"""
Study Material Processor - Complete solution for processing lecture videos
Combines transcription, video analysis, screenshot extraction, and PDF linking
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
import subprocess
import shutil

# Suppress warnings
warnings.filterwarnings("ignore", message="The MPEG_LAYER_III subtype is unknown to TorchAudio")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Language mapping and models
LANGUAGE_MAP = {
    "english": "en", "german": "de", "french": "fr", "spanish": "es", 
    "italian": "it", "japanese": "ja", "chinese": "zh", "portuguese": "pt",
    "russian": "ru", "arabic": "ar", "korean": "ko", "turkish": "tr",
    "hindi": "hi", "dutch": "nl", "swedish": "sv", "indonesian": "id",
    "polish": "pl", "finnish": "fi", "czech": "cs", "danish": "da",
    "auto": None
}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

class AudioTranscriber:
    """Enhanced audio transcriber based on the original script."""
    
    def __init__(self, model_name="large-v3", language="german", device=None):
        self.model_name = model_name
        self.language = LANGUAGE_MAP.get(language.lower(), "de")
        self.device = device
        self.model = None
        
    def load_model(self):
        """Load the Whisper model."""
        if self.model is None:
            import torch
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Whisper model: {self.model_name} on {self.device}")
            self.model = whisper.load_model(self.model_name, device=self.device)
    
    def analyze_background_noise(self, audio_file: str, adjustment: float = 3.0) -> float:
        """Analyze audio to determine silence threshold."""
        audio = AudioSegment.from_file(audio_file)
        noise_level = audio.dBFS
        silence_thresh = noise_level + adjustment
        logger.info(f"Background noise: {noise_level:.2f} dBFS, silence threshold: {silence_thresh:.2f} dBFS")
        return silence_thresh
    
    def split_audio_on_silence(self, audio_file: str, silence_thresh: float, 
                              min_silence_len: int = 1000, padding: int = 750) -> List[Tuple[str, int, int]]:
        """Split audio into segments based on silence."""
        audio = AudioSegment.from_file(audio_file)
        total_duration_ms = len(audio)
        
        logger.info(f"Splitting audio based on {min_silence_len}ms silence...")
        
        nonsilent_ranges = silence.detect_nonsilent(
            audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
        )
        
        if not nonsilent_ranges:
            logger.warning("No speech detected, using entire file as one segment")
            segment_file = f"{audio_file}_segment_0.wav"
            audio.export(segment_file, format="wav")
            return [(segment_file, 0, total_duration_ms)]
        
        segments = []
        for i, (start, end) in enumerate(nonsilent_ranges):
            segment_start = max(0, start - padding)
            segment_end = min(total_duration_ms, end + padding)
            segment = audio[segment_start:segment_end]
            segment_file = f"{audio_file}_segment_{i}.wav"
            segment.export(segment_file, format="wav")
            segments.append((segment_file, segment_start, segment_end))
        
        logger.info(f"Split audio into {len(segments)} segments")
        return segments
    
    def transcribe_segment(self, segment_info: Tuple[str, int, int], segment_idx: int) -> Dict[str, Any]:
        """Transcribe a single audio segment."""
        segment_file, start_time, end_time = segment_info
        
        try:
            result = self.model.transcribe(
                segment_file, 
                language=self.language, 
                verbose=False, 
                fp16=True,
                condition_on_previous_text=True
            )
            
            return {
                "segment_id": segment_idx,
                "text": result["text"].strip(),
                "confidence": float(result.get("confidence", 0.0)),
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "segment_file": segment_file
            }
            
        except Exception as e:
            logger.error(f"Error transcribing {segment_file}: {e}")
            return {
                "segment_id": segment_idx,
                "text": "",
                "error": str(e),
                "start_time": start_time,
                "end_time": end_time
            }
    
    def transcribe_audio_file(self, audio_file: str, cleanup_segments: bool = True) -> Dict[str, Any]:
        """Transcribe entire audio file."""
        self.load_model()
        
        # Split audio into segments
        silence_thresh = self.analyze_background_noise(audio_file)
        segments = self.split_audio_on_silence(audio_file, silence_thresh)
        
        # Transcribe segments
        results = []
        for i, segment_info in enumerate(tqdm.tqdm(segments, desc="Transcribing segments")):
            result = self.transcribe_segment(segment_info, i)
            results.append(result)
            
            # Cleanup segment file
            if cleanup_segments:
                try:
                    os.remove(segment_info[0])
                except Exception as e:
                    logger.warning(f"Could not remove segment file: {e}")
        
        # Combine results
        full_text = " ".join([r["text"] for r in results if r["text"]])
        total_duration = max([r["end_time"] for r in results if "end_time" in r], default=0)
        
        return {
            "segments": results,
            "full_text": full_text,
            "total_duration": total_duration,
            "language": self.language,
            "model": self.model_name
        }

class VideoScreenshotExtractor:
    """Extract screenshots from video at scene changes."""
    
    def __init__(self, similarity_threshold=0.85, min_time_between_shots=2.0):
        self.similarity_threshold = similarity_threshold
        self.min_time_between_shots = min_time_between_shots
    
    def extract_screenshots(self, video_path: str, output_dir: str) -> List[Dict]:
        """Extract screenshots at scene changes."""
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1.0)  # Check every second
        
        screenshots = []
        previous_frame = None
        frame_count = 0
        last_screenshot_time = -self.min_time_between_shots
        
        video_name = Path(video_path).stem
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                
                # Compare with previous frame
                if previous_frame is not None:
                    gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Resize for faster comparison
                    gray_current = cv2.resize(gray_current, (320, 240))
                    gray_previous = cv2.resize(gray_previous, (320, 240))
                    
                    similarity = ssim(gray_current, gray_previous)
                    
                    # Check for scene change
                    if (similarity < self.similarity_threshold and 
                        timestamp - last_screenshot_time >= self.min_time_between_shots):
                        
                        # Save screenshot
                        timestamp_str = f"{int(timestamp//3600):02d}-{int((timestamp%3600)//60):02d}-{int(timestamp%60):02d}"
                        filename = f"{video_name}_screenshot_{len(screenshots):03d}_{timestamp_str}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        
                        cv2.imwrite(filepath, frame)
                        
                        screenshot_info = {
                            "index": len(screenshots),
                            "timestamp": timestamp,
                            "timestamp_formatted": self.format_timestamp(timestamp),
                            "filename": filename,
                            "filepath": filepath,
                            "similarity_score": similarity
                        }
                        screenshots.append(screenshot_info)
                        last_screenshot_time = timestamp
                        
                        logger.debug(f"Screenshot at {timestamp:.2f}s (similarity: {similarity:.3f})")
                
                previous_frame = frame.copy()
            
            frame_count += 1
        
        cap.release()
        
        # Always include first frame if no screenshots taken
        if not screenshots:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                filename = f"{video_name}_screenshot_000_00-00-00.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, frame)
                screenshots.append({
                    "index": 0,
                    "timestamp": 0.0,
                    "timestamp_formatted": "00:00:00.000",
                    "filename": filename,
                    "filepath": filepath,
                    "similarity_score": 1.0
                })
            cap.release()
        
        logger.info(f"Extracted {len(screenshots)} screenshots")
        return screenshots
    
    def format_timestamp(self, seconds: float) -> str:
        """Format timestamp as HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"

class PDFMatcher:
    """Match and extract content from related PDFs."""
    
    def __init__(self, studies_dir: str):
        self.studies_dir = studies_dir
    
    def find_related_pdfs(self, video_filename: str) -> List[Dict]:
        """Find PDFs related to video by date/name similarity."""
        video_path = Path(video_filename)
        video_name = video_path.stem
        
        # Extract date from video filename
        date_match = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', video_name)
        video_date = None
        if date_match:
            day, month, year = date_match.groups()
            video_date = f"{year}{month}{day}"
        
        pdf_files = glob(os.path.join(self.studies_dir, "*.pdf"))
        related_pdfs = []
        
        for pdf_path in pdf_files:
            pdf_name = Path(pdf_path).stem
            relevance_score = 0
            
            # Date matching
            if video_date:
                if video_date in pdf_name:
                    relevance_score += 10
                # Partial date matching
                elif video_date[:6] in pdf_name:  # Year + month
                    relevance_score += 5
            
            # Keyword matching
            video_keywords = set(re.findall(r'\w+', video_name.lower()))
            pdf_keywords = set(re.findall(r'\w+', pdf_name.lower()))
            common_keywords = video_keywords.intersection(pdf_keywords)
            relevance_score += len(common_keywords) * 2
            
            if relevance_score > 0:
                pdf_info = {
                    "filepath": pdf_path,
                    "filename": os.path.basename(pdf_path),
                    "relevance_score": relevance_score,
                    "content_preview": self.extract_pdf_preview(pdf_path)
                }
                related_pdfs.append(pdf_info)
        
        related_pdfs.sort(key=lambda x: x["relevance_score"], reverse=True)
        logger.info(f"Found {len(related_pdfs)} related PDFs")
        return related_pdfs
    
    def extract_pdf_preview(self, pdf_path: str, max_chars: int = 1000) -> str:
        """Extract preview text from PDF."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
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

class HTMLReportGenerator:
    """Generate searchable HTML reports."""
    
    def generate_report(self, results: Dict, output_path: str):
        """Generate comprehensive HTML report."""
        html_content = self._generate_html_template(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
    
    def _generate_html_template(self, results: Dict) -> str:
        """Generate HTML template with all content."""
        video_name = Path(results["video_path"]).stem
        
        # Generate transcript HTML
        transcript_html = self._generate_transcript_html(results.get("transcription", {}))
        
        # Generate screenshots HTML
        screenshots_html = self._generate_screenshots_html(results.get("screenshots", []))
        
        # Generate PDFs HTML
        pdfs_html = self._generate_pdfs_html(results.get("related_pdfs", []))
        
        # Generate mapping HTML
        mapping_html = self._generate_mapping_html(results.get("screenshot_transcript_mapping", []))
        
        return f"""
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Studienanalyse: {video_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .search-box {{ margin-bottom: 20px; }}
        .search-box input {{ width: 100%; padding: 10px; font-size: 16px; border: 1px solid #ddd; border-radius: 4px; }}
        .transcript-segment {{ margin-bottom: 15px; padding: 10px; background: #f9f9f9; border-radius: 4px; }}
        .timestamp {{ color: #007bff; font-weight: bold; }}
        .screenshot {{ display: inline-block; margin: 10px; text-align: center; }}
        .screenshot img {{ max-width: 200px; border: 1px solid #ddd; border-radius: 4px; }}
        .pdf-item {{ margin-bottom: 15px; padding: 10px; background: #fff3cd; border-radius: 4px; }}
        .mapping-item {{ margin-bottom: 20px; padding: 15px; background: #e7f3ff; border-radius: 4px; }}
        .highlight {{ background-color: yellow; }}
        .tabs {{ border-bottom: 1px solid #ddd; margin-bottom: 20px; }}
        .tab {{ display: inline-block; padding: 10px 20px; cursor: pointer; border-bottom: 2px solid transparent; }}
        .tab.active {{ border-bottom-color: #007bff; background: #f8f9fa; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Studienanalyse: {video_name}</h1>
            <p><strong>Bearbeitet am:</strong> {results.get('processing_timestamp', 'N/A')}</p>
            <p><strong>Video:</strong> {results.get('video_path', 'N/A')}</p>
        </div>
        
        <div class="search-box">
            <input type="text" id="searchInput" placeholder="Durchsuchen Sie das Transkript, Screenshots und PDFs...">
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="showTab('transcript')">Transkript</div>
            <div class="tab" onclick="showTab('screenshots')">Screenshots</div>
            <div class="tab" onclick="showTab('pdfs')">Verwandte PDFs</div>
            <div class="tab" onclick="showTab('mapping')">Zuordnung</div>
        </div>
        
        <div id="transcript" class="tab-content active">
            <div class="section">
                <h2>Transkript</h2>
                {transcript_html}
            </div>
        </div>
        
        <div id="screenshots" class="tab-content">
            <div class="section">
                <h2>Screenshots</h2>
                {screenshots_html}
            </div>
        </div>
        
        <div id="pdfs" class="tab-content">
            <div class="section">
                <h2>Verwandte PDFs</h2>
                {pdfs_html}
            </div>
        </div>
        
        <div id="mapping" class="tab-content">
            <div class="section">
                <h2>Screenshot-Transkript Zuordnung</h2>
                {mapping_html}
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            var contents = document.getElementsByClassName('tab-content');
            for (var i = 0; i < contents.length; i++) {{
                contents[i].classList.remove('active');
            }}
            
            // Remove active class from all tabs
            var tabs = document.getElementsByClassName('tab');
            for (var i = 0; i < tabs.length; i++) {{
                tabs[i].classList.remove('active');
            }}
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }}
        
        // Search functionality
        document.getElementById('searchInput').addEventListener('input', function(e) {{
            var searchTerm = e.target.value.toLowerCase();
            var elements = document.querySelectorAll('.transcript-segment, .pdf-item, .mapping-item');
            
            elements.forEach(function(element) {{
                var text = element.textContent.toLowerCase();
                if (text.includes(searchTerm)) {{
                    element.style.display = 'block';
                    // Highlight search term
                    if (searchTerm) {{
                        var regex = new RegExp('(' + searchTerm + ')', 'gi');
                        element.innerHTML = element.innerHTML.replace(regex, '<span class="highlight">$1</span>');
                    }}
                }} else {{
                    element.style.display = 'none';
                }}
            }});
        }});
    </script>
</body>
</html>
        """
    
    def _generate_transcript_html(self, transcription: Dict) -> str:
        """Generate HTML for transcript."""
        if not transcription or not transcription.get("segments"):
            return "<p>Kein Transkript verfügbar.</p>"
        
        html_parts = []
        for segment in transcription["segments"]:
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            text = segment.get("text", "")
            
            start_formatted = self._format_timestamp(start_time)
            end_formatted = self._format_timestamp(end_time)
            
            html_parts.append(f"""
            <div class="transcript-segment">
                <span class="timestamp">[{start_formatted} - {end_formatted}]</span>
                <p>{text}</p>
            </div>
            """)
        
        return "".join(html_parts)
    
    def _generate_screenshots_html(self, screenshots: List[Dict]) -> str:
        """Generate HTML for screenshots."""
        if not screenshots:
            return "<p>Keine Screenshots verfügbar.</p>"
        
        html_parts = []
        for screenshot in screenshots:
            filename = screenshot.get("filename", "")
            timestamp = screenshot.get("timestamp_formatted", "")
            
            html_parts.append(f"""
            <div class="screenshot">
                <img src="screenshots/{filename}" alt="Screenshot bei {timestamp}">
                <p>{timestamp}</p>
            </div>
            """)
        
        return "".join(html_parts)
    
    def _generate_pdfs_html(self, pdfs: List[Dict]) -> str:
        """Generate HTML for related PDFs."""
        if not pdfs:
            return "<p>Keine verwandten PDFs gefunden.</p>"
        
        html_parts = []
        for pdf in pdfs:
            filename = pdf.get("filename", "")
            relevance = pdf.get("relevance_score", 0)
            preview = pdf.get("content_preview", "")
            
            html_parts.append(f"""
            <div class="pdf-item">
                <h3>{filename} (Relevanz: {relevance})</h3>
                <p><strong>Vorschau:</strong></p>
                <p>{preview[:500]}...</p>
            </div>
            """)
        
        return "".join(html_parts)
    
    def _generate_mapping_html(self, mappings: List[Dict]) -> str:
        """Generate HTML for screenshot-transcript mappings."""
        if not mappings:
            return "<p>Keine Zuordnungen verfügbar.</p>"
        
        html_parts = []
        for mapping in mappings:
            screenshot = mapping.get("screenshot", {})
            transcript = mapping.get("transcript_segment", {})
            
            screenshot_time = screenshot.get("timestamp_formatted", "")
            screenshot_file = screenshot.get("filename", "")
            transcript_text = transcript.get("text", "")
            
            html_parts.append(f"""
            <div class="mapping-item">
                <div style="display: flex; align-items: start; gap: 20px;">
                    <div>
                        <img src="screenshots/{screenshot_file}" alt="Screenshot" style="max-width: 300px;">
                        <p><strong>Zeit:</strong> {screenshot_time}</p>
                    </div>
                    <div style="flex: 1;">
                        <h4>Entsprechender Text:</h4>
                        <p>{transcript_text}</p>
                    </div>
                </div>
            </div>
            """)
        
        return "".join(html_parts)
    
    def _format_timestamp(self, ms: int) -> str:
        """Format milliseconds to HH:MM:SS"""
        seconds = ms // 1000
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class StudyMaterialProcessor:
    """Main processor combining all components."""
    
    def __init__(self, args):
        self.args = args
        self.transcriber = AudioTranscriber(
            model_name=args.model,
            language=args.language,
            device=getattr(args, 'device', None)
        )
        self.screenshot_extractor = VideoScreenshotExtractor(
            similarity_threshold=args.similarity_threshold,
            min_time_between_shots=args.min_time_between_shots
        )
        self.pdf_matcher = PDFMatcher(args.studies_dir)
        self.html_generator = HTMLReportGenerator()
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video using ffmpeg."""
        audio_path = f"{video_path}.wav"
        
        if os.path.exists(audio_path):
            logger.info(f"Audio file already exists: {audio_path}")
            return audio_path
        
        logger.info(f"Extracting audio from {video_path}")
        
        cmd = [
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
            "-ar", "44100", "-ac", "2", audio_path, "-y"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Audio extracted to: {audio_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio: {e}")
            raise
        
        return audio_path
    
    def map_screenshots_to_transcript(self, screenshots: List[Dict], transcription: Dict) -> List[Dict]:
        """Map screenshots to transcript segments."""
        mappings = []
        
        for screenshot in screenshots:
            screenshot_time = screenshot["timestamp"] * 1000  # Convert to ms
            
            closest_segment = None
            min_distance = float('inf')
            
            for segment in transcription.get("segments", []):
                if "start_time" in segment:
                    segment_time = segment["start_time"]
                    distance = abs(screenshot_time - segment_time)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_segment = segment
            
            if closest_segment:
                mappings.append({
                    "screenshot": screenshot,
                    "transcript_segment": closest_segment,
                    "time_difference": min_distance / 1000.0  # Convert back to seconds
                })
        
        return mappings
    
    def process_video(self, video_path: str) -> Dict:
        """Process a single video file completely."""
        logger.info(f"Processing video: {video_path}")
        
        video_name = Path(video_path).stem
        output_dir = os.path.join(self.args.output_dir, video_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract audio
        audio_path = self.extract_audio_from_video(video_path)
        
        # Transcribe audio
        logger.info("Transcribing audio...")
        transcription = self.transcriber.transcribe_audio_file(audio_path, cleanup_segments=True)
        
        # Extract screenshots
        screenshots = []
        if self.args.extract_screenshots:
            logger.info("Extracting screenshots...")
            screenshots_dir = os.path.join(output_dir, "screenshots")
            screenshots = self.screenshot_extractor.extract_screenshots(video_path, screenshots_dir)
        
        # Find related PDFs
        logger.info("Finding related PDFs...")
        related_pdfs = self.pdf_matcher.find_related_pdfs(video_path)
        
        # Map screenshots to transcript
        screenshot_transcript_mapping = []
        if screenshots and transcription:
            screenshot_transcript_mapping = self.map_screenshots_to_transcript(screenshots, transcription)
        
        # Compile results
        result = {
            "video_path": video_path,
            "audio_path": audio_path,
            "transcription": transcription,
            "screenshots": screenshots,
            "related_pdfs": related_pdfs,
            "screenshot_transcript_mapping": screenshot_transcript_mapping,
            "processing_timestamp": datetime.now().isoformat(),
            "processing_args": vars(self.args)
        }
        
        # Save JSON result
        json_output = os.path.join(output_dir, f"{video_name}_analysis.json")
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Generate HTML report
        html_output = os.path.join(output_dir, f"{video_name}_report.html")
        self.html_generator.generate_report(result, html_output)
        
        # Clean up audio file if requested
        if self.args.cleanup_audio and os.path.exists(audio_path):
            os.remove(audio_path)
        
        logger.info(f"Processing complete. Results saved to: {output_dir}")
        return result

def setup_argparse():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(description="Study Material Processor")
    
    # Input/Output
    parser.add_argument("--input", type=str, required=True,
                       help="Video file or directory containing videos")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory for results")
    parser.add_argument("--studies_dir", type=str, default="./studies",
                       help="Directory containing study materials (PDFs)")
    
    # Video processing
    parser.add_argument("--extract_screenshots", action="store_true", default=True,
                       help="Extract screenshots at scene changes")
    parser.add_argument("--similarity_threshold", type=float, default=0.85,
                       help="Similarity threshold for scene change detection")
    parser.add_argument("--min_time_between_shots", type=float, default=2.0,
                       help="Minimum time between screenshots (seconds)")
    
    # Transcription
    parser.add_argument("--language", type=str, default="german",
                       help="Language for transcription")
    parser.add_argument("--model", type=str, default="large-v3", choices=WHISPER_MODELS,
                       help="Whisper model to use")
    parser.add_argument("--device", type=str, default=None,
                       help="Device for transcription (cpu, cuda, etc.)")
    
    # Processing options
    parser.add_argument("--batch_process", action="store_true",
                       help="Process all videos in input directory")
    parser.add_argument("--cleanup_audio", action="store_true",
                       help="Remove extracted audio files after processing")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    return parser

def main():
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize processor
    processor = StudyMaterialProcessor(args)
    
    # Determine input files
    if os.path.isfile(args.input):
        if args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
            video_files = [args.input]
        else:
            logger.error("Input file is not a video file")
            return
    elif os.path.isdir(args.input) and args.batch_process:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob(os.path.join(args.input, f"*{ext}")))
    else:
        logger.error("Input must be a video file or directory with --batch_process")
        return
    
    if not video_files:
        logger.error("No video files found")
        return
    
    logger.info(f"Processing {len(video_files)} video file(s)")
    
    # Process videos
    all_results = []
    for video_path in tqdm.tqdm(video_files, desc="Processing videos"):
        try:
            result = processor.process_video(video_path)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            continue
    
    # Generate combined index
    if len(all_results) > 1:
        index_path = os.path.join(args.output_dir, "index.html")
        generate_index_html(all_results, index_path)
    
    logger.info("Processing completed!")

def generate_index_html(results: List[Dict], output_path: str):
    """Generate an index HTML file linking to all processed videos."""
    html_content = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Studienanalyse - Übersicht</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .container { max-width: 800px; margin: 0 auto; }
        .video-item { margin-bottom: 20px; padding: 15px; background: #f9f9f9; border-radius: 4px; }
        .video-item h3 { margin-top: 0; }
        .stats { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Studienanalyse - Übersicht</h1>
        <p>Generiert am: """ + datetime.now().strftime("%d.%m.%Y %H:%M:%S") + """</p>
        
        <h2>Verarbeitete Videos</h2>
    """
    
    for result in results:
        video_name = Path(result["video_path"]).stem
        transcription = result.get("transcription", {})
        screenshots_count = len(result.get("screenshots", []))
        pdfs_count = len(result.get("related_pdfs", []))
        
        duration = transcription.get("total_duration", 0) / 1000 / 60  # Convert to minutes
        
        html_content += f"""
        <div class="video-item">
            <h3><a href="{video_name}/{video_name}_report.html">{video_name}</a></h3>
            <div class="stats">
                <p>Dauer: {duration:.1f} Minuten | Screenshots: {screenshots_count} | Verwandte PDFs: {pdfs_count}</p>
                <p>Verarbeitet: {result.get('processing_timestamp', 'N/A')}</p>
            </div>
        </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Index file generated: {output_path}")

if __name__ == "__main__":
    main()