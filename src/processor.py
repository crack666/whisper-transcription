"""
Main processor module that orchestrates the entire study material processing workflow
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from .enhanced_transcriber import EnhancedAudioTranscriber
from .video_processor import VideoScreenshotExtractor
from .pdf_matcher import PDFMatcher
from .html_generator import HTMLReportGenerator
from .benchmark_logger import BenchmarkLogger
from .utils import (
    extract_audio_from_video, 
    ensure_directory, 
    sanitize_filename, 
    get_video_info,
    check_dependencies,
    estimate_processing_time
)
from .config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class StudyMaterialProcessor:
    """
    Main processor that orchestrates the complete study material analysis workflow.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the study material processor.
        
        Args:
            config: Configuration dictionary for all components
        """
        self.config = self._merge_config(config)
        
        # Initialize components - always use EnhancedAudioTranscriber
        self.transcriber = EnhancedAudioTranscriber(
            model_name=self.config['transcription']['model'],
            language=self.config['transcription']['language'],
            device=self.config['transcription']['device'],
            config=self.config['transcription']
        )
        
        # VideoScreenshotExtractor is initialized later, once speech_segments are available
        self.screenshot_extractor = None
        
        self.pdf_matcher = None  # Will be initialized when studies_dir is provided
        self.html_generator = HTMLReportGenerator()
        
        # Initialize benchmark logger
        self.benchmark_logger = BenchmarkLogger()
        
        logger.info("StudyMaterialProcessor initialized")
    
    def _merge_config(self, user_config: Optional[Dict]) -> Dict:
        """
        Merge user configuration with defaults.
        
        Args:
            user_config: User-provided configuration
            
        Returns:
            Merged configuration dictionary
        """
        config = DEFAULT_CONFIG.copy()
        
        if user_config:
            for section, values in user_config.items():
                if section in config:
                    config[section].update(values)
                else:
                    config[section] = values
        
        return config
    
    def set_studies_directory(self, studies_dir: str) -> None:
        """
        Set the studies directory and initialize PDF matcher.
        
        Args:
            studies_dir: Path to directory containing study materials
        """
        if not os.path.exists(studies_dir):
            logger.warning(f"Studies directory does not exist: {studies_dir}")
        else:
            self.pdf_matcher = PDFMatcher(studies_dir, self.config['pdf_matching'])
            logger.info(f"Studies directory set to: {studies_dir}")
    
    def process_video(self, video_path: str, output_dir: str, 
                     studies_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single video file completely.
        
        Args:
            video_path: Path to video file
            output_dir: Directory for output files
            studies_dir: Directory containing study materials (optional)
            
        Returns:
            Complete processing results dictionary
        """
        start_time = time.time()
        
        # Start benchmark logging
        self.benchmark_logger.start_run(video_path, self.config)
        
        logger.info(f"Starting processing of video: {video_path}")
        
        # Validate inputs
        if not os.path.exists(video_path):
            error_msg = f"Video file not found: {video_path}"
            self.benchmark_logger.log_error(error_msg, "validation")
            self.benchmark_logger.end_run(success=False)
            raise FileNotFoundError(error_msg)
        
        # Set studies directory if provided
        if studies_dir:
            self.set_studies_directory(studies_dir)
        
        # Prepare output structure
        video_name = sanitize_filename(Path(video_path).stem)
        video_path_obj = Path(video_path)
        
        # Determine output directory:
        # - If output_dir is None: use source directory, screenshots in {name}_screenshots/
        # - Otherwise: use output_dir/{video_name}/, screenshots in screenshots/
        if output_dir is None:
            # Save in source directory - reports go there, screenshots in subdirectory
            video_output_dir = video_path_obj.parent.resolve()
            screenshots_subdir_name = f"{video_name}_screenshots"
        else:
            # Use specified output directory structure
            video_output_dir = ensure_directory(os.path.join(output_dir, video_name))
            screenshots_subdir_name = "screenshots"
        
        logger.info(f"Output directory: {video_output_dir}")
        logger.info(f"Screenshots subdirectory: {screenshots_subdir_name}")
        
        try:
            # Get video information
            video_info = get_video_info(video_path)
            logger.info(f"Video info: {video_info['duration_seconds']:.1f}s, {video_info['width']}x{video_info['height']}")
            
            # Estimate processing time
            time_estimate = estimate_processing_time(video_info, self.config['transcription']['model'])
            logger.info(f"Estimated processing time: {time_estimate['estimated_total_minutes']:.1f} minutes")
            
            # Step 1: Extract audio from video
            self.benchmark_logger.start_phase("audio_extraction")
            logger.info("Step 1/5: Extracting audio from video...")
            audio_path = extract_audio_from_video(video_path)
            self.benchmark_logger.end_phase("audio_extraction")
            
            # Step 2: Transcribe audio
            self.benchmark_logger.start_phase("transcription")
            logger.info("Step 2/5: Transcribing audio...")
            transcription_result = self.transcriber.transcribe_audio_file_enhanced(audio_path)

            # Log transcription metadata
            speech_segments = transcription_result.get('transcription', {}).get('segments', [])
            word_count = sum(len(seg.get('text', '').split()) for seg in speech_segments)
            self.benchmark_logger.end_phase("transcription", metadata={
                "segment_count": len(speech_segments),
                "word_count": word_count
            })
            
            # Save transcription result as a side-car JSON file
            input_file_path = Path(video_path)
            sidecar_transcription_path = input_file_path.with_suffix('.json')
            try:
                with open(sidecar_transcription_path, 'w', encoding='utf-8') as f:
                    json.dump(transcription_result, f, indent=2, ensure_ascii=False, default=str)
                logger.info(f"Transcription saved as side-car JSON: {sidecar_transcription_path}")
            except Exception as e:
                logger.error(f"Failed to save side-car transcription JSON to {sidecar_transcription_path}: {e}")

            # Initialize ScreenshotExtractor with actual speech segments
            self.screenshot_extractor = VideoScreenshotExtractor(
                similarity_threshold=self.config['screenshots']['similarity_threshold'],
                min_time_between_shots=self.config['screenshots']['min_time_between_shots'],
                config=self.config['screenshots'],
                speech_segments=speech_segments # Pass actual segments
            )
            
            # Step 3: Extract screenshots (if enabled and if it's a video file)
            screenshots = []
            is_video_file = self._is_video_file(video_path)
            
            if self.config['output'].get('extract_screenshots', True) and is_video_file:
                if self.screenshot_extractor: # Check if initialized
                    self.benchmark_logger.start_phase("screenshot_extraction")
                    print(f"\n📸 Step 3/5: Extracting screenshots...")
                    logger.info("Step 3/5: Extracting screenshots...")
                    screenshots_dir = video_output_dir / screenshots_subdir_name
                    # Pass speech_segments again or ensure it's used from init
                    # The extractor is now initialized with segments, so this call is fine
                    screenshots = self.screenshot_extractor.extract_screenshots(video_path, str(screenshots_dir))
                    self.benchmark_logger.end_phase("screenshot_extraction", metadata={
                        "screenshot_count": len(screenshots)
                    })
                else:
                    logger.warning("Screenshot extractor not initialized, skipping screenshot extraction.")
            elif not is_video_file:
                logger.info("Step 3/5: Skipping screenshot extraction (audio-only file)")
            else:
                logger.info("Step 3/5: Screenshot extraction disabled")
            
            # Step 4: Find related PDFs
            related_pdfs = []
            if self.pdf_matcher:
                self.benchmark_logger.start_phase("pdf_matching")
                print(f"\n📚 Step 4/5: Finding related PDFs...")
                logger.info("Step 4/5: Finding related PDFs...")
                related_pdfs = self.pdf_matcher.find_related_pdfs(video_path)
                self.benchmark_logger.end_phase("pdf_matching", metadata={
                    "pdf_count": len(related_pdfs)
                })
            else:
                logger.info("Step 4/5: PDF matching disabled (no studies directory)")
            
            # Step 5: Map screenshots to transcript
            self.benchmark_logger.start_phase("mapping_and_report_generation")
            print(f"\n🔗 Step 5/5: Creating mappings and generating reports...")
            logger.info("Step 5/5: Creating screenshot-transcript mappings...")
            screenshot_transcript_mapping = self._map_screenshots_to_transcript(
                screenshots, transcription_result
            )
            
            # Compile complete results
            processing_time = time.time() - start_time
            
            result = {
                "video_path": video_path,
                "audio_path": audio_path,
                "output_directory": str(video_output_dir),
                "video_info": video_info,
                "transcription": transcription_result,
                "screenshots": screenshots,
                "related_pdfs": related_pdfs,
                "screenshot_transcript_mapping": screenshot_transcript_mapping,
                "processing_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": processing_time,
                "config_used": self.config,
                "time_estimates": time_estimate
            }
            
            # Save results
            self._save_results(result, video_output_dir, video_name)
            self.benchmark_logger.end_phase("mapping_and_report_generation")
            
            # End benchmark logging with success
            self.benchmark_logger.end_run(success=True, result_data={
                "word_count": word_count,
                "segment_count": len(speech_segments),
                "screenshot_count": len(screenshots),
                "pdf_count": len(related_pdfs)
            })
            
            # Cleanup if requested
            if self.config['output']['cleanup_audio'] and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info("Cleaned up extracted audio file")
            
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            
            # Print quick benchmark summary
            print(f"\n📊 Performance: {processing_time:.1f}s total ({video_info['duration_seconds']/processing_time:.2f}x realtime)")
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing video {video_path}: {e}"
            logger.error(error_msg)
            self.benchmark_logger.log_error(str(e), "processing")
            self.benchmark_logger.end_run(success=False)
            raise
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     studies_dir: Optional[str] = None, 
                     video_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Process all videos in a directory.
        
        Args:
            input_dir: Directory containing video files
            output_dir: Directory for output files
            studies_dir: Directory containing study materials (optional)
            video_extensions: List of video file extensions to process
            
        Returns:
            List of processing results for all videos
        """
        if video_extensions is None:
            from .config import VIDEO_EXTENSIONS
            video_extensions = VIDEO_EXTENSIONS
        
        # Find all video files
        input_path = Path(input_dir)
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(input_path.glob(f"*{ext}"))
        
        if not video_files:
            logger.warning(f"No video files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(video_files)} video files to process")
        
        # Process each video
        all_results = []
        for i, video_path in enumerate(video_files, 1):
            file_start_time = time.time()
            logger.info(f"Processing video {i}/{len(video_files)}: {video_path.name}")
            print(f"\n[{i}/{len(video_files)}] 🎬 {video_path.name} - Started at {time.strftime('%H:%M:%S')}")
            
            try:
                result = self.process_video(str(video_path), output_dir, studies_dir)
                file_time = time.time() - file_start_time
                print(f"     ✅ Completed in {file_time/60:.2f} minutes ({file_time:.1f}s)")
                all_results.append(result)
            except Exception as e:
                file_time = time.time() - file_start_time
                print(f"     ❌ Failed after {file_time/60:.2f} minutes: {e}")
                logger.error(f"Failed to process {video_path.name}: {e}")
                # Continue with next video
                continue
        
        # Generate combined index if multiple videos processed
        if len(all_results) > 1:
            index_path = os.path.join(output_dir, "index.html")
            self.html_generator.generate_index_page(all_results, index_path)
        
        logger.info(f"Batch processing completed: {len(all_results)}/{len(video_files)} videos successful")
        return all_results
    
    def _map_screenshots_to_transcript(self, screenshots: List[Dict], 
                                     transcription: Dict) -> List[Dict]:
        """
        Map screenshots to corresponding transcript segments.
        
        Args:
            screenshots: List of screenshot information
            transcription: Transcription results
            
        Returns:
            List of mapping dictionaries
        """
        if not screenshots or not transcription.get("segments"):
            return []
        
        mappings = []
        
        for screenshot in screenshots:
            screenshot_time_ms = screenshot["timestamp"] * 1000  # Convert to milliseconds
            
            # Find the closest transcript segment
            closest_segment = None
            min_distance = float('inf')
            
            for segment in transcription["segments"]:
                if "start_time" in segment and segment["text"].strip():
                    segment_start = segment["start_time"]
                    segment_end = segment.get("end_time", segment_start)
                    
                    # Calculate distance to segment (prefer segments that contain the timestamp)
                    if segment_start <= screenshot_time_ms <= segment_end:
                        distance = 0  # Perfect match
                    else:
                        distance = min(
                            abs(screenshot_time_ms - segment_start),
                            abs(screenshot_time_ms - segment_end)
                        )
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_segment = segment
            
            if closest_segment:
                mapping = {
                    "screenshot": screenshot,
                    "transcript_segment": closest_segment,
                    "time_difference_seconds": min_distance / 1000.0,
                    "is_exact_match": min_distance == 0
                }
                mappings.append(mapping)
        
        logger.info(f"Created {len(mappings)} screenshot-transcript mappings")
        return mappings
    
    def _save_results(self, result: Dict, output_dir: Path, video_name: str) -> None:
        """
        Save processing results in multiple formats.
        
        Args:
            result: Complete processing results
            output_dir: Output directory
            video_name: Sanitized video name
        """
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results if enabled
        if self.config['output']['generate_json']:
            print(f"💾 Saving analysis data...")
            json_path = output_dir / f"{video_name}_analysis.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"JSON results saved to: {json_path}")
        
        # Generate HTML report if enabled
        if self.config['output']['generate_html']:
            print(f"📄 Generating HTML report...")
            html_path = output_dir / f"{video_name}_report.html"
            self.html_generator.generate_report(result, str(html_path))
            print(f"✅ HTML report generated: {html_path.name}")
            logger.info(f"HTML report saved to: {html_path}")
        
        # Extract plain text transcript (always enabled)
        if result.get('transcription'):
            print(f"📝 Extracting plain text transcript...")
            try:
                text_content = self._extract_plain_text(result['transcription'])
                txt_path = output_dir / f"{video_name}_transcript.txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                print(f"✅ Plain text transcript saved: {txt_path.name}")
                logger.info(f"Plain text transcript saved to: {txt_path}")
            except Exception as e:
                logger.error(f"Failed to extract plain text transcript: {e}")
    
    def _extract_plain_text(self, transcription_data: Dict) -> str:
        """
        Extract plain text from transcription data.
        
        Args:
            transcription_data: Transcription dictionary
            
        Returns:
            Formatted plain text transcript
        """
        # Handle nested transcription structure
        actual_transcription = transcription_data.get('transcription', transcription_data)
        segments = actual_transcription.get('segments', [])
        
        if not segments:
            return ""
        
        # Build text with timestamps
        lines = []
        lines.append("=" * 80)
        lines.append("TRANSCRIPTION")
        lines.append("=" * 80)
        lines.append("")
        
        # Add metadata if available
        if actual_transcription.get('language'):
            lines.append(f"Language: {actual_transcription['language']}")
        
        duration = segments[-1].get('end', 0) if segments else 0
        lines.append(f"Duration: {duration / 60:.1f} minutes ({duration:.1f} seconds)")
        lines.append(f"Segments: {len(segments)}")
        
        word_count = sum(len(seg.get('text', '').split()) for seg in segments)
        lines.append(f"Words: {word_count}")
        lines.append("")
        lines.append("=" * 80)
        lines.append("")
        
        # Add segments with timestamps
        for i, segment in enumerate(segments, 1):
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            if not text:
                continue
            
            # Format timestamp
            start_h = int(start // 3600)
            start_m = int((start % 3600) // 60)
            start_s = int(start % 60)
            
            timestamp = f"[{start_h:02d}:{start_m:02d}:{start_s:02d}]"
            
            lines.append(f"{timestamp} {text}")
            lines.append("")
        
        return "\n".join(lines)
    
    def analyze_video_complexity(self, video_path: str) -> Dict:
        """
        Analyze video complexity to suggest optimal processing settings.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Analysis results with recommendations
        """
        logger.info(f"Analyzing video complexity: {video_path}")
        
        # Get basic video info
        video_info = get_video_info(video_path)
        
        # Analyze screenshot complexity
        screenshot_analysis = self.screenshot_extractor.analyze_video_complexity(video_path)
        
        # Estimate processing requirements
        time_estimates = estimate_processing_time(video_info, self.config['transcription']['model'])
        
        # Generate recommendations
        recommendations = {
            "video_info": video_info,
            "screenshot_analysis": screenshot_analysis,
            "time_estimates": time_estimates,
            "recommended_settings": {
                "model": self._recommend_model(video_info['duration_seconds']),
                "screenshot_threshold": screenshot_analysis.get('recommended_threshold', 0.85),
                "screenshot_interval": screenshot_analysis.get('recommended_interval', 2.0)
            }
        }
        
        return recommendations
    
    def _recommend_model(self, duration_seconds: float) -> str:
        """
        Recommend Whisper model based on video duration.
        
        Args:
            duration_seconds: Video duration in seconds
            
        Returns:
            Recommended model name
        """
        duration_minutes = duration_seconds / 60
        
        if duration_minutes < 10:
            return "large-v3"  # High quality for short videos
        elif duration_minutes < 60:
            return "large"     # Good balance
        elif duration_minutes < 120:
            return "medium"    # Faster for longer videos
        else:
            return "base"      # Fastest for very long videos
    
    def get_processing_status(self) -> Dict:
        """
        Get current status of all processor components.
        
        Returns:
            Status information dictionary
        """
        return {
            "transcriber": self.transcriber.get_model_info(),
            "pdf_matcher_ready": self.pdf_matcher is not None,
            "config": self.config,
            "dependencies_checked": True  # Assuming we've made it this far
        }
    
    def validate_setup(self) -> Dict:
        """
        Validate that all required dependencies and components are properly set up.
        
        Returns:
            Validation results dictionary
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check dependencies
            check_dependencies()
        except (ImportError, FileNotFoundError) as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Dependency error: {e}")
        
        # Check transcriber
        try:
            model_info = self.transcriber.get_model_info()
            if not model_info.get("model_loaded", False):
                validation_results["warnings"].append("Whisper model not yet loaded")
        except Exception as e:
            validation_results["errors"].append(f"Transcriber error: {e}")
        
        # Check PDF matcher setup
        if self.pdf_matcher is None:
            validation_results["warnings"].append("PDF matcher not initialized (no studies directory set)")
        
        return validation_results
    
    def _is_video_file(self, file_path: str) -> bool:
        """
        Check if the file is a video file or audio-only file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if it's a video file with video streams, False if audio-only
        """
        import mimetypes
        
        # Get file extension and mime type
        file_extension = Path(file_path).suffix.lower()
        mime_type, _ = mimetypes.guess_type(file_path)
        
        # Check for common audio-only extensions
        audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
        if file_extension in audio_extensions:
            return False
        
        # Check for common video extensions
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
        if file_extension in video_extensions:
            return True
        
        # Check mime type if extension is ambiguous
        if mime_type:
            if mime_type.startswith('audio/'):
                return False
            elif mime_type.startswith('video/'):
                return True
        
        # Default to video for unknown types (will fail gracefully in OpenCV)
        logger.warning(f"Unknown file type for {file_path}, treating as video")
        return True