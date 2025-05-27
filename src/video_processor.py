"""
Video processing module for screenshot extraction and scene analysis
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os
from skimage.metrics import structural_similarity as ssim

from .config import DEFAULT_CONFIG
from .utils import format_timestamp_seconds, ensure_directory, sanitize_filename

logger = logging.getLogger(__name__)

class VideoScreenshotExtractor:
    """
    Extract screenshots from videos at scene changes with intelligent detection.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 min_time_between_shots: float = 2.0,
                 config: Optional[Dict] = None):
        """
        Initialize the video screenshot extractor.
        
        Args:
            similarity_threshold: Threshold for scene change detection (0.0-1.0)
            min_time_between_shots: Minimum time between screenshots in seconds
            config: Additional configuration options
        """
        self.similarity_threshold = similarity_threshold
        self.min_time_between_shots = min_time_between_shots
        
        # Merge config with defaults
        self.config = DEFAULT_CONFIG['screenshots'].copy()
        if config:
            self.config.update(config)
        
        # Override with direct parameters
        self.config['similarity_threshold'] = similarity_threshold
        self.config['min_time_between_shots'] = min_time_between_shots
        
        logger.info(f"Initialized VideoScreenshotExtractor with threshold: {similarity_threshold}")
    
    def extract_screenshots(self, video_path: str, output_dir: str) -> List[Dict]:
        """
        Extract screenshots from video at detected scene changes.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save screenshots
            
        Returns:
            List of screenshot information dictionaries
        """
        output_path = ensure_directory(output_dir)
        video_name = sanitize_filename(Path(video_path).stem)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Processing video: {duration_seconds:.1f}s, {fps:.1f} FPS, {total_frames} frames")
        
        frame_interval = int(fps * self.config['frame_check_interval'])
        resize_dimensions = self.config['resize_for_comparison']
        
        screenshots = []
        previous_frame_gray = None
        frame_count = 0
        last_screenshot_time = -self.min_time_between_shots
        
        # Always capture first frame
        first_frame_captured = False
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_count / fps
                
                # Check frame at intervals
                if frame_count % frame_interval == 0:
                    # Convert to grayscale and resize for comparison
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    resized_gray = cv2.resize(gray_frame, resize_dimensions)
                    
                    should_capture = False
                    similarity_score = 1.0
                    
                    if not first_frame_captured:
                        # Always capture first frame
                        should_capture = True
                        first_frame_captured = True
                        logger.debug("Capturing first frame")
                    elif previous_frame_gray is not None:
                        # Calculate similarity with previous frame
                        similarity_score = ssim(resized_gray, previous_frame_gray)
                        
                        # Check for scene change
                        if (similarity_score < self.similarity_threshold and 
                            timestamp - last_screenshot_time >= self.min_time_between_shots):
                            should_capture = True
                            logger.debug(f"Scene change at {timestamp:.2f}s (similarity: {similarity_score:.3f})")
                    
                    if should_capture:
                        screenshot_info = self._save_screenshot(
                            frame, video_name, len(screenshots), timestamp, 
                            output_path, similarity_score
                        )
                        screenshots.append(screenshot_info)
                        last_screenshot_time = timestamp
                    
                    previous_frame_gray = resized_gray
                
                frame_count += 1
        
        finally:
            cap.release()
        
        # Ensure we have at least one screenshot
        if not screenshots:
            logger.warning("No screenshots captured, creating one from first frame")
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                screenshot_info = self._save_screenshot(
                    frame, video_name, 0, 0.0, output_path, 1.0
                )
                screenshots.append(screenshot_info)
            cap.release()
        
        logger.info(f"Extracted {len(screenshots)} screenshots from {video_name}")
        return screenshots
    
    def _save_screenshot(self, frame: np.ndarray, video_name: str, index: int, 
                        timestamp: float, output_dir: Path, similarity_score: float) -> Dict:
        """
        Save a screenshot and return its metadata.
        
        Args:
            frame: Video frame to save
            video_name: Name of the source video
            index: Screenshot index
            timestamp: Timestamp in seconds
            output_dir: Output directory
            similarity_score: Similarity score that triggered the capture
            
        Returns:
            Screenshot information dictionary
        """
        # Generate filename with timestamp
        timestamp_str = format_timestamp_seconds(timestamp).replace(':', '-')
        filename = f"{video_name}_screenshot_{index:03d}_{timestamp_str}.jpg"
        filepath = output_dir / filename
        
        # Save image with good quality
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        screenshot_info = {
            "index": index,
            "timestamp": timestamp,
            "timestamp_formatted": format_timestamp_seconds(timestamp),
            "filename": filename,
            "filepath": str(filepath),
            "similarity_score": similarity_score,
            "file_size_bytes": filepath.stat().st_size if filepath.exists() else 0
        }
        
        logger.debug(f"Saved screenshot {index}: {filename}")
        return screenshot_info
    
    def analyze_video_complexity(self, video_path: str, sample_duration: int = 60) -> Dict:
        """
        Analyze video complexity to suggest optimal screenshot settings.
        
        Args:
            video_path: Path to video file
            sample_duration: Duration in seconds to sample for analysis
            
        Returns:
            Analysis results with recommendations
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video"}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Sample frames for analysis
        sample_frames = min(sample_duration * fps, total_frames) if fps > 0 else total_frames
        frame_step = max(1, int(total_frames / sample_frames))
        
        similarities = []
        previous_gray = None
        frames_analyzed = 0
        
        for frame_num in range(0, total_frames, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 240))
            
            if previous_gray is not None:
                similarity = ssim(gray, previous_gray)
                similarities.append(similarity)
            
            previous_gray = gray
            frames_analyzed += 1
        
        cap.release()
        
        if not similarities:
            return {"error": "Could not analyze video"}
        
        # Calculate statistics
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        min_similarity = np.min(similarities)
        
        # Generate recommendations
        if avg_similarity > 0.95:
            complexity = "low"
            recommended_threshold = 0.90
            recommended_interval = 5.0
        elif avg_similarity > 0.85:
            complexity = "medium"
            recommended_threshold = 0.85
            recommended_interval = 3.0
        else:
            complexity = "high"
            recommended_threshold = 0.80
            recommended_interval = 2.0
        
        return {
            "duration_seconds": duration,
            "frames_analyzed": frames_analyzed,
            "average_similarity": avg_similarity,
            "similarity_std": std_similarity,
            "min_similarity": min_similarity,
            "complexity": complexity,
            "recommended_threshold": recommended_threshold,
            "recommended_interval": recommended_interval,
            "estimated_screenshots": int(duration / recommended_interval)
        }
    
    def extract_frames_at_timestamps(self, video_path: str, timestamps: List[float], 
                                   output_dir: str) -> List[Dict]:
        """
        Extract frames at specific timestamps.
        
        Args:
            video_path: Path to video file
            timestamps: List of timestamps in seconds
            output_dir: Output directory
            
        Returns:
            List of extracted frame information
        """
        output_path = ensure_directory(output_dir)
        video_name = sanitize_filename(Path(video_path).stem)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        for i, timestamp in enumerate(timestamps):
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if ret:
                frame_info = self._save_screenshot(
                    frame, video_name, i, timestamp, output_path, 1.0
                )
                frames.append(frame_info)
            else:
                logger.warning(f"Could not extract frame at {timestamp}s")
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames at specified timestamps")
        return frames
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        Get comprehensive video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video information dictionary
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video"}
        
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
            "file_size_bytes": Path(video_path).stat().st_size
        }
        
        if info['fps'] > 0:
            info['duration_seconds'] = info['frame_count'] / info['fps']
            info['duration_formatted'] = format_timestamp_seconds(info['duration_seconds'])
        else:
            info['duration_seconds'] = 0
            info['duration_formatted'] = "00:00:00.000"
        
        # Convert fourcc to readable format
        fourcc_bytes = info['fourcc'].to_bytes(4, byteorder='little')
        info['codec'] = fourcc_bytes.decode('ascii', errors='ignore')
        
        cap.release()
        return info