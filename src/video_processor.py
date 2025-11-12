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
from tqdm import tqdm

from config import DEFAULT_CONFIG
from utils import format_timestamp_seconds, ensure_directory, sanitize_filename

logger = logging.getLogger(__name__)

class VideoScreenshotExtractor:
    """
    Extract screenshots from videos at scene changes with intelligent detection.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 min_time_between_shots: float = 2.0,
                 config: Optional[Dict] = None,
                 speech_segments: Optional[List[Dict]] = None): # Added speech_segments
        """
        Initialize the video screenshot extractor.
        
        Args:
            similarity_threshold: Threshold for scene change detection (0.0-1.0)
            min_time_between_shots: Minimum time between screenshots in seconds
            config: Additional configuration options
            speech_segments: List of speech segments with 'start' and 'end' times.
        """
        self.similarity_threshold = similarity_threshold
        self.min_time_between_shots = min_time_between_shots
        self.speech_segments = speech_segments if speech_segments else []
        
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
        Extract screenshots from video based on speech segments and visual changes.
        
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
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            logger.error(f"Video FPS is 0, cannot process: {video_path}")
            cap.release()
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps
        
        logger.info(f"Processing video: {duration_seconds:.1f}s, {fps:.1f} FPS, {total_frames} frames")
        
        screenshots = []
        # Extract screenshot at the beginning of each speech segment
        segment_start_timestamps = sorted(list(set([seg['start'] for seg in self.speech_segments if 'start' in seg])))
        
        # Calculate total frames to process for progress tracking (clip to video duration)
        total_frames_to_process = 0
        for segment in self.speech_segments:
            segment_start = segment.get('start', 0)
            segment_end = segment.get('end', 0)
            
            # Skip segments completely outside video duration
            if segment_start >= duration_seconds:
                continue
            
            # Clip segment end to video duration
            segment_end = min(segment_end, duration_seconds)
            
            # Only count frames within valid range
            if segment_end > segment_start:
                segment_frames = int((segment_end - segment_start) * fps)
                total_frames_to_process += segment_frames
        
        # Progress bar for screenshot extraction
        print(f"\n📸 Extracting screenshots from {len(self.speech_segments)} speech segments...")
        print(f"   Video duration: {duration_seconds:.1f}s, Processing {total_frames_to_process:,} frames")
        print(f"   Phase 1/2: Capturing segment start frames ({len(segment_start_timestamps)} positions)...")
        
        pbar = tqdm(total=total_frames_to_process, unit='frames', desc="Screenshot extraction", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        # Extract screenshots at segment starts (with progress updates)
        segment_start_progress = 0
        for i, timestamp in enumerate(segment_start_timestamps):
            # Skip timestamps beyond video duration
            if timestamp >= duration_seconds:
                logger.debug(f"Skipping segment start at {timestamp:.2f}s (beyond video duration {duration_seconds:.2f}s)")
                continue
                
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                screenshot_info = self._save_screenshot(
                    frame, video_name, f"segment_start_{i}", timestamp, 
                    output_path, 1.0 # Similarity score 1.0 for forced captures
                )
                screenshots.append(screenshot_info)
                # Update progress bar with actual progress
                segment_start_progress += 1
                pbar.set_postfix({'phase': f'segment starts ({segment_start_progress}/{len(segment_start_timestamps)})', 'screenshots': len(screenshots)}, refresh=True)
            else:
                logger.warning(f"Could not extract frame for segment start at {timestamp:.2f}s")

        # Existing logic for detecting visual changes within segments or throughout the video
        # This part needs to be adapted to consider speech segments' durations
        
        print(f"   Phase 2/2: Detecting visual changes within segments...")
        
        frame_interval = int(fps * self.config['frame_check_interval'])
        resize_dimensions = self.config['resize_for_comparison']
        
        previous_frame_gray = None
        frame_count = 0
        last_screenshot_time = -self.min_time_between_shots # Initialize to allow first capture

        # Iterate through speech segments to detect changes within them
        for segment in self.speech_segments:
            segment_start_time = segment.get('start')
            segment_end_time = segment.get('end')

            if segment_start_time is None or segment_end_time is None:
                logger.warning(f"Segment missing start or end time: {segment}")
                continue

            # Skip segments that start beyond video duration
            if segment_start_time >= duration_seconds:
                logger.debug(f"Skipping segment starting at {segment_start_time:.2f}s (beyond video duration)")
                continue
            
            # Clip segment end to video duration
            segment_end_time = min(segment_end_time, duration_seconds)

            # Set video capture to the start of the current segment
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(segment_start_time * fps))
            current_time_in_segment = segment_start_time
            
            # Process frames within the current segment
            frames_processed_in_segment = 0
            while current_time_in_segment < segment_end_time:
                ret, frame = cap.read()
                if not ret:
                    break # End of video or error

                frames_processed_in_segment += 1
                # Update progress bar every N frames to avoid slowdown
                if frames_processed_in_segment % 30 == 0:
                    pbar.update(30)

                # Only process frames at the specified interval for efficiency
                # And ensure we are within the current segment's time boundaries
                current_frame_num_abs = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) -1 # current frame number
                
                if current_frame_num_abs % frame_interval == 0:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    resized_gray = cv2.resize(gray_frame, resize_dimensions)
                    
                    similarity_score = 1.0 # Default for first frame of segment or if no previous
                    if previous_frame_gray is not None:
                        similarity_score = ssim(resized_gray, previous_frame_gray)

                    # Capture if significant change and enough time has passed since last screenshot
                    if (similarity_score < self.similarity_threshold and
                        current_time_in_segment - last_screenshot_time >= self.min_time_between_shots):
                        
                        # Avoid duplicate if a segment-start screenshot was already taken at this exact time
                        is_duplicate_of_segment_start = any(
                            s['timestamp'] == current_time_in_segment and s['filename'].startswith(f"{video_name}_screenshot_segment_start_")
                            for s in screenshots
                        )
                        if not is_duplicate_of_segment_start:
                            screenshot_info = self._save_screenshot(
                                frame, video_name, f"change_{len(screenshots)}", current_time_in_segment,
                                output_path, similarity_score
                            )
                            screenshots.append(screenshot_info)
                            last_screenshot_time = current_time_in_segment
                            # Update progress bar description with screenshot count
                            pbar.set_postfix({'screenshots': len(screenshots)}, refresh=False)
                    
                    previous_frame_gray = resized_gray
                
                current_time_in_segment = (cap.get(cv2.CAP_PROP_POS_FRAMES)) / fps
                if current_time_in_segment >= segment_end_time : # ensure we don't go past segment end
                    break
            
            # Update progress bar with remaining frames from this segment
            remaining_frames = frames_processed_in_segment % 30
            if remaining_frames > 0:
                pbar.update(remaining_frames)

        # Close progress bar
        pbar.close()
        
        # Ensure first frame of the video is captured if no speech segments and no other screenshots
        if not screenshots and not self.speech_segments:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if ret:
                screenshot_info = self._save_screenshot(
                    frame, video_name, "initial_0", 0.0, output_path, 1.0
                )
                screenshots.append(screenshot_info)
        
        cap.release()
        
        # Sort all screenshots by timestamp
        screenshots.sort(key=lambda s: s['timestamp'])
        # Re-index after sorting to ensure sequential numbering for change-detected ones if desired,
        # or maintain unique IDs like "segment_start_X" and "change_Y"
        # For simplicity, we'll keep the mixed naming for now. If a strict numerical index is needed:
        # for idx, s_info in enumerate(screenshots):
        # s_info['index'] = idx 
        # s_info['filename'] = f"{video_name}_screenshot_{idx:03d}_{format_timestamp_seconds(s_info['timestamp']).replace(':', '-')}.jpg"
        # s_info['filepath'] = str(output_path / s_info['filename'])
        # This would require re-saving or renaming files if filenames need to reflect the new index.
        # Current _save_screenshot uses the passed index directly.

        print(f"✅ Screenshot extraction completed: {len(screenshots)} screenshots saved")
        logger.info(f"Extracted {len(screenshots)} screenshots from {video_name}")
        return screenshots
    
    def _save_screenshot(self, frame: np.ndarray, video_name: str, index: any, # Index can be string or int
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
        # Adapt filename if index is a string (e.g., "segment_start_0" or "change_12")
        if isinstance(index, str):
             filename = f"{video_name}_screenshot_{index}_{timestamp_str}.jpg"
        else: # Original behavior if index is an int
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