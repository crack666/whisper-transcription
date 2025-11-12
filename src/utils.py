"""
Utility functions for the study material processor
"""

import os
import subprocess
import logging
from typing import Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

def format_timestamp(ms: int) -> str:
    """
    Format milliseconds timestamp to HH:MM:SS.mmm.
    
    Args:
        ms: Timestamp in milliseconds
        
    Returns:
        Formatted timestamp string
    """
    seconds, ms = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"

def format_timestamp_seconds(seconds: float) -> str:
    """
    Format seconds timestamp to HH:MM:SS.mmm
    
    Args:
        seconds: Timestamp in seconds
        
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object of the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def extract_audio_from_video(video_path: str, audio_path: Optional[str] = None) -> str:
    """
    Extract audio from video using ffmpeg.
    
    Args:
        video_path: Path to video file
        audio_path: Output audio path (optional, defaults to video_path.wav)
        
    Returns:
        Path to extracted audio file
        
    Raises:
        subprocess.CalledProcessError: If ffmpeg fails
        FileNotFoundError: If ffmpeg not found
    """
    if audio_path is None:
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
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Audio extracted to: {audio_path}")
        return audio_path
    except FileNotFoundError:
        raise FileNotFoundError("ffmpeg not found. Please install ffmpeg to extract audio from videos.")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg stderr: {e.stderr}")
        raise RuntimeError(f"Failed to extract audio: {e}")

def check_dependencies():
    """
    Check if required dependencies are available.
    
    Raises:
        ImportError: If required packages are missing
        FileNotFoundError: If external tools are missing
    """
    required_packages = [
        'whisper', 'cv2', 'numpy', 'PIL', 'skimage', 'PyPDF2', 
        'pydub', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(f"Missing required packages: {', '.join(missing_packages)}")
    
    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise FileNotFoundError("ffmpeg not found. Please install ffmpeg.")
    
    logger.info("All dependencies are available")

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system operations.
    Preserves spaces for better readability in file explorer.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace problematic characters
    import re
    # Keep alphanumeric, spaces, dots, dashes, underscores
    # Replace only truly problematic characters (like /, \, :, *, ?, ", <, >, |)
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Collapse multiple spaces into single space
    sanitized = re.sub(r'\s+', ' ', sanitized)
    # Remove leading/trailing dots or spaces
    sanitized = sanitized.strip(' ._')
    return sanitized

def get_video_info(video_path: str) -> dict:
    """
    Get basic information about a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration_seconds': 0
    }
    
    if info['fps'] > 0:
        info['duration_seconds'] = info['frame_count'] / info['fps']
    
    cap.release()
    return info

def estimate_processing_time(video_info: dict, model_name: str = "large-v3") -> dict:
    """
    Estimate processing time based on video duration and model.
    
    Args:
        video_info: Video information from get_video_info
        model_name: Whisper model name
        
    Returns:
        Dictionary with time estimates
    """
    duration_minutes = video_info['duration_seconds'] / 60
    
    # Rough estimates based on model size (times are per minute of video)
    model_factors = {
        'tiny': 0.1,
        'base': 0.2,
        'small': 0.4,
        'medium': 0.8,
        'large': 1.5,
        'large-v2': 1.5,
        'large-v3': 1.8
    }
    
    factor = model_factors.get(model_name, 1.0)
    
    # Add overhead for video processing and screenshot extraction
    transcription_time = duration_minutes * factor
    screenshot_time = duration_minutes * 0.1  # Rough estimate
    total_time = transcription_time + screenshot_time + 1  # +1 for overhead
    
    return {
        'video_duration_minutes': duration_minutes,
        'estimated_transcription_minutes': transcription_time,
        'estimated_screenshot_minutes': screenshot_time,
        'estimated_total_minutes': total_time,
        'model_factor': factor
    }