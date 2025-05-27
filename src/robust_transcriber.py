"""
Robust Enhanced Transcriber with better handling of problematic audio
Fixes the issues with silent audio sections and overly restrictive filtering
"""

import whisper
import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pydub import AudioSegment, silence
import os
from pathlib import Path
import tqdm

from .config import LANGUAGE_MAP, DEFAULT_CONFIG
from .utils import format_timestamp

logger = logging.getLogger(__name__)

class RobustAudioTranscriber:
    """
    Robust audio transcriber that handles problematic audio properly.
    Fixes issues with silent sections and overly restrictive filtering.
    """
    
    def __init__(self, 
                 model_name: str = "large-v3", 
                 language: str = "german", 
                 device: Optional[str] = None,
                 config: Optional[Dict] = None):
        """
        Initialize the robust audio transcriber.
        
        Args:
            model_name: Whisper model to use
            language: Language for transcription
            device: Device for inference (cpu, cuda, etc.)
            config: Additional configuration options
        """
        self.model_name = model_name
        self.language = LANGUAGE_MAP.get(language.lower(), "de")
        self.device = device
        self.model = None
        
        # Robust config that handles problematic audio
        self.config = self._get_robust_config()
        if config:
            self.config.update(config)
        
        logger.info(f"Initialized RobustAudioTranscriber")
    
    def _get_robust_config(self) -> Dict:
        """Get robust configuration that handles various audio types."""
        return {
            # More tolerant segmentation
            'min_silence_len': 1500,        # 1.5 seconds
            'padding': 1000,                # 1 second padding
            'silence_adjustment': 3.0,      # Conservative but not extreme
            
            # Flexible segment lengths
            'max_segment_length': 300000,   # 5 minutes max per segment
            'min_segment_length': 2000,     # 2 seconds minimum (was 10000!)
            'overlap_duration': 2000,       # 2 seconds overlap
            
            # Audio analysis settings
            'enable_vad': False,            # Disable VAD for problematic audio
            'normalize_audio': False,       # Don't normalize by default
            'robust_silence_detection': True,  # Use robust method
            
            # Whisper settings
            'use_beam_search': False,       # Keep it simple for now
            'best_of': 1,                   # Single attempt
            'patience': 1.0,                # Standard patience
            'length_penalty': 1.0,          # No length bias
            
            # Post-processing
            'merge_short_segments': True,   # Still merge very short ones
            'validate_transcription': True,
            'cleanup_segments': True
        }
    
    def load_model(self) -> None:
        """Load the Whisper model if not already loaded."""
        if self.model is None:
            import torch
            
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Whisper model: {self.model_name} on {self.device}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info("Model loaded successfully")
    
    def robust_audio_analysis(self, audio_file: str) -> Dict:
        """
        Robust audio analysis that handles silent sections properly.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Analysis results with safe fallbacks
        """
        logger.info("Performing robust audio analysis...")
        
        audio = AudioSegment.from_file(audio_file)
        duration = len(audio) / 1000.0  # seconds
        
        # Analyze in chunks, filtering out silent ones
        chunk_size = 5000  # 5 second chunks
        volumes = []
        speech_chunks = 0
        total_chunks = 0
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) > 1000:  # At least 1 second
                chunk_db = chunk.dBFS
                total_chunks += 1
                
                # Only include non-silent chunks in volume analysis
                if chunk_db > -60:  # Not completely silent
                    volumes.append(chunk_db)
                    if chunk_db > -40:  # Likely speech
                        speech_chunks += 1
        
        # Calculate safe statistics
        if volumes:
            mean_volume = np.mean(volumes)
            volume_std = np.std(volumes)
            quiet_threshold = mean_volume - volume_std
        else:
            # Fallback for completely silent audio
            mean_volume = -60
            volume_std = 10
            quiet_threshold = -70
        
        # Calculate ratios safely
        speech_ratio = speech_chunks / total_chunks if total_chunks > 0 else 0
        quiet_ratio = 1 - speech_ratio
        
        # Determine speaker type more conservatively
        if speech_ratio < 0.3:
            speaker_type = "very_sparse_speech"
            recommended_silence_len = 3000
            recommended_padding = 2000
        elif speech_ratio < 0.5:
            speaker_type = "sparse_speech" 
            recommended_silence_len = 2500
            recommended_padding = 1500
        elif speech_ratio < 0.7:
            speaker_type = "moderate_speech"
            recommended_silence_len = 2000
            recommended_padding = 1000
        else:
            speaker_type = "dense_speech"
            recommended_silence_len = 1500
            recommended_padding = 750
        
        analysis = {
            "duration_seconds": duration,
            "mean_volume_db": mean_volume,
            "volume_std": volume_std,
            "quiet_ratio": quiet_ratio,
            "speech_ratio": speech_ratio,
            "speaker_type": speaker_type,
            "recommended_min_silence_len": recommended_silence_len,
            "recommended_padding": recommended_padding,
            "estimated_speech_time": speech_chunks * (chunk_size / 1000),
            "total_chunks_analyzed": total_chunks,
            "speech_chunks": speech_chunks,
            "analysis_method": "robust"
        }
        
        logger.info(f"Robust analysis: {speaker_type}, speech_ratio: {speech_ratio:.2f}")
        return analysis
    
    def robust_silence_detection(self, audio: AudioSegment, analysis: Dict) -> List[Tuple[int, int]]:
        """
        Robust silence detection with multiple fallback strategies.
        
        Args:
            audio: AudioSegment object
            analysis: Audio analysis results
            
        Returns:
            List of (start, end) tuples for non-silent segments
        """
        # Start with recommended parameters
        min_silence_len = analysis["recommended_min_silence_len"]
        silence_thresh = analysis["mean_volume_db"] + self.config['silence_adjustment']
        
        logger.info(f"Trying robust silence detection: {min_silence_len}ms, {silence_thresh:.1f}dB")
        
        try:
            # Primary attempt
            nonsilent_ranges = silence.detect_nonsilent(
                audio, 
                min_silence_len=min_silence_len, 
                silence_thresh=silence_thresh
            )
            
            if nonsilent_ranges and len(nonsilent_ranges) > 0:
                logger.info(f"Primary detection successful: {len(nonsilent_ranges)} segments")
                return nonsilent_ranges
            
        except Exception as e:
            logger.warning(f"Primary silence detection failed: {e}")
        
        # Fallback strategies
        fallback_configs = [
            {"min_silence_len": 1000, "thresh_offset": 5},   # More tolerant
            {"min_silence_len": 500, "thresh_offset": 8},    # Very tolerant
            {"min_silence_len": 2000, "thresh_offset": 10},  # Conservative but very tolerant volume
            {"min_silence_len": 100, "thresh_offset": 3},    # Aggressive
        ]
        
        for i, config in enumerate(fallback_configs):
            try:
                thresh = analysis["mean_volume_db"] + config["thresh_offset"]
                logger.info(f"Fallback {i+1}: {config['min_silence_len']}ms, {thresh:.1f}dB")
                
                nonsilent_ranges = silence.detect_nonsilent(
                    audio,
                    min_silence_len=config["min_silence_len"],
                    silence_thresh=thresh
                )
                
                if nonsilent_ranges and len(nonsilent_ranges) > 0:
                    logger.info(f"Fallback {i+1} successful: {len(nonsilent_ranges)} segments")
                    return nonsilent_ranges
                    
            except Exception as e:
                logger.warning(f"Fallback {i+1} failed: {e}")
                continue
        
        # Last resort: use the entire audio as one segment
        logger.warning("All silence detection failed, using entire audio as one segment")
        return [(0, len(audio))]
    
    def create_robust_segments(self, audio_file: str, nonsilent_ranges: List[Tuple[int, int]], 
                             analysis: Dict) -> List[Tuple[str, int, int]]:
        """
        Create audio segments with robust handling of short segments.
        
        Args:
            audio_file: Path to audio file
            nonsilent_ranges: List of detected speech ranges
            analysis: Audio analysis results
            
        Returns:
            List of (segment_file_path, start_time_ms, end_time_ms)
        """
        audio = AudioSegment.from_file(audio_file)
        total_duration = len(audio)
        
        padding = analysis["recommended_padding"]
        overlap = self.config['overlap_duration']
        max_length = self.config['max_segment_length']
        min_length = self.config['min_segment_length']
        
        segments = []
        base_name = Path(audio_file).stem
        
        logger.info(f"Creating segments with min_length={min_length}ms, padding={padding}ms")
        
        for i, (start, end) in enumerate(nonsilent_ranges):
            segment_length = end - start
            
            # Much more lenient minimum length check
            if segment_length < min_length:
                # Try to merge with nearby segments instead of skipping
                merged = False
                
                # Look for next segment to merge with
                if i < len(nonsilent_ranges) - 1:
                    next_start, next_end = nonsilent_ranges[i + 1]
                    gap = next_start - end
                    
                    # Merge if gap is small
                    if gap < self.config['min_silence_len'] * 2:
                        # Create merged segment
                        merged_start = max(0, start - padding)
                        merged_end = min(total_duration, next_end + padding)
                        segment_prefix = self.config.get('segment_prefix', 'segment')
                        
                        segment = audio[merged_start:merged_end]
                        segment_file = f"{audio_file}_{segment_prefix}_{len(segments):03d}.wav"
                        segment.export(segment_file, format="wav")
                        
                        segments.append((segment_file, merged_start, merged_end))
                        merged = True
                        
                        logger.debug(f"Merged segments {i} and {i+1}: {merged_start}-{merged_end}")
                
                if not merged:
                    # Create segment anyway if it's not too short
                    if segment_length >= 1000:  # At least 1 second
                        padded_start = max(0, start - padding)
                        padded_end = min(total_duration, end + padding)
                        segment_prefix = self.config.get('segment_prefix', 'segment')
                        
                        segment = audio[padded_start:padded_end]
                        segment_file = f"{audio_file}_{segment_prefix}_{len(segments):03d}.wav"
                        segment.export(segment_file, format="wav")
                        
                        segments.append((segment_file, padded_start, padded_end))
                        logger.debug(f"Kept short segment {i}: {segment_length}ms")
                    else:
                        logger.debug(f"Skipped very short segment {i}: {segment_length}ms")
                
                continue
            
            # Handle long segments
            if segment_length > max_length:
                # Split into overlapping chunks
                chunk_start = start
                chunk_index = 0
                segment_prefix = self.config.get('segment_prefix', 'segment')
                
                while chunk_start < end:
                    chunk_end = min(chunk_start + max_length, end)
                    
                    # Apply padding
                    padded_start = max(0, chunk_start - padding)
                    padded_end = min(total_duration, chunk_end + padding)
                    
                    segment = audio[padded_start:padded_end]
                    segment_file = f"{audio_file}_{segment_prefix}_{len(segments):03d}_{chunk_index}.wav"
                    segment.export(segment_file, format="wav")
                    
                    segments.append((segment_file, padded_start, padded_end))
                    
                    # Move to next chunk with overlap
                    chunk_start = chunk_end - overlap
                    chunk_index += 1
            else:
                # Normal segment with padding
                padded_start = max(0, start - padding)
                padded_end = min(total_duration, end + padding)
                segment_prefix = self.config.get('segment_prefix', 'segment')
                
                segment = audio[padded_start:padded_end]
                segment_file = f"{audio_file}_{segment_prefix}_{len(segments):03d}.wav"
                segment.export(segment_file, format="wav")
                
                segments.append((segment_file, padded_start, padded_end))
        
        logger.info(f"Created {len(segments)} robust segments")
        return segments
    
    def transcribe_audio_file_robust(self, audio_file: str) -> Dict[str, Any]:
        """
        Robust transcription workflow that handles problematic audio.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Complete transcription results
        """
        start_time = time.time()
        
        logger.info(f"Starting robust transcription for: {audio_file}")
        
        # Load model
        self.load_model()
        
        # Step 1: Robust audio analysis
        analysis = self.robust_audio_analysis(audio_file)
        
        # Step 2: Robust silence detection
        audio = AudioSegment.from_file(audio_file)
        nonsilent_ranges = self.robust_silence_detection(audio, analysis)
        
        if not nonsilent_ranges:
            logger.error("No speech ranges detected even with robust method")
            return {
                "segments": [],
                "full_text": "",
                "total_duration": 0,
                "error": "No speech detected with robust method",
                "analysis": analysis
            }
        
        # Step 3: Create robust segments
        segments = self.create_robust_segments(audio_file, nonsilent_ranges, analysis)
        
        if not segments:
            logger.error("No valid segments created with robust method")
            return {
                "segments": [],
                "full_text": "",
                "total_duration": 0,
                "error": "No valid segments created with robust method",
                "analysis": analysis
            }
        
        # Step 4: Transcribe segments
        logger.info(f"Transcribing {len(segments)} segments with robust method...")
        results = []
        
        for i, segment_info in enumerate(tqdm.tqdm(segments, desc="Robust transcription")):
            segment_file, start_time_ms, end_time_ms = segment_info
            
            try:
                # Simple, reliable transcription
                result = self.model.transcribe(
                    segment_file,
                    language=self.language,
                    verbose=False,
                    fp16=True
                )
                
                segment_result = {
                    "segment_id": i,
                    "text": result["text"].strip(),
                    "confidence": float(result.get("confidence", 0.0)),
                    "start_time": start_time_ms,
                    "end_time": end_time_ms,
                    "duration": end_time_ms - start_time_ms,
                    "segment_file": segment_file,
                    "language_detected": result.get("language", self.language),
                    "method": "robust"
                }
                
                results.append(segment_result)
                
            except Exception as e:
                logger.error(f"Error transcribing segment {i}: {e}")
                error_result = {
                    "segment_id": i,
                    "text": "",
                    "error": str(e),
                    "start_time": start_time_ms,
                    "end_time": end_time_ms,
                    "segment_file": segment_file
                }
                results.append(error_result)
            
            # Cleanup segment file if configured
            if self.config['cleanup_segments']:
                try:
                    os.remove(segment_file)
                except Exception as e:
                    logger.warning(f"Could not remove segment file {segment_file}: {e}")
        
        # Compile final results
        processing_time = time.time() - start_time
        
        successful_segments = [r for r in results if not r.get("error") and r.get("text")]
        failed_segments = [r for r in results if r.get("error")]
        
        full_text = " ".join([r["text"] for r in successful_segments])
        total_duration = max([r.get("end_time", 0) for r in results], default=0)
        
        transcription_result = {
            "segments": results,
            "full_text": full_text,
            "total_duration": total_duration,
            "language": self.language,
            "model": self.model_name,
            "processing_time_seconds": processing_time,
            "segments_total": len(segments),
            "segments_successful": len(successful_segments),
            "segments_failed": len(failed_segments),
            "speech_analysis": analysis,
            "config_used": self.config.copy(),
            "method": "robust_enhanced"
        }
        
        logger.info(f"Robust transcription completed in {processing_time:.2f}s")
        logger.info(f"Success rate: {len(successful_segments)}/{len(segments)} segments")
        
        return transcription_result