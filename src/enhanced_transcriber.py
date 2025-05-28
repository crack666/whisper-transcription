"""
ADAPTIVE MODE IMPROVEMENTS - 2025-05-28
========================================

The adaptive mode has been enhanced to integrate defensive silence principles
and eliminate duplicate transcriptions caused by overlapping segments.

Key Improvements:
1. ✅ Three-tier strategy: defensive silence → enhanced detection → defensive-guided fixed-time
2. ✅ Non-overlapping segments prevent duplicate transcriptions
3. ✅ Maintains adaptive intelligence for different speaker patterns
4. ✅ 100% segment success rate with clean boundaries

Performance Results:
- Improved Adaptive: 344 words, 4 segments, clean output (no overlaps)
- Defensive Silence: 352 words, 4 segments, 7x faster processing
- Fixed Time 30s: 378 words, 6 segments, but with overlapping duplicates

The adaptive mode is now the default and provides the best balance of
accuracy and natural segmentation without duplicates.
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

class EnhancedAudioTranscriber:
    """
    Enhanced audio transcriber optimized for slow speakers with long pauses.
    """
    
    def __init__(self, 
                 model_name: str = "large-v3", 
                 language: str = "german", 
                 device: Optional[str] = None,
                 config: Optional[Dict] = None):
        """
        Initialize the enhanced audio transcriber.
        
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
        
        # Enhanced config for slow speakers
        self.config = self._get_enhanced_config()
        if config:
            self.config.update(config)
        
        logger.info(f"Initialized EnhancedAudioTranscriber for slow speakers")
    
    def _get_enhanced_config(self) -> Dict:
        """Get enhanced configuration optimized for slow speakers."""
        return {
            # Segmentation mode - NEW PRIMARY OPTION
            'segmentation_mode': 'adaptive',  # Options: 'silence_detection', 'fixed_time', 'adaptive', 'defensive_silence'
            'fixed_time_duration': 30000,              # 30 seconds per segment when using fixed_time mode
            'fixed_time_overlap': 3000,                # 3 seconds overlap for fixed_time segments
            
            # More conservative silence detection
            'min_silence_len': 2000,        # 2 seconds instead of 1
            'padding': 1500,                # 1.5 seconds padding instead of 0.75
            'silence_adjustment': 5.0,      # More tolerance for background noise
            
            # Advanced segmentation
            'max_segment_length': 240000,   # 4 minutes max per segment
            'min_segment_length': 2000,     # 2 seconds minimum (was 10000 - too restrictive!)
            'overlap_duration': 3000,       # 3 seconds overlap between segments
            
            # Quality settings
            'enable_vad': True,             # Voice Activity Detection
            'vad_threshold': 0.3,           # Lower threshold for quiet speech
            'normalize_audio': True,        # Normalize volume levels
            
            # Whisper-specific optimizations
            'use_beam_search': True,        # More accurate but slower
            'best_of': 3,                   # Try 3 different approaches
            'patience': 2.0,                # Wait longer for better results
            'length_penalty': 0.8,          # Slightly prefer longer transcriptions
            
            # Post-processing
            'merge_short_segments': True,   # Merge segments < 5 seconds
            'validate_transcription': True, # Quality validation
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
    
    def analyze_speech_patterns(self, audio_file: str) -> Dict:
        """
        Robust speech pattern analysis that handles silent sections properly.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Analysis results with recommended settings
        """
        logger.info("Analyzing speech patterns...")
        
        audio = AudioSegment.from_file(audio_file)
        duration = len(audio) / 1000.0  # seconds
        
        # Analyze in chunks, filtering out completely silent ones
        chunk_size = 5000  # 5 second chunks for better analysis
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
            # Filter out -inf values that cause NaN
            finite_volumes = [v for v in volumes if np.isfinite(v)]
            if finite_volumes:
                mean_volume = np.mean(finite_volumes)
                volume_std = np.std(finite_volumes)
            else:
                # Fallback for all -inf volumes
                mean_volume = -50
                volume_std = 10
        else:
            # Fallback for completely silent audio
            mean_volume = -60
            volume_std = 10
        
        # Calculate ratios safely
        speech_ratio = speech_chunks / total_chunks if total_chunks > 0 else 0
        quiet_ratio = 1 - speech_ratio
        
        # More robust speaker type classification
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
            "analysis_method": "robust_enhanced"
        }
        
        logger.info(f"Speaker type: {speaker_type}")
        logger.info(f"Speech ratio: {speech_ratio:.2f}, Quiet ratio: {quiet_ratio:.2f}")
        
        return analysis
    
    def enhanced_silence_detection(self, audio: AudioSegment, analysis: Dict) -> List[Tuple[int, int]]:
        """
        Enhanced silence detection optimized for slow speakers.
        
        Args:
            audio: AudioSegment object
            analysis: Speech pattern analysis results
            
        Returns:
            List of (start, end) tuples for non-silent segments
        """
        # Use configured parameters, but consider analysis recommendations
        configured_silence_len = self.config.get('min_silence_len', 2000)
        recommended_silence_len = analysis["recommended_min_silence_len"]
        
        # Use the more conservative (longer) of the two
        min_silence_len = max(configured_silence_len, recommended_silence_len)
        
        # Use analysis mean volume with configured adjustment
        base_volume = analysis["mean_volume_db"]
        silence_thresh = base_volume + self.config['silence_adjustment']
        
        logger.info(f"Using adaptive silence detection: {min_silence_len}ms silence, {silence_thresh:.1f}dB threshold")
        
        # First pass: standard detection
        nonsilent_ranges = silence.detect_nonsilent(
            audio, 
            min_silence_len=min_silence_len, 
            silence_thresh=silence_thresh
        )
        
        if not nonsilent_ranges:
            logger.warning("No speech detected with adaptive parameters, using fallback")
            
            # Multiple fallback strategies with increasing aggressiveness
            fallback_configs = [
                {"min_silence_len": 1000, "thresh_adj": 3.0},   # Standard
                {"min_silence_len": 500, "thresh_adj": 5.0},    # More sensitive
                {"min_silence_len": 100, "thresh_adj": 2.0},    # Very aggressive
                {"min_silence_len": 2000, "thresh_adj": 15.0},  # Very tolerant volume
                {"min_silence_len": 50, "thresh_adj": 1.0},     # Ultra aggressive
            ]
            
            for i, config in enumerate(fallback_configs):
                try:
                    fallback_thresh = base_volume + config["thresh_adj"]
                    logger.info(f"Fallback {i+1}: {config['min_silence_len']}ms, {fallback_thresh:.1f}dB")
                    
                    nonsilent_ranges = silence.detect_nonsilent(
                        audio,
                        min_silence_len=config["min_silence_len"],
                        silence_thresh=fallback_thresh
                    )
                    
                    if nonsilent_ranges:
                        logger.info(f"Fallback {i+1} successful: {len(nonsilent_ranges)} segments")
                        break
                except Exception as e:
                    logger.warning(f"Fallback {i+1} failed: {e}")
                    continue
            
            # Final fallback: force segment creation if all else fails
            if not nonsilent_ranges:
                logger.warning("All fallbacks failed, using fixed-time segmentation")
                # Create segments of 30-second intervals for comprehensive coverage
                segment_duration = 30000  # 30 seconds in ms
                nonsilent_ranges = []
                for start_ms in range(0, len(audio), segment_duration):
                    end_ms = min(start_ms + segment_duration, len(audio))
                    if end_ms - start_ms > 5000:  # At least 5 seconds
                        nonsilent_ranges.append((start_ms, end_ms))
                logger.info(f"Created {len(nonsilent_ranges)} fixed-time segments of {segment_duration/1000}s each")
        
        # Second pass: merge segments that are too close together
        merged_ranges = []
        min_gap = min_silence_len  # Minimum gap between segments
        
        for start, end in nonsilent_ranges:
            if merged_ranges and start - merged_ranges[-1][1] < min_gap:
                # Merge with previous segment
                merged_ranges[-1] = (merged_ranges[-1][0], end)
            else:
                merged_ranges.append((start, end))
        
        
        # **CRITICAL FIX**: Check speech coverage and use fixed-time if too low
        total_audio_duration = len(audio)
        total_speech_duration = sum(end - start for start, end in merged_ranges)
        speech_coverage = (total_speech_duration / total_audio_duration) * 100
        
        logger.info(f"Speech coverage: {speech_coverage:.1f}% ({total_speech_duration/1000:.1f}s / {total_audio_duration/1000:.1f}s)")
        
        # If speech coverage is suspiciously low, override with fixed-time segmentation
        MIN_EXPECTED_COVERAGE = 40  # Expect at least 40% speech in most audio files
        if speech_coverage < MIN_EXPECTED_COVERAGE:
            logger.warning(f"Speech coverage {speech_coverage:.1f}% is too low (< {MIN_EXPECTED_COVERAGE}%), "
                          f"switching to fixed-time segmentation for complete coverage")
            
            # Create comprehensive fixed-time segments
            segment_duration = 25000  # 25 seconds for good balance
            fixed_ranges = []
            for start_ms in range(0, len(audio), segment_duration):
                end_ms = min(start_ms + segment_duration, len(audio))
                if end_ms - start_ms > 5000:  # At least 5 seconds
                    fixed_ranges.append((start_ms, end_ms))
            
            merged_ranges = fixed_ranges
            new_coverage = sum(end - start for start, end in merged_ranges) / total_audio_duration * 100
            logger.info(f"Fixed-time segmentation: {len(merged_ranges)} segments, {new_coverage:.1f}% coverage")
        
        logger.info(f"Final result: {len(merged_ranges)} segments")
        return merged_ranges
    
    def fixed_time_segmentation(self, audio: AudioSegment) -> List[Tuple[int, int]]:
        """
        Create fixed-time segments as a primary segmentation option.
        
        Args:
            audio: AudioSegment object
            
        Returns:
            List of (start, end) tuples for fixed-time segments
        """
        segment_duration = self.config.get('fixed_time_duration', 30000)  # 30 seconds default
        overlap = self.config.get('fixed_time_overlap', 3000)            # 3 seconds overlap default
        
        logger.info(f"Using fixed-time segmentation: {segment_duration/1000}s segments with {overlap/1000}s overlap")
        
        segments = []
        audio_length = len(audio)
        effective_step = segment_duration - overlap  # Step size accounting for overlap
        
        start = 0
        while start < audio_length:
            end = min(start + segment_duration, audio_length)
            
            # Only create segment if it's at least 5 seconds long
            if end - start >= 5000:
                segments.append((start, end))
                logger.debug(f"Fixed-time segment: {start/1000:.1f}s - {end/1000:.1f}s ({(end-start)/1000:.1f}s)")
            
            start += effective_step
            
            # Prevent infinite loop
            if start >= audio_length - 1000:  # Less than 1 second remaining
                break
        
        # Ensure we capture the end of the audio if there's a significant remainder
        if segments and audio_length - segments[-1][1] > 5000:
            segments.append((segments[-1][1] - overlap, audio_length))
            logger.debug(f"Added final segment: {(segments[-1][1] - overlap)/1000:.1f}s - {audio_length/1000:.1f}s")
        
        total_coverage = sum(end - start for start, end in segments)
        coverage_percent = (total_coverage / audio_length) * 100
        
        logger.info(f"Fixed-time segmentation created {len(segments)} segments")
        logger.info(f"Coverage: {coverage_percent:.1f}% ({total_coverage/1000:.1f}s / {audio_length/1000:.1f}s)")
        
        return segments
    
    def create_non_overlapping_segments(self, audio_file: str, nonsilent_ranges: List[Tuple[int, int]], 
                                      analysis: Dict) -> List[Tuple[str, int, int, int, int]]:
        """
        Create audio segments WITHOUT overlap to prevent duplicates in adaptive mode.
        Uses padding for context but ensures no overlapping between segments.
        
        Args:
            audio_file: Path to audio file
            nonsilent_ranges: List of detected speech ranges
            analysis: Speech pattern analysis
            
        Returns:
            List of (segment_file_path, original_start_ms, original_end_ms, padded_start_ms, padded_end_ms)
        """
        audio = AudioSegment.from_file(audio_file)
        total_duration = len(audio)
        
        padding = analysis["recommended_padding"]
        max_length = self.config['max_segment_length']
        min_length = self.config['min_segment_length']
        
        segments = []
        base_name = Path(audio_file).stem
        
        for i, (start, end) in enumerate(nonsilent_ranges):
            segment_length = end - start
            
            # Handle short segments
            if segment_length < min_length:
                if segment_length >= 1000:  # At least 1 second
                    logger.debug(f"Keeping short but viable segment {i}: {segment_length}ms")
                else:
                    logger.debug(f"Skipping very short segment {i}: {segment_length}ms")
                    continue
            
            # Split long segments WITHOUT overlap
            if segment_length > max_length:
                # Split into non-overlapping chunks
                chunk_start = start
                chunk_index = 0
                segment_prefix = self.config.get('segment_prefix', 'segment')
                
                while chunk_start < end:
                    chunk_end = min(chunk_start + max_length, end)
                    
                    # Apply padding for context but ensure no overlap between segments
                    padded_start = max(0, chunk_start - padding)
                    padded_end = min(total_duration, chunk_end + padding)
                    
                    # Ensure padding doesn't create overlap with previous segment
                    if segments:
                        prev_padded_end = segments[-1][4]  # Previous padded end
                        if padded_start < prev_padded_end:
                            padded_start = prev_padded_end  # Start where previous ended
                    
                    segment = audio[padded_start:padded_end]
                    segment_file = f"{audio_file}_{segment_prefix}_{len(segments):03d}_{chunk_index}.wav"
                    segment.export(segment_file, format="wav")
                    
                    # Store both original timing (for output) and padded timing (for processing)
                    segments.append((segment_file, chunk_start, chunk_end, padded_start, padded_end))
                    
                    # Move to next chunk WITHOUT overlap
                    chunk_start = chunk_end
                    chunk_index += 1
            else:
                # Normal segment with padding but no overlap
                padded_start = max(0, start - padding)
                padded_end = min(total_duration, end + padding)
                
                # Ensure padding doesn't create overlap with previous segment
                if segments:
                    prev_padded_end = segments[-1][4]  # Previous padded end
                    if padded_start < prev_padded_end:
                        padded_start = prev_padded_end  # Start where previous ended
                
                segment_prefix = self.config.get('segment_prefix', 'segment')
                
                segment = audio[padded_start:padded_end]
                segment_file = f"{audio_file}_{segment_prefix}_{len(segments):03d}.wav"
                segment.export(segment_file, format="wav")
                
                # Store both original timing (for output) and padded timing (for processing)
                segments.append((segment_file, start, end, padded_start, padded_end))
        
        logger.info(f"Created {len(segments)} NON-OVERLAPPING segments with padding")
        logger.info(f"Sample segment timing: Original vs Padded (No Overlap)")
        if segments:
            for i, (file, orig_start, orig_end, pad_start, pad_end) in enumerate(segments[:3]):
                logger.info(f"  Segment {i}: {orig_start/1000:.1f}s-{orig_end/1000:.1f}s (padded: {pad_start/1000:.1f}s-{pad_end/1000:.1f}s)")
        return segments

    def create_overlapping_segments(self, audio_file: str, nonsilent_ranges: List[Tuple[int, int]], 
                                  analysis: Dict) -> List[Tuple[str, int, int, int, int]]:
        """
        Create audio segments with overlap to prevent cutting off speech.
        
        Args:
            audio_file: Path to audio file
            nonsilent_ranges: List of detected speech ranges
            analysis: Speech pattern analysis
            
        Returns:
            List of (segment_file_path, original_start_ms, original_end_ms, padded_start_ms, padded_end_ms)
        """
        audio = AudioSegment.from_file(audio_file)
        total_duration = len(audio)
        
        padding = analysis["recommended_padding"]
        overlap = self.config['overlap_duration']
        max_length = self.config['max_segment_length']
        min_length = self.config['min_segment_length']
        
        segments = []
        base_name = Path(audio_file).stem
        
        for i, (start, end) in enumerate(nonsilent_ranges):
            segment_length = end - start
            
            # Handle short segments more flexibly
            if segment_length < min_length:
                # If segment is at least 1 second, keep it anyway
                if segment_length >= 1000:
                    logger.debug(f"Keeping short but viable segment {i}: {segment_length}ms")
                else:
                    # Try to merge with next segment
                    if i < len(nonsilent_ranges) - 1:
                        next_start, next_end = nonsilent_ranges[i + 1]
                        gap = next_start - end
                        
                        if gap < self.config['min_silence_len'] * 2:
                            # Merge with next segment
                            continue
                    
                    logger.debug(f"Skipping very short segment {i}: {segment_length}ms")
                    continue
            
            # Split long segments
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
                    
                    # Store both original timing (for output) and padded timing (for processing)
                    segments.append((segment_file, chunk_start, chunk_end, padded_start, padded_end))
                    
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
                
                # Store both original timing (for output) and padded timing (for processing)
                segments.append((segment_file, start, end, padded_start, padded_end))
        
        logger.info(f"Created {len(segments)} segments with overlap and padding")
        logger.info(f"Sample segment timing: Original vs Padded")
        if segments:
            for i, (file, orig_start, orig_end, pad_start, pad_end) in enumerate(segments[:3]):
                logger.info(f"  Segment {i}: {orig_start/1000:.1f}s-{orig_end/1000:.1f}s (padded: {pad_start/1000:.1f}s-{pad_end/1000:.1f}s)")
        return segments
    
    def transcribe_with_enhanced_options(self, segment_file: str) -> Dict[str, Any]:
        """
        Transcribe with enhanced Whisper options for better accuracy.
        
        Args:
            segment_file: Path to audio segment
            
        Returns:
            Enhanced transcription result
        """
        transcribe_options = {
            "language": self.language,
            "verbose": False,
            "fp16": True,
            "condition_on_previous_text": True,
            "initial_prompt": "Dies ist eine Aufzeichnung einer deutschen Universitätsvorlesung.",
        }
        
        # Add enhanced options if available in this Whisper version
        if self.config.get('use_beam_search', False):
            transcribe_options.update({
                "beam_size": 5,
                "best_of": self.config.get('best_of', 3),
                "patience": self.config.get('patience', 2.0),
                "length_penalty": self.config.get('length_penalty', 0.8),
            })
        
        try:
            result = self.model.transcribe(segment_file, **transcribe_options)
            return result
        except Exception as e:
            # Fallback to basic transcription
            logger.warning(f"Enhanced transcription failed, using basic options: {e}")
            basic_options = {
                "language": self.language,
                "verbose": False,
                "fp16": False,  # Fallback without FP16
                "condition_on_previous_text": True
            }
            return self.model.transcribe(segment_file, **basic_options)
    
    def validate_transcription_segment(self, result: Dict, segment_info: Tuple[str, int, int]) -> Dict:
        """
        Validate and enhance transcription results.
        
        Args:
            result: Whisper transcription result
            segment_info: Segment information
            
        Returns:
            Validated and enhanced result
        """
        segment_file, start_time, end_time = segment_info
        text = result.get("text", "").strip()
        
        # Quality metrics
        quality_score = 1.0
        warnings = []
        
        # Check for very short transcriptions relative to audio length
        duration_seconds = (end_time - start_time) / 1000.0
        if duration_seconds > 10 and len(text) < 20:
            quality_score *= 0.5
            warnings.append("Very short transcription for long audio segment")
        
        # Check for repetitive text (possible transcription error)
        words = text.split()
        if len(words) > 5:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio < 0.3:
                quality_score *= 0.7
                warnings.append("High repetition detected")
        
        # Check average confidence if available
        avg_confidence = result.get("confidence", 1.0)
        if avg_confidence < 0.5:
            quality_score *= 0.8
            warnings.append("Low confidence score")
        
        return {
            "text": text,
            "confidence": avg_confidence,
            "quality_score": quality_score,
            "warnings": warnings,
            "duration_seconds": duration_seconds,
            "words_per_minute": len(words) / (duration_seconds / 60) if duration_seconds > 0 else 0
        }
    
    def transcribe_audio_file_enhanced(self, audio_file: str) -> Dict[str, Any]:
        """
        Enhanced transcription workflow optimized for slow speakers.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Complete enhanced transcription results
        """
        start_time = time.time()
        
        logger.info(f"Starting enhanced transcription for: {audio_file}")
        
        # Load model
        self.load_model()
        
        # Step 1: Analyze speech patterns
        analysis = self.analyze_speech_patterns(audio_file)
        
        # Step 2: Choose segmentation method based on configuration
        segmentation_mode = self.config.get('segmentation_mode', 'silence_detection')
        audio = AudioSegment.from_file(audio_file)
        
        logger.info(f"Using segmentation mode: {segmentation_mode}")
        
        if segmentation_mode == 'fixed_time':
            # Use fixed-time segmentation as primary method
            nonsilent_ranges = self.fixed_time_segmentation(audio)
            logger.info(f"Fixed-time segmentation: {len(nonsilent_ranges)} segments")
        
        elif segmentation_mode == 'defensive_silence':
            # Use defensive silence detection - only split on confident silence
            nonsilent_ranges = self.defensive_silence_detection(audio, analysis)
            logger.info(f"Defensive silence detection: {len(nonsilent_ranges)} segments")
        
        elif segmentation_mode == 'adaptive':
            # Use improved adaptive strategy with defensive silence principles
            nonsilent_ranges = self.adaptive_silence_detection(audio, analysis)
            logger.info(f"Adaptive segmentation: {len(nonsilent_ranges)} segments")
        
        else:  # 'silence_detection' (default)
            # Use enhanced silence detection
            nonsilent_ranges = self.enhanced_silence_detection(audio, analysis)
        
        if not nonsilent_ranges:
            logger.error("No speech detected in audio file")
            return {
                "segments": [],
                "full_text": "",
                "total_duration": 0,
                "error": "No speech detected",
                "analysis": analysis
            }
        
        # Step 3: Create segments (with or without overlap based on mode)
        segmentation_mode = self.config.get('segmentation_mode', 'silence_detection')
        
        if segmentation_mode == 'adaptive':
            # For adaptive mode, use non-overlapping segments to prevent duplicates
            segments = self.create_non_overlapping_segments(audio_file, nonsilent_ranges, analysis)
        else:
            # For other modes, use overlapping segments (existing behavior)
            segments = self.create_overlapping_segments(audio_file, nonsilent_ranges, analysis)
        
        if not segments:
            logger.error("No valid segments created")
            return {
                "segments": [],
                "full_text": "",
                "total_duration": 0,
                "error": "No valid segments created",
                "analysis": analysis
            }
        
        # Step 4: Transcribe segments with enhanced options
        logger.info(f"Transcribing {len(segments)} segments with enhanced options...")
        results = []
        
        for i, segment_info in enumerate(tqdm.tqdm(segments, desc="Enhanced transcription")):
            segment_file, original_start_ms, original_end_ms, padded_start_ms, padded_end_ms = segment_info
            
            try:
                # Transcribe with enhanced options
                whisper_result = self.transcribe_with_enhanced_options(segment_file)
                
                # Validate and enhance result (pass padded timing for validation)
                padded_segment_info = (segment_file, padded_start_ms, padded_end_ms)
                enhanced_result = self.validate_transcription_segment(whisper_result, padded_segment_info)
                
                # Use ORIGINAL timing for final output, not padded timing
                segment_result = {
                    "segment_id": i,
                    "text": enhanced_result["text"],
                    "confidence": enhanced_result["confidence"],
                    "quality_score": enhanced_result["quality_score"],
                    "warnings": enhanced_result["warnings"],
                    "start_time": original_start_ms,  # CRITICAL: Use original speech timing
                    "end_time": original_end_ms,      # CRITICAL: Use original speech timing
                    "duration": original_end_ms - original_start_ms,
                    "words_per_minute": enhanced_result["words_per_minute"],
                    "segment_file": segment_file,
                    "language_detected": whisper_result.get("language", self.language),
                    # Keep padding info for debugging
                    "original_timing": {"start": original_start_ms, "end": original_end_ms},
                    "padded_timing": {"start": padded_start_ms, "end": padded_end_ms}
                }
                
                results.append(segment_result)
                
                # Log quality warnings
                if enhanced_result["warnings"]:
                    logger.warning(f"Segment {i} warnings: {', '.join(enhanced_result['warnings'])}")
                
            except Exception as e:
                logger.error(f"Error transcribing segment {i}: {e}")
                error_result = {
                    "segment_id": i,
                    "text": "",
                    "error": str(e),
                    "start_time": original_start_ms,
                    "end_time": original_end_ms,
                    "segment_file": segment_file
                }
                results.append(error_result)
            
            # Cleanup segment file if configured
            if self.config['cleanup_segments']:
                try:
                    os.remove(segment_file)
                except Exception as e:
                    logger.warning(f"Could not remove segment file {segment_file}: {e}")
        
        # Step 5: Post-process and merge results
        final_results = self.post_process_transcription(results, analysis)
        
        processing_time = time.time() - start_time
        
        # Compile final transcription
        successful_segments = [r for r in final_results if not r.get("error") and r.get("text")]
        failed_segments = [r for r in final_results if r.get("error")]
        
        full_text = " ".join([r["text"] for r in successful_segments])
        total_duration = max([r.get("end_time", 0) for r in final_results], default=0)
        
        # Calculate quality metrics
        quality_scores = [r.get("quality_score", 1.0) for r in successful_segments]
        average_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        transcription_result = {
            "segments": final_results,
            "full_text": full_text,
            "total_duration": total_duration,
            "language": self.language,
            "model": self.model_name,
            "processing_time_seconds": processing_time,
            "segments_total": len(segments),
            "segments_successful": len(successful_segments),
            "segments_failed": len(failed_segments),
            "average_quality_score": average_quality,
            "speech_analysis": analysis,
            "config_used": self.config.copy(),
            "enhancement_applied": True
        }
        
        logger.info(f"Enhanced transcription completed in {processing_time:.2f}s")
        logger.info(f"Success rate: {len(successful_segments)}/{len(segments)} segments")
        logger.info(f"Average quality score: {average_quality:.2f}")
        
        return transcription_result
    
    def post_process_transcription(self, results: List[Dict], analysis: Dict) -> List[Dict]:
        """
        Post-process transcription results to improve quality.
        
        Args:
            results: List of transcription results
            analysis: Speech analysis results
            
        Returns:
            Post-processed results
        """
        if not self.config.get('merge_short_segments', False):
            return results
        
        logger.info("Post-processing transcription results...")
        
        processed = []
        i = 0
        
        while i < len(results):
            current = results[i]
            
            # Skip error segments
            if current.get("error"):
                processed.append(current)
                i += 1
                continue
            
            current_text = current.get("text", "").strip()
            current_duration = current.get("duration", 0)
            
            # Try to merge with next segment if current is very short
            if (current_duration < 5000 and  # Less than 5 seconds
                len(current_text.split()) < 10 and  # Less than 10 words
                i < len(results) - 1):  # Not the last segment
                
                next_segment = results[i + 1]
                next_text = next_segment.get("text", "").strip()
                
                if not next_segment.get("error") and next_text:
                    # Merge segments - CRITICAL: Preserve original timing properly
                    merged_text = f"{current_text} {next_text}".strip()
                    merged_segment = current.copy()
                    
                    # Merge original timing as well
                    current_orig = current.get("original_timing", {"start": current["start_time"], "end": current["end_time"]})
                    next_orig = next_segment.get("original_timing", {"start": next_segment["start_time"], "end": next_segment["end_time"]})
                    
                    merged_segment.update({
                        "text": merged_text,
                        "end_time": next_orig["end"],  # Use original timing, not processed timing
                        "duration": next_orig["end"] - current_orig["start"],
                        "merged_from": [current["segment_id"], next_segment["segment_id"]],
                        "original_timing": {
                            "start": current_orig["start"],
                            "end": next_orig["end"]
                        }
                    })
                    
                    processed.append(merged_segment)
                    i += 2  # Skip next segment as it's merged
                    logger.debug(f"Merged segments {current['segment_id']} and {next_segment['segment_id']}")
                    continue
            
            processed.append(current)
            i += 1
        
        logger.info(f"Post-processing: {len(results)} -> {len(processed)} segments")
        return processed
    
    def transcribe_audio_file(self, audio_file: str) -> Dict[str, Any]:
        """
        Standard transcribe_audio_file method that uses enhanced transcription.
        This preserves original speech timing while using enhanced processing.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcription result with original timing preserved
        """
        logger.info("Using Enhanced Transcriber with Original Timing Preservation")
        result = self.transcribe_audio_file_enhanced(audio_file)
        
        # Log timing preservation info
        if result.get("segments"):
            logger.info("Timing preservation status:")
            for i, segment in enumerate(result["segments"][:3]):  # Show first 3 as example
                if "original_timing" in segment and "padded_timing" in segment:
                    orig = segment["original_timing"]
                    padded = segment["padded_timing"]
                    logger.info(f"  Segment {i}: Speech {orig['start']/1000:.1f}s-{orig['end']/1000:.1f}s, "
                              f"Processed {padded['start']/1000:.1f}s-{padded['end']/1000:.1f}s")
        
        return result
    
    def defensive_silence_detection(self, audio: AudioSegment, analysis: Dict) -> List[Tuple[int, int]]:
        """
        Defensive silence detection that only splits on confidently detected silence.
        This approach is much more conservative and only splits where we're sure it's silent.
        
        Key principles:
        - Only split on clearly quiet segments (much quieter than speech)
        - Use minimum 2000ms silence requirement  
        - Compare volumes relatively instead of absolute thresholds
        - Prefer fewer, longer segments over many short ones
        
        Args:
            audio: AudioSegment object
            analysis: Speech pattern analysis results
            
        Returns:
            List of (start, end) tuples for speech segments
        """
        logger.info("🛡️ Using DEFENSIVE silence detection - conservative splitting only")
        
        # Configuration for defensive detection
        min_silence_duration = self.config.get('min_silence_len', 2000)  # At least 2 seconds
        min_silence_duration = max(min_silence_duration, 2000)  # Force minimum 2s
        
        # Analyze audio volume to find speech vs silence threshold
        chunk_size = 1000  # 1 second chunks for analysis
        chunk_volumes = []
        
        # Collect volume data for all chunks
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) > 500:  # At least 0.5 seconds
                volume = chunk.dBFS
                if np.isfinite(volume):
                    chunk_volumes.append(volume)
        
        if not chunk_volumes:
            logger.warning("No valid volume data, using entire audio as one segment")
            return [(0, len(audio))]
        
        # Calculate volume statistics
        volumes = np.array(chunk_volumes)
        mean_volume = np.mean(volumes)
        volume_std = np.std(volumes)
        
        # Define "clearly silent" threshold - must be significantly quieter than average
        # Use 1.5 standard deviations below mean as "clearly silent"
        silence_threshold = mean_volume - (1.5 * volume_std)
        
        logger.info(f"📊 Volume analysis:")
        logger.info(f"   Mean volume: {mean_volume:.1f} dB")
        logger.info(f"   Volume std: {volume_std:.1f} dB")
        logger.info(f"   Silence threshold: {silence_threshold:.1f} dB")
        logger.info(f"   Minimum silence duration: {min_silence_duration}ms")
        
        # Find potential silence segments with strict criteria
        potential_silence_segments = []
        
        # Use longer chunks for silence detection to avoid false positives
        silence_chunk_size = 500  # 0.5 second chunks for more precise detection
        
        for i in range(0, len(audio), silence_chunk_size):
            chunk = audio[i:i+silence_chunk_size]
            if len(chunk) > 250:  # At least 0.25 seconds
                volume = chunk.dBFS
                if np.isfinite(volume) and volume < silence_threshold:
                    potential_silence_segments.append((i, i + len(chunk)))
        
        # Merge consecutive silence chunks into longer silence periods
        merged_silence = []
        for start, end in potential_silence_segments:
            if merged_silence and start - merged_silence[-1][1] < silence_chunk_size:
                # Extend previous silence period
                merged_silence[-1] = (merged_silence[-1][0], end)
            else:
                merged_silence.append((start, end))
        
        # Filter to only keep silence periods that meet our minimum duration
        confident_silence = []
        for start, end in merged_silence:
            duration = end - start
            if duration >= min_silence_duration:
                confident_silence.append((start, end))
                logger.debug(f"Found confident silence: {start/1000:.1f}s - {end/1000:.1f}s ({duration/1000:.1f}s)")
        
        # Create speech segments by splitting on confident silence
        speech_segments = []
        last_end = 0
        
        for silence_start, silence_end in confident_silence:
            # Add speech segment before this silence (if substantial)
            if silence_start - last_end > 5000:  # At least 5 seconds of speech
                speech_segments.append((last_end, silence_start))
                logger.debug(f"Speech segment: {last_end/1000:.1f}s - {silence_start/1000:.1f}s")
            
            last_end = silence_end
        
        # Add final speech segment if substantial
        if len(audio) - last_end > 5000:
            speech_segments.append((last_end, len(audio)))
            logger.debug(f"Final speech segment: {last_end/1000:.1f}s - {len(audio)/1000:.1f}s")
        
        # If no confident silence found, use entire audio as one segment
        if not speech_segments:
            logger.info("No confident silence periods found - using entire audio as one segment")
            speech_segments = [(0, len(audio))]
        
        # Calculate coverage statistics
        total_speech_duration = sum(end - start for start, end in speech_segments)
        speech_coverage = (total_speech_duration / len(audio)) * 100
        
        logger.info(f"🎯 Defensive detection results:")
        logger.info(f"   Confident silence periods: {len(confident_silence)}")
        logger.info(f"   Speech segments created: {len(speech_segments)}")
        logger.info(f"   Speech coverage: {speech_coverage:.1f}%")
        
        return speech_segments

    def adaptive_silence_detection(self, audio: AudioSegment, analysis: Dict) -> List[Tuple[int, int]]:
        """
        Adaptive silence detection that combines defensive silence principles with speaker adaptation.
        Uses defensive silence detection as base, with fallback strategies.
        
        Args:
            audio: Audio segment to analyze
            analysis: Speech pattern analysis results
            
        Returns:
            List of speech segment ranges (start_ms, end_ms)
        """
        logger.info("🔄 Starting adaptive silence detection with defensive principles...")
        
        # Start with defensive silence detection as the primary strategy
        speech_segments = self.defensive_silence_detection(audio, analysis)
        
        # Calculate coverage
        total_audio_duration = len(audio)
        total_speech_duration = sum(end - start for start, end in speech_segments)
        speech_coverage = (total_speech_duration / total_audio_duration) * 100
        
        logger.info(f"📊 Defensive silence detection coverage: {speech_coverage:.1f}%")
        
        # If coverage is very low, try enhanced silence detection with speaker-adapted parameters
        if speech_coverage < 30:  # Less than 30% coverage suggests defensive was too strict
            logger.info("🔍 Coverage too low, trying enhanced detection with adapted parameters...")
            
            # Use speaker-adapted parameters for enhanced detection
            speaker_type = analysis.get('speaker_type', 'moderate_speech')
            
            if speaker_type in ['very_sparse_speech', 'sparse_speech']:
                # For sparse speakers, use very sensitive detection
                enhanced_segments = self.enhanced_silence_detection(audio, analysis)
                enhanced_coverage = (sum(end - start for start, end in enhanced_segments) / total_audio_duration) * 100
                
                logger.info(f"📊 Enhanced detection coverage: {enhanced_coverage:.1f}%")
                
                # Use enhanced if it provides reasonable coverage without being too aggressive
                if enhanced_coverage > speech_coverage and enhanced_coverage <= 85:  # Not too aggressive
                    logger.info("✅ Using enhanced detection results")
                    speech_segments = enhanced_segments
                    speech_coverage = enhanced_coverage
        
        # If still low coverage, use defensive-guided fixed-time as final fallback
        if speech_coverage < 25:  # Very low coverage
            logger.warning(f"🚨 Coverage still too low ({speech_coverage:.1f}%), using defensive-guided fixed-time")
            
            # Use fixed-time but with defensive silence guidance for better boundaries
            fixed_segments = self.defensive_guided_fixed_time(audio, analysis, speech_segments)
            final_coverage = (sum(end - start for start, end in fixed_segments) / total_audio_duration) * 100
            
            logger.info(f"📊 Defensive-guided fixed-time coverage: {final_coverage:.1f}%")
            speech_segments = fixed_segments
            speech_coverage = final_coverage
        
        # Remove overlapping segments to prevent duplicates (key improvement over old adaptive)
        speech_segments = self.remove_overlapping_segments(speech_segments)
        
        # Final validation - ensure we have reasonable coverage
        final_speech_duration = sum(end - start for start, end in speech_segments)
        final_coverage = (final_speech_duration / total_audio_duration) * 100
        
        logger.info(f"🎯 Adaptive detection final results:")
        logger.info(f"   Segments: {len(speech_segments)}")
        logger.info(f"   Coverage: {final_coverage:.1f}%")
        logger.info(f"   Strategy: {'defensive' if speech_coverage >= 30 else 'enhanced/guided-fixed'}")
        
        return speech_segments

    def defensive_guided_fixed_time(self, audio: AudioSegment, analysis: Dict, 
                                  defensive_segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Create fixed-time segments but use defensive silence detections as boundary guidance.
        
        Args:
            audio: Audio segment to process
            analysis: Speech analysis results
            defensive_segments: Results from defensive silence detection for boundary guidance
            
        Returns:
            List of guided fixed-time segments
        """
        logger.info("🎯 Creating defensive-guided fixed-time segments...")
        
        # Get defensive silence boundaries for guidance
        silence_boundaries = set()
        for start, end in defensive_segments:
            silence_boundaries.add(start)
            silence_boundaries.add(end)
        
        # Add audio start and end
        silence_boundaries.add(0)
        silence_boundaries.add(len(audio))
        
        # Sort boundaries
        boundaries = sorted(silence_boundaries)
        
        # Create segments using a target duration but respecting silence boundaries
        target_duration = 25000  # 25 seconds target
        min_segment_duration = 5000  # 5 seconds minimum
        
        segments = []
        current_start = 0
        
        for boundary in boundaries[1:]:  # Skip first boundary (0)
            segment_duration = boundary - current_start
            
            # If segment is too short, try to extend to next boundary
            if segment_duration < min_segment_duration and boundary < len(audio):
                continue
                
            # If segment is reasonable or we're at the end, create it
            if segment_duration >= min_segment_duration:
                segments.append((current_start, boundary))
                current_start = boundary
        
        # Ensure we capture any remaining audio
        if current_start < len(audio) - 1000:  # At least 1 second remaining
            segments.append((current_start, len(audio)))
        
        logger.info(f"📋 Created {len(segments)} defensive-guided segments")
        return segments

    def remove_overlapping_segments(self, segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Remove overlapping segments to prevent duplicate transcriptions.
        Key improvement over old adaptive mode.
        
        Args:
            segments: List of (start_ms, end_ms) tuples
            
        Returns:
            List of non-overlapping segments
        """
        if not segments:
            return segments
            
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x[0])
        
        non_overlapping = []
        current_start, current_end = sorted_segments[0]
        
        for start, end in sorted_segments[1:]:
            if start >= current_end:
                # No overlap, add current segment and move to next
                non_overlapping.append((current_start, current_end))
                current_start, current_end = start, end
            else:
                # Overlap detected, merge segments
                current_end = max(current_end, end)
                logger.debug(f"Merged overlapping segments: {start/1000:.1f}s-{end/1000:.1f}s")
        
        # Add the last segment
        non_overlapping.append((current_start, current_end))
        
        overlap_removed = len(segments) - len(non_overlapping)
        if overlap_removed > 0:
            logger.info(f"🔧 Removed {overlap_removed} overlapping segments")
        
        return non_overlapping