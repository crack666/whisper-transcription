"""
Enhanced audio transcriber with improved handling for slow speakers and long pauses
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
                logger.warning("All fallbacks failed, creating full-duration segment")
                nonsilent_ranges = [(0, len(audio))]
        
        # Second pass: merge segments that are too close together
        merged_ranges = []
        min_gap = min_silence_len  # Minimum gap between segments
        
        for start, end in nonsilent_ranges:
            if merged_ranges and start - merged_ranges[-1][1] < min_gap:
                # Merge with previous segment
                merged_ranges[-1] = (merged_ranges[-1][0], end)
            else:
                merged_ranges.append((start, end))
        
        logger.info(f"Detected {len(nonsilent_ranges)} segments, merged to {len(merged_ranges)}")
        return merged_ranges
    
    def create_overlapping_segments(self, audio_file: str, nonsilent_ranges: List[Tuple[int, int]], 
                                  analysis: Dict) -> List[Tuple[str, int, int]]:
        """
        Create audio segments with overlap to prevent cutting off speech.
        
        Args:
            audio_file: Path to audio file
            nonsilent_ranges: List of detected speech ranges
            analysis: Speech pattern analysis
            
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
                
                while chunk_start < end:
                    chunk_end = min(chunk_start + max_length, end)
                    
                    # Apply padding
                    padded_start = max(0, chunk_start - padding)
                    padded_end = min(total_duration, chunk_end + padding)
                    
                    segment = audio[padded_start:padded_end]
                    segment_file = f"{audio_file}_segment_{len(segments):03d}_{chunk_index}.wav"
                    segment.export(segment_file, format="wav")
                    
                    segments.append((segment_file, padded_start, padded_end))
                    
                    # Move to next chunk with overlap
                    chunk_start = chunk_end - overlap
                    chunk_index += 1
            else:
                # Normal segment with padding
                padded_start = max(0, start - padding)
                padded_end = min(total_duration, end + padding)
                
                segment = audio[padded_start:padded_end]
                segment_file = f"{audio_file}_segment_{len(segments):03d}.wav"
                segment.export(segment_file, format="wav")
                
                segments.append((segment_file, padded_start, padded_end))
        
        logger.info(f"Created {len(segments)} segments with overlap and padding")
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
        
        # Step 2: Enhanced silence detection
        audio = AudioSegment.from_file(audio_file)
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
        
        # Step 3: Create overlapping segments
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
            segment_file, start_time_ms, end_time_ms = segment_info
            
            try:
                # Transcribe with enhanced options
                whisper_result = self.transcribe_with_enhanced_options(segment_file)
                
                # Validate and enhance result
                enhanced_result = self.validate_transcription_segment(whisper_result, segment_info)
                
                segment_result = {
                    "segment_id": i,
                    "text": enhanced_result["text"],
                    "confidence": enhanced_result["confidence"],
                    "quality_score": enhanced_result["quality_score"],
                    "warnings": enhanced_result["warnings"],
                    "start_time": start_time_ms,
                    "end_time": end_time_ms,
                    "duration": end_time_ms - start_time_ms,
                    "words_per_minute": enhanced_result["words_per_minute"],
                    "segment_file": segment_file,
                    "language_detected": whisper_result.get("language", self.language)
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
                    # Merge segments
                    merged_text = f"{current_text} {next_text}".strip()
                    merged_segment = current.copy()
                    merged_segment.update({
                        "text": merged_text,
                        "end_time": next_segment.get("end_time", current["end_time"]),
                        "duration": next_segment.get("end_time", current["end_time"]) - current["start_time"],
                        "merged_from": [current["segment_id"], next_segment["segment_id"]]
                    })
                    
                    processed.append(merged_segment)
                    i += 2  # Skip next segment as it's merged
                    logger.debug(f"Merged segments {current['segment_id']} and {next_segment['segment_id']}")
                    continue
            
            processed.append(current)
            i += 1
        
        logger.info(f"Post-processing: {len(results)} -> {len(processed)} segments")
        return processed