"""
Audio transcription module using OpenAI Whisper
"""

import whisper
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from pydub import AudioSegment, silence
import os
from pathlib import Path
import tqdm

from .config import LANGUAGE_MAP, DEFAULT_CONFIG
from .utils import format_timestamp

logger = logging.getLogger(__name__)

class AudioTranscriber:
    """
    Enhanced audio transcriber based on OpenAI Whisper with intelligent segmentation.
    """
    
    def __init__(self, 
                 model_name: str = "large-v3", 
                 language: str = "german", 
                 device: Optional[str] = None,
                 config: Optional[Dict] = None):
        """
        Initialize the audio transcriber.
        
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
        
        # Merge config with defaults
        self.config = DEFAULT_CONFIG['transcription'].copy()
        if config:
            self.config.update(config)
        
        logger.info(f"Initialized AudioTranscriber with model: {model_name}, language: {language}")
    
    def load_model(self) -> None:
        """Load the Whisper model if not already loaded."""
        if self.model is None:
            import torch
            
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Whisper model: {self.model_name} on {self.device}")
            
            if self.device.startswith("cuda"):
                try:
                    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info(f"CUDA memory: {memory_gb:.2f} GB")
                except Exception as e:
                    logger.warning(f"Could not get CUDA info: {e}")
            
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info("Model loaded successfully")
    
    def analyze_background_noise(self, audio_file: str) -> float:
        """
        Analyze audio to determine optimal silence threshold.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Calculated silence threshold in dBFS
        """
        logger.info("Analyzing background noise levels...")
        audio = AudioSegment.from_file(audio_file)
        
        # Calculate noise level from the first and last 10 seconds
        sample_duration = min(10000, len(audio) // 4)  # 10 seconds or 1/4 of audio
        
        start_sample = audio[:sample_duration]
        end_sample = audio[-sample_duration:]
        
        start_noise = start_sample.dBFS
        end_noise = end_sample.dBFS
        
        # Use the higher (less negative) value as baseline
        noise_level = max(start_noise, end_noise)
        silence_thresh = noise_level + self.config['silence_adjustment']
        
        logger.info(f"Background noise level: {noise_level:.2f} dBFS")
        logger.info(f"Silence threshold: {silence_thresh:.2f} dBFS")
        
        return silence_thresh
    
    def split_audio_on_silence(self, audio_file: str, silence_thresh: float) -> List[Tuple[str, int, int]]:
        """
        Split audio into segments based on silence detection.
        
        Args:
            audio_file: Path to audio file
            silence_thresh: Silence threshold in dBFS
            
        Returns:
            List of tuples (segment_file_path, start_time_ms, end_time_ms)
        """
        audio = AudioSegment.from_file(audio_file)
        total_duration_ms = len(audio)
        
        logger.info(f"Splitting audio based on {self.config['min_silence_len']}ms silence detection...")
        
        # Detect non-silent ranges
        nonsilent_ranges = silence.detect_nonsilent(
            audio, 
            min_silence_len=self.config['min_silence_len'], 
            silence_thresh=silence_thresh
        )
        
        if not nonsilent_ranges:
            logger.warning("No speech detected, using entire file as single segment")
            segment_file = f"{audio_file}_segment_0.wav"
            audio.export(segment_file, format="wav")
            return [(segment_file, 0, total_duration_ms)]
        
        # Create segments with padding
        segments = []
        base_name = Path(audio_file).stem
        
        for i, (start, end) in enumerate(nonsilent_ranges):
            # Apply padding while staying within bounds
            segment_start = max(0, start - self.config['padding'])
            segment_end = min(total_duration_ms, end + self.config['padding'])
            
            # Extract segment
            segment = audio[segment_start:segment_end]
            segment_file = f"{audio_file}_segment_{i:03d}.wav"
            segment.export(segment_file, format="wav")
            
            segments.append((segment_file, segment_start, segment_end))
            
            logger.debug(f"Segment {i}: {segment_start}ms - {segment_end}ms ({segment_end-segment_start}ms)")
        
        logger.info(f"Created {len(segments)} audio segments")
        return segments
    
    def transcribe_segment(self, segment_info: Tuple[str, int, int], segment_idx: int) -> Dict[str, Any]:
        """
        Transcribe a single audio segment.
        
        Args:
            segment_info: Tuple of (segment_file_path, start_time_ms, end_time_ms)
            segment_idx: Index of the segment
            
        Returns:
            Dictionary with transcription results
        """
        segment_file, start_time, end_time = segment_info
        
        try:
            # Transcribe with context and error handling
            result = self._transcribe_with_fallback(segment_file)
            
            return {
                "segment_id": segment_idx,
                "text": result["text"].strip(),
                "confidence": float(result.get("confidence", 0.0)),
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "segment_file": segment_file,
                "language_detected": result.get("language", self.language)
            }
            
        except Exception as e:
            logger.error(f"Error transcribing segment {segment_idx} ({segment_file}): {e}")
            return {
                "segment_id": segment_idx,
                "text": "",
                "error": str(e),
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "segment_file": segment_file
            }
    
    def _transcribe_with_fallback(self, segment_file: str) -> Dict[str, Any]:
        """
        Transcribe with fallback options for error recovery.
        
        Args:
            segment_file: Path to audio segment
            
        Returns:
            Whisper transcription result
        """
        # Try with FP16 first (faster)
        try:
            result = self.model.transcribe(
                segment_file,
                language=self.language,
                verbose=False,
                fp16=True,
                condition_on_previous_text=True
            )
            return result
            
        except RuntimeError as e:
            if "Expected key.size(1) == value.size(1)" in str(e):
                # Common CUDA attention error, retry with FP16=False
                logger.warning(f"CUDA attention error, retrying with FP16=False for {segment_file}")
                result = self.model.transcribe(
                    segment_file,
                    language=self.language,
                    verbose=False,
                    fp16=False,
                    condition_on_previous_text=True
                )
                return result
            else:
                raise
    
    def transcribe_audio_file(self, audio_file: str) -> Dict[str, Any]:
        """
        Transcribe entire audio file with intelligent segmentation.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Complete transcription results
        """
        start_time = time.time()
        
        # Load model
        self.load_model()
        
        # Analyze and split audio
        silence_thresh = self.analyze_background_noise(audio_file)
        segments = self.split_audio_on_silence(audio_file, silence_thresh)
        
        if not segments:
            logger.error("No audio segments created")
            return {
                "segments": [],
                "full_text": "",
                "total_duration": 0,
                "language": self.language,
                "model": self.model_name,
                "error": "No audio segments could be created"
            }
        
        # Transcribe segments
        logger.info(f"Transcribing {len(segments)} segments...")
        results = []
        
        for i, segment_info in enumerate(tqdm.tqdm(segments, desc="Transcribing")):
            result = self.transcribe_segment(segment_info, i)
            results.append(result)
            
            # Cleanup segment file if configured
            if self.config['cleanup_segments']:
                try:
                    os.remove(segment_info[0])
                except Exception as e:
                    logger.warning(f"Could not remove segment file {segment_info[0]}: {e}")
        
        # Compile final results
        successful_segments = [r for r in results if not r.get("error")]
        failed_segments = [r for r in results if r.get("error")]
        
        full_text = " ".join([r["text"] for r in successful_segments if r["text"]])
        total_duration = max([r.get("end_time", 0) for r in results], default=0)
        
        processing_time = time.time() - start_time
        
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
            "config_used": self.config.copy()
        }
        
        logger.info(f"Transcription completed in {processing_time:.2f}s")
        logger.info(f"Success rate: {len(successful_segments)}/{len(segments)} segments")
        
        if failed_segments:
            logger.warning(f"Failed to transcribe {len(failed_segments)} segments")
        
        return transcription_result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        if self.model is None:
            return {"model_loaded": False}
        
        return {
            "model_loaded": True,
            "model_name": self.model_name,
            "language": self.language,
            "device": self.device,
            "config": self.config
        }