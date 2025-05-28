#!/usr/bin/env python3
"""
Process Isolation Manager for Whisper Transcription Optimization.

This module handles isolated process execution for parallel transcription testing,
preventing CUDA conflicts, memory issues, and file access race conditions.
"""

import os
import sys
import time
import uuid
import tempfile
import shutil
import logging
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import torch
except ImportError:
    torch = None

from .audio_profile_analyzer import AudioProfile, OptimizationResult
from .enhanced_transcriber import EnhancedAudioTranscriber

class ProcessIsolationManager:
    """Manages isolated process execution for transcription testing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.temp_dirs = []  # Track temporary directories for cleanup
    
    def test_configurations_parallel(self, configs: List[Dict], audio_file: str, 
                                   audio_profile: AudioProfile, max_workers: int = 2) -> List[OptimizationResult]:
        """
        Test multiple configurations in parallel with process isolation.
        
        Args:
            configs: List of configuration dictionaries to test
            audio_file: Path to audio file
            audio_profile: Audio characteristics
            max_workers: Maximum number of parallel processes
            
        Returns:
            List of optimization results
        """
        self.logger.info(f"Testing {len(configs)} configurations with {max_workers} parallel processes")
        
        results = []
        
        try:
            # Use ProcessPoolExecutor for true process isolation
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Prepare isolated configurations
                isolated_configs = []
                for i, config in enumerate(configs):
                    isolated_config = config.copy()
                    isolated_config['process_id'] = i
                    isolated_config['total_processes'] = len(configs)
                    isolated_configs.append(isolated_config)
                
                # Submit tasks
                future_to_config = {
                    executor.submit(
                        self._test_configuration_isolated_static,
                        isolated_config,
                        audio_file,
                        audio_profile.to_dict()
                    ): isolated_config for isolated_config in isolated_configs
                }
                
                # Collect results with extended timeout
                for future in as_completed(future_to_config, timeout=300):
                    config = future_to_config[future]
                    try:
                        result = future.result(timeout=180)
                        if result:
                            results.append(result)
                            self.logger.info(f"✅ {config['name']}: {result.word_count} words")
                        else:
                            self.logger.warning(f"❌ {config['name']}: No result returned")
                    except Exception as e:
                        self.logger.error(f"❌ {config['name']}: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Parallel processing failed: {e}")
            
        finally:
            # Cleanup temporary directories
            self._cleanup_temp_dirs()
            
        self.logger.info(f"Parallel testing completed: {len(results)}/{len(configs)} successful")
        return results
    
    @staticmethod
    def _test_configuration_isolated_static(config: Dict, audio_file: str, audio_profile_dict: Dict) -> Optional[OptimizationResult]:
        """
        Static method for isolated configuration testing in separate process.
        
        Args:
            config: Configuration dictionary with process_id
            audio_file: Path to audio file
            audio_profile_dict: Audio profile as dictionary
            
        Returns:
            OptimizationResult or None if failed
        """
        # This needs to be a static method for ProcessPoolExecutor
        # Create a new instance of the manager in the isolated process
        manager = ProcessIsolationManager()
        audio_profile = AudioProfile.from_dict(audio_profile_dict)
        return manager._test_configuration_isolated(config, audio_file, audio_profile)
    
    def _test_configuration_isolated(self, config: Dict, audio_file: str, audio_profile: AudioProfile) -> Optional[OptimizationResult]:
        """
        Test a single configuration in isolation with unique process identifiers.
        
        Args:
            config: Configuration with process_id
            audio_file: Path to audio file
            audio_profile: Audio characteristics
            
        Returns:
            OptimizationResult or None if failed
        """
        start_time = time.time()
        
        # Generate unique process identifier for file isolation
        process_id = config.get('process_id', 0)
        unique_id = f"{process_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        # Create process-specific temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"whisper_proc_{unique_id}_")
        self.temp_dirs.append(temp_dir)
        
        try:
            # Clear CUDA cache at start and set unique random seed
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.manual_seed(hash(unique_id) % 2**32)
                self.logger.debug(f"Process {unique_id}: CUDA cache cleared, unique seed set")
            
            # Thread-safe transcriber config with unique naming
            transcriber_config = {
                'segment_prefix': f"seg_{unique_id}",  # Unique segment file prefix
                'temp_dir': temp_dir,
                'process_id': unique_id
            }
            
            # Preprocess audio with isolation
            processed_audio = self._preprocess_audio_isolated(audio_file, audio_profile, unique_id, temp_dir)
            
            # Create transcriber with isolated configuration
            transcriber = self._create_transcriber(config, transcriber_config)
            
            # Perform transcription with timeout and isolation
            transcript, segments = self._transcribe_with_timeout_isolated(
                transcriber, processed_audio, timeout=150, unique_id=unique_id
            )
            
            # Validate transcription quality
            if not self._validate_transcription_quality(transcript, segments, audio_profile):
                self.logger.warning(f"Process {unique_id}: Low quality transcription detected")
                return None
              # Count words
            word_count = len(transcript.split()) if transcript else 0
            processing_time = time.time() - start_time
            self.logger.info(f"Process {unique_id}: {config['name']} -> {word_count} words in {processing_time:.1f}s")
            
            return OptimizationResult(
                config_name=config['name'],
                parameters=config,
                word_count=word_count,
                char_count=len(transcript) if transcript else 0,
                segment_count=len(segments),
                processing_time=processing_time,
                success=word_count > 0,
                quality_score=min(1.0, word_count / 100.0),
                audio_profile=audio_profile,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
        except Exception as e:
            self.logger.error(f"Process {unique_id}: Configuration {config['name']} failed: {e}")
            return None
            
        finally:
            # Cleanup process-specific resources
            try:
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
            except Exception as e:
                self.logger.warning(f"Process {unique_id}: Cleanup failed: {e}")
    
    def _preprocess_audio_isolated(self, audio_file: str, audio_profile: AudioProfile, 
                                 unique_id: str, temp_dir: str) -> str:
        """
        Preprocess audio with process isolation and unique file naming.
        
        Args:
            audio_file: Original audio file path
            audio_profile: Audio characteristics
            unique_id: Unique process identifier
            temp_dir: Process-specific temporary directory
            
        Returns:
            Path to processed audio file
        """
        try:
            # For very sparse content, create segments to improve processing
            if audio_profile.speaker_type == "very_sparse" or audio_profile.silence_ratio > 0.6:
                return self._preprocess_audio_for_sparse_content_isolated(
                    audio_file, audio_profile, unique_id, temp_dir
                )
            else:
                # For normal content, return original file
                return audio_file
                
        except Exception as e:
            self.logger.error(f"Process {unique_id}: Audio preprocessing failed: {e}")
            return audio_file
    
    def _preprocess_audio_for_sparse_content_isolated(self, audio_file: str, audio_profile: AudioProfile,
                                                    unique_id: str, temp_dir: str) -> str:
        """
        Preprocess sparse audio content with process isolation.
        
        Args:
            audio_file: Original audio file
            audio_profile: Audio characteristics
            unique_id: Unique process identifier
            temp_dir: Process-specific temporary directory
            
        Returns:
            Path to processed audio file
        """
        try:
            # Import audio processing libraries
            try:
                from pydub import AudioSegment
                from pydub.silence import split_on_silence
            except ImportError:
                self.logger.warning(f"Process {unique_id}: pydub not available, skipping preprocessing")
                return audio_file
            
            # Load audio
            audio = AudioSegment.from_file(audio_file)
            
            # Determine silence threshold based on audio profile
            if audio_profile.background_noise_level > -40:
                silence_thresh = audio.dBFS - 16  # Noisy environment
            else:
                silence_thresh = audio.dBFS - 20  # Quiet environment
            
            # Split on silence with conservative settings for sparse content
            min_silence_len = max(800, int(audio_profile.estimated_pause_length * 600))
            
            chunks = split_on_silence(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=500  # Keep some silence for context
            )
            
            if chunks and len(chunks) > 1:
                # Combine chunks with reduced silence
                combined_audio = AudioSegment.empty()
                for i, chunk in enumerate(chunks):
                    combined_audio += chunk
                    if i < len(chunks) - 1:  # Add minimal silence between chunks
                        combined_audio += AudioSegment.silent(duration=300)
                
                # Export processed audio to temporary directory
                processed_file = os.path.join(temp_dir, f"processed_{unique_id}.wav")
                combined_audio.export(processed_file, format="wav")
                
                self.logger.info(f"Process {unique_id}: Preprocessed sparse audio: {len(chunks)} segments")
                return processed_file
            else:                return audio_file
                
        except Exception as e:
            self.logger.error(f"Process {unique_id}: Sparse content preprocessing failed: {e}")
            return audio_file
    
    def _transcribe_with_timeout_isolated(self, transcriber, audio_file: str, 
                                        timeout: int = 150, unique_id: str = None) -> tuple:
        """
        Perform transcription with timeout and process isolation.
        
        Args:
            transcriber: Transcriber instance
            audio_file: Audio file to transcribe
            timeout: Timeout in seconds
            unique_id: Unique process identifier
            
        Returns:
            Tuple of (transcript, segments)
        """
        if unique_id is None:
            unique_id = f"proc_{int(time.time())}"
            
        try:
            self.logger.debug(f"Process {unique_id}: Starting transcription with {timeout}s timeout")
            
            # Use subprocess for additional isolation if needed
            if hasattr(transcriber, 'transcribe_with_isolation'):
                return transcriber.transcribe_with_isolation(audio_file, timeout=timeout, process_id=unique_id)
            else:
                # Standard transcription with timeout handling
                result = transcriber.transcribe_audio_file(audio_file)
                
                # All transcriber methods return a dictionary with 'full_text' and 'segments'
                if isinstance(result, dict):
                    transcript = result.get('full_text', '')
                    segments = result.get('segments', [])
                else:
                    # Fallback for unexpected return types
                    transcript = str(result) if result else ''
                    segments = []
                
                return transcript, segments
                
        except Exception as e:
            self.logger.error(f"Process {unique_id}: Transcription failed: {e}")
            # Clear CUDA cache on error
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "", []
    
    def _create_transcriber(self, config: Dict, transcriber_config: Dict):
        """
        Create transcriber instance with isolated configuration.
        
        Args:
            config: Base transcription configuration
            transcriber_config: Process-specific transcriber configuration
            
        Returns:
            Configured transcriber instance
        """
        model_name = config.get('model', 'large-v3')
        
        # Merge configurations
        full_config = {**config, **transcriber_config}
        
        # Always use EnhancedAudioTranscriber
        return EnhancedAudioTranscriber(model_name=model_name, config=full_config)
    
    def _validate_transcription_quality(self, transcript: str, segments: list, audio_profile: AudioProfile) -> bool:
        """
        Validate transcription quality based on expected characteristics.
        
        Args:
            transcript: Transcribed text
            segments: Transcription segments
            audio_profile: Audio characteristics
            
        Returns:
            True if quality is acceptable
        """
        if not transcript or not transcript.strip():
            return False
        
        word_count = len(transcript.split())
        
        # Adjust expectations based on speaker type
        if audio_profile.speaker_type in ["very_sparse", "sparse"]:
            min_expected_words = max(3, int(audio_profile.duration_seconds * 0.2))
        else:
            min_expected_words = max(5, int(audio_profile.duration_seconds * 0.5))
        
        if word_count < min_expected_words:
            self.logger.warning(f"Low word count: {word_count} < {min_expected_words}")
            return False
        
        # Check for reasonable segment distribution
        if len(segments) == 0:
            return False
        
        # Check for overly repetitive content (sign of processing error)
        words = transcript.lower().split()
        if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
            self.logger.warning("Transcript appears overly repetitive")
            return False
        
        return True
    
    def _cleanup_temp_dirs(self):
        """Clean up all temporary directories created by this manager."""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")
        
        self.temp_dirs.clear()
    
    def __del__(self):
        """Cleanup on destruction."""
        self._cleanup_temp_dirs()
