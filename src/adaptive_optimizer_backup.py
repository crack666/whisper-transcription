#!/usr/bin/env python3
"""
Adaptive Configuration Optimizer for Whisper Transcription.

This module automatically determines optimal transcription settings based on
audio characteristics and speaker patterns. It learns from previous optimizations
and can adapt to different audio profiles (speakers, recording conditions, etc.).
"""

import os
import sys
import json
import time
import uuid
import tempfile
import shutil
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Add src directory to path if needed
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import extract_audio_from_video

@dataclass
class AudioProfile:
    """Audio characteristics profile for optimization."""
    duration_seconds: float
    mean_volume_db: float
    volume_std: float
    silence_ratio: float
    speech_ratio: float
    speaker_type: str
    estimated_pause_length: float
    background_noise_level: float
    dynamic_range: float
    file_hash: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AudioProfile':
        return cls(**data)

@dataclass 
class OptimizationResult:
    """Result from a single optimization test."""
    config_name: str
    parameters: Dict
    word_count: int
    char_count: int
    segment_count: int
    processing_time: float
    success: bool
    quality_score: float
    audio_profile: AudioProfile
    timestamp: str
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['audio_profile'] = self.audio_profile.to_dict()
        return result

class AdaptiveOptimizer:
    """
    Adaptive configuration optimizer that learns optimal settings
    for different audio profiles and speaker patterns.
    """
    
    def __init__(self, optimization_db_path: str = "optimization_database.json", default_model: str = "large-v3"):
        self.db_path = optimization_db_path
        self.default_model = default_model
        self.logger = logging.getLogger(__name__)
        self.optimization_history = self.load_optimization_history()
        
        # Configuration templates for different optimization strategies
        self.base_configs = {
            "conservative": {
                "min_silence_len": 1000,
                "silence_adjustment": 5.0,
                "min_segment_length": 1000,
                "padding": 1000
            },
            "moderate": {
                "min_silence_len": 1500,
                "silence_adjustment": 4.0,
                "min_segment_length": 1500,
                "padding": 1500
            },
            "aggressive": {
                "min_silence_len": 2000,
                "silence_adjustment": 3.0,
                "min_segment_length": 800,
                "padding": 2000
            },
            "ultra_sensitive": {
                "min_silence_len": 500,
                "silence_adjustment": 8.0,
                "min_segment_length": 500,
                "padding": 750
            },
            "lecture_optimized": {
                "min_silence_len": 2500,
                "silence_adjustment": 4.0,
                "min_segment_length": 1200,
                "padding": 2000
            }
        }
    
    def analyze_audio_profile(self, audio_file: str) -> AudioProfile:
        """Analyze audio characteristics to create a profile."""
        try:
            from pydub import AudioSegment
            from pydub.silence import detect_nonsilent
            
            # Load audio
            audio = AudioSegment.from_file(audio_file)
            duration = len(audio) / 1000.0  # seconds
            
            # Calculate volume statistics
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            
            # Convert to dBFS
            rms = np.sqrt(np.mean(samples**2))
            mean_volume_db = 20 * np.log10(rms / (2**15)) if rms > 0 else -np.inf
            
            # Volume standard deviation
            volume_chunks = []
            chunk_size = len(samples) // 100  # 100 chunks
            for i in range(0, len(samples), chunk_size):
                chunk = samples[i:i+chunk_size]
                if len(chunk) > 0:
                    chunk_rms = np.sqrt(np.mean(chunk**2))
                    if chunk_rms > 0:
                        chunk_db = 20 * np.log10(chunk_rms / (2**15))
                        volume_chunks.append(chunk_db)
            
            volume_std = np.std(volume_chunks) if volume_chunks else 0
            
            # Silence analysis
            silence_thresh = mean_volume_db + 10  # Rough threshold
            nonsilent_ranges = detect_nonsilent(audio, min_silence_len=1000, silence_thresh=silence_thresh)
            
            if nonsilent_ranges:
                speech_duration = sum(end - start for start, end in nonsilent_ranges) / 1000.0
                speech_ratio = speech_duration / duration
                silence_ratio = 1 - speech_ratio
                
                # Estimate pause lengths
                if len(nonsilent_ranges) > 1:
                    gaps = []
                    for i in range(len(nonsilent_ranges) - 1):
                        gap = nonsilent_ranges[i+1][0] - nonsilent_ranges[i][1]
                        gaps.append(gap)
                    estimated_pause_length = np.mean(gaps) if gaps else 1000
                else:
                    estimated_pause_length = 1000
            else:
                speech_ratio = 0.1
                silence_ratio = 0.9
                estimated_pause_length = 2000
            
            # Speaker type classification
            if speech_ratio > 0.8:
                speaker_type = "dense_speech"
            elif speech_ratio > 0.6:
                speaker_type = "moderate_speech"
            elif speech_ratio > 0.3:
                speaker_type = "sparse_speech"
            else:
                speaker_type = "very_sparse"
            
            # Background noise estimation
            silent_chunks = []
            for i in range(0, len(samples), chunk_size):
                chunk = samples[i:i+chunk_size]
                if len(chunk) > 0:
                    chunk_rms = np.sqrt(np.mean(chunk**2))
                    if chunk_rms > 0:
                        chunk_db = 20 * np.log10(chunk_rms / (2**15))
                        silent_chunks.append(chunk_db)
            
            background_noise_level = np.percentile(silent_chunks, 10) if silent_chunks else -40
            dynamic_range = np.percentile(silent_chunks, 90) - np.percentile(silent_chunks, 10) if silent_chunks else 20
            
            # File hash for uniqueness
            with open(audio_file, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:16]
            
            return AudioProfile(
                duration_seconds=duration,
                mean_volume_db=mean_volume_db,
                volume_std=volume_std,
                silence_ratio=silence_ratio,
                speech_ratio=speech_ratio,
                speaker_type=speaker_type,
                estimated_pause_length=estimated_pause_length,
                background_noise_level=background_noise_level,
                dynamic_range=dynamic_range,
                file_hash=file_hash
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze audio profile: {e}")
            # Return default profile
            return AudioProfile(
                duration_seconds=150,
                mean_volume_db=-20,
                volume_std=5,
                silence_ratio=0.3,
                speech_ratio=0.7,
                speaker_type="moderate_speech",
                estimated_pause_length=1500,
                background_noise_level=-30,
                dynamic_range=20,
                file_hash="unknown"
            )
    
    def find_similar_profiles(self, target_profile: AudioProfile, threshold: float = 0.8) -> List[OptimizationResult]:
        """Find optimization results from similar audio profiles."""
        similar_results = []
        
        for result in self.optimization_history:
            if not result.success:
                continue
                
            profile = result.audio_profile
            
            # Calculate similarity score based on multiple factors
            similarity_factors = []
            
            # Speaker type match (high weight)
            if profile.speaker_type == target_profile.speaker_type:
                similarity_factors.append(1.0)
            else:
                similarity_factors.append(0.3)
            
            # Speech ratio similarity
            speech_diff = abs(profile.speech_ratio - target_profile.speech_ratio)
            speech_sim = max(0, 1 - speech_diff / 0.5)  # Normalize by 0.5 range
            similarity_factors.append(speech_sim)
            
            # Pause length similarity
            pause_diff = abs(profile.estimated_pause_length - target_profile.estimated_pause_length)
            pause_sim = max(0, 1 - pause_diff / 2000)  # Normalize by 2 second range
            similarity_factors.append(pause_sim)
            
            # Volume level similarity
            volume_diff = abs(profile.mean_volume_db - target_profile.mean_volume_db)
            volume_sim = max(0, 1 - volume_diff / 20)  # Normalize by 20dB range
            similarity_factors.append(volume_sim)
            
            # Dynamic range similarity
            range_diff = abs(profile.dynamic_range - target_profile.dynamic_range)
            range_sim = max(0, 1 - range_diff / 30)  # Normalize by 30dB range
            similarity_factors.append(range_sim)
            
            # Weighted average (speaker type gets double weight)
            weights = [2.0, 1.0, 1.0, 1.0, 1.0]
            overall_similarity = sum(f * w for f, w in zip(similarity_factors, weights)) / sum(weights)
            
            if overall_similarity >= threshold:
                similar_results.append(result)
        
        # Sort by similarity (higher first)
        return sorted(similar_results, key=lambda r: self.calculate_similarity_score(r.audio_profile, target_profile), reverse=True)
    
    def calculate_similarity_score(self, profile1: AudioProfile, profile2: AudioProfile) -> float:
        """Calculate detailed similarity score between two profiles."""
        if profile1.speaker_type == profile2.speaker_type:
            type_sim = 1.0
        else:
            type_sim = 0.3
        
        speech_sim = max(0, 1 - abs(profile1.speech_ratio - profile2.speech_ratio) / 0.5)
        pause_sim = max(0, 1 - abs(profile1.estimated_pause_length - profile2.estimated_pause_length) / 2000)
        volume_sim = max(0, 1 - abs(profile1.mean_volume_db - profile2.mean_volume_db) / 20)
        range_sim = max(0, 1 - abs(profile1.dynamic_range - profile2.dynamic_range) / 30)
        
        weights = [2.0, 1.0, 1.0, 1.0, 1.0]
        factors = [type_sim, speech_sim, pause_sim, volume_sim, range_sim]
        
        return sum(f * w for f, w in zip(factors, weights)) / sum(weights)
    
    def generate_optimized_configs(self, audio_profile: AudioProfile) -> List[Dict]:
        """Generate optimized configurations based on audio profile and learning history."""
        configs = []
        
        # 1. Find the best performing configuration from all history
        best_config = self.find_best_performing_config()
        
        if best_config:
            self.logger.info(f"🎯 Found best performing config with {best_config['word_count']} words")
            
            # Generate intelligent variations around the best config
            intelligent_configs = self.generate_intelligent_parameter_variations(audio_profile, best_config)
            configs.extend(intelligent_configs)
            
            # Also include the exact best config as a baseline
            best_params = best_config['parameters'].copy()
            best_params.pop('model', None)  # Remove old model reference
            configs.append({
                "name": "proven_best_config",
                "transcriber": "standard",
                "model": self.default_model,
                **{k: v for k, v in best_params.items() if k not in ['name', 'transcriber', 'source']},
                "source": "proven_best"
            })
        
        # 2. Check for similar profiles  
        similar_results = self.find_similar_profiles(audio_profile)
        
        if similar_results:
            self.logger.info(f"📊 Found {len(similar_results)} similar audio profiles")
            
            # Use top 2 configurations from similar profiles
            best_similar = sorted(similar_results, key=lambda r: r.word_count, reverse=True)[:2]
            
            for i, result in enumerate(best_similar):
                config = result.parameters.copy()
                config_name = f"learned_similar_{i+1}"
                
                # Force use of default model (large-v3) regardless of what's in history
                config.pop('model', None)  # Remove old model reference
                
                configs.append({
                    "name": config_name,
                    "transcriber": "standard",
                    "model": self.default_model,
                    **{k: v for k, v in config.items() if k not in ['name', 'transcriber', 'source']},
                    "source": "learned_from_similar"
                })
        
        # 3. Generate smart parameter variations if no good history
        if not best_config and not similar_results:
            self.logger.info("🧠 No optimization history - generating smart variations")
            intelligent_configs = self.generate_intelligent_parameter_variations(audio_profile)
            configs.extend(intelligent_configs)
        
        # 4. Add only the most promising base configurations (reduce redundancy)
        promising_bases = ["conservative", "moderate"]  # These have shown better results
        for base_name in promising_bases:
            if base_name in self.base_configs:
                base_config = self.base_configs[base_name]
                configs.append({
                    "name": f"base_{base_name}",
                    "transcriber": "standard",
                    "model": self.default_model,
                    **base_config,
                    "source": "base_template"
                })
        
        return configs
    
    def generate_profile_adaptive_configs(self, profile: AudioProfile) -> List[Dict]:
        """Generate configurations specifically adapted to the audio profile."""
        configs = []
        
        # Adaptive configuration based on speech patterns
        if profile.speaker_type == "dense_speech":
            # Dense speech - need more aggressive silence detection
            configs.append({
                "name": "adaptive_dense_speech",
                "transcriber": "standard",
                "model": self.default_model,
                "min_silence_len": max(1000, int(profile.estimated_pause_length * 0.5)),
                "silence_adjustment": 3.0,
                "min_segment_length": 800,
                "padding": 1500,
                "source": "adaptive_dense"
            })
            
        elif profile.speaker_type == "sparse_speech":
            # Sparse speech - longer pauses, more conservative
            configs.append({
                "name": "adaptive_sparse_speech",
                "transcriber": "standard",
                "model": self.default_model,
                "min_silence_len": min(3000, int(profile.estimated_pause_length * 1.2)),
                "silence_adjustment": 5.0,
                "min_segment_length": 1500,
                "padding": 2000,
                "source": "adaptive_sparse"
            })
            
        else:
            # Moderate speech - balanced approach
            configs.append({
                "name": "adaptive_moderate_speech",
                "transcriber": "standard",
                "model": self.default_model,
                "min_silence_len": int(profile.estimated_pause_length),
                "silence_adjustment": 4.0,
                "min_segment_length": 1200,
                "padding": 1500,
                "source": "adaptive_moderate"
            })
        
        # Volume-adaptive configuration
        if profile.mean_volume_db < -25:
            # Quiet audio - more sensitive detection
            silence_adj = 8.0
        elif profile.mean_volume_db > -15:
            # Loud audio - less sensitive
            silence_adj = 2.0
        else:
            # Normal volume
            silence_adj = 4.0
        
        configs.append({
            "name": "adaptive_volume_optimized",
            "transcriber": "standard",
            "model":  self.default_model,
            "min_silence_len": 1500,
            "silence_adjustment": silence_adj,
            "min_segment_length": 1000,
            "padding": 1500,
            "source": "adaptive_volume"
        })
        
        return configs
    
    def run_adaptive_optimization(self, audio_file: str, max_configs: int = 8, use_parallel: bool = True) -> Dict:
        """Run adaptive optimization on an audio file.
        
        Args:
            audio_file: Path to audio file
            max_configs: Maximum number of configurations to test
            use_parallel: Whether to use parallel processing (False for memory-constrained scenarios)
        """
        self.logger.info(f"🎯 Starting adaptive optimization for: {audio_file}")
        
        # Analyze previous optimization results for insights
        if self.optimization_history:
            self.logger.info("📊 Analyzing parameter impact from history...")
            parameter_analysis = self.analyze_parameter_impact()
            
            if "error" not in parameter_analysis:
                insights = parameter_analysis.get("optimization_insights", [])
                for insight in insights[:3]:  # Show top 3 insights
                    self.logger.info(f"   💡 {insight}")
                
                # Show best parameter ranges
                silence_rec = parameter_analysis["min_silence_len_analysis"]["recommendation"]
                adj_rec = parameter_analysis["silence_adjustment_analysis"]["recommendation"] 
                self.logger.info(f"   🎛️  Silence detection: {silence_rec}")
                self.logger.info(f"   🎛️  Adjustment level: {adj_rec}")
        
        # Pre-extract audio if needed (one-time operation)
        self.logger.info("🎵 Pre-extracting audio for optimization...")
        processed_audio_file = self._ensure_audio_extracted(audio_file)
        
        # Analyze audio profile
        self.logger.info("📊 Analyzing audio profile...")
        audio_profile = self.analyze_audio_profile(processed_audio_file)
        
        self.logger.info(f"   Speaker type: {audio_profile.speaker_type}")
        self.logger.info(f"   Speech ratio: {audio_profile.speech_ratio:.2f}")
        self.logger.info(f"   Estimated pause length: {audio_profile.estimated_pause_length:.0f}ms")
        self.logger.info(f"   Mean volume: {audio_profile.mean_volume_db:.1f}dB")
        
        # Generate optimized configurations using new intelligent system
        self.logger.info("🧠 Generating adaptive configurations...")
        test_configs = self.generate_optimized_configs(audio_profile)[:max_configs]
        
        self.logger.info(f"   Generated {len(test_configs)} configurations")
        for config in test_configs:
            model = config.get('model', 'unknown')
            source = config.get('source', 'unknown')
            silence_len = config.get('min_silence_len', 'N/A')
            silence_adj = config.get('silence_adjustment', 'N/A')
            self.logger.info(f"   - {config['name']} (silence: {silence_len}ms, adj: {silence_adj}, source: {source})")
        
        # Run optimization tests
        results = []
        
        if use_parallel:
            self.logger.info("🚀 Running tests in parallel (process isolation)...")
            # Use very conservative worker count to prevent CUDA conflicts
            max_workers = min(2, max(1, os.cpu_count() // 4))  # Ultra-conservative for CUDA stability
            
            # Use ProcessPoolExecutor for better isolation
            from concurrent.futures import ProcessPoolExecutor
            
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit each config with unique process ID for file isolation
                    future_to_config = {}
                    for i, config in enumerate(test_configs):
                        # Add process isolation info to config
                        isolated_config = config.copy()
                        isolated_config['process_id'] = i
                        isolated_config['total_processes'] = len(test_configs)
                        
                        future = executor.submit(self._test_configuration_isolated, isolated_config, processed_audio_file, audio_profile)
                        future_to_config[future] = config
                    
                    for future in as_completed(future_to_config):
                        config = future_to_config[future]
                        try:
                            result = future.result(timeout=180)  # Extended timeout for process overhead
                            results.append(result)
                            
                            efficiency = result.word_count / result.processing_time if result.processing_time > 0 else 0
                            self.logger.info(f"   ✅ {config['name']}: {result.word_count} words, {result.segment_count} segments, {result.processing_time:.1f}s ({efficiency:.1f} w/s)")
                        
                        except Exception as e:
                            self.logger.error(f"   ❌ {config['name']} failed: {e}")
                            # Add failed result
                            results.append(OptimizationResult(
                                config_name=config['name'],
                                parameters=config,
                                word_count=0,
                                char_count=0,
                                segment_count=0,
                                processing_time=0,
                                success=False,
                                quality_score=0.0,
                                audio_profile=audio_profile,
                                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                            ))
            
            except Exception as e:
                self.logger.error(f"Parallel processing failed, falling back to sequential: {e}")
                # Fall back to sequential processing
                use_parallel = False
        else:
            self.logger.info("🔄 Running tests sequentially (memory-safe)...")
            for i, config in enumerate(test_configs, 1):
                self.logger.info(f"   Testing {i}/{len(test_configs)}: {config['name']}")
                try:
                    result = self.test_configuration(config, processed_audio_file, audio_profile)
                    results.append(result)
                    
                    efficiency = result.word_count / result.processing_time if result.processing_time > 0 else 0
                    self.logger.info(f"   ✅ {result.word_count} words, {result.segment_count} segments, {result.processing_time:.1f}s ({efficiency:.1f} w/s)")
                
                except Exception as e:
                    self.logger.error(f"   ❌ Test failed: {e}")
        
        # Analyze results and save to history
        successful_results = [r for r in results if r.success]
        
        if successful_results:
            # Add to optimization history
            self.optimization_history.extend(successful_results)
            self.save_optimization_history()
            
            # Find best configuration
            best_result = max(successful_results, key=lambda r: r.word_count)
            
            # Compare with previous best
            previous_best = self.find_best_performing_config()
            improvement = ""
            if previous_best:
                improvement_pct = ((best_result.word_count - previous_best['word_count']) / previous_best['word_count']) * 100
                if improvement_pct > 0:
                    improvement = f" (+{improvement_pct:.1f}% vs previous best)"
                elif improvement_pct < -10:
                    improvement = f" ({improvement_pct:.1f}% vs previous best)"
            
            analysis = {
                "audio_profile": audio_profile.to_dict(),
                "total_tests": len(results),
                "successful_tests": len(successful_results),
                "best_config": {
                    "name": best_result.config_name,
                    "word_count": best_result.word_count,
                    "parameters": best_result.parameters,
                    "processing_time": best_result.processing_time,
                    "efficiency": best_result.word_count / best_result.processing_time if best_result.processing_time > 0 else 0
                },
                "all_results": [r.to_dict() for r in results],
                "parameter_analysis": self.analyze_parameter_impact() if len(self.optimization_history) >= 3 else {"note": "Need more data for parameter analysis"}
            }
            
            self.logger.info(f"\n🏆 Best configuration: {best_result.config_name}")
            self.logger.info(f"   Words: {best_result.word_count}{improvement}")
            self.logger.info(f"   Time: {best_result.processing_time:.1f}s")
            self.logger.info(f"   Efficiency: {best_result.word_count / best_result.processing_time:.1f} words/sec")
            
            # Show parameter insights if available
            if "parameter_analysis" in analysis and "error" not in analysis["parameter_analysis"]:
                param_insights = analysis["parameter_analysis"]["optimization_insights"]
                if param_insights:
                    self.logger.info(f"\n💡 Key insights:")
                    for insight in param_insights[:2]:
                        self.logger.info(f"   {insight}")
            
            return analysis
        else:
            self.logger.error("❌ No successful configurations found")
            return {
                "audio_profile": audio_profile.to_dict(),
                "total_tests": len(results),
                "successful_tests": 0,
                "error": "No configurations succeeded"
            }
    
    def _test_configuration_isolated(self, config: Dict, audio_file: str, audio_profile: AudioProfile) -> OptimizationResult:
        """Test configuration in isolated process to prevent CUDA conflicts."""
        import os
        import time
        import threading
        import tempfile
        import uuid
        from pathlib import Path
        
        # Generate unique process identifier for file isolation
        process_id = config.get('process_id', 0)
        unique_id = f"{process_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        start_time = time.time()
        max_retries = 2  # Reduced retries for process isolation
        last_error = None
        
        # Create thread-safe transcriber config with unique naming
        transcriber_config = {
            'min_silence_len': config['min_silence_len'],
            'padding': config['padding'],
            'silence_adjustment': config['silence_adjustment'],
            'cleanup_segments': True,
            'segment_prefix': f"seg_{unique_id}"  # Unique segment file prefix
        }
        
        # Force process to use different CUDA context
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache at start
            # Set different random seed to avoid identical CUDA states
            torch.manual_seed(hash(unique_id) % 2**32)
        
        for attempt in range(max_retries):
            try:
                # Force CPU fallback on first failure for stability
                if attempt == 0:
                    device_strategy = None  # Auto-detect
                else:
                    device_strategy = "cpu"  # Force CPU fallback
                    
                # Import here to avoid module conflicts in multiprocessing
                from src.transcriber import AudioTranscriber
                
                transcriber = AudioTranscriber(
                    model_name=config['model'],
                    language='german',
                    config=transcriber_config,
                    device=device_strategy
                )
                
                # Create process-specific temporary directory
                temp_dir = tempfile.mkdtemp(prefix=f"whisper_process_{unique_id}_")
                
                try:
                    # Preprocess audio with unique naming
                    processed_audio = self._preprocess_audio_isolated(audio_file, audio_profile, unique_id, temp_dir)
                    
                    # Run transcription with extended timeout for process overhead
                    result = self._transcribe_with_timeout_isolated(transcriber, processed_audio, timeout=150, unique_id=unique_id)
                    
                    # Extract metrics
                    transcript = result.get('full_text', '')
                    segments = result.get('segments', [])
                    
                    # Validate result quality
                    if self._validate_transcription_quality(transcript, segments, audio_profile):
                        word_count = len(transcript.split()) if transcript else 0
                        char_count = len(transcript) if transcript else 0
                        segment_count = len(segments)
                        processing_time = time.time() - start_time
                        quality_score = min(1.0, word_count / 100.0)
                        
                        return OptimizationResult(
                            config_name=config['name'],
                            parameters=config,
                            word_count=word_count,
                            char_count=char_count,
                            segment_count=segment_count,
                            processing_time=processing_time,
                            success=word_count > 0,
                            quality_score=quality_score,
                            audio_profile=audio_profile,
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                        )
                        
                finally:
                    # Safe cleanup of process-specific files
                    try:
                        import shutil
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)
                    except Exception as cleanup_error:
                        pass  # Ignore cleanup errors
                    
            except Exception as e:
                last_error = e
                # Clear CUDA cache on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Wait before retry
                if attempt < max_retries - 1:
                    time.sleep(3)  # Longer wait for process isolation
        
        # All attempts failed
        processing_time = time.time() - start_time
        
        return OptimizationResult(
            config_name=config['name'],
            parameters=config,
            word_count=0,
            char_count=0,
            segment_count=0,
            processing_time=processing_time,
            success=False,
            quality_score=0.0,
            audio_profile=audio_profile,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

    def _preprocess_audio_isolated(self, audio_file: str, audio_profile: AudioProfile, unique_id: str, temp_dir: str) -> str:
        """Preprocess audio with process isolation."""
        import subprocess
        import os
        
        # For very sparse content, apply audio enhancement
        if audio_profile.speaker_type == "very_sparse" or audio_profile.silence_ratio > 0.8:
            try:
                # Create process-specific enhanced audio file
                enhanced_audio = os.path.join(temp_dir, f"enhanced_{unique_id}.wav")
                
                # Apply audio enhancements: normalize, reduce noise, boost speech
                enhancement_filters = [
                    "highpass=f=80",           # Remove low-frequency noise
                    "lowpass=f=8000",          # Remove high-frequency noise  
                    "volume=1.5",              # Boost volume
                    "compand=0.02,0.20:-60/-60,-30/-15,-20/-10,-5/-5,0/-3:6:0:0:0", # Dynamic range compression
                    "speechnorm=e=25:r=0.00001:l=1"  # Speech normalization
                ]
                
                cmd = [
                    'ffmpeg', '-y', '-i', audio_file,
                    '-af', ','.join(enhancement_filters),
                    '-ar', '16000',  # Standard sample rate for Whisper
                    '-ac', '1',      # Mono
                    enhanced_audio
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and os.path.exists(enhanced_audio):
                    return enhanced_audio
                    
            except Exception:
                pass  # Fall back to original
        
        return audio_file

    def _transcribe_with_timeout_isolated(self, transcriber, audio_file: str, timeout: int = 150, unique_id: str = None):
        """Run transcription with timeout protection in isolated process."""
        import threading
        import time
        
        result = None
        exception = None
        
        def transcribe():
            nonlocal result, exception
            try:
                # Set process-specific environment
                import os
                os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Force synchronous CUDA
                
                result = transcriber.transcribe_audio_file(audio_file)
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=transcribe, name=f"transcribe_{unique_id}")
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Timeout occurred
            raise TimeoutError(f"Transcription timed out after {timeout}s in process {unique_id}")
        
        if exception:
            raise exception
            
        return result if result else {"full_text": "", "segments": []}

    def test_configuration(self, config: Dict, audio_file: str, audio_profile: AudioProfile) -> OptimizationResult:
        """Test a single configuration with comprehensive error handling."""
        from src.transcriber import AudioTranscriber
        
        start_time = time.time()
        max_retries = 3
        last_error = None
        
        # Create transcriber with config
        transcriber_config = {
            'min_silence_len': config['min_silence_len'],
            'padding': config['padding'],
            'silence_adjustment': config['silence_adjustment'],
            'cleanup_segments': True
        }
        
        for attempt in range(max_retries):
            try:
                # Try different device strategies on failures
                device_strategy = "auto"
                if attempt == 1:
                    device_strategy = "cuda"
                elif attempt == 2:
                    device_strategy = "cpu"
                    
                transcriber = AudioTranscriber(
                    model_name=config['model'],
                    language='german',
                    config=transcriber_config,
                    device=device_strategy if device_strategy != "auto" else None
                )
                
                # Preprocess audio for very sparse content
                processed_audio = self._preprocess_audio_for_sparse_content(audio_file, audio_profile)
                
                # Run transcription with timeout
                result = self._transcribe_with_timeout(transcriber, processed_audio, timeout=120)
                
                # Extract metrics
                transcript = result.get('full_text', '')
                segments = result.get('segments', [])
                
                # Validate result quality
                if self._validate_transcription_quality(transcript, segments, audio_profile):
                    word_count = len(transcript.split()) if transcript else 0
                    char_count = len(transcript) if transcript else 0
                    segment_count = len(segments)
                    processing_time = time.time() - start_time
                    quality_score = min(1.0, word_count / 100.0)
                    
                    # Clean up temporary files
                    if processed_audio != audio_file and os.path.exists(processed_audio):
                        os.remove(processed_audio)
                    
                    return OptimizationResult(
                        config_name=config['name'],
                        parameters=config,
                        word_count=word_count,
                        char_count=char_count,
                        segment_count=segment_count,
                        processing_time=processing_time,
                        success=word_count > 0,
                        quality_score=quality_score,
                        audio_profile=audio_profile,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed for {config['name']}: {e}")
                
                # Clean up on error
                try:
                    if 'processed_audio' in locals() and processed_audio != audio_file and os.path.exists(processed_audio):
                        os.remove(processed_audio)
                except:
                    pass
                
                # Wait before retry
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        # All attempts failed
        processing_time = time.time() - start_time
        self.logger.error(f"All attempts failed for {config['name']}: {last_error}")
        
        return OptimizationResult(
            config_name=config['name'],
            parameters=config,
            word_count=0,
            char_count=0,
            segment_count=0,
            processing_time=processing_time,
            success=False,
            quality_score=0.0,
            audio_profile=audio_profile,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _preprocess_audio_for_sparse_content(self, audio_file: str, audio_profile: AudioProfile) -> str:
        """Preprocess audio file specifically for very sparse content."""
        import subprocess
        import tempfile
        
        # For very sparse content, apply audio enhancement
        if audio_profile.speaker_type == "very_sparse" or audio_profile.silence_ratio > 0.8:
            try:
                # Create temporary enhanced audio file
                temp_dir = tempfile.gettempdir()
                enhanced_audio = os.path.join(temp_dir, f"enhanced_{os.path.basename(audio_file)}")
                
                # Apply audio enhancements: normalize, reduce noise, boost speech
                enhancement_filters = [
                    "highpass=f=80",           # Remove low-frequency noise
                    "lowpass=f=8000",          # Remove high-frequency noise  
                    "volume=1.5",              # Boost volume
                    "compand=0.02,0.20:-60/-60,-30/-15,-20/-10,-5/-5,0/-3:6:0:0:0", # Dynamic range compression
                    "speechnorm=e=25:r=0.00001:l=1"  # Speech normalization
                ]
                
                cmd = [
                    'ffmpeg', '-y', '-i', audio_file,
                    '-af', ','.join(enhancement_filters),
                    '-ar', '16000',  # Standard sample rate for Whisper
                    '-ac', '1',      # Mono
                    enhanced_audio
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and os.path.exists(enhanced_audio):
                    self.logger.info(f"Enhanced audio for sparse content: {enhanced_audio}")
                    return enhanced_audio
                else:
                    self.logger.warning(f"Audio enhancement failed, using original: {result.stderr}")
                    
            except Exception as e:
                self.logger.warning(f"Audio enhancement failed: {e}")
        
        return audio_file
    
    def _transcribe_with_timeout(self, transcriber, audio_file: str, timeout: int = 120):
        """Run transcription with timeout protection."""
        import signal
        import threading
        
        result = None
        exception = None
        
        def transcribe():
            nonlocal result, exception
            try:
                result = transcriber.transcribe_audio_file(audio_file)
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=transcribe)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Timeout occurred
            raise TimeoutError(f"Transcription timed out after {timeout}s")
        
        if exception:
            raise exception
            
        return result if result else {"full_text": "", "segments": []}
    
    def _validate_transcription_quality(self, transcript: str, segments: list, audio_profile: AudioProfile) -> bool:
        """Validate transcription quality and detect obvious failures."""
        if not transcript or not transcript.strip():
            return False
            
        word_count = len(transcript.split())
        
        # For very sparse content, even small word counts can be valid
        if audio_profile.speaker_type == "very_sparse":
            min_expected_words = max(1, int(audio_profile.duration_seconds * 0.1))  # Very low threshold
        else:
            min_expected_words = max(5, int(audio_profile.duration_seconds * 0.5))  # Normal threshold
        
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
    
    def generate_intelligent_parameter_variations(self, base_config: Dict, audio_profile: AudioProfile, count: int = 6) -> List[Dict]:
        """Generate intelligent parameter variations based on analysis and audio profile."""
        variations = []
        
        # Get parameter insights from historical data
        param_analysis = self.analyze_parameter_impact()
        
        # Base proven configuration
        proven_best = self.find_best_performing_config()
        if proven_best:
            base_variant = {
                "name": "proven_best_config",
                "transcriber": "standard",
                "model": base_config['model'],
                "min_silence_len": proven_best['min_silence_len'],
                "silence_adjustment": proven_best['silence_adjustment'],
                "min_segment_length": proven_best.get('min_segment_length', 1000),
                "padding": proven_best.get('padding', 1000),
                "source": "proven_best"
            }
            variations.append(base_variant)
        
        # Audio profile adapted configurations
        if audio_profile.speaker_type == "very_sparse":
            # Very conservative approach for very sparse content
            variations.extend([
                {
                    "name": "very_sparse_optimized",
                    "transcriber": "standard", 
                    "model": base_config['model'],
                    "min_silence_len": 3000,  # Longer silence detection
                    "silence_adjustment": 3.0,  # Lower sensitivity to avoid false positives
                    "min_segment_length": 2000,
                    "padding": 2500,
                    "source": "very_sparse_optimized"
                },
                {
                    "name": "very_sparse_enhanced",
                    "transcriber": "standard",
                    "model": base_config['model'], 
                    "min_silence_len": 2500,
                    "silence_adjustment": 4.0,
                    "min_segment_length": 1500,
                    "padding": 2000,
                    "source": "very_sparse_enhanced"
                },
                {
                    "name": "very_sparse_aggressive",
                    "transcriber": "standard",
                    "model": base_config['model'],
                    "min_silence_len": 1500,
                    "silence_adjustment": 6.0,  # Higher sensitivity for quiet speech
                    "min_segment_length": 1000,
                    "padding": 1500,
                    "source": "very_sparse_aggressive"
                }
            ])
        
        elif audio_profile.speaker_type == "sparse":
            # Standard sparse optimizations
            variations.extend([
                {
                    "name": "sparse_balanced",
                    "transcriber": "standard",
                    "model": base_config['model'],
                    "min_silence_len": 2000,
                    "silence_adjustment": 4.0,
                    "min_segment_length": 1200,
                    "padding": 1500,
                    "source": "sparse_balanced"
                },
                {
                    "name": "sparse_sensitive",
                    "transcriber": "standard",
                    "model": base_config['model'],
                    "min_silence_len": 1200,
                    "silence_adjustment": 5.5,
                    "min_segment_length": 1000,
                    "padding": 1200,
                    "source": "sparse_sensitive"
                }
            ])
        
        # Parameter analysis based variations
        if "error" not in param_analysis:
            # Best silence length range
            if param_analysis.get("min_silence_len_analysis", {}).get("best_range") == "short":
                variations.append({
                    "name": "optimized_variant_1",
                    "transcriber": "standard",
                    "model": base_config['model'],
                    "min_silence_len": 1000,  # Short range
                    "silence_adjustment": 5.0,
                    "min_segment_length": 800,
                    "padding": 1000,
                    "source": "intelligent_variation"
                })
            
            # Best adjustment range
            if param_analysis.get("silence_adjustment_analysis", {}).get("best_range") == "high":
                variations.append({
                    "name": "optimized_variant_2", 
                    "transcriber": "standard",
                    "model": base_config['model'],
                    "min_silence_len": 1200,
                    "silence_adjustment": 4.0,  # Reduced from previous high values
                    "min_segment_length": 800,
                    "padding": 1200,
                    "source": "intelligent_variation"
                })
        
        # Conservative fallback configurations
        variations.extend([
            {
                "name": "conservative_baseline", 
                "transcriber": "standard",
                "model": base_config['model'],
                "min_silence_len": 1500,
                "silence_adjustment": 3.5,
                "min_segment_length": 1000,
                "padding": 1500,
                "source": "conservative_baseline"
            },
            {
                "name": "medium_sensitivity",
                "transcriber": "standard", 
                "model": base_config['model'],
                "min_silence_len": 1200,
                "silence_adjustment": 4.5,
                "min_segment_length": 900,
                "padding": 1200,
                "source": "medium_sensitivity"
            }
        ])
        
        # Remove duplicates and limit count
        seen_configs = set()
        unique_variations = []
        
        for variant in variations:
            # Create a signature based on key parameters
            signature = (variant['min_silence_len'], variant['silence_adjustment'], variant['padding'])
            if signature not in seen_configs:
                seen_configs.add(signature)
                unique_variations.append(variant)
                
                if len(unique_variations) >= count:
                    break
        
        self.logger.info(f"Generated {len(unique_variations)} intelligent parameter variations")
        for i, variant in enumerate(unique_variations[:3], 1):
            self.logger.info(f"  {i}. {variant['name']} (silence: {variant['min_silence_len']}ms, adj: {variant['silence_adjustment']}, source: {variant['source']})")
        
        return unique_variations[:count]