#!/usr/bin/env python3
"""
Adaptive Configuration Optimizer for Whisper Transcription.

This module coordinates the optimization process by orchestrating audio analysis,
configuration generation, and parallel testing to find optimal transcription settings.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Add src directory to path if needed
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .audio_profile_analyzer import AudioProfile, OptimizationResult, AudioProfileAnalyzer
from .configuration_generator import ConfigurationGenerator
from .process_isolation_manager import ProcessIsolationManager
from .utils import extract_audio_from_video

class AdaptiveOptimizer:
    """
    Orchestrates the adaptive optimization process using specialized components
    for audio analysis, configuration generation, and parallel testing.
    """
    
    def __init__(self, optimization_db_path: str = "optimization_database.json", default_model: str = "large-v3"):
        self.db_path = optimization_db_path
        self.default_model = default_model
        self.logger = logging.getLogger(__name__)
        
        # Initialize specialized components
        self.audio_analyzer = AudioProfileAnalyzer()
        self.config_generator = ConfigurationGenerator()
        self.process_manager = ProcessIsolationManager()
        
        # Load optimization history
        self.optimization_history = self.load_optimization_history()
        
        self.logger.info(f"Initialized AdaptiveOptimizer with {len(self.optimization_history)} historical results")
    
    def run_adaptive_optimization(self, audio_file: str, max_configs: int = 8, use_parallel: bool = True) -> Dict:
        """
        Run adaptive optimization on an audio file.
        
        Args:
            audio_file: Path to audio file
            max_configs: Maximum number of configurations to test
            use_parallel: Whether to use parallel processing
            
        Returns:
            Dictionary with optimization results and analysis
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
        
        # Pre-extract audio if needed (one-time operation)
        self.logger.info("🎵 Pre-extracting audio for optimization...")
        processed_audio_file = self._ensure_audio_extracted(audio_file)
        
        # Analyze audio profile using specialized analyzer
        self.logger.info("📊 Analyzing audio profile...")
        audio_profile = self.audio_analyzer.analyze_audio_profile(processed_audio_file)
        
        self.logger.info(f"   Speaker type: {audio_profile.speaker_type}")
        self.logger.info(f"   Speech ratio: {audio_profile.speech_ratio:.2f}")
        self.logger.info(f"   Estimated pause length: {audio_profile.estimated_pause_length:.0f}ms")
        self.logger.info(f"   Mean volume: {audio_profile.mean_volume_db:.1f}dB")
        
        # Generate optimized configurations using specialized generator
        self.logger.info("🧠 Generating adaptive configurations...")
        test_configs = self.config_generator.generate_optimized_configs(
            audio_profile, self.optimization_history
        )[:max_configs]
        
        self.logger.info(f"   Generated {len(test_configs)} configurations")
        for config in test_configs:
            source = config.get('source', 'unknown')
            silence_len = config.get('min_silence_len', 'N/A')
            silence_adj = config.get('silence_adjustment', 'N/A')
            self.logger.info(f"   - {config['name']} (silence: {silence_len}ms, adj: {silence_adj}, source: {source})")
        
        # Run optimization tests using specialized process manager
        if use_parallel:
            self.logger.info("🚀 Running tests in parallel (process isolation)...")
            max_workers = min(2, max(1, os.cpu_count() // 4))  # Conservative for CUDA stability
            results = self.process_manager.test_configurations_parallel(
                test_configs, processed_audio_file, audio_profile, max_workers
            )
        else:
            self.logger.info("🔄 Running tests sequentially (memory-safe)...")
            results = self._test_configurations_sequential(test_configs, processed_audio_file, audio_profile)
        
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
            
            return analysis
        else:
            self.logger.error("❌ No successful configurations found")
            return {
                "audio_profile": audio_profile.to_dict(),
                "total_tests": len(results),
                "successful_tests": 0,
                "error": "No configurations succeeded"
            }
    
    def _test_configurations_sequential(self, configs: List[Dict], audio_file: str, 
                                      audio_profile: AudioProfile) -> List[OptimizationResult]:
        """Test configurations sequentially (fallback for memory constraints)."""
        results = []
        
        for i, config in enumerate(configs, 1):
            self.logger.info(f"   Testing {i}/{len(configs)}: {config['name']}")
            try:
                result = self._test_single_configuration(config, audio_file, audio_profile)
                results.append(result)
                
                efficiency = result.word_count / result.processing_time if result.processing_time > 0 else 0
                self.logger.info(f"   ✅ {result.word_count} words, {result.processing_time:.1f}s ({efficiency:.1f} w/s)")
            
            except Exception as e:
                self.logger.error(f"   ❌ Test failed: {e}")
                # Add failed result for tracking
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
        
        return results
    
    def _test_single_configuration(self, config: Dict, audio_file: str, 
                                 audio_profile: AudioProfile) -> OptimizationResult:
        """Test a single configuration with fallback handling."""
        from .transcriber import AudioTranscriber
        
        start_time = time.time()
        
        try:
            # Create transcriber with config
            transcriber_config = {
                'min_silence_len': config['min_silence_len'],
                'padding': config['padding'],
                'silence_adjustment': config['silence_adjustment'],
                'cleanup_segments': True            }
            
            transcriber = AudioTranscriber(
                model_name=config['model'],
                config=transcriber_config
            )
            
            # Run transcription
            result = transcriber.transcribe_audio_file(audio_file)
            
            # Extract metrics - transcriber returns a dictionary
            if isinstance(result, dict):
                transcript = result.get('full_text', '')
                segments = result.get('segments', [])
            else:
                # Fallback for unexpected return types
                transcript = str(result) if result else ''
                segments = []
            
            # Validate and return result
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
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Configuration {config['name']} failed: {e}")
            
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
    
    def _ensure_audio_extracted(self, audio_file: str) -> str:
        """Ensure we have an audio file (extract from video if needed)."""
        if audio_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            self.logger.info("Converting video to audio for optimization...")
            return extract_audio_from_video(audio_file)
        return audio_file
    
    def load_optimization_history(self) -> List[OptimizationResult]:
        """Load optimization history from database."""
        if not os.path.exists(self.db_path):
            return []
        
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = []
            for item in data:
                try:
                    # Convert audio profile dict back to object
                    if 'audio_profile' in item and isinstance(item['audio_profile'], dict):
                        item['audio_profile'] = AudioProfile.from_dict(item['audio_profile'])
                    
                    results.append(OptimizationResult(**item))
                except Exception as e:
                    self.logger.warning(f"Skipping invalid history item: {e}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to load optimization history: {e}")
            return []
    
    def save_optimization_history(self):
        """Save optimization history to database."""
        try:
            data = [result.to_dict() for result in self.optimization_history]
            
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save optimization history: {e}")
    
    def find_best_performing_config(self) -> Optional[Dict]:
        """Find the best performing configuration from history."""
        if not self.optimization_history:
            return None
        
        successful_results = [r for r in self.optimization_history if r.success]
        if not successful_results:
            return None
        
        best_result = max(successful_results, key=lambda r: r.word_count)
        return {
            'word_count': best_result.word_count,
            'parameters': best_result.parameters,
            'config_name': best_result.config_name
        }
    
    def analyze_parameter_impact(self) -> Dict:
        """Analyze parameter impact from historical data."""
        if not self.optimization_history:
            return {"error": "No optimization history available"}
        
        successful_results = [r for r in self.optimization_history if r.success]
        if len(successful_results) < 3:
            return {"error": "Need at least 3 successful results for analysis"}
        
        # Group results by parameter ranges
        silence_len_groups = {"short": [], "medium": [], "long": []}
        adjustment_groups = {"low": [], "medium": [], "high": []}
        
        for result in successful_results:
            params = result.parameters
            silence_len = params.get('min_silence_len', 1500)
            adjustment = params.get('silence_adjustment', 4.0)
            
            # Categorize silence length
            if silence_len < 1300:
                silence_len_groups["short"].append(result)
            elif silence_len < 2000:
                silence_len_groups["medium"].append(result)
            else:
                silence_len_groups["long"].append(result)
            
            # Categorize adjustment
            if adjustment < 3.5:
                adjustment_groups["low"].append(result)
            elif adjustment < 5.0:
                adjustment_groups["medium"].append(result)
            else:
                adjustment_groups["high"].append(result)
        
        # Calculate averages
        def calc_avg_words(group):
            return sum(r.word_count for r in group) / len(group) if group else 0
        
        silence_analysis = {
            "short_avg": calc_avg_words(silence_len_groups["short"]),
            "medium_avg": calc_avg_words(silence_len_groups["medium"]),
            "long_avg": calc_avg_words(silence_len_groups["long"])
        }
        
        adjustment_analysis = {
            "low_avg": calc_avg_words(adjustment_groups["low"]),
            "medium_avg": calc_avg_words(adjustment_groups["medium"]),
            "high_avg": calc_avg_words(adjustment_groups["high"])
        }
        
        # Find best ranges
        best_silence_range = max(silence_analysis.keys(), key=lambda k: silence_analysis[k])
        best_adjustment_range = max(adjustment_analysis.keys(), key=lambda k: adjustment_analysis[k])
        
        # Generate insights
        insights = []
        top_performers = sorted(successful_results, key=lambda r: r.word_count, reverse=True)[:5]
        
        if len(top_performers) >= 3:
            avg_silence = sum(r.parameters.get('min_silence_len', 1500) for r in top_performers[:3]) / 3
            avg_adjustment = sum(r.parameters.get('silence_adjustment', 4.0) for r in top_performers[:3]) / 3
            
            insights.append(f"Top performers average {avg_silence:.0f}ms silence length and {avg_adjustment:.1f} adjustment")
            insights.append(f"Best silence length range: {best_silence_range} ({silence_analysis[best_silence_range]:.1f} avg words)")
            insights.append(f"Best adjustment range: {best_adjustment_range} ({adjustment_analysis[best_adjustment_range]:.1f} avg words)")
        
        return {
            "total_results_analyzed": len(successful_results),
            "min_silence_len_analysis": {
                "best_range": best_silence_range,
                "best_range_avg_words": silence_analysis[best_silence_range],
                "recommendation": f"Use {best_silence_range} silence length values"
            },
            "silence_adjustment_analysis": {
                "best_range": best_adjustment_range,
                "best_range_avg_words": adjustment_analysis[best_adjustment_range],
                "recommendation": f"Use {best_adjustment_range} adjustment values"
            },
            "top_performing_configs": [
                {
                    "name": r.config_name,
                    "word_count": r.word_count,
                    "efficiency": r.word_count / r.processing_time if r.processing_time > 0 else 0,
                    "parameters": r.parameters
                } for r in top_performers[:5]
            ],
            "optimization_insights": insights
        }
    
    def create_optimized_config_file(self, best_result: OptimizationResult, output_path: str):
        """Create optimized configuration file from best result."""
        config = {
            "optimization_info": {
                "config_name": best_result.config_name,
                "word_count": best_result.word_count,
                "processing_time": best_result.processing_time,
                "efficiency": best_result.word_count / best_result.processing_time if best_result.processing_time > 0 else 0,
                "audio_profile": best_result.audio_profile.to_dict(),
                "timestamp": best_result.timestamp
            },
            "transcription_config": {
                "model": best_result.parameters.get('model', self.default_model),
                "language": "german",
                "min_silence_len": best_result.parameters.get('min_silence_len', 1500),
                "silence_adjustment": best_result.parameters.get('silence_adjustment', 4.0),
                "padding": best_result.parameters.get('padding', 1500),
                "min_segment_length": best_result.parameters.get('min_segment_length', 1000)
            }
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Optimized configuration saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config file: {e}")
            raise
