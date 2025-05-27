#!/usr/bin/env python3
"""
Audio-only optimization for transcription settings.
Tests different parameters directly with transcriber classes.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_logging():
    """Setup logging for optimization tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class AudioTranscriptionOptimizer:
    """Optimize audio transcription settings using word count as metric."""
    
    def __init__(self, test_file: str):
        self.test_file = test_file
        self.results = []
        self.logger = logging.getLogger(__name__)
        
        # Test configurations - focus on audio parameters
        self.test_configs = [
            # Standard Transcriber tests
            {
                "name": "standard_conservative",
                "transcriber": "standard",
                "model": "tiny",
                "min_silence_len": 1000,
                "padding": 1000,
                "silence_adjustment": 5.0,
                "cleanup_segments": True
            },
            {
                "name": "standard_moderate",
                "transcriber": "standard", 
                "model": "tiny",
                "min_silence_len": 1500,
                "padding": 1500,
                "silence_adjustment": 4.0,
                "cleanup_segments": True
            },
            {
                "name": "standard_aggressive",
                "transcriber": "standard",
                "model": "tiny", 
                "min_silence_len": 2000,
                "padding": 2000,
                "silence_adjustment": 3.0,
                "cleanup_segments": True
            },
            {
                "name": "standard_ultra_sensitive",
                "transcriber": "standard",
                "model": "tiny",
                "min_silence_len": 500,
                "padding": 750,
                "silence_adjustment": 8.0,
                "cleanup_segments": True
            },
            # Enhanced Transcriber tests
            {
                "name": "enhanced_conservative",
                "transcriber": "enhanced",
                "model": "tiny",
                "min_silence_len": 2000,
                "silence_adjustment": 4.0,
                "min_segment_length": 2000,
                "overlap_duration": 1000,
                "padding": 2000
            },
            {
                "name": "enhanced_moderate",
                "transcriber": "enhanced",
                "model": "tiny",
                "min_silence_len": 1500,
                "silence_adjustment": 5.0,
                "min_segment_length": 1500,
                "overlap_duration": 750,
                "padding": 1500
            },
            {
                "name": "enhanced_aggressive",
                "transcriber": "enhanced",
                "model": "tiny",
                "min_silence_len": 1000,
                "silence_adjustment": 6.0,
                "min_segment_length": 1000,
                "overlap_duration": 500,
                "padding": 1000
            },
            {
                "name": "enhanced_ultra_sensitive",
                "transcriber": "enhanced",
                "model": "tiny",
                "min_silence_len": 800,
                "silence_adjustment": 7.0,
                "min_segment_length": 800,
                "overlap_duration": 300,
                "padding": 800
            },
            # Lecture-specific optimizations
            {
                "name": "lecture_optimized",
                "transcriber": "enhanced",
                "model": "tiny",
                "min_silence_len": 2500,
                "silence_adjustment": 4.0,
                "min_segment_length": 3000,
                "overlap_duration": 1500,
                "padding": 2000
            }
        ]
    
    def extract_audio_if_needed(self, video_file: str) -> str:
        """Extract audio from video if needed."""
        if video_file.endswith('.mp3') or video_file.endswith('.wav'):
            return video_file
        
        # Extract audio
        try:
            from src.utils import extract_audio_from_video
            audio_file = extract_audio_from_video(video_file)
            self.logger.info(f"📹 Extracted audio: {audio_file}")
            return audio_file
        except Exception as e:
            self.logger.error(f"Failed to extract audio: {e}")
            raise
    
    def test_standard_transcriber(self, params: Dict, audio_file: str) -> Dict[str, Any]:
        """Test standard transcriber configuration."""
        try:
            from src.transcriber import AudioTranscriber
            
            config = {
                'min_silence_len': params['min_silence_len'],
                'padding': params['padding'],
                'silence_adjustment': params['silence_adjustment'],
                'cleanup_segments': params.get('cleanup_segments', True)
            }
            
            transcriber = AudioTranscriber(
                model_name=params['model'],
                language='german',
                config=config
            )
            
            result = transcriber.transcribe_audio_file(audio_file)
            return result
            
        except Exception as e:
            raise Exception(f"Standard transcriber failed: {e}")
    
    def test_enhanced_transcriber(self, params: Dict, audio_file: str) -> Dict[str, Any]:
        """Test enhanced transcriber configuration."""
        try:
            from src.enhanced_transcriber import EnhancedAudioTranscriber
            
            config = {
                'min_silence_len': params['min_silence_len'],
                'silence_adjustment': params['silence_adjustment'],
                'min_segment_length': params['min_segment_length'],
                'overlap_duration': params['overlap_duration'],
                'padding': params['padding']
            }
            
            transcriber = EnhancedAudioTranscriber(
                model_name=params['model'],
                language='german',
                config=config
            )
            
            result = transcriber.transcribe_audio_file_enhanced(audio_file)
            return result
            
        except Exception as e:
            raise Exception(f"Enhanced transcriber failed: {e}")
    
    def test_configuration(self, params: Dict) -> Dict[str, Any]:
        """Test a single configuration and return results."""
        self.logger.info(f"🧪 Testing: {params['name']} ({params['transcriber']})")
        
        start_time = time.time()
        
        try:
            # Extract audio if needed
            audio_file = self.extract_audio_if_needed(self.test_file)
            
            # Run transcription based on type
            if params['transcriber'] == 'standard':
                result = self.test_standard_transcriber(params, audio_file)
            elif params['transcriber'] == 'enhanced':
                result = self.test_enhanced_transcriber(params, audio_file)
            else:
                raise ValueError(f"Unknown transcriber type: {params['transcriber']}")
            
            # Extract metrics
            transcript = result.get('full_text', '') or result.get('transcript', '')
            segments = result.get('segments', [])
            word_count = len(transcript.split()) if transcript else 0
            char_count = len(transcript) if transcript else 0
            segment_count = len(segments)
            
            processing_time = time.time() - start_time
            
            # Calculate quality metrics
            avg_segment_length = sum(len(seg.get('text', '').split()) for seg in segments) / max(segment_count, 1)
            
            test_result = {
                "config_name": params['name'],
                "transcriber_type": params['transcriber'],
                "parameters": params,
                "word_count": word_count,
                "char_count": char_count,
                "segment_count": segment_count,
                "avg_segment_length": avg_segment_length,
                "processing_time": processing_time,
                "success": word_count > 0,
                "transcript_preview": transcript[:300] if transcript else "No text",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.logger.info(f"   ✅ {word_count} words, {segment_count} segments, {processing_time:.1f}s")
            
            # Cleanup temporary audio file if created
            if audio_file != self.test_file and os.path.exists(audio_file):
                os.remove(audio_file)
            
            return test_result
            
        except Exception as e:
            self.logger.error(f"   ❌ Failed: {e}")
            
            return {
                "config_name": params['name'],
                "transcriber_type": params['transcriber'],
                "parameters": params,
                "word_count": 0,
                "char_count": 0,
                "segment_count": 0,
                "avg_segment_length": 0,
                "processing_time": time.time() - start_time,
                "success": False,
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def run_optimization(self) -> List[Dict]:
        """Run optimization tests on all configurations."""
        self.logger.info(f"🎯 Starting audio transcription optimization")
        self.logger.info(f"📁 Test file: {self.test_file}")
        self.logger.info(f"📊 Testing {len(self.test_configs)} configurations")
        
        for i, config in enumerate(self.test_configs, 1):
            self.logger.info(f"\n--- Test {i}/{len(self.test_configs)} ---")
            result = self.test_configuration(config)
            self.results.append(result)
            
            # Short pause between tests
            time.sleep(1)
        
        return self.results
    
    def analyze_results(self) -> Dict:
        """Analyze results and find optimal settings."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Sort by word count (primary metric)
        successful_results = [r for r in self.results if r['success']]
        successful_results.sort(key=lambda x: x['word_count'], reverse=True)
        
        # Find best configurations
        best_overall = successful_results[0] if successful_results else None
        best_speed = min(successful_results, key=lambda x: x['processing_time']) if successful_results else None
        
        # Find best per transcriber type
        standard_results = [r for r in successful_results if r['transcriber_type'] == 'standard']
        enhanced_results = [r for r in successful_results if r['transcriber_type'] == 'enhanced']
        
        best_standard = standard_results[0] if standard_results else None
        best_enhanced = enhanced_results[0] if enhanced_results else None
        
        # Calculate statistics
        word_counts = [r['word_count'] for r in successful_results]
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        
        analysis = {
            "total_tests": len(self.results),
            "successful_tests": len(successful_results),
            "failed_tests": len(self.results) - len(successful_results),
            "best_overall": {
                "config": best_overall['config_name'] if best_overall else None,
                "transcriber": best_overall['transcriber_type'] if best_overall else None,
                "words": best_overall['word_count'] if best_overall else 0,
                "segments": best_overall['segment_count'] if best_overall else 0,
                "time": best_overall['processing_time'] if best_overall else 0,
                "parameters": best_overall['parameters'] if best_overall else None
            },
            "best_standard": {
                "config": best_standard['config_name'] if best_standard else None,
                "words": best_standard['word_count'] if best_standard else 0,
                "parameters": best_standard['parameters'] if best_standard else None
            },
            "best_enhanced": {
                "config": best_enhanced['config_name'] if best_enhanced else None,
                "words": best_enhanced['word_count'] if best_enhanced else 0,
                "parameters": best_enhanced['parameters'] if best_enhanced else None
            },
            "fastest_config": {
                "config": best_speed['config_name'] if best_speed else None,
                "time": best_speed['processing_time'] if best_speed else 0,
                "words": best_speed['word_count'] if best_speed else 0
            },
            "statistics": {
                "average_words": avg_words,
                "word_count_range": {
                    "min": min(word_counts) if word_counts else 0,
                    "max": max(word_counts) if word_counts else 0
                },
                "standard_count": len(standard_results),
                "enhanced_count": len(enhanced_results)
            }
        }
        
        return analysis
    
    def save_results(self, filename: str = "audio_optimization_results.json"):
        """Save results to file."""
        analysis = self.analyze_results()
        
        output = {
            "test_file": self.test_file,
            "analysis": analysis,
            "detailed_results": self.results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Results saved to: {filename}")
        return analysis
    
    def print_summary(self):
        """Print a summary of optimization results."""
        analysis = self.analyze_results()
        
        print(f"\n🎯 AUDIO TRANSCRIPTION OPTIMIZATION SUMMARY")
        print(f"=" * 60)
        print(f"Test file: {self.test_file}")
        print(f"Total tests: {analysis['total_tests']}")
        print(f"Successful: {analysis['successful_tests']}")
        print(f"Failed: {analysis['failed_tests']}")
        
        if analysis['successful_tests'] > 0:
            print(f"\n🏆 BEST OVERALL CONFIGURATION:")
            best = analysis['best_overall']
            print(f"   Config: {best['config']} ({best['transcriber']})")
            print(f"   Words: {best['words']} | Segments: {best['segments']} | Time: {best['time']:.1f}s")
            
            print(f"\n🔧 BEST BY TRANSCRIBER TYPE:")
            if analysis['best_standard']['config']:
                std = analysis['best_standard']
                print(f"   Standard: {std['config']} ({std['words']} words)")
            
            if analysis['best_enhanced']['config']:
                enh = analysis['best_enhanced']
                print(f"   Enhanced: {enh['config']} ({enh['words']} words)")
            
            print(f"\n⚡ FASTEST CONFIGURATION:")
            fastest = analysis['fastest_config']
            print(f"   Config: {fastest['config']}")
            print(f"   Time: {fastest['time']:.1f}s | Words: {fastest['words']}")
            
            print(f"\n📊 STATISTICS:")
            stats = analysis['statistics']
            print(f"   Average words: {stats['average_words']:.1f}")
            print(f"   Word range: {stats['word_count_range']['min']} - {stats['word_count_range']['max']}")
            print(f"   Standard tests: {stats['standard_count']}")
            print(f"   Enhanced tests: {stats['enhanced_count']}")
            
            print(f"\n📝 TOP 5 CONFIGURATIONS:")
            successful_results = [r for r in self.results if r['success']]
            successful_results.sort(key=lambda x: x['word_count'], reverse=True)
            
            for i, result in enumerate(successful_results[:5], 1):
                print(f"   {i}. {result['config_name']} ({result['transcriber_type']}): "
                      f"{result['word_count']} words ({result['segment_count']} segments, "
                      f"{result['processing_time']:.1f}s)")

def main():
    """Main function."""
    setup_logging()
    
    test_file = "TestFile_cut.mp4"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return 1
    
    print(f"🚀 Starting audio transcription optimization")
    print(f"📁 Test file: {test_file} ({os.path.getsize(test_file)/1024/1024:.1f} MB)")
    
    optimizer = AudioTranscriptionOptimizer(test_file)
    
    # Run optimization
    results = optimizer.run_optimization()
    
    # Save and analyze results
    analysis = optimizer.save_results()
    optimizer.print_summary()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())