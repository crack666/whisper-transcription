#!/usr/bin/env python3
"""
Transcription Diagnostics Tool - Enhanced debugging for transcription failures
"""

import os
import sys
import json
import time
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_logging():
    """Setup enhanced logging for diagnostics."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('transcription_diagnostics.log')
        ]
    )
    return logging.getLogger(__name__)

class TranscriptionDiagnostics:
    """Comprehensive diagnostics for transcription issues."""
    
    def __init__(self, audio_file: str):
        self.audio_file = audio_file
        self.logger = setup_logging()
        self.results = {}
        
    def run_full_diagnostics(self) -> Dict:
        """Run comprehensive diagnostics suite."""
        self.logger.info("🔬 Starting transcription diagnostics")
        self.logger.info(f"📁 File: {self.audio_file}")
        
        diagnostics = {
            "file_info": self.analyze_file_info(),
            "audio_analysis": self.analyze_audio_properties(),
            "system_checks": self.check_system_resources(),
            "model_tests": self.test_model_performance(),
            "parameter_sensitivity": self.test_parameter_sensitivity(),
            "error_patterns": self.analyze_error_patterns()
        }
        
        # Generate recommendations
        diagnostics["recommendations"] = self.generate_recommendations(diagnostics)
        
        return diagnostics
    
    def analyze_file_info(self) -> Dict:
        """Analyze basic file information."""
        try:
            import mutagen
            from mutagen.mp3 import MP3
            from mutagen.mp4 import MP4
            
            file_stats = os.stat(self.audio_file)
            file_info = {
                "size_mb": file_stats.st_size / (1024 * 1024),
                "extension": os.path.splitext(self.audio_file)[1].lower(),
                "exists": os.path.exists(self.audio_file),
                "readable": os.access(self.audio_file, os.R_OK)
            }
            
            # Try to get audio metadata
            try:
                if file_info["extension"] in [".mp3"]:
                    audio = MP3(self.audio_file)
                    file_info["duration"] = audio.info.length
                    file_info["bitrate"] = audio.info.bitrate
                elif file_info["extension"] in [".mp4", ".m4a"]:
                    audio = MP4(self.audio_file)
                    file_info["duration"] = audio.info.length
                    file_info["bitrate"] = audio.info.bitrate
            except Exception as e:
                file_info["metadata_error"] = str(e)
            
            return file_info
            
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_audio_properties(self) -> Dict:
        """Analyze audio properties using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', self.audio_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Extract audio stream info
                audio_streams = [s for s in data.get('streams', []) if s.get('codec_type') == 'audio']
                
                if audio_streams:
                    stream = audio_streams[0]
                    return {
                        "codec": stream.get('codec_name'),
                        "sample_rate": int(stream.get('sample_rate', 0)),
                        "channels": int(stream.get('channels', 0)),
                        "duration": float(stream.get('duration', 0)),
                        "bit_rate": int(stream.get('bit_rate', 0)) if stream.get('bit_rate') else None,
                        "format": data.get('format', {}).get('format_name')
                    }
            
            return {"error": f"ffprobe failed: {result.stderr}"}
            
        except Exception as e:
            return {"error": str(e)}
    
    def check_system_resources(self) -> Dict:
        """Check system resources and capabilities."""
        try:
            import psutil
            import torch
            
            # CPU info
            cpu_info = {
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "memory_percent": psutil.virtual_memory().percent
            }
            
            # GPU info
            gpu_info = {
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_info[f"gpu_{i}"] = {
                        "name": gpu_props.name,
                        "memory_total_gb": gpu_props.total_memory / (1024**3),
                        "memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                        "memory_cached_gb": torch.cuda.memory_reserved(i) / (1024**3)
                    }
            
            return {**cpu_info, **gpu_info}
            
        except Exception as e:
            return {"error": str(e)}
    
    def test_model_performance(self) -> Dict:
        """Test different model configurations."""
        model_tests = {}
        
        models_to_test = ["tiny", "base", "small"]
        
        for model in models_to_test:
            self.logger.info(f"Testing model: {model}")
            
            try:
                start_time = time.time()
                
                # Simple test with minimal configuration
                from src.transcriber import AudioTranscriber
                
                config = {
                    'min_silence_len': 2000,
                    'padding': 1500,
                    'silence_adjustment': 3.0,
                    'cleanup_segments': True
                }
                
                transcriber = AudioTranscriber(
                    model_name=model,
                    language='german',
                    config=config
                )
                
                result = transcriber.transcribe_audio_file(self.audio_file)
                
                processing_time = time.time() - start_time
                transcript = result.get('full_text', '')
                
                model_tests[model] = {
                    "success": True,
                    "word_count": len(transcript.split()) if transcript else 0,
                    "char_count": len(transcript) if transcript else 0,
                    "segment_count": len(result.get('segments', [])),
                    "processing_time": processing_time,
                    "transcript_preview": transcript[:200] if transcript else "No text"
                }
                
            except Exception as e:
                model_tests[model] = {
                    "success": False,
                    "error": str(e),
                    "processing_time": time.time() - start_time
                }
        
        return model_tests
    
    def test_parameter_sensitivity(self) -> Dict:
        """Test sensitivity to different parameters."""
        param_tests = {}
        
        # Test different silence lengths
        silence_lengths = [1000, 2000, 3000, 4000]
        
        for silence_len in silence_lengths:
            test_name = f"silence_{silence_len}ms"
            self.logger.info(f"Testing {test_name}")
            
            try:
                from src.transcriber import AudioTranscriber
                
                config = {
                    'min_silence_len': silence_len,
                    'padding': silence_len,
                    'silence_adjustment': 3.0,
                    'cleanup_segments': True
                }
                
                transcriber = AudioTranscriber(
                    model_name="tiny",  # Fast model for testing
                    language='german',
                    config=config
                )
                
                start_time = time.time()
                result = transcriber.transcribe_audio_file(self.audio_file)
                processing_time = time.time() - start_time
                
                transcript = result.get('full_text', '')
                
                param_tests[test_name] = {
                    "success": True,
                    "word_count": len(transcript.split()) if transcript else 0,
                    "segment_count": len(result.get('segments', [])),
                    "processing_time": processing_time
                }
                
            except Exception as e:
                param_tests[test_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        return param_tests
    
    def analyze_error_patterns(self) -> Dict:
        """Analyze common error patterns in recent optimizations."""
        try:
            # Load optimization database to find error patterns
            db_file = "optimization_database.json"
            if not os.path.exists(db_file):
                return {"note": "No optimization database found"}
            
            with open(db_file, 'r') as f:
                history = json.load(f)
            
            # Analyze recent failures
            recent_results = history[-20:]  # Last 20 results
            
            error_analysis = {
                "total_tests": len(recent_results),
                "failed_tests": sum(1 for r in recent_results if not r.get('success', True)),
                "low_word_counts": sum(1 for r in recent_results if r.get('word_count', 0) < 10),
                "processing_time_issues": sum(1 for r in recent_results if r.get('processing_time', 0) > 60),
                "quality_score_distribution": [r.get('quality_score', 0) for r in recent_results]
            }
            
            # Find configurations that consistently fail
            config_performance = {}
            for result in recent_results:
                config_name = result.get('config_name', 'unknown')
                if config_name not in config_performance:
                    config_performance[config_name] = {"tests": 0, "failures": 0, "low_quality": 0}
                
                config_performance[config_name]["tests"] += 1
                if not result.get('success', True):
                    config_performance[config_name]["failures"] += 1
                if result.get('word_count', 0) < 10:
                    config_performance[config_name]["low_quality"] += 1
            
            error_analysis["config_performance"] = config_performance
            
            return error_analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def generate_recommendations(self, diagnostics: Dict) -> List[str]:
        """Generate actionable recommendations based on diagnostics."""
        recommendations = []
        
        # File-based recommendations
        file_info = diagnostics.get("file_info", {})
        if file_info.get("size_mb", 0) > 100:
            recommendations.append("Large file detected - consider using CPU processing to avoid CUDA memory issues")
        
        # Audio property recommendations
        audio_props = diagnostics.get("audio_analysis", {})
        if audio_props.get("sample_rate", 0) > 22050:
            recommendations.append("High sample rate detected - consider downsampling to 16kHz for better Whisper performance")
        
        if audio_props.get("channels", 0) > 1:
            recommendations.append("Multi-channel audio detected - convert to mono for optimal transcription")
        
        # System resource recommendations
        system = diagnostics.get("system_checks", {})
        if system.get("memory_percent", 0) > 85:
            recommendations.append("High memory usage detected - use smaller model or CPU processing")
        
        if not system.get("cuda_available", False):
            recommendations.append("CUDA not available - ensure GPU drivers and PyTorch CUDA are properly installed")
        
        # Model performance recommendations
        model_tests = diagnostics.get("model_tests", {})
        best_model = None
        best_words = 0
        
        for model, result in model_tests.items():
            if result.get("success") and result.get("word_count", 0) > best_words:
                best_words = result["word_count"]
                best_model = model
        
        if best_model:
            recommendations.append(f"Best performing model: {best_model} with {best_words} words")
        
        # Parameter sensitivity recommendations
        param_tests = diagnostics.get("parameter_sensitivity", {})
        best_silence = None
        best_param_words = 0
        
        for test_name, result in param_tests.items():
            if result.get("success") and result.get("word_count", 0) > best_param_words:
                best_param_words = result["word_count"]
                best_silence = test_name
        
        if best_silence:
            recommendations.append(f"Best silence parameter: {best_silence} with {best_param_words} words")
        
        # Error pattern recommendations
        error_patterns = diagnostics.get("error_patterns", {})
        if error_patterns.get("low_word_counts", 0) > error_patterns.get("total_tests", 1) * 0.5:
            recommendations.append("High rate of low word count results - audio may need preprocessing or different approach")
        
        if error_patterns.get("processing_time_issues", 0) > 0:
            recommendations.append("Processing time issues detected - consider timeout limits and CPU fallback")
        
        return recommendations
    
    def save_report(self, diagnostics: Dict, filename: str = None) -> str:
        """Save comprehensive diagnostics report."""
        if not filename:
            timestamp = int(time.time())
            filename = f"transcription_diagnostics_{timestamp}.json"
        
        report = {
            "audio_file": self.audio_file,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "diagnostics": diagnostics
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📊 Diagnostics report saved: {filename}")
        return filename

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python transcription_diagnostics.py <audio_file>")
        return 1
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        return 1
    
    diagnostics = TranscriptionDiagnostics(audio_file)
    results = diagnostics.run_full_diagnostics()
    
    # Print summary
    print("\n" + "="*60)
    print("🔬 TRANSCRIPTION DIAGNOSTICS SUMMARY")
    print("="*60)
    
    print(f"\n📁 File: {audio_file}")
    
    # File info
    file_info = results.get("file_info", {})
    print(f"📊 Size: {file_info.get('size_mb', 0):.1f} MB")
    
    # Audio properties
    audio_props = results.get("audio_analysis", {})
    if "error" not in audio_props:
        print(f"🎵 Duration: {audio_props.get('duration', 0):.1f}s")
        print(f"🎵 Sample Rate: {audio_props.get('sample_rate', 0)} Hz")
        print(f"🎵 Channels: {audio_props.get('channels', 0)}")
    
    # System status
    system = results.get("system_checks", {})
    if "error" not in system:
        print(f"💾 Memory: {system.get('memory_percent', 0):.1f}% used")
        print(f"🔥 CUDA: {'Available' if system.get('cuda_available') else 'Not available'}")
    
    # Model performance
    model_tests = results.get("model_tests", {})
    print(f"\n🧠 MODEL PERFORMANCE:")
    for model, result in model_tests.items():
        if result.get("success"):
            print(f"   {model}: {result['word_count']} words in {result['processing_time']:.1f}s")
        else:
            print(f"   {model}: FAILED - {result.get('error', 'Unknown error')}")
    
    # Recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Save report
    report_file = diagnostics.save_report(results)
    print(f"\n📋 Full report saved: {report_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
