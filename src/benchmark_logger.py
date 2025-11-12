"""
Centralized benchmark logging system for tracking performance across runs.
Logs processing times, hardware info, model performance, and file characteristics.
"""

import os
import json
import time
import platform
import psutil
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import subprocess

class BenchmarkLogger:
    """
    Centralized benchmark logger that tracks performance metrics across multiple runs.
    Stores data in a JSON file for analysis and hardware-specific recommendations.
    """
    
    def __init__(self, log_dir: str = "./benchmark_logs"):
        """
        Initialize the benchmark logger.
        
        Args:
            log_dir: Directory to store benchmark log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main benchmark log file
        self.log_file = self.log_dir / "benchmark_history.jsonl"
        
        # Current run data
        self.current_run = {}
        self.start_time = None
        self.phase_times = {}
        
    def start_run(self, video_path: str, config: Dict[str, Any]) -> None:
        """
        Start tracking a new processing run.
        
        Args:
            video_path: Path to the video/audio file being processed
            config: Configuration used for this run
        """
        self.start_time = time.time()
        self.phase_times = {}
        
        # Get file info
        file_path = Path(video_path)
        file_stats = file_path.stat()
        
        # Get video/audio metadata
        duration, fps, resolution = self._get_media_info(video_path)
        
        # Collect hardware info
        hardware_info = self._get_hardware_info()
        
        # Initialize run data
        self.current_run = {
            "timestamp": datetime.now().isoformat(),
            "file": {
                "path": str(file_path),
                "name": file_path.name,
                "size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                "duration_seconds": duration,
                "fps": fps,
                "resolution": resolution,
                "type": "video" if file_path.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov', '.webm'] else "audio"
            },
            "config": {
                "model": config.get('transcription', {}).get('model', 'unknown'),
                "language": config.get('transcription', {}).get('language', 'unknown'),
                "device": config.get('transcription', {}).get('device', 'unknown'),
                "segmentation_enabled": not config.get('transcription', {}).get('disable_segmentation', False),
                "segmentation_mode": config.get('transcription', {}).get('segmentation_mode', 'none'),
                "screenshots_enabled": config.get('video_processing', {}).get('extract_screenshots', True)
            },
            "hardware": hardware_info,
            "phases": {},
            "results": {},
            "errors": []
        }
        
    def start_phase(self, phase_name: str) -> None:
        """
        Start tracking a processing phase.
        
        Args:
            phase_name: Name of the phase (e.g., 'transcription', 'screenshot_extraction')
        """
        self.phase_times[phase_name] = {
            "start": time.time(),
            "end": None,
            "duration_seconds": None
        }
        
    def end_phase(self, phase_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        End tracking a processing phase.
        
        Args:
            phase_name: Name of the phase
            metadata: Additional metadata for this phase (e.g., segment count, screenshot count)
        """
        if phase_name in self.phase_times:
            self.phase_times[phase_name]["end"] = time.time()
            self.phase_times[phase_name]["duration_seconds"] = round(
                self.phase_times[phase_name]["end"] - self.phase_times[phase_name]["start"], 2
            )
            
            if metadata:
                self.phase_times[phase_name]["metadata"] = metadata
                
    def end_run(self, success: bool = True, result_data: Optional[Dict[str, Any]] = None) -> None:
        """
        End tracking the current run and save to log file.
        
        Args:
            success: Whether the run completed successfully
            result_data: Results from the processing (e.g., word count, segment count)
        """
        if not self.start_time:
            return
            
        total_time = time.time() - self.start_time
        
        # Finalize run data
        self.current_run["total_duration_seconds"] = round(total_time, 2)
        self.current_run["success"] = success
        self.current_run["phases"] = self.phase_times
        
        if result_data:
            self.current_run["results"] = result_data
            
        # Calculate performance metrics
        self.current_run["metrics"] = self._calculate_metrics()
        
        # Append to log file (JSONL format - one JSON object per line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(self.current_run, ensure_ascii=False) + '\n')
            
    def log_error(self, error_message: str, phase: Optional[str] = None) -> None:
        """
        Log an error that occurred during processing.
        
        Args:
            error_message: Description of the error
            phase: Phase where the error occurred
        """
        self.current_run["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "message": error_message
        })
        
    def _get_media_info(self, file_path: str) -> tuple:
        """
        Extract media metadata using ffprobe.
        
        Returns:
            Tuple of (duration_seconds, fps, resolution_string)
        """
        try:
            import subprocess
            import json
            
            # Use ffprobe to get media info
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                
                # Get duration
                duration = float(info.get('format', {}).get('duration', 0))
                
                # Get video stream info
                video_stream = next(
                    (s for s in info.get('streams', []) if s.get('codec_type') == 'video'),
                    None
                )
                
                if video_stream:
                    # Calculate FPS
                    fps_str = video_stream.get('r_frame_rate', '0/1')
                    if '/' in fps_str:
                        num, den = map(int, fps_str.split('/'))
                        fps = num / den if den != 0 else 0
                    else:
                        fps = float(fps_str)
                        
                    # Get resolution
                    width = video_stream.get('width', 0)
                    height = video_stream.get('height', 0)
                    resolution = f"{width}x{height}" if width and height else "unknown"
                else:
                    fps = 0
                    resolution = "audio-only"
                    
                return round(duration, 2), round(fps, 2), resolution
                
        except Exception as e:
            pass
            
        return 0, 0, "unknown"
        
    def _get_hardware_info(self) -> Dict[str, Any]:
        """
        Collect hardware and system information.
        
        Returns:
            Dictionary with hardware specs
        """
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_version": platform.python_version()
        }
        
        # Try to detect GPU
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = {
                    "available": True,
                    "name": torch.cuda.get_device_name(0),
                    "count": torch.cuda.device_count(),
                    "cuda_version": torch.version.cuda
                }
            else:
                info["gpu"] = {"available": False}
        except:
            info["gpu"] = {"available": False}
            
        return info
        
    def _calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics for this run.
        
        Returns:
            Dictionary with calculated metrics
        """
        metrics = {}
        
        # Real-time factor (RTF) - processing_time / media_duration
        duration = self.current_run["file"]["duration_seconds"]
        total_time = self.current_run["total_duration_seconds"]
        
        if duration > 0:
            metrics["rtf"] = round(total_time / duration, 2)
            metrics["speedup"] = round(duration / total_time, 2)
        else:
            metrics["rtf"] = None
            metrics["speedup"] = None
            
        # Processing speed (seconds of media per second of processing)
        if total_time > 0:
            metrics["processing_speed_x"] = round(duration / total_time, 2)
        else:
            metrics["processing_speed_x"] = None
            
        # MB per second processed
        file_size_mb = self.current_run["file"]["size_mb"]
        if total_time > 0:
            metrics["throughput_mb_per_sec"] = round(file_size_mb / total_time, 2)
        else:
            metrics["throughput_mb_per_sec"] = None
            
        # Words per second (if available)
        if "word_count" in self.current_run.get("results", {}):
            word_count = self.current_run["results"]["word_count"]
            if total_time > 0:
                metrics["words_per_second"] = round(word_count / total_time, 2)
                
        return metrics
        
    def get_statistics(self, filter_by: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze benchmark history and generate statistics.
        
        Args:
            filter_by: Optional filters (e.g., {"config.model": "large-v3"})
            
        Returns:
            Dictionary with aggregated statistics
        """
        if not self.log_file.exists():
            return {"error": "No benchmark data available"}
            
        # Load all runs
        runs = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    runs.append(json.loads(line.strip()))
                except:
                    continue
                    
        if not runs:
            return {"error": "No valid benchmark data"}
            
        # Apply filters
        if filter_by:
            filtered_runs = []
            for run in runs:
                match = True
                for key, value in filter_by.items():
                    # Support nested keys like "config.model"
                    parts = key.split('.')
                    current = run
                    for part in parts:
                        current = current.get(part, {})
                    if current != value:
                        match = False
                        break
                if match:
                    filtered_runs.append(run)
            runs = filtered_runs
            
        if not runs:
            return {"error": "No runs matching filter criteria"}
            
        # Calculate statistics
        stats = {
            "total_runs": len(runs),
            "successful_runs": sum(1 for r in runs if r.get("success", False)),
            "failed_runs": sum(1 for r in runs if not r.get("success", False)),
            "models_used": list(set(r.get("config", {}).get("model", "unknown") for r in runs)),
            "average_metrics": {},
            "by_model": {},
            "recent_runs": runs[-10:]  # Last 10 runs
        }
        
        # Average metrics across all runs
        metrics_keys = ["rtf", "speedup", "processing_speed_x", "throughput_mb_per_sec"]
        for key in metrics_keys:
            values = [r.get("metrics", {}).get(key) for r in runs if r.get("metrics", {}).get(key) is not None]
            if values:
                stats["average_metrics"][key] = {
                    "mean": round(sum(values) / len(values), 2),
                    "min": round(min(values), 2),
                    "max": round(max(values), 2)
                }
                
        # Statistics by model
        for model in stats["models_used"]:
            model_runs = [r for r in runs if r.get("config", {}).get("model") == model]
            if model_runs:
                stats["by_model"][model] = {
                    "count": len(model_runs),
                    "success_rate": round(sum(1 for r in model_runs if r.get("success", False)) / len(model_runs) * 100, 1),
                    "avg_processing_time": round(sum(r.get("total_duration_seconds", 0) for r in model_runs) / len(model_runs), 2)
                }
                
                # Average metrics for this model
                for key in metrics_keys:
                    values = [r.get("metrics", {}).get(key) for r in model_runs if r.get("metrics", {}).get(key) is not None]
                    if values:
                        stats["by_model"][model][f"avg_{key}"] = round(sum(values) / len(values), 2)
                        
        return stats
        
    def print_report(self, filter_by: Optional[Dict[str, Any]] = None) -> None:
        """
        Print a formatted benchmark report to console.
        
        Args:
            filter_by: Optional filters for the report
        """
        stats = self.get_statistics(filter_by)
        
        if "error" in stats:
            print(f"❌ {stats['error']}")
            return
            
        print("\n" + "="*80)
        print("📊 BENCHMARK REPORT")
        print("="*80)
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total Runs: {stats['total_runs']}")
        print(f"   Successful: {stats['successful_runs']} ({stats['successful_runs']/stats['total_runs']*100:.1f}%)")
        print(f"   Failed: {stats['failed_runs']}")
        print(f"   Models Used: {', '.join(stats['models_used'])}")
        
        if stats.get("average_metrics"):
            print(f"\n⚡ Average Performance:")
            for key, values in stats["average_metrics"].items():
                print(f"   {key}: {values['mean']} (min: {values['min']}, max: {values['max']})")
                
        if stats.get("by_model"):
            print(f"\n🎯 Performance by Model:")
            for model, model_stats in stats["by_model"].items():
                print(f"\n   {model}:")
                print(f"      Runs: {model_stats['count']}")
                print(f"      Success Rate: {model_stats['success_rate']}%")
                print(f"      Avg Processing Time: {model_stats['avg_processing_time']}s")
                if "avg_speedup" in model_stats:
                    print(f"      Avg Speedup: {model_stats['avg_speedup']}x")
                    
        print("\n" + "="*80 + "\n")
