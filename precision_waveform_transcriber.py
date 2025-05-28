#!/usr/bin/env python3
"""
Precision Waveform Segmentation Mode

Integration of the scientific waveform analysis into the enhanced transcriber
to replace the current inaccurate silence detection with precise waveform-based segmentation.
"""

import sys
import os
sys.path.append('src')

from enhanced_transcriber import EnhancedAudioTranscriber
import json
from advanced_waveform_analysis import WaveformSpeechDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrecisionWaveformTranscriber(EnhancedAudioTranscriber):
    """
    Enhanced transcriber with precision waveform-based segmentation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize waveform detector
        self.waveform_detector = WaveformSpeechDetector(
            min_speech_duration_ms=2000,  # 2 seconds minimum speech
            min_silence_duration_ms=1000,  # 1 second minimum silence  
            energy_threshold_factor=0.15   # 15% energy threshold
        )
    
    def precision_waveform_detection(self, audio_file: str) -> list:
        """
        Use scientific waveform analysis for precise speech segment detection.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            List of (start_ms, end_ms) tuples with precise speech segments
        """
        logger.info("🔬 Using PRECISION WAVEFORM ANALYSIS for speech detection")
        
        # Perform waveform analysis
        analysis = self.waveform_detector.analyze_audio_file(audio_file)
        
        # Extract segments and convert to milliseconds
        segments = analysis['final_segments']
        segments_ms = [(int(start * 1000), int(end * 1000)) for start, end in segments]
        
        logger.info(f"🎯 Precision waveform detected {len(segments_ms)} speech segments")
        logger.info(f"📊 Total speech coverage: {analysis['statistics']['speech_percentage']:.1f}%")
        
        # Log first few segments for verification
        for i, (start_ms, end_ms) in enumerate(segments_ms[:5]):
            duration_s = (end_ms - start_ms) / 1000.0
            logger.info(f"  Segment {i+1}: {start_ms/1000:.1f}s - {end_ms/1000:.1f}s ({duration_s:.1f}s)")
        
        if len(segments_ms) > 5:
            logger.info(f"  ... and {len(segments_ms) - 5} more segments")
        
        return segments_ms
    
    def transcribe_with_precision_waveform(self, audio_file: str, output_dir: str = "precision_waveform_results"):
        """
        Transcribe audio using precision waveform segmentation.
        """
        logger.info("🚀 Starting PRECISION WAVEFORM TRANSCRIPTION")
        
        # Get precision segments
        speech_segments = self.precision_waveform_detection(audio_file)
        
        if not speech_segments:
            logger.error("No speech segments detected")
            return None
        
        # Prepare modified config for precision mode
        config = self.config.copy()
        config['segmentation_mode'] = 'precision_waveform'
        
        # Create a temporary enhanced transcriber with custom segments
        return self._transcribe_custom_segments(audio_file, speech_segments, output_dir)
    
    def _transcribe_custom_segments(self, audio_file: str, segments: list, output_dir: str):
        """
        Transcribe using custom pre-defined segments.
        """
        from pydub import AudioSegment
        import time
        import os
        from pathlib import Path
        
        # Load audio
        audio = AudioSegment.from_file(audio_file)
        total_duration = len(audio)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results
        results = {
            "segments": [],
            "full_text": "",
            "total_duration": total_duration,
            "segments_successful": 0,
            "segments_failed": 0,
            "processing_start_time": time.time()
        }
        
        logger.info(f"🎵 Processing {len(segments)} precision segments...")
        
        for i, (start_ms, end_ms) in enumerate(segments):
            try:
                # Extract segment with small padding for context
                padding_ms = 500  # 0.5 second padding
                padded_start = max(0, start_ms - padding_ms)
                padded_end = min(total_duration, end_ms + padding_ms)
                
                # Extract audio segment
                segment_audio = audio[padded_start:padded_end]
                
                # Save segment file
                segment_file = f"{output_dir}/segment_{i:03d}.wav"
                segment_audio.export(segment_file, format="wav")
                
                # Transcribe segment
                logger.info(f"🎯 Transcribing segment {i+1}/{len(segments)}: {start_ms/1000:.1f}s - {end_ms/1000:.1f}s")
                
                result = self.model.transcribe(
                    segment_file,
                    language=self.language_code,
                    verbose=False
                )
                
                text = result.get("text", "").strip()
                
                if text:
                    segment_result = {
                        "segment_id": i,
                        "text": text,
                        "start_time": start_ms,
                        "end_time": end_ms,
                        "duration": end_ms - start_ms,
                        "segment_file": segment_file,
                        "confidence": 1.0,
                        "language_detected": result.get("language", self.language_code)
                    }
                    
                    results["segments"].append(segment_result)
                    results["full_text"] += text + " "
                    results["segments_successful"] += 1
                    
                    logger.info(f"✅ Segment {i+1} success: {len(text.split())} words")
                else:
                    logger.warning(f"⚠️ Segment {i+1} produced no text")
                    results["segments_failed"] += 1
                
                # Clean up segment file
                if os.path.exists(segment_file):
                    os.remove(segment_file)
                    
            except Exception as e:
                logger.error(f"❌ Error processing segment {i+1}: {e}")
                results["segments_failed"] += 1
        
        # Calculate final statistics
        processing_time = time.time() - results["processing_start_time"]
        total_words = len(results["full_text"].split())
        
        results.update({
            "processing_time": processing_time,
            "total_words": total_words,
            "words_per_second": total_words / processing_time if processing_time > 0 else 0,
            "success_rate": results["segments_successful"] / len(segments) * 100
        })
        
        # Save results
        results_file = f"{output_dir}/precision_transcription.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        transcript_file = f"{output_dir}/precision_transcript.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write("=== PRECISION WAVEFORM TRANSCRIPTION ===\n\n")
            f.write(results["full_text"])
            f.write(f"\n\n=== STATISTICS ===\n")
            f.write(f"Processing Time: {processing_time:.1f}s\n")
            f.write(f"Segments Processed: {len(segments)}\n") 
            f.write(f"Successful Segments: {results['segments_successful']}\n")
            f.write(f"Failed Segments: {results['segments_failed']}\n")
            f.write(f"Success Rate: {results['success_rate']:.1f}%\n")
            f.write(f"Total Words: {total_words}\n")
            f.write(f"Processing Speed: {results['words_per_second']:.1f} words/sec\n")
        
        # Print summary
        print("\n🎯 PRECISION WAVEFORM TRANSCRIPTION RESULTS")
        print("=" * 55)
        print(f"📁 Audio file: {audio_file}")
        print(f"⏱️  Processing time: {processing_time:.1f}s")
        print(f"🎵 Audio duration: {total_duration/1000:.1f}s")
        print(f"📊 Segments: {len(segments)} ({results['segments_successful']} successful)")
        print(f"✅ Success rate: {results['success_rate']:.1f}%")
        print(f"📝 Total words: {total_words}")
        print(f"🚀 Speed: {results['words_per_second']:.1f} words/sec")
        print(f"⚡ Real-time factor: {(total_duration/1000)/processing_time:.1f}x")
        print(f"💾 Results saved to: {results_file}")
        print(f"📄 Transcript saved to: {transcript_file}")
        
        return results

def test_precision_waveform():
    """Test the precision waveform transcription."""
    
    audio_file = "studies/Aufzeichnung - 20.05.2025.mp4.wav"
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    # Initialize precision transcriber
    transcriber = PrecisionWaveformTranscriber(
        model_name="base",
        language="german", 
        device="cpu",
        config={
            "min_segment_length": 2000,
            "max_segment_length": 30000,
            "padding": 500
        }
    )
    
    # Run precision transcription
    result = transcriber.transcribe_with_precision_waveform(
        audio_file, 
        "precision_waveform_results"
    )
    
    return result

if __name__ == "__main__":
    test_precision_waveform()
