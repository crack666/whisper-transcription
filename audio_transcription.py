import whisper
import time
import argparse
import json
import logging
import concurrent.futures
from datetime import datetime, timedelta
import tqdm
from pydub import AudioSegment, silence
import os
from glob import glob
import re
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Mapping of language name to Whisper language code
LANGUAGE_MAP = {
    "english": "en", "german": "de", "french": "fr", "spanish": "es", 
    "italian": "it", "japanese": "ja", "chinese": "zh", "portuguese": "pt",
    "russian": "ru", "arabic": "ar", "korean": "ko", "turkish": "tr",
    "hindi": "hi", "dutch": "nl", "swedish": "sv", "indonesian": "id",
    "polish": "pl", "finnish": "fi", "czech": "cs", "danish": "da",
    "auto": None  # None tells Whisper to auto-detect the language
}

# Available Whisper models
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

def setup_argparse():
    """Set up argument parser with all available options."""
    parser = argparse.ArgumentParser(description="Advanced audio transcription script using Whisper")
    
    # Required arguments
    parser.add_argument("--audio_file", type=str, required=True, help="Path to the audio file to transcribe")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the transcription output")
    
    # Audio processing options
    parser.add_argument("--language", type=str, default="german", 
                        help=f"Language for transcription (default: german). Options: {', '.join(LANGUAGE_MAP.keys())}")
    parser.add_argument("--min_silence_len", type=int, default=1000, 
                        help="Minimum silence length in ms (default: 1000)")
    parser.add_argument("--padding", type=int, default=750, 
                        help="Padding in ms added to start and end of segments (default: 750)")
    parser.add_argument("--silence_adjustment", type=float, default=3.0, 
                        help="Adjustment to silence threshold in dB (default: 3.0)")
    
    # Whisper model options
    parser.add_argument("--model", type=str, default="large-v3", choices=WHISPER_MODELS,
                        help=f"Whisper model to use (default: large-v3). Options: {', '.join(WHISPER_MODELS)}")
    parser.add_argument("--no_fp16", action="store_true", 
                        help="Disable FP16 for faster processing on CPUs without AVX2")
    
    # Processing options
    parser.add_argument("--workers", type=int, default=1, 
                        help="Number of parallel workers for transcription (default: 1)")
    parser.add_argument("--retry_attempts", type=int, default=3, 
                        help="Number of retry attempts for failed segments (default: 3)")
    parser.add_argument("--resume", action="store_true", 
                        help="Resume transcription from existing segments")
    parser.add_argument("--keep_segments", action="store_true", 
                        help="Keep audio segments after processing (default: false)")
    
    # Output format options
    parser.add_argument("--include_timestamps", action="store_true", 
                        help="Include timestamps in the output (default: false)")
    parser.add_argument("--output_format", type=str, default="txt", choices=["txt", "json", "srt", "vtt"],
                        help="Output format (default: txt)")
    parser.add_argument("--diarize", action="store_true", 
                        help="Attempt speaker diarization (experimental, requires pyannote.audio)")
    parser.add_argument("--max_speakers", type=int, default=2, 
                        help="Maximum number of speakers for diarization (default: 2)")
    
    # Debug options
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug mode")
    
    return parser

def analyze_background_noise(audio_file: str, adjustment: float = 3.0) -> float:
    """
    Analyze the audio to determine a suitable silence threshold.
    
    Args:
        audio_file: Path to the audio file
        adjustment: Adjustment to silence threshold in dB
        
    Returns:
        Calculated silence threshold
    """
    audio = AudioSegment.from_file(audio_file)
    # Calculate average dBFS (RMS of silence)
    noise_level = audio.dBFS
    silence_thresh = noise_level + adjustment
    logger.info(f"Estimated background noise level: {noise_level:.2f} dBFS")
    logger.info(f"Using silence threshold: {silence_thresh:.2f} dBFS")
    return silence_thresh

def split_audio_on_silence(
    audio_file: str, 
    silence_thresh: float, 
    min_silence_len: int = 1000, 
    padding: int = 750
) -> List[Tuple[str, int, int]]:
    """
    Split audio into segments based on silence.
    
    Args:
        audio_file: Path to the audio file
        silence_thresh: Threshold for silence detection in dBFS
        min_silence_len: Minimum silence length in ms
        padding: Padding in ms added to start and end of segments
        
    Returns:
        List of tuples containing (segment_file_path, start_time_ms, end_time_ms)
    """
    audio = AudioSegment.from_file(audio_file)
    total_duration_ms = len(audio)
    
    logger.info(f"Splitting audio into segments based on {min_silence_len}ms silence...")
    
    nonsilent_ranges = silence.detect_nonsilent(
        audio, 
        min_silence_len=min_silence_len, 
        silence_thresh=silence_thresh
    )
    
    if not nonsilent_ranges:
        logger.warning(f"No speech detected in audio file. Using entire file as one segment.")
        segment_file = f"{audio_file}_segment_0.wav"
        audio.export(segment_file, format="wav")
        return [(segment_file, 0, total_duration_ms)]
    
    # Add padding and split audio
    segments = []
    for i, (start, end) in enumerate(nonsilent_ranges):
        # Apply padding, ensuring start and end are within bounds
        segment_start = max(0, start - padding)
        segment_end = min(total_duration_ms, end + padding)
        segment = audio[segment_start:segment_end]
        segment_file = f"{audio_file}_segment_{i}.wav"
        segment.export(segment_file, format="wav")
        segments.append((segment_file, segment_start, segment_end))
    
    logger.info(f"Split audio into {len(segments)} segments")
    return segments

def get_unprocessed_segments(audio_file: str, processed_ids: set) -> List[Tuple[str, int, int]]:
    """
    Identify segments that still need to be processed.
    
    Args:
        audio_file: Path to the audio file
        processed_ids: Set of already processed segment IDs
        
    Returns:
        List of tuples containing (segment_file_path, start_time_ms, end_time_ms)
    """
    all_segments = glob(f"{audio_file}_segment_*.wav")
    unprocessed_segments = []
    
    # Extract timing information from filenames if available
    pattern = re.compile(r"_segment_(\d+)\.wav$")
    
    for segment_path in all_segments:
        match = pattern.search(segment_path)
        if match:
            segment_id = int(match.group(1))
            if segment_id not in processed_ids:
                # We don't have timing info for existing segments, use placeholders
                # This will only affect resume scenarios
                unprocessed_segments.append((segment_path, -1, -1))
    
    return unprocessed_segments

def get_processed_segments(output_file: str, output_format: str) -> set:
    """
    Identify segments that have already been processed.
    
    Args:
        output_file: Path to the output file
        output_format: Format of the output file
        
    Returns:
        Set of already processed segment IDs
    """
    processed_segments = set()
    
    if not os.path.exists(output_file):
        return processed_segments
    
    if output_format == "json":
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    if "segment_id" in entry:
                        processed_segments.add(entry["segment_id"])
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is empty or invalid JSON, treat as no processed segments
            pass
    else:
        # For text and subtitle formats
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract segment IDs using regex pattern
                for match in re.finditer(r"segment_(\d+)", content):
                    processed_segments.add(int(match.group(1)))
        except FileNotFoundError:
            pass
    
    logger.info(f"Found {len(processed_segments)} already processed segments")
    return processed_segments

def format_timestamp(ms: int) -> str:
    """
    Format milliseconds timestamp to HH:MM:SS.mmm.
    
    Args:
        ms: Timestamp in milliseconds
        
    Returns:
        Formatted timestamp string
    """
    seconds, ms = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"

def load_diarization_model():
    """
    Load the speaker diarization model.
    
    Returns:
        Diarization pipeline or None if not available
    """
    try:
        from pyannote.audio import Pipeline
        
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=os.environ.get("HF_TOKEN")
            )
            return pipeline
        except Exception as e:
            logger.error(f"Failed to load diarization model: {e}")
            logger.warning("Speaker diarization requires a Hugging Face token. Set the HF_TOKEN environment variable.")
            return None
            
    except ImportError:
        logger.warning("Speaker diarization requires pyannote.audio. Install with: pip install pyannote.audio")
        return None

def process_diarization(audio_file: str, max_speakers: int = 2):
    """
    Process speaker diarization on an audio file.
    
    Args:
        audio_file: Path to the audio file
        max_speakers: Maximum number of speakers
        
    Returns:
        Dictionary mapping time ranges to speaker IDs or None if diarization failed
    """
    pipeline = load_diarization_model()
    if not pipeline:
        return None

    try:
        diarization = pipeline(audio_file, num_speakers=max_speakers)
        speaker_map = {}
        
        # Convert diarization to a speaker map
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_time = int(turn.start * 1000)  # Convert to ms
            end_time = int(turn.end * 1000)  # Convert to ms
            speaker_id = int(speaker.strip('SPEAKER_'))
            
            # Store in 500ms increments for easier lookup
            current_time = start_time
            while current_time < end_time:
                speaker_map[current_time] = speaker_id
                current_time += 500
                
        return speaker_map
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        return None

def estimate_completion_time(
    processed: int, 
    total: int, 
    elapsed: float
) -> str:
    """
    Estimate completion time based on current progress.
    
    Args:
        processed: Number of processed items
        total: Total number of items
        elapsed: Elapsed time in seconds
        
    Returns:
        Estimated time of completion as a string
    """
    if processed == 0:
        return "calculating..."
    
    items_per_second = processed / elapsed
    remaining_items = total - processed
    estimated_seconds = remaining_items / items_per_second
    
    eta = datetime.now() + timedelta(seconds=estimated_seconds)
    time_remaining = str(timedelta(seconds=int(estimated_seconds)))
    
    return f"{time_remaining} (ETA: {eta.strftime('%H:%M:%S')})"

def transcribe_segment(
    model, 
    segment_info: Tuple[str, int, int],
    language: Optional[str] = None,
    fp16: bool = True,
    include_timestamps: bool = False,
    segment_idx: Optional[int] = None,
    diarization_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Transcribe a single segment.
    
    Args:
        model: Whisper model
        segment_info: Tuple of (segment_file_path, start_time_ms, end_time_ms)
        language: Language code for transcription
        fp16: Use FP16 precision
        include_timestamps: Include timestamps in the output
        segment_idx: Index of the segment for identification
        diarization_data: Speaker diarization data
        
    Returns:
        Dictionary with transcription results
    """
    segment_file, start_time, end_time = segment_info
    
    try:
        result = model.transcribe(segment_file, language=language, verbose=False, fp16=fp16)
        
        # Extract segment ID from filename
        segment_id = segment_idx
        if segment_id is None:
            match = re.search(r"_segment_(\d+)\.wav$", segment_file)
            segment_id = int(match.group(1)) if match else -1
        
        # Determine speaker if diarization data is available
        speaker_id = None
        if diarization_data and start_time >= 0:
            # Find the most common speaker in this segment
            speakers = {}
            current_time = start_time
            while current_time < end_time:
                if current_time in diarization_data:
                    speaker = diarization_data[current_time]
                    speakers[speaker] = speakers.get(speaker, 0) + 1
                current_time += 500
            
            if speakers:
                speaker_id = max(speakers.items(), key=lambda x: x[1])[0]
        
        # Create result object
        transcription_result = {
            "segment_id": segment_id,
            "text": result["text"].strip(),
            "confidence": float(result.get("confidence", 0.0)),
            "segment_file": segment_file
        }
        
        # Add timestamps and durations if requested
        if include_timestamps and start_time >= 0 and end_time >= 0:
            transcription_result["start_time"] = start_time
            transcription_result["end_time"] = end_time
            transcription_result["duration"] = end_time - start_time
        
        # Add speaker information if available
        if speaker_id is not None:
            transcription_result["speaker"] = f"Speaker {speaker_id}"
        
        return transcription_result
    
    except Exception as e:
        logger.error(f"Error transcribing {segment_file}: {str(e)}")
        return {
            "segment_file": segment_file,
            "error": str(e),
            "segment_id": -1,
            "text": ""
        }

def save_result(
    result: Dict[str, Any],
    output_file: str,
    output_format: str,
    include_timestamps: bool
) -> bool:
    """
    Save transcription result to the output file.
    
    Args:
        result: Transcription result dictionary
        output_file: Path to the output file
        output_format: Format of the output file
        include_timestamps: Include timestamps in the output
        
    Returns:
        True if the result was saved successfully
    """
    try:
        # Skip if there was an error
        if "error" in result and not result.get("text"):
            return False
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        if output_format == "json":
            # Check if file exists to append or create new
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                with open(output_file, 'r+', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        data.append(result)
                        f.seek(0)
                        json.dump(data, f, indent=2, ensure_ascii=False)
                        f.truncate()
                    except json.JSONDecodeError:
                        # If the file exists but isn't valid JSON, overwrite it
                        f.seek(0)
                        json.dump([result], f, indent=2, ensure_ascii=False)
                        f.truncate()
            else:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump([result], f, indent=2, ensure_ascii=False)
                    
        elif output_format == "srt":
            with open(output_file, 'a', encoding='utf-8') as f:
                segment_num = result.get("segment_id", 0) + 1
                
                if include_timestamps and "start_time" in result and "end_time" in result:
                    start_time = format_timestamp(result["start_time"]).replace('.', ',')
                    end_time = format_timestamp(result["end_time"]).replace('.', ',')
                    
                    f.write(f"{segment_num}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    
                    # Add speaker if available
                    if "speaker" in result:
                        f.write(f"[{result['speaker']}] {result['text']}\n\n")
                    else:
                        f.write(f"{result['text']}\n\n")
                else:
                    # SRT requires timestamps, so we create dummy ones if not provided
                    f.write(f"{segment_num}\n")
                    f.write(f"00:00:00,000 --> 00:00:00,000\n")
                    f.write(f"{result['text']}\n\n")
                    
        elif output_format == "vtt":
            with open(output_file, 'a', encoding='utf-8') as f:
                # Write header if file is empty
                if os.path.getsize(output_file) == 0:
                    f.write("WEBVTT\n\n")
                
                if include_timestamps and "start_time" in result and "end_time" in result:
                    start_time = format_timestamp(result["start_time"])
                    end_time = format_timestamp(result["end_time"])
                    
                    # Add speaker if available
                    if "speaker" in result:
                        f.write(f"{start_time} --> {end_time} <v {result['speaker']}>\n")
                        f.write(f"{result['text']}\n\n")
                    else:
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{result['text']}\n\n")
                else:
                    # VTT requires timestamps, so we create dummy ones if not provided
                    f.write(f"00:00:00.000 --> 00:00:00.000\n")
                    f.write(f"{result['text']}\n\n")
                    
        else:  # Default to text format
            with open(output_file, 'a', encoding='utf-8') as f:
                line = ""
                
                # Add speaker if available
                if "speaker" in result:
                    line += f"[{result['speaker']}] "
                    
                # Add timestamps if requested
                if include_timestamps and "start_time" in result and "end_time" in result:
                    start_time = format_timestamp(result["start_time"])
                    end_time = format_timestamp(result["end_time"])
                    line += f"[{start_time} - {end_time}] "
                    
                # Add text
                line += result["text"]
                f.write(line + "\n")
                
        return True
        
    except Exception as e:
        logger.error(f"Error saving result: {str(e)}")
        return False

def process_audio(args):
    """
    Process audio file with all the configured options.
    
    Args:
        args: Command line arguments
    """
    # Validate input file
    if not os.path.exists(args.audio_file):
        logger.error(f"Audio file not found: {args.audio_file}")
        return
    
    # Set log level based on arguments
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
    # Normalize language input and get code
    language_key = args.language.lower()
    language_code = LANGUAGE_MAP.get(language_key)
    if language_key not in LANGUAGE_MAP:
        logger.warning(f"Unknown language: {args.language}. Using auto-detection.")
        language_code = None  # None means auto-detection
    
    # Clear output file if not resuming
    if not args.resume and os.path.exists(args.output_file):
        # Create a backup if the file exists
        if os.path.getsize(args.output_file) > 0:
            backup_file = f"{args.output_file}.bak"
            try:
                with open(args.output_file, 'r', encoding='utf-8') as src:
                    with open(backup_file, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
                logger.info(f"Created backup of existing output file: {backup_file}")
            except Exception as e:
                logger.warning(f"Could not create backup: {e}")
                
        # Initialize the output file
        if args.output_format == "vtt":
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
        elif args.output_format == "json":
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write("[]")
        else:
            open(args.output_file, 'w', encoding='utf-8').close()
    
    # Initialize diarization if requested
    diarization_data = None
    if args.diarize:
        logger.info("Initializing speaker diarization...")
        diarization_data = process_diarization(args.audio_file, args.max_speakers)
        if diarization_data is None:
            logger.warning("Speaker diarization failed or is not available. Continuing without it.")
    
    # Get segments (either by splitting audio or finding existing segments)
    segment_infos = []
    processed_ids = set()
    
    if args.resume:
        # Get IDs of already processed segments
        processed_ids = get_processed_segments(args.output_file, args.output_format)
        # Get unprocessed segments
        segment_infos = get_unprocessed_segments(args.audio_file, processed_ids)
        logger.info(f"Resuming with {len(segment_infos)} unprocessed segments...")
    else:
        # Analyze audio and split into segments
        silence_thresh = analyze_background_noise(args.audio_file, args.silence_adjustment)
        segment_infos = split_audio_on_silence(
            args.audio_file, silence_thresh, args.min_silence_len, args.padding
        )
    
    if not segment_infos:
        logger.info("No segments to process. Exiting.")
        return
    
    # Load the Whisper model
    logger.info(f"Loading Whisper model: {args.model}")
    model = whisper.load_model(args.model)
    
    # Prepare for processing
    total_segments = len(segment_infos)
    start_time = time.time()
    
    logger.info(f"Starting transcription of {total_segments} segments with {args.workers} worker{'s' if args.workers > 1 else ''}...")
    
    # Create a progress bar
    pbar = tqdm.tqdm(total=total_segments, desc="Transcribing")
    
    # Process segments
    processed_count = 0
    successful_count = 0
    
    if args.workers <= 1:
        # Single-threaded processing
        for i, segment_info in enumerate(segment_infos):
            segment_file = segment_info[0]
            
            for attempt in range(args.retry_attempts):
                result = transcribe_segment(
                    model, 
                    segment_info, 
                    language_code, 
                    not args.no_fp16,
                    args.include_timestamps,
                    None,
                    diarization_data
                )
                
                success = save_result(
                    result, 
                    args.output_file, 
                    args.output_format, 
                    args.include_timestamps
                )
                
                if success:
                    successful_count += 1
                    # Clean up segment file if we're not keeping them
                    if not args.keep_segments:
                        try:
                            os.remove(segment_file)
                        except Exception as e:
                            logger.warning(f"Could not remove segment file {segment_file}: {e}")
                    break
                
                if attempt < args.retry_attempts - 1:
                    logger.warning(f"Retry {attempt+1}/{args.retry_attempts} for segment {segment_file}")
            else:
                logger.error(f"Failed to transcribe {segment_file} after {args.retry_attempts} attempts.")
            
            processed_count += 1
            pbar.update(1)
            
            # Update ETA information
            if i % 5 == 0 or i == total_segments - 1:
                elapsed = time.time() - start_time
                eta = estimate_completion_time(processed_count, total_segments, elapsed)
                pbar.set_description(f"Transcribing (ETA: {eta})")
    else:
        # Parallel processing with thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            futures = []
            for segment_info in segment_infos:
                future = executor.submit(
                    transcribe_segment,
                    model, 
                    segment_info, 
                    language_code, 
                    not args.no_fp16,
                    args.include_timestamps,
                    None,
                    diarization_data
                )
                futures.append((future, segment_info[0]))
            
            # Process results as they complete
            for future, segment_file in futures:
                try:
                    result = future.result()
                    success = save_result(
                        result, 
                        args.output_file, 
                        args.output_format, 
                        args.include_timestamps
                    )
                    
                    if success:
                        successful_count += 1
                        # Clean up segment file if we're not keeping them
                        if not args.keep_segments:
                            try:
                                os.remove(segment_file)
                            except Exception as e:
                                logger.warning(f"Could not remove segment file {segment_file}: {e}")
                except Exception as e:
                    logger.error(f"Exception processing segment {segment_file}: {e}")
                
                processed_count += 1
                pbar.update(1)
                
                # Update ETA information
                if processed_count % 5 == 0 or processed_count == total_segments:
                    elapsed = time.time() - start_time
                    eta = estimate_completion_time(processed_count, total_segments, elapsed)
                    pbar.set_description(f"Transcribing (ETA: {eta})")
    
    pbar.close()
    
    # Print summary statistics
    elapsed_time = time.time() - start_time
    logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")
    logger.info(f"Successfully transcribed {successful_count}/{total_segments} segments")
    
    # Print output information
    logger.info(f"Output saved to: {os.path.abspath(args.output_file)}")

def main():
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Check that required arguments are provided
    if not args.audio_file or not args.output_file:
        parser.print_help()
        return
    
    # Process audio file
    process_audio(args)

if __name__ == "__main__":
    main()
