import whisper
import time
import argparse
from pydub import AudioSegment, silence
import os
from glob import glob

def analyze_background_noise(audio_file):
    """
    Analyze the audio to determine a suitable silence threshold.
    """
    audio = AudioSegment.from_file(audio_file)
    # Calculate average dBFS (RMS of silence)
    noise_level = audio.dBFS
    print(f"Estimated background noise level: {noise_level:.2f} dBFS")
    return noise_level

def split_audio_on_silence(audio_file, silence_thresh, min_silence_len=1000, padding=750):
    """
    Split audio into segments based on silence.
    """
    audio = AudioSegment.from_file(audio_file)
    nonsilent_ranges = silence.detect_nonsilent(
        audio, 
        min_silence_len=min_silence_len, 
        silence_thresh=silence_thresh
    )

    # Add padding and split audio
    segments = []
    for start, end in nonsilent_ranges:
        # Apply padding, ensuring start and end are within bounds
        segment_start = max(0, start - padding)
        segment_end = min(len(audio), end + padding)
        segment = audio[segment_start:segment_end]
        segment_file = f"{audio_file}_segment_{len(segments)}.wav"
        segment.export(segment_file, format="wav")
        segments.append(segment_file)

    return segments

def get_unprocessed_segments(audio_file, output_file):
    """
    Identify segments that still need to be processed.
    """
    all_segments = glob(f"{audio_file}_segment_*.wav")
    processed_segments = set()
    
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if "_segment_" in line:
                    processed_segments.add(line.strip())
    
    return [seg for seg in all_segments if seg not in processed_segments]

def transcribe_segment(model, segment_file, output_file, language):
    """
    Transcribe a single segment.
    """
    try:
        result = model.transcribe(segment_file, language=language, verbose=True, fp16=True)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(result["text"] + "\n")
        return True
    except UnicodeEncodeError as e:
        print(f"UnicodeEncodeError encountered while transcribing {segment_file}: {e}.")
    except Exception as e:
        print(f"An error occurred while transcribing {segment_file}: {e}.")
    return False

def transcribe_with_logging(audio_file, output_file, language="de", min_silence_len=1000, padding=750, resume=False):
    """
    Transcribe audio file by splitting it at silent intervals.
    """
    if not resume:
        # Analyze background noise to determine silence threshold
        silence_thresh = analyze_background_noise(audio_file) + 3  # Adjust by 10 dB for better segmentation
        print(f"Using silence threshold: {silence_thresh:.2f} dBFS")

        # Split audio into segments based on silence
        print(f"Splitting audio into segments based on {min_silence_len}ms silence...")
        segments = split_audio_on_silence(audio_file, silence_thresh, min_silence_len, padding)
    else:
        # Resume from existing segments
        print(f"Resuming transcription from existing segments...")
        segments = get_unprocessed_segments(audio_file, output_file)

    if not segments:
        print("No segments found to process. Exiting.")
        return

    # Load the Whisper model
    model = whisper.load_model("large-v3")

    # Start time tracking
    start_time = time.time()

    # Transcribe each segment
    for segment_file in segments:
        retry_attempts = 3
        for attempt in range(retry_attempts):
            success = transcribe_segment(model, segment_file, output_file, language)
            if success:
                os.remove(segment_file)  # Remove the segment file after successful processing
                break
            print(f"Retry attempt {attempt + 1} for {segment_file} failed.")
        else:
            print(f"Failed to transcribe {segment_file} after {retry_attempts} attempts. Skipping this segment.")

    # Print total processing time
    elapsed_time = time.time() - start_time
    print(f"Total transcription completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    # Set up argument parser for command-line usage
    parser = argparse.ArgumentParser(description="Audio transcription script using Whisper")
    parser.add_argument("--audio_file", type=str, help="Path to the audio file to transcribe")
    parser.add_argument("--output_file", type=str, help="Path to save the transcription output")
    parser.add_argument("--language", type=str, default="German", help="Language for transcription (default: German)")
    parser.add_argument("--min_silence_len", type=int, default=1000, help="Minimum silence length in ms (default: 1000)")
    parser.add_argument("--padding", type=int, default=750, help="Padding in ms added to start and end of segments (default: 750)")
    parser.add_argument("--resume", action="store_true", help="Resume transcription from existing segments")

    args = parser.parse_args()

    # Run transcription
    transcribe_with_logging(args.audio_file, args.output_file, args.language, args.min_silence_len, args.padding, args.resume)
