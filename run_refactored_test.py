import sys
import os
import json
import logging
from datetime import datetime # For timestamping the report

# Add the parent directory of 'src' to Python path to treat 'src' as a package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.enhanced_transcriber import EnhancedAudioTranscriber
from src.html_generator import HTMLReportGenerator # Import HTMLReportGenerator

# Configure basic logging to see output from the transcriber and other modules
# Change logging level to DEBUG to see detailed logs
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

def main():
    script_dir = os.path.dirname(__file__)
    
    # Define the list of files to process
    # Using relative paths from the script_dir (project root)
    files_to_process = [
        "interview_cut.mp3",
        "TestFile_cut.mp4" 
    ]
    
    all_analysis_results = []
    
    # Initialize the transcriber once
    transcriber = EnhancedAudioTranscriber()

    for audio_file_name in files_to_process:
        audio_file_path = os.path.join(script_dir, audio_file_name)
        
        # Check if the audio file exists and handle fallbacks
        if not os.path.exists(audio_file_path):
            logging.warning(f"File not found at {audio_file_path}. Attempting fallback...")
            # Fallback for .mp3.wav or .mp4.wav
            base_name, ext = os.path.splitext(audio_file_name)
            fallback_wav_name = base_name + ext + ".wav" # e.g. interview_cut.mp3.wav
            fallback_direct_wav_name = base_name + ".wav" # e.g. TestFile_cut.wav (if original was TestFile_cut.mp4)
            
            potential_fallbacks = [fallback_wav_name]
            if ext.lower() == ".mp4": # For mp4, also try direct .wav version
                potential_fallbacks.append(fallback_direct_wav_name)

            found_fallback = False
            for fallback_name in potential_fallbacks:
                fallback_path = os.path.join(script_dir, fallback_name)
                if os.path.exists(fallback_path):
                    logging.info(f"Found fallback: {fallback_name}. Using this file.")
                    audio_file_path = fallback_path
                    found_fallback = True
                    break
            
            if not found_fallback:
                logging.error(f"No suitable file or fallback found for {audio_file_name}. Skipping this file.")
                continue # Skip to the next file

        logging.info(f"Starting transcription for: {audio_file_path}")

        try:
            result = transcriber.transcribe_audio_file_enhanced(audio_file_path)
            
            # DEBUG: Print the structure of transcription results, especially segments
            logging.debug(f"Full result for {audio_file_name}: {json.dumps(result, indent=2, ensure_ascii=False)}")
            if 'transcription' in result and 'segments' in result['transcription']:
                logging.debug(f"Segments for {audio_file_name} (count: {len(result['transcription']['segments'])}): {json.dumps(result['transcription']['segments'], indent=2, ensure_ascii=False)}")
            elif 'transcription' in result:
                logging.debug(f"No 'segments' key in result['transcription'] for {audio_file_name}. Keys: {list(result['transcription'].keys())}")
            else:
                logging.debug(f"No 'transcription' key in result for {audio_file_name}. Top-level keys: {list(result.keys())}")

            # Add a processing timestamp to the results for the report
            result['processing_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Ensure audio_file_path is the one actually processed (could be a fallback)
            result['audio_file_path'] = audio_file_path 

            all_analysis_results.append(result)
            
            #print(f"\n" + "="*20 + f" TRANSCRIPTION RESULT for {audio_file_name} " + "="*20)
            #print(json.dumps(result, indent=4, ensure_ascii=False))

            if "text" in result and result["text"]:
                print(f"\n" + "="*20 + f" FULL TRANSCRIBED TEXT for {audio_file_name} is DONE " + "="*20)
                #print(result['text'])
            else:
                print(f"\n" + "="*20 + f" NOTES for {audio_file_name} " + "="*20)
                print("No text transcribed or an error might have occurred.")
                if "warnings" in result and result["warnings"]:
                    print(f"Warnings: {result['warnings']}")

        except Exception as e:
            logging.error(f"An error occurred during the transcription of {audio_file_name}: {e}", exc_info=True)
            # Add a placeholder result for the report if a file fails
            all_analysis_results.append({
                "audio_file_path": audio_file_path,
                "error": str(e),
                "processing_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "transcription": {"text": "Error during processing.", "segments": []},
                "transcription_config": {},
                "speech_pattern_analysis": {}
            })

    # After processing all files, generate the HTML report if there are any results
    if all_analysis_results:
        logging.info("Generating combined HTML report for all processed files...")
        report_generator = HTMLReportGenerator()
        output_dir = os.path.join(script_dir, "output", "test_reports") # Define an output directory
        os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists
        
        # Save all_analysis_results to a debug JSON file
        debug_json_path = os.path.join(output_dir, f"debug_all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(debug_json_path, 'w', encoding='utf-8') as f_json:
                json.dump(all_analysis_results, f_json, indent=4, ensure_ascii=False)
            logging.info(f"Saved debug results to: {debug_json_path}")
        except Exception as e:
            logging.error(f"Failed to save debug JSON: {e}", exc_info=True)

        report_file_path = os.path.join(output_dir, f"combined_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        try:
            report_generator.generate_report(all_analysis_results, report_file_path)
            logging.info(f"Combined HTML report saved to: {report_file_path}")
            print(f"\nCombined report generated: {report_file_path}")
        except Exception as e:
            logging.error(f"Failed to generate HTML report: {e}", exc_info=True)
            print(f"Failed to generate HTML report: {e}")
    else:
        logging.warning("No analysis results to generate a report.")
        print("\nNo files were successfully processed to generate a report.")

if __name__ == "__main__":
    main()
