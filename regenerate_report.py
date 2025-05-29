import sys
import os
import json
import logging
from datetime import datetime
import glob

# Add the parent directory of 'src' to Python path to treat 'src' as a package
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.html_generator import HTMLReportGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

def find_latest_debug_json(directory: str) -> str | None:
    """Finds the most recent 'debug_all_results_*.json' file in the given directory."""
    try:
        list_of_files = glob.glob(os.path.join(directory, 'debug_all_results_*.json'))
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        logging.error(f"Error finding latest debug JSON file: {e}")
        return None

def main():
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(script_dir, "output", "test_reports")

    logging.info(f"Looking for the latest debug JSON in: {output_dir}")
    latest_json_path = find_latest_debug_json(output_dir)

    if not latest_json_path:
        logging.error("No 'debug_all_results_*.json' file found. Cannot regenerate report.")
        print("No debug JSON file found. Please run 'run_refactored_test.py' first to generate data.")
        return

    logging.info(f"Found latest debug JSON: {latest_json_path}")

    try:
        with open(latest_json_path, 'r', encoding='utf-8') as f_json:
            all_analysis_results = json.load(f_json)
        logging.info(f"Successfully loaded data from {latest_json_path}")
    except Exception as e:
        logging.error(f"Failed to load or parse JSON file {latest_json_path}: {e}", exc_info=True)
        print(f"Error loading data from {latest_json_path}. Report generation aborted.")
        return

    if not all_analysis_results:
        logging.warning("Loaded data is empty. No report will be generated.")
        print("The debug JSON file is empty. No report to generate.")
        return

    logging.info("Regenerating HTML report from loaded data...")
    report_generator = HTMLReportGenerator()
    
    # Generate a new report file path with a new timestamp
    new_report_file_name = f"regenerated_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_file_path = os.path.join(output_dir, new_report_file_name)
    
    try:
        report_generator.generate_report(all_analysis_results, report_file_path)
        logging.info(f"Successfully regenerated HTML report: {report_file_path}")
        print(f"\\nRegenerated HTML report saved to: {report_file_path}")
    except Exception as e:
        logging.error(f"Failed to regenerate HTML report: {e}", exc_info=True)
        print(f"Failed to regenerate HTML report: {e}")

if __name__ == "__main__":
    main()
