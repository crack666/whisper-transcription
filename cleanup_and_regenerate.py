#!/usr/bin/env python3
"""
Cleanup and regenerate MAD HTML reports with fixed JavaScript
"""

import os
import json
from pathlib import Path
from src.html_generator import HTMLReportGenerator

def cleanup_and_regenerate_directory(target_dir: Path, dir_name: str):
    """
    1. Delete old HTML reports in target directory
    2. Regenerate new HTML reports from JSON files
    """
    
    if not target_dir.exists():
        print(f"⚠️  {target_dir} does not exist, skipping...")
        return 0
    
    print(f"\n{'='*60}")
    print(f"🔄 Processing: {dir_name}")
    print(f"{'='*60}\n")
    
    print("🧹 Cleaning up old HTML reports...")
    
    # Delete old HTML reports
    old_reports = list(target_dir.glob("*_report.html"))
    for report in old_reports:
        report.unlink()
        print(f"   🗑️  Deleted: {report.name}")
    
    print()
    print("🔄 Regenerating HTML reports from JSON files...")
    print()
    
    html_generator = HTMLReportGenerator()
    
    # Find all JSON analysis files (new format: Video Name.mp4.json)
    json_files = list(target_dir.glob("*.mp4.json"))
    
    if not json_files:
        print("   ⚠️  No JSON analysis files found")
        return 0
    
    print(f"📄 Found {len(json_files)} JSON files to process\n")
    
    success_count = 0
    
    for json_file in json_files:
        # Extract base name from JSON filename
        # e.g., "Video Name.mp4.json" -> "Video Name.mp4"
        base_name = json_file.stem  # This gives us "Video Name.mp4"
        print(f"🔄 Processing: {base_name}")
        
        # Load JSON data
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
        except Exception as e:
            print(f"   ❌ Failed to load JSON: {e}")
            continue
        
        # Generate HTML report using new naming convention: Video Name.mp4_report.html
        html_dest = target_dir / f"{base_name}_report.html"
        try:
            html_generator.generate_report(analysis_data, str(html_dest))
            print(f"   ✅ Generated HTML: {html_dest.name}")
            success_count += 1
        except Exception as e:
            print(f"   ❌ Failed to generate HTML: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    return success_count


def cleanup_and_regenerate():
    """
    Process all directories: mad, pqm, datenbanken
    """
    print("🚀 Starting HTML report regeneration...")
    
    directories = {
        "mad": Path("./mad"),
        "pqm": Path("./pqm"),
        "datenbanken": Path("./datenbanken")
    }
    
    total_success = 0
    total_files = 0
    
    for dir_name, dir_path in directories.items():
        count = cleanup_and_regenerate_directory(dir_path, dir_name.upper())
        total_success += count
        if dir_path.exists():
            total_files += len(list(dir_path.glob("*.mp4.json")))
    
    print("\n" + "="*60)
    print("✨ All directories processed!")
    print("="*60)
    print(f"\n📋 Summary:")
    print(f"   - Total JSON files: {total_files}")
    print(f"   - Successfully generated: {total_success} HTML reports")

if __name__ == "__main__":
    cleanup_and_regenerate()
