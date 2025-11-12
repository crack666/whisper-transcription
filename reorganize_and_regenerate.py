#!/usr/bin/env python3
"""
Cleanup and regenerate MAD HTML reports with fixed JavaScript
"""

import os
import json
from pathlib import Path
from src.html_generator import HTMLReportGenerator

def cleanup_and_regenerate():
    """
    1. Delete old HTML reports in ./mad/
    2. Regenerate new HTML reports from JSON files
    """
    
    target_mad = Path("./mad")
    
    if not target_mad.exists():
        print(f"❌ {target_mad} does not exist")
        return
    
    print("🧹 Cleaning up old HTML reports...")
    
    # Delete old HTML reports
    old_reports = list(target_mad.glob("*_report.html"))
    for report in old_reports:
        report.unlink()
        print(f"   🗑️  Deleted: {report.name}")
    
    print()
    print("� Regenerating HTML reports from JSON files...")
    print()
    
    html_generator = HTMLReportGenerator()
    
    # Find all JSON analysis files
    json_files = list(target_mad.glob("*_analysis.json"))
    
    if not json_files:
        print("❌ No JSON analysis files found")
        return
    
    print(f"📄 Found {len(json_files)} JSON files to process")
    print()
    
    for json_file in json_files:
        video_name = subdir.name
        print(f"🔄 Processing: {video_name}")
        
        # Find JSON file
        json_files = list(subdir.glob("*_analysis.json"))
        if not json_files:
            print(f"   ⚠️  No JSON file found, skipping")
            continue
        
        json_file = json_files[0]
        print(f"   📄 Found JSON: {json_file.name}")
        
        # Load JSON data
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
        except Exception as e:
            print(f"   ❌ Failed to load JSON: {e}")
            continue
        
        # Handle screenshots folder
        screenshots_src = subdir / "screenshots"
        if screenshots_src.exists():
            # Rename to {name}_screenshots in target directory
            screenshots_dest = target_mad / f"{video_name}_screenshots"
            
            if screenshots_dest.exists():
                print(f"   ⚠️  Screenshot folder already exists: {screenshots_dest.name}")
            else:
                try:
                    shutil.move(str(screenshots_src), str(screenshots_dest))
                    print(f"   📸 Moved screenshots to: {screenshots_dest.name}")
                except Exception as e:
                    print(f"   ❌ Failed to move screenshots: {e}")
                    continue
            
            # Update screenshot paths in JSON
            if "screenshots" in analysis_data:
                for screenshot in analysis_data["screenshots"]:
                    if "filepath" in screenshot:
                        # Update path to new location
                        old_path = screenshot["filepath"]
                        filename = Path(old_path).name
                        screenshot["filepath"] = str(screenshots_dest / filename)
        else:
            print(f"   ⚠️  No screenshots folder found")
        
        # Move JSON to target directory
        json_dest = target_mad / json_file.name
        try:
            shutil.copy2(json_file, json_dest)
            print(f"   💾 Copied JSON to: {json_dest.name}")
        except Exception as e:
            print(f"   ❌ Failed to copy JSON: {e}")
            continue
        
        # Regenerate HTML report
        html_dest = target_mad / f"{video_name}_report.html"
        try:
            html_generator.generate_report(analysis_data, str(html_dest))
            print(f"   ✅ Generated HTML: {html_dest.name}")
        except Exception as e:
            print(f"   ❌ Failed to generate HTML: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("✨ Done!")
    print()
    print("📋 Summary:")
    print(f"   - Processed {len(subdirs)} directories")
    print(f"   - Files are now in: {target_mad.absolute()}")
    print()
    print("🗑️  You can now delete ./results/mad/ if everything looks good")

if __name__ == "__main__":
    reorganize_and_regenerate()
