"""
HTML report generation module for creating interactive study material reports
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from src.utils import format_timestamp_seconds

logger = logging.getLogger(__name__)

class HTMLReportGenerator:
    """
    Generate interactive HTML reports for study material analysis.
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the HTML report generator.
        
        Args:
            template_dir: Directory containing custom templates (optional)
        """
        self.template_dir = Path(template_dir) if template_dir else None
        logger.info("Initialized HTMLReportGenerator")
    
    def generate_report(self, results: Dict, output_path: str) -> None:
        """
        Generate comprehensive HTML report from analysis results.
        
        Args:
            results: Complete analysis results dictionary for a single file.
            output_path: Path to save the HTML report
        """
        logger.info(f"Generating HTML report: {output_path}")
        
        # Ensure results is a list for _generate_html_document, 
        # as it's designed to handle multiple results for the embedded JSON data,
        # even if this specific report is for a single file.
        results_for_html_doc = [results] if isinstance(results, dict) else results
        
        html_content = self._generate_html_document(results_for_html_doc)
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {output_path}")
    
    def _generate_html_document(self, results: List[Dict]) -> str: # Changed results type to List[Dict] for clarity
        """
        Generate complete HTML document with a file selector and dynamic content area.
        
        Args:
            results: A list of analysis results for multiple files.
            
        Returns:
            Complete HTML document as string
        """
        if not isinstance(results, list) or not results:
            logger.error("Invalid or empty results provided. Expected a list of analysis dicts.")
            # Return a minimal HTML page indicating the error
            return '''
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Error</title></head>
<body><p>Error: No analysis data to display or data is in an invalid format.</p></body></html>
'''

        first_result_data = results[0]
        
        report_title_base = "Transcription Analysis Report"
        try:
            audio_path = first_result_data.get("audio_file_path")
            if audio_path:
                report_title_base = Path(audio_path).stem
            elif first_result_data.get("error"):
                 # Use a simpler way to construct the f-string for error titles
                 file_name_for_error = Path(first_result_data.get("audio_file_path", "Unknown File")).name
                 report_title_base = f"Error processing: {file_name_for_error}"
        except Exception as e:
            logger.warning(f"Could not determine report title from first file: {e}")
            # report_title_base remains "Transcription Analysis Report"

        return f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title_base} - Analysis Report</title>
    <script type="application/json" id="allData">
        {json.dumps(results, ensure_ascii=False, indent=2)}
    </script>
    <style>
        {self._get_embedded_css()}
    </style>
</head>
<body>
    {self._generate_body_content(first_result_data, report_title_base)}
    <script>
        {self._get_embedded_javascript(results)}
    </script>
</body>
</html>
'''

    def _get_embedded_css(self) -> str:
        """Returns the embedded CSS for the HTML report."""
        return '''
        body { font-family: sans-serif; margin: 0; background-color: #f4f4f4; color: #333; }
        .container { max-width: 1200px; margin: 20px auto; background-color: #fff; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; }
        .header { background-color: #007bff; color: white; padding: 20px; border-radius: 8px 8px 0 0; margin: -20px -20px 20px -20px; }
        .header h1 { margin: 0; font-size: 1.8em; }
        .header-info { display: flex; flex-wrap: wrap; gap: 15px; margin-top: 10px; font-size: 0.9em; }
        .header-info div { background-color: rgba(255,255,255,0.1); padding: 8px; border-radius: 4px; }
        .file-selector-container { margin-bottom: 20px; padding: 10px; background-color: #e9ecef; border-radius: 4px; }
        .file-selector-container label { margin-right: 10px; font-weight: bold; }
        #fileSelector { padding: 8px; border-radius: 4px; border: 1px solid #ccc; min-width: 300px; }
        .tabs { display: flex; border-bottom: 1px solid #ccc; margin-bottom: 20px; }
        .tab { padding: 10px 15px; cursor: pointer; border: 1px solid transparent; border-bottom: none; margin-right: 5px; border-radius: 4px 4px 0 0; background-color: #e9ecef; }
        .tab.active { background-color: #fff; border-color: #ccc; border-bottom-color: #fff; position: relative; top: 1px; font-weight: bold; }
        .tab-content { display: none; padding: 15px; border: 1px solid #ccc; border-top: none; border-radius: 0 0 4px 4px; }
        .tab-content.active { display: block; }
        .section { margin-bottom: 20px; }
        .section h2 { font-size: 1.5em; color: #007bff; border-bottom: 2px solid #007bff; padding-bottom: 5px; margin-top: 0; }
        .section-stats { display: flex; gap: 20px; margin-bottom: 10px; font-size: 0.9em; color: #555; }
        .transcript-segment { border: 1px solid #eee; padding: 10px; margin-bottom: 10px; border-radius: 4px; background-color: #f9f9f9; }
        .segment-header { display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 0.9em; color: #555; }
        .timestamp { font-weight: bold; }
        .confidence { padding: 2px 6px; border-radius: 3px; color: white; }
        .high-confidence { background-color: #28a745; }
        .medium-confidence { background-color: #ffc107; color: #333; }
        .low-confidence { background-color: #dc3545; }
        .segment-text { margin-top: 0; white-space: pre-wrap; }
        .stats-card { background-color: #f9f9f9; border: 1px solid #eee; border-radius: 4px; padding: 15px; margin-bottom: 15px; }
        .stats-card h3 { margin-top: 0; font-size: 1.2em; color: #333; }
        .stats-card p { margin: 5px 0; font-size: 0.95em; }
        .stats-card strong { color: #007bff; }
        .empty-state { text-align: center; color: #777; padding: 20px; font-style: italic; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 15px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 0.9em; }
        th { background-color: #007bff; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        
        /* Timeline-based UI Styles */
        .timeline-container { background: #f8f9fa; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .timeline-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .timeline-controls { display: flex; gap: 10px; align-items: center; }
        .timeline-button { background: #007bff; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 14px; }
        .timeline-button:hover { background: #0056b3; }
        .timeline-button:disabled { background: #6c757d; cursor: not-allowed; }
        .timeline-info { font-size: 14px; color: #666; }
        
        .timeline-main { display: flex; gap: 20px; height: 500px; }
        .timeline-sidebar { width: 300px; display: flex; flex-direction: column; }
        .timeline-content { flex: 1; display: flex; flex-direction: column; }
        
        .timeline-slider-container { margin-bottom: 15px; }
        .timeline-slider { width: 100%; height: 8px; background: #ddd; border-radius: 4px; outline: none; cursor: pointer; }
        .timeline-slider::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 20px; height: 20px; background: #007bff; border-radius: 50%; cursor: pointer; }
        .timeline-slider::-moz-range-thumb { width: 20px; height: 20px; background: #007bff; border-radius: 50%; cursor: pointer; border: none; }
        
        .timeline-time-display { text-align: center; font-family: monospace; font-size: 16px; margin-bottom: 10px; color: #333; }
        
        .segments-list { flex: 1; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; background: white; }
        .segment-item { padding: 12px; border-bottom: 1px solid #eee; cursor: pointer; transition: background-color 0.2s; }
        .segment-item:hover { background-color: #f8f9fa; }
        .segment-item.active { background-color: #e3f2fd; border-left: 4px solid #007bff; }
        .segment-time { font-family: monospace; font-size: 12px; color: #666; margin-bottom: 4px; }
        .segment-text { font-size: 14px; line-height: 1.4; }
        
        .screenshot-viewer { flex: 1; display: flex; flex-direction: column; border: 1px solid #ddd; border-radius: 4px; background: white; }
        .screenshot-header { padding: 15px; border-bottom: 1px solid #eee; background: #f8f9fa; }
        .screenshot-content { flex: 1; padding: 15px; display: flex; align-items: center; justify-content: center; background: #fafafa; }
        .screenshot-image { max-width: 100%; max-height: 100%; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .screenshot-placeholder { color: #999; font-style: italic; text-align: center; }
        
        .timeline-stats { display: flex; gap: 20px; margin-top: 15px; font-size: 14px; color: #666; }
        .timeline-stat { display: flex; align-items: center; gap: 5px; }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .timeline-main { flex-direction: column; height: auto; }
            .timeline-sidebar { width: 100%; }
            .screenshot-viewer { height: 300px; }
        }
        
        /* Add more styles as needed */
        '''
    
    def _generate_benchmark_info_html(self, benchmark_data: Dict) -> str:
        """
        Generate HTML for benchmark information display in header.
        
        Args:
            benchmark_data: Benchmark data from BenchmarkLogger
            
        Returns:
            HTML string with benchmark info
        """
        if not benchmark_data:
            return ""
        
        # Extract key metrics
        total_duration = benchmark_data.get("total_duration_seconds", 0)
        
        # Hardware info
        hw_info = benchmark_data.get("hardware", {})
        cpu_name = hw_info.get("processor", "N/A")
        cpu_cores = hw_info.get("cpu_count", "N/A")
        ram_gb = hw_info.get("ram_gb", "N/A")
        
        # GPU info
        gpu_info = hw_info.get("gpu", {})
        if gpu_info.get("available"):
            gpu_name = gpu_info.get("name", "Unknown GPU")
            gpu_count = gpu_info.get("count", 1)
            cuda_version = gpu_info.get("cuda_version", "N/A")
            gpu_display = f"{gpu_name} (CUDA {cuda_version})"
        else:
            gpu_display = "CPU only"
        
        # Config info
        config = benchmark_data.get("config", {})
        model_name = config.get("model", "N/A")
        device = config.get("device", "N/A")
        processing_mode = config.get("processing_mode", "Unknown")
        segmentation_mode = config.get("segmentation_mode", "N/A")
        
        # Metrics
        metrics = benchmark_data.get("metrics", {})
        speedup = metrics.get("speedup", 0)
        rtf = metrics.get("rtf", 0)
        
        # Format processing time
        processing_time_str = f"{total_duration:.1f}s"
        if total_duration >= 60:
            minutes = int(total_duration // 60)
            seconds = total_duration % 60
            processing_time_str = f"{minutes}m {seconds:.0f}s"
        
        # Processing mode badge color
        mode_color = "rgba(76, 175, 80, 0.2)" if processing_mode == "Segmented" else "rgba(33, 150, 243, 0.2)"
        
        return f'''
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.3);">
                <div style="font-size: 0.95em; margin-bottom: 8px;"><strong>⚡ Processing Information</strong></div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; font-size: 0.85em;">
                    <div style="background-color: rgba(255,255,255,0.1); padding: 8px 12px; border-radius: 4px;">
                        <strong>⏱️ Processing Time:</strong> {processing_time_str}
                    </div>
                    <div style="background-color: rgba(255,255,255,0.1); padding: 8px 12px; border-radius: 4px;">
                        <strong>⚡ Speedup:</strong> {speedup:.2f}x realtime (RTF: {rtf:.3f})
                    </div>
                    <div style="background-color: {mode_color}; padding: 8px 12px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.2);">
                        <strong>🎯 Mode:</strong> {processing_mode}
                    </div>
                    <div style="background-color: rgba(255,255,255,0.1); padding: 8px 12px; border-radius: 4px;">
                        <strong>🤖 Model:</strong> {model_name}
                    </div>
                    <div style="background-color: rgba(255,255,255,0.1); padding: 8px 12px; border-radius: 4px;">
                        <strong>🎮 GPU:</strong> {gpu_display}
                    </div>
                    <div style="background-color: rgba(255,255,255,0.1); padding: 8px 12px; border-radius: 4px;">
                        <strong>💻 CPU:</strong> {cpu_cores} cores, {ram_gb}GB RAM
                    </div>
                </div>
            </div>
        '''

    def _generate_body_content(self, first_result_data: dict, report_title_base: str) -> str:
        """Generates the initial HTML structure for the report body."""
        # Use os.path.basename for cleaner file display name
        file_path = first_result_data.get("audio_file_path", "N/A")
        display_file_name = os.path.basename(file_path) if file_path != "N/A" else "N/A"
        
        # Initial values from first_result_data, JavaScript will update these
        processing_timestamp = first_result_data.get("processing_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        duration_seconds = 0
        if first_result_data.get("transcription", {}).get("segments"):
            segments = first_result_data.get("transcription", {}).get("segments", [])
            if segments:
                duration_seconds = segments[-1].get("end", 0)
        duration_minutes_text = f"{(duration_seconds / 60):.1f} Minuten"
        
        language = first_result_data.get("transcription_config", {}).get("language") or first_result_data.get("transcription", {}).get("language", "N/A")
        screenshots_count = len(first_result_data.get("screenshots", []))
        pdfs_count = len(first_result_data.get("related_pdfs", []))
        
        # Extract benchmark data if available
        benchmark_data = first_result_data.get("benchmark", {})
        benchmark_html = self._generate_benchmark_info_html(benchmark_data) if benchmark_data else ""

        return f'''
    <div class="container" id="active_file_container">
        <div class="header">
            <h1>📚 {report_title_base} - Analysis Report</h1>
            <div class="header-info">
                <div id="active_info_timestamp_container"><strong>Verarbeitet:</strong> <span id="active_info_timestamp">{processing_timestamp}</span></div>
                <div id="active_info_filePath_container"><strong>Datei:</strong> <span id="active_info_filePath">{display_file_name}</span></div>
                <div id="active_info_duration_container"><strong>Dauer:</strong> <span id="active_info_duration">{duration_minutes_text}</span></div>
                <div id="active_info_language_container"><strong>Sprache:</strong> <span id="active_info_language">{language}</span></div>
                <div id="active_info_screenshotsCount_container"><strong>Screenshots:</strong> <span id="active_info_screenshotsCount">{screenshots_count}</span></div>
                <div id="active_info_pdfsCount_container"><strong>PDFs:</strong> <span id="active_info_pdfsCount">{pdfs_count}</span></div>
            </div>
            {benchmark_html}
        </div>

        <div class="file-selector-container">
            <label for="fileSelector">Analysierte Datei auswählen:</label>
            <select id="fileSelector" onchange="handleFileSelectionChange(this.value)">
                <!-- Options will be populated by JavaScript -->
            </select>
        </div>

        <div class="tabs">
            <div class="tab active" onclick="showTab('active_transcript', 'active_file_container')" data-tab="active_transcript">📝 Transkript</div>
            <div class="tab" onclick="showTab('active_statistics', 'active_file_container')" data-tab="active_statistics">📊 Statistiken & Parameter</div>
            <div class="tab" onclick="showTab('active_screenshots', 'active_file_container')" data-tab="active_screenshots">🖼️ Screenshots</div>
            <div class="tab" onclick="showTab('active_pdfs', 'active_file_container')" data-tab="active_pdfs">📄 PDFs</div>
            <div class="tab" onclick="showTab('active_mapping', 'active_file_container')" data-tab="active_mapping">🔗 Mapping</div>
        </div>

        <div id="active_transcript" class="tab-content active">
            {self._initial_transcript_html(first_result_data.get("transcription", {}))}
        </div>
        <div id="active_statistics" class="tab-content">
            {self._initial_statistics_html(first_result_data)}
        </div>
        <div id="active_screenshots" class="tab-content">
            {self._initial_screenshots_html(first_result_data.get("screenshots", []))}
        </div>
        <div id="active_pdfs" class="tab-content">
            {self._initial_pdfs_html(first_result_data.get("related_pdfs", []))}
        </div>
        <div id="active_mapping" class="tab-content">
            {self._initial_mapping_html(first_result_data.get("screenshot_transcript_mapping", []))}
        </div>
    </div>
'''

    def _initial_transcript_html(self, transcription_data: dict) -> str:
        # This is a helper for initial rendering. JS will use its own generator.
        # Handle nested transcription structure
        actual_transcription_data = transcription_data
        if transcription_data and "transcription" in transcription_data:
            actual_transcription_data = transcription_data["transcription"]
            
        if not actual_transcription_data or not actual_transcription_data.get("segments"):
            return '<div class="empty-state"><p>Kein Transkript verfügbar.</p></div>'
        
        segments_html = ""
        for segment in actual_transcription_data.get("segments", []):
            start_formatted = format_timestamp_seconds(segment.get("start", 0))
            end_formatted = format_timestamp_seconds(segment.get("end", 0))
            text = (segment.get("text") or "").strip()
            confidence = segment.get("confidence", 0)
            # Simplified confidence class for initial render, JS can be more detailed
            confidence_class = "high-confidence" if confidence > 0.8 else ("medium-confidence" if confidence > 0.6 else "low-confidence")
            if not text: continue
            segments_html += f'''
                <div class="transcript-segment">
                    <div class="segment-header">
                        <span class="timestamp">[{start_formatted} - {end_formatted}]</span>
                        <span class="confidence {confidence_class}">Vertrauen: {confidence:.2f}</span>
                    </div>
                    <p class="segment-text">{text}</p>
                </div>
            '''
        full_text = actual_transcription_data.get("text", "")
        word_count = len(full_text.split()) if full_text else 0
        return f'''
            <div class="section">
                <h2>📝 Transkript</h2>
                <div class="section-stats">
                    <span>Segmente: {len(actual_transcription_data.get("segments", []))}</span>
                    <span>Wörter: {word_count}</span>
                    <span>Zeichen: {len(full_text)}</span>
                </div>
                <div class="transcript-container">{segments_html}</div>
            </div>
        '''

    def _initial_statistics_html(self, file_data: dict) -> str:
        # Placeholder for initial statistics. JS will generate the full content.
        return '<div class="section"><h2>📊 Statistiken & Parameter</h2><div class="empty-state"><p>Statistiken werden geladen...</p></div></div>'

    def _initial_screenshots_html(self, screenshots: list) -> str:
        return '<div class="section"><h2>🖼️ Screenshots</h2><div class="empty-state"><p>Screenshots werden geladen...</p></div></div>'
        
    def _initial_pdfs_html(self, pdfs: list) -> str:
        return '<div class="section"><h2>📄 PDFs</h2><div class="empty-state"><p>PDFs werden geladen...</p></div></div>'

    def _initial_mapping_html(self, mapping_data: list) -> str:
        return '<div class="section"><h2>🔗 Mapping</h2><div class="empty-state"><p>Mapping-Daten werden geladen...</p></div></div>'

    def _get_embedded_javascript(self, all_analysis_results: List[Dict]) -> str:
        """Generate JavaScript for tab navigation, search, and dynamic content updates."""
        # The all_analysis_results are embedded in the HTML separately and parsed by this JS.
        # This method now primarily returns the static JS code.
        # The actual data (all_analysis_results) is picked up by the JS from the <script id="allData"> tag.
        return r'''
        // JavaScript code for HTML report interactivity

        // --- Global Timeline Variables and Functions ---
        // Helper functions (moved to global scope for HTML event handlers)
        function formatTimestamp(seconds) {
            if (isNaN(seconds) || seconds === null) return "00:00:00";
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return [h, m, s].map(v => v.toString().padStart(2, '0')).join(':');
        }
        
        function getConfidenceClass(confidence) {
            if (confidence > 0.8) return "high-confidence";
            if (confidence > 0.6) return "medium-confidence";
            return "low-confidence";
        }

        // Timeline functionality
        let currentTimelineData = {
            segments: [],
            screenshots: [],
            currentTime: 0,
            currentSegmentIndex: -1,
            totalDuration: 0,
            isPlaying: false,
            playInterval: null
        };
        
        function initializeTimeline(segments, screenshots, totalDuration) {
            currentTimelineData = {
                segments: segments,
                screenshots: screenshots,
                currentTime: 0,
                currentSegmentIndex: -1,
                totalDuration: totalDuration,
                isPlaying: false,
                playInterval: null
            };
            
            // Set initial state
            timelineSeek(0);
        }
        
        function timelineSeek(time) {
            const timeValue = parseFloat(time);
            currentTimelineData.currentTime = timeValue;
            
            // Update slider
            const slider = document.getElementById('timelineSlider');
            if (slider) slider.value = timeValue;
            
            // Update time display
            updateTimeDisplay(timeValue);
            
            // Find current segment
            const segmentIndex = findSegmentAtTime(timeValue);
            updateCurrentSegment(segmentIndex);
            
            // Find and display screenshot
            updateScreenshotForTime(timeValue);
        }
        
        function findSegmentAtTime(time) {
            for (let i = 0; i < currentTimelineData.segments.length; i++) {
                const segment = currentTimelineData.segments[i];
                if (time >= segment.start && time <= segment.end) {
                    return i;
                }
            }
            return -1;
        }
        
        function updateCurrentSegment(segmentIndex) {
            currentTimelineData.currentSegmentIndex = segmentIndex;
            
            // Update segment highlighting
            const segmentItems = document.querySelectorAll('.segment-item');
            segmentItems.forEach((item, index) => {
                if (index === segmentIndex) {
                    item.classList.add('active');
                    // Scroll into view
                    item.scrollIntoView({ behavior: 'smooth', block: 'center' });
                } else {
                    item.classList.remove('active');
                }
            });
            
            // Update current segment stat
            const statElement = document.getElementById('currentSegmentStat');
            if (statElement) {
                if (segmentIndex >= 0) {
                    statElement.innerHTML = `<span>🎯</span><span>Segment: ${segmentIndex + 1}/${currentTimelineData.segments.length}</span>`;
                } else {
                    statElement.innerHTML = `<span>🎯</span><span>Segment: -</span>`;
                }
            }
        }
        
        function updateScreenshotForTime(time) {
            // Find the best screenshot for this time
            let bestScreenshot = null;
            let minTimeDiff = Infinity;
            
            for (const screenshot of currentTimelineData.screenshots) {
                const timeDiff = Math.abs(screenshot.timestamp - time);
                if (timeDiff < minTimeDiff) {
                    minTimeDiff = timeDiff;
                    bestScreenshot = screenshot;
                }
            }
            
            const screenshotContent = document.getElementById('screenshotContent');
            const screenshotTitle = document.getElementById('screenshotTitle');
            const screenshotInfo = document.getElementById('screenshotInfo');
            
            if (bestScreenshot && minTimeDiff <= 30) { // Show screenshot if within 30 seconds
                // Fix screenshot path - support both screenshots and name_screenshots folders
                let screenshotPath = bestScreenshot.filepath || bestScreenshot.path || '';
                if (screenshotPath) {
                    // Convert backslashes to forward slashes
                    screenshotPath = screenshotPath.replace(/\\/g, '/');
                    
                    // Extract folder ending with screenshots + filename
                    // This matches patterns like:
                    // - results/mad/VideoName/screenshots/file.jpg -> VideoName_screenshots/file.jpg
                    // - VideoName_screenshots/file.jpg -> VideoName_screenshots/file.jpg
                    const screenshotsMatch = screenshotPath.match(/([^\/]+)\/screenshots\/([^\/]+)$/i);
                    if (screenshotsMatch) {
                        // Use VideoName_screenshots/filename format
                        const videoName = screenshotsMatch[1];
                        const filename = screenshotsMatch[2];
                        screenshotPath = videoName + '_screenshots/' + filename;
                    } else {
                        // Already in correct format or try to extract last folder + filename
                        const pathParts = screenshotPath.split('/');
                        const filename = pathParts[pathParts.length - 1];
                        const folderName = pathParts[pathParts.length - 2] || '';
                        if (folderName) {
                            screenshotPath = folderName + '/' + filename;
                        } else {
                            screenshotPath = filename;
                        }
                    }
                }
                
                screenshotContent.innerHTML = `
                    <img src="${screenshotPath}" alt="Screenshot at ${formatTimestamp(bestScreenshot.timestamp)}" 
                         class="screenshot-image" 
                         onerror="this.style.display='none'; this.parentElement.innerHTML='&lt;div class=&quot;screenshot-placeholder&quot;&gt;🚫 Screenshot konnte nicht geladen werden&lt;/div&gt;';">
                `;
                
                screenshotTitle.textContent = `Screenshot bei ${formatTimestamp(bestScreenshot.timestamp)}`;
                screenshotInfo.textContent = `Zeitdifferenz: ${minTimeDiff.toFixed(1)}s | Datei: ${bestScreenshot.filename || 'Unbekannt'}`;
            } else {
                screenshotContent.innerHTML = '<div class="screenshot-placeholder">🖼️ Kein Screenshot für diese Zeit verfügbar</div>';
                screenshotTitle.textContent = 'Kein Screenshot verfügbar';
                screenshotInfo.textContent = `Aktuelle Zeit: ${formatTimestamp(time)}`;
            }
        }
        
        function updateTimeDisplay(time) {
            const timeDisplay = document.getElementById('timeDisplay');
            if (timeDisplay) {
                // Use formatTimestamp for both current time and total duration
                timeDisplay.textContent = `${formatTimestamp(time)} / ${formatTimestamp(currentTimelineData.totalDuration)}`;
            }
        }
        
        function timelineJumpToSegment(segmentIndex) {
            if (segmentIndex >= 0 && segmentIndex < currentTimelineData.segments.length) {
                const segment = currentTimelineData.segments[segmentIndex];
                timelineSeek(segment.start);
            }
        }
        
        function timelineGoToPrevious() {
            if (currentTimelineData.currentSegmentIndex > 0) {
                timelineJumpToSegment(currentTimelineData.currentSegmentIndex - 1);
            } else if (currentTimelineData.segments.length > 0) {
                timelineJumpToSegment(0);
            }
        }
        
        function timelineGoToNext() {
            if (currentTimelineData.currentSegmentIndex < currentTimelineData.segments.length - 1) {
                timelineJumpToSegment(currentTimelineData.currentSegmentIndex + 1);
            } else if (currentTimelineData.segments.length > 0) {
                timelineJumpToSegment(currentTimelineData.segments.length - 1);
            }
        }
        
        function timelineReset() {
            timelineSeek(0);
            if (currentTimelineData.isPlaying) {
                timelineTogglePlay();
            }
        }
        
        function timelineTogglePlay() {
            const playBtn = document.getElementById('playPauseBtn');
            
            if (currentTimelineData.isPlaying) {
                // Pause
                clearInterval(currentTimelineData.playInterval);
                currentTimelineData.isPlaying = false;
                if (playBtn) playBtn.textContent = '▶️ Play';
            } else {
                // Play
                currentTimelineData.isPlaying = true;
                if (playBtn) playBtn.textContent = '⏸️ Pause';
                
                currentTimelineData.playInterval = setInterval(() => {
                    const newTime = currentTimelineData.currentTime + 1; // Advance 1 second
                    if (newTime >= currentTimelineData.totalDuration) {
                        timelineTogglePlay(); // Auto-pause at end
                        return;
                    }
                    timelineSeek(newTime);
                }, 1000); // Update every second
            }
        }

        // Wait for the DOM to be fully loaded before executing scripts
        document.addEventListener('DOMContentLoaded', function() {
            const allData = JSON.parse(document.getElementById('allData').textContent);
            const fileSelector = document.getElementById('fileSelector');
            const activeFileContainer = document.getElementById('active_file_container');

            // --- Content Update Functions ---
            function updateHeader(fileData) {
                if (!fileData) return;
                const header = activeFileContainer.querySelector('.header');
                const headerInfo = activeFileContainer.querySelector('.header-info');

                let reportTitleBase = "Transcription Analysis Report";
                let displayFileName = "N/A";

                // Fix: Use audio_path instead of audio_file_path
                const audioPath = fileData.audio_path || fileData.audio_file_path;
                if (audioPath) {
                    reportTitleBase = audioPath.split(/[\\\\/]/).pop().split('.').slice(0, -1).join('.') || audioPath.split(/[\\\\/]/).pop();
                    displayFileName = audioPath.split(/[\\\\/]/).pop();
                } else if (fileData.error) {
                    const errorFileName = (audioPath || "Unknown File").split(/[\\\\/]/).pop();
                    reportTitleBase = `Error processing: ${errorFileName}`;
                    displayFileName = errorFileName;
                }
                
                header.querySelector('h1').textContent = `📚 ${reportTitleBase} - Analysis Report`;
                
                // Fix: Access nested transcription structure
                const transcriptionData = fileData.transcription && fileData.transcription.transcription ? fileData.transcription.transcription : fileData.transcription;
                const durationSeconds = transcriptionData && transcriptionData.segments && transcriptionData.segments.length > 0 ?
                                        transcriptionData.segments[transcriptionData.segments.length -1].end : 0;

                headerInfo.querySelector('#active_info_timestamp').textContent = fileData.processing_timestamp || new Date().toISOString();
                headerInfo.querySelector('#active_info_filePath').textContent = displayFileName;
                headerInfo.querySelector('#active_info_duration').textContent = `${(durationSeconds / 60).toFixed(1)} Minuten`;
                headerInfo.querySelector('#active_info_language').textContent = (fileData.transcription_config && fileData.transcription_config.language) || (transcriptionData && transcriptionData.language) || "N/A";
                headerInfo.querySelector('#active_info_screenshotsCount').textContent = (fileData.screenshots || []).length;
                headerInfo.querySelector('#active_info_pdfsCount').textContent = (fileData.related_pdfs || []).length;
            }

            function updateTranscriptTab(transcriptionData) {
                const tabContent = document.getElementById('active_transcript');
                
                // Fix: Handle nested transcription structure
                const actualTranscriptionData = transcriptionData && transcriptionData.transcription ? transcriptionData.transcription : transcriptionData;
                
                if (!actualTranscriptionData || !actualTranscriptionData.segments || actualTranscriptionData.segments.length === 0) {
                    tabContent.innerHTML = '<div class="section"><h2>📝 Transkript</h2><div class="empty-state"><p>Kein Transkript verfügbar.</p></div></div>';
                    return;
                }

                let segmentsHtml = "";
                for (const segment of actualTranscriptionData.segments) {
                    const startFormatted = formatTimestamp(segment.start);
                    const endFormatted = formatTimestamp(segment.end);
                    const text = (segment.text || "").trim();
                    const confidence = segment.confidence || 0;
                    const confidenceClass = getConfidenceClass(confidence);
                    if (!text) continue;
                    segmentsHtml += `
                        <div class="transcript-segment">
                            <div class="segment-header">
                                <span class="timestamp">[${startFormatted} - ${endFormatted}]</span>
                                <span class="confidence ${confidenceClass}">Vertrauen: ${confidence.toFixed(2)}</span>
                            </div>
                            <p class="segment-text">${text}</p>
                        </div>
                    `;
                }
                const fullText = actualTranscriptionData.text || "";
                const wordCount = fullText ? fullText.split(/\s+/).filter(Boolean).length : 0;
                
                tabContent.innerHTML = `
                    <div class="section">
                        <h2>📝 Transkript</h2>
                        <div class="section-stats">
                            <span>Segmente: ${actualTranscriptionData.segments.length}</span>
                            <span>Wörter: ${wordCount}</span>
                            <span>Zeichen: ${fullText.length}</span>
                        </div>
                        <div class="transcript-container">${segmentsHtml}</div>
                    </div>
                `;
            }

            function updateStatisticsTab(fileData) {
                const tabContent = document.getElementById('active_statistics');
                if (!fileData) {
                    tabContent.innerHTML = '<div class="section"><h2>📊 Statistiken & Parameter</h2><div class="empty-state"><p>Keine Daten für Statistiken verfügbar.</p></div></div>';
                    return;
                }

                let html = '<div class="section"><h2>📊 Statistiken & Parameter</h2>';
                // Transcription Config
                if (fileData.transcription_config) {
                    html += '<div class="stats-card"><h3>Transkriptionsparameter</h3>';
                    for (const [key, value] of Object.entries(fileData.transcription_config)) {
                        html += `<p><strong>${key}:</strong> ${typeof value === 'object' ? JSON.stringify(value) : value}</p>`;
                    }
                    html += '</div>';
                }
                // Speech Pattern Analysis (if available)
                if (fileData.transcription && fileData.transcription.speech_pattern_analysis) {
                    html += '<div class="stats-card"><h3>Sprachmusteranalyse</h3>';
                    const spa = fileData.transcription.speech_pattern_analysis;
                    for (const [key, value] of Object.entries(spa)) {
                         html += `<p><strong>${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</strong> ${typeof value === 'number' ? value.toFixed(2) : value}</p>`;
                    }
                    html += '</div>';
                }
                 // Screenshot Config
                if (fileData.screenshot_config) {
                    html += '<div class="stats-card"><h3>Screenshot-Parameter</h3>';
                    for (const [key, value] of Object.entries(fileData.screenshot_config)) {
                        html += `<p><strong>${key}:</strong> ${value}</p>`;
                    }
                    html += '</div>';
                }
                html += '</div>';
                tabContent.innerHTML = html;
            }

            function updateScreenshotsTab(screenshots) {
                const tabContent = document.getElementById('active_screenshots');
                if (!screenshots || screenshots.length === 0) {
                    tabContent.innerHTML = '<div class="section"><h2>🖼️ Screenshots</h2><div class="empty-state"><p>Keine Screenshots verfügbar.</p></div></div>';
                    return;
                }
                let imagesHtml = screenshots.map(ss => {
                    // Fix: Use correct property (filepath instead of path) and handle relative paths
                    let imagePath = ss.filepath || ss.path;
                    if (imagePath) {
                        // Convert backslashes to forward slashes
                        imagePath = imagePath.replace(/\\/g, '/');
                        
                        // Extract folder ending with screenshots + filename
                        // This matches patterns like:
                        // - results/mad/VideoName/screenshots/file.jpg -> VideoName_screenshots/file.jpg
                        // - VideoName_screenshots/file.jpg -> VideoName_screenshots/file.jpg
                        const screenshotsMatch = imagePath.match(/([^\/]+)\/screenshots\/([^\/]+)$/i);
                        if (screenshotsMatch) {
                            // Use VideoName_screenshots/filename format
                            const videoName = screenshotsMatch[1];
                            const filename = screenshotsMatch[2];
                            imagePath = videoName + '_screenshots/' + filename;
                        } else {
                            // Already in correct format or try to extract last folder + filename
                            const pathParts = imagePath.split('/');
                            const filename = pathParts[pathParts.length - 1];
                            const folderName = pathParts[pathParts.length - 2] || '';
                            if (folderName) {
                                imagePath = folderName + '/' + filename;
                            } else {
                                imagePath = filename;
                            }
                        }
                    }
                    return `
                        <div class="stats-card">
                            <img src="${imagePath}" alt="Screenshot at ${formatTimestamp(ss.timestamp)}" style="max-width: 100%; height: auto; border-radius: 4px;" onerror="this.style.display='none'; this.nextElementSibling.innerHTML='❌ Image not found: ${imagePath}';">
                            <p style="text-align: center; margin-top: 5px;">Timestamp: ${formatTimestamp(ss.timestamp)}</p>
                        </div>
                    `;
                }).join('');
                tabContent.innerHTML = `<div class="section"><h2>🖼️ Screenshots</h2><div style="display: flex; flex-wrap: wrap; gap: 15px;">${imagesHtml}</div></div>`;
            }

            function updatePDFsTab(pdfs) {
                const tabContent = document.getElementById('active_pdfs');
                if (!pdfs || pdfs.length === 0) {
                    tabContent.innerHTML = '<div class="section"><h2>📄 PDFs</h2><div class="empty-state"><p>Keine verknüpften PDFs gefunden.</p></div></div>';
                    return;
                }
                let pdfsHtml = '<table><thead><tr><th>Dateiname</th><th>Pfad</th><th>Relevanz (Score)</th></tr></thead><tbody>';
                pdfs.forEach(pdf => {
                    // Fix: Use correct property names (filename/filepath instead of file_name/file_path)
                    const fileName = pdf.filename || pdf.file_name || 'N/A';
                    const filePath = pdf.filepath || pdf.file_path || '#';
                    const relevanceScore = pdf.relevance_score ? pdf.relevance_score.toFixed(2) : 'N/A';
                    pdfsHtml += `<tr><td>${fileName}</td><td><a href="${filePath}" target="_blank">${filePath}</a></td><td>${relevanceScore}</td></tr>`;
                });
                pdfsHtml += '</tbody></table>';
                tabContent.innerHTML = `<div class="section"><h2>📄 PDFs</h2>${pdfsHtml}</div>`;
            }
            
            function updateMappingTab(mappingData, segments, screenshots) {
                const tabContent = document.getElementById('active_mapping');
                
                // Prepare data for timeline
                const timelineSegments = segments || [];
                const timelineScreenshots = screenshots || [];
                
                if (!timelineSegments.length && !timelineScreenshots.length) {
                    tabContent.innerHTML = '<div class="section"><h2>🔗 Timeline Navigation</h2><div class="empty-state"><p>Keine Daten für Timeline verfügbar.</p></div></div>';
                    return;
                }
                
                // Calculate total duration
                const totalDuration = timelineSegments.length > 0 ? timelineSegments[timelineSegments.length - 1].end : 0;
                const totalMinutes = Math.floor(totalDuration / 60);
                const totalSeconds = Math.floor(totalDuration % 60);
                
                // Generate timeline HTML
                tabContent.innerHTML = `
                    <div class="section">
                        <h2>🔗 Timeline Navigation</h2>
                        <div class="timeline-container">
                            <div class="timeline-header">
                                <div class="timeline-info">
                                    <strong>Video-ähnliche Navigation durch ${timelineSegments.length} Transkript-Segmente</strong>
                                </div>
                                <div class="timeline-controls">
                                    <button class="timeline-button" onclick="timelineGoToPrevious()">⏮️ Vorheriges</button>
                                    <button class="timeline-button" onclick="timelineTogglePlay()" id="playPauseBtn">▶️ Play</button>
                                    <button class="timeline-button" onclick="timelineGoToNext()">⏭️ Nächstes</button>
                                    <button class="timeline-button" onclick="timelineReset()">🔄 Reset</button>
                                </div>
                            </div>
                            
                            <div class="timeline-slider-container">
                                <input type="range" id="timelineSlider" class="timeline-slider" 
                                       min="0" max="${totalDuration}" value="0" step="0.1"
                                       oninput="timelineSeek(this.value)">
                            </div>
                            
                            <div class="timeline-time-display" id="timeDisplay">
                                00:00:00 / ${String(totalMinutes).padStart(2, '0')}:${String(totalSeconds).padStart(2, '0')}:00
                            </div>
                            
                            <div class="timeline-main">
                                <div class="timeline-sidebar">
                                    <h4 style="margin: 0 0 10px 0;">📝 Segmente</h4>
                                    <div class="segments-list" id="segmentsList">
                                        ${timelineSegments.map((segment, index) => `
                                            <div class="segment-item" data-index="${index}" data-start="${segment.start}" data-end="${segment.end}"
                                                 onclick="timelineJumpToSegment(${index})">
                                                <div class="segment-time">${formatTimestamp(segment.start)} - ${formatTimestamp(segment.end)}</div>
                                                <div class="segment-text">${(segment.text || '').substring(0, 100)}${segment.text && segment.text.length > 100 ? '...' : ''}</div>
                                            </div>
                                        `).join('')}
                                    </div>
                                </div>
                                
                                <div class="timeline-content">
                                    <div class="screenshot-viewer">
                                        <div class="screenshot-header">
                                            <strong id="screenshotTitle">Screenshot wird geladen...</strong>
                                            <div id="screenshotInfo" style="font-size: 12px; color: #666; margin-top: 4px;"></div>
                                        </div>
                                        <div class="screenshot-content" id="screenshotContent">
                                            <div class="screenshot-placeholder">
                                                🖼️ Bewegen Sie den Slider oder klicken Sie auf ein Segment, um Screenshots anzuzeigen
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="timeline-stats">
                                <div class="timeline-stat">
                                    <span>📊</span>
                                    <span>${timelineSegments.length} Segmente</span>
                                </div>
                                <div class="timeline-stat">
                                    <span>🖼️</span>
                                    <span>${timelineScreenshots.length} Screenshots</span>
                                </div>
                                <div class="timeline-stat">
                                    <span>⏱️</span>
                                    <span>${Math.floor(totalDuration / 60)} min ${Math.floor(totalDuration % 60)} sec</span>
                                </div>
                                <div class="timeline-stat" id="currentSegmentStat">
                                    <span>🎯</span>
                                    <span>Segment: -</span>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                // Initialize timeline functionality
                initializeTimeline(timelineSegments, timelineScreenshots, totalDuration);
            }


            // --- Event Handlers ---
            window.showTab = function(tabId, containerId) { // Make it global for inline HTML onclick
                const container = document.getElementById(containerId);
                if (!container) return;

                const tabs = container.querySelectorAll('.tab');
                tabs.forEach(t => t.classList.remove('active'));
                
                const contents = container.querySelectorAll('.tab-content');
                contents.forEach(c => c.classList.remove('active'));
                
                const selectedTabButton = container.querySelector(`.tab[data-tab='${tabId}']`);
                if (selectedTabButton) selectedTabButton.classList.add('active');
                
                const selectedContent = container.querySelector(`#${tabId}`);
                if (selectedContent) selectedContent.classList.add('active');
            }

            window.handleFileSelectionChange = function(selectedIndex) { // Make it global
                if (!allData || !allData[selectedIndex]) {
                    console.error("Selected data not found for index:", selectedIndex);
                    // Optionally clear content or show an error message in the UI
                    return;
                }
                const selectedFileData = allData[selectedIndex];
                
                updateHeader(selectedFileData);
                updateTranscriptTab(selectedFileData.transcription);
                updateStatisticsTab(selectedFileData);
                updateScreenshotsTab(selectedFileData.screenshots);
                updatePDFsTab(selectedFileData.related_pdfs);
                
                // Extract segments for timeline
                const transcriptionData = selectedFileData.transcription && selectedFileData.transcription.transcription ? 
                                        selectedFileData.transcription.transcription : selectedFileData.transcription;
                const segments = transcriptionData ? transcriptionData.segments || [] : [];
                const screenshots = selectedFileData.screenshots || [];
                
                updateMappingTab(selectedFileData.screenshot_transcript_mapping, segments, screenshots);
                
                // Ensure the first tab is active after changing file
                showTab('active_transcript', 'active_file_container');
            }
            
            // --- Initialization ---
            if (fileSelector && allData && allData.length > 0) {
                allData.forEach((fileResult, index) => {
                    const option = document.createElement('option');
                    option.value = index;
                    let displayName = `Datei ${index + 1}`;
                    if (fileResult.audio_file_path) {
                        displayName = fileResult.audio_file_path.split(/[\\\\/]/).pop() || `Unbenannte Datei ${index + 1}`;
                    } else if (fileResult.error) {
                        displayName = `Fehler bei Verarbeitung: ${(fileResult.audio_file_path || 'Unbekannte Datei').split(/[\\\\/]/).pop()}`;
                    }
                    option.textContent = displayName;
                    fileSelector.appendChild(option);
                });
                
                // Load data for the first file initially
                if (allData.length > 0) {
                    handleFileSelectionChange(0); // Load the first file's data
                }

            } else if (fileSelector) {
                const option = document.createElement('option');
                option.textContent = "Keine Dateien verarbeitet oder Datenfehler.";
                fileSelector.appendChild(option);
                fileSelector.disabled = true;
                // Clear content areas or show a general error message
                updateHeader(null); // Clear header
                document.getElementById('active_transcript').innerHTML = '<div class="section"><h2>📝 Transkript</h2><div class="empty-state"><p>Keine Daten zum Anzeigen.</p></div></div>';
                document.getElementById('active_statistics').innerHTML = '<div class="section"><h2>📊 Statistiken & Parameter</h2><div class="empty-state"><p>Keine Daten zum Anzeigen.</p></div></div>';
                document.getElementById('active_screenshots').innerHTML = '<div class="section"><h2>🖼️ Screenshots</h2><div class="empty-state"><p>Keine Daten zum Anzeigen.</p></div></div>';
                document.getElementById('active_pdfs').innerHTML = '<div class="section"><h2>📄 PDFs</h2><div class="empty-state"><p>Keine Daten zum Anzeigen.</p></div></div>';
                document.getElementById('active_mapping').innerHTML = '<div class="section"><h2>🔗 Mapping</h2><div class="empty-state"><p>Keine Daten zum Anzeigen.</p></div></div>';
            }
        });
        '''
    def _get_empty_result_structure(self) -> dict:
        """Returns a default structure for when no results are available or an error occurs."""
        return {
            "audio_file_path": "N/A",
            "processing_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # Use Python's datetime for initial server-side render
            "transcription": {"text": "No data available.", "segments": []},
            "transcription_config": {},
            "speech_pattern_analysis": {},
            "error": "No data loaded"
        }

    def generate_index_page(self, all_results: List[Dict], index_path: str) -> None:
        """
        Generate an index page with links to all individual HTML reports.
        
        Args:
            all_results: List of analysis result dictionaries
            index_path: Output path for the index.html file
        """
        if not all_results:
            logger.warning("No results provided for index page generation")
            return
            
        # Calculate some summary statistics
        total_files = len(all_results)
        successful_files = len([r for r in all_results if not r.get("error")])
        failed_files = total_files - successful_files
        
        # Get total duration and segments
        total_duration = 0
        total_segments = 0
        total_screenshots = 0
        total_pdfs = 0
        
        for result in all_results:
            if not result.get("error"):
                # Handle nested transcription structure
                transcription_data = result.get("transcription", {})
                if "transcription" in transcription_data:
                    transcription_data = transcription_data["transcription"]
                
                segments = transcription_data.get("segments", [])
                if segments:
                    total_duration += segments[-1].get("end", 0)
                    total_segments += len(segments)
                
                total_screenshots += len(result.get("screenshots", []))
                total_pdfs += len(result.get("related_pdfs", []))
        
        # Generate HTML content for index page
        html_content = f'''<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📚 Study Material Analysis - Index</title>
    <style>
        {self._get_embedded_css()}
        .index-summary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }}
        .index-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        .stat-card {{
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 2rem;
            font-weight: bold;
            display: block;
        }}
        .file-list {{
            display: grid;
            gap: 1rem;
            margin: 2rem 0;
        }}
        .file-card {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .file-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .file-card.error {{
            border-color: #ff6b6b;
            background: #fff5f5;
        }}
        .file-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}
        .file-title {{
            font-size: 1.2rem;
            font-weight: bold;
            color: #333;
            text-decoration: none;
        }}
        .file-title:hover {{
            color: #667eea;
        }}
        .file-status {{
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        .status-success {{
            background: #d4edda;
            color: #155724;
        }}
        .status-error {{
            background: #f8d7da;
            color: #721c24;
        }}
        .file-details {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            font-size: 0.9rem;
            color: #666;
        }}
        .detail-item {{
            display: flex;
            justify-content: space-between;
        }}
        .error-message {{
            color: #721c24;
            font-style: italic;
            margin-top: 0.5rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="index-summary">
            <h1>📚 Study Material Analysis Dashboard</h1>
            <p>Übersicht über alle verarbeiteten Videodateien</p>
            
            <div class="index-stats">
                <div class="stat-card">
                    <span class="stat-number">{total_files}</span>
                    <span>Dateien gesamt</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{successful_files}</span>
                    <span>Erfolgreich</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{failed_files}</span>
                    <span>Fehler</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{total_duration/60:.0f}</span>
                    <span>Minuten gesamt</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{total_segments}</span>
                    <span>Transkript-Segmente</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{total_screenshots}</span>
                    <span>Screenshots</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{total_pdfs}</span>
                    <span>Verknüpfte PDFs</span>
                </div>
            </div>
        </div>
        
        <div class="file-list">'''
        
        # Add individual file cards
        for i, result in enumerate(all_results):
            has_error = bool(result.get("error"))
            
            # Extract file information
            video_path = result.get("video_path", "")
            audio_path = result.get("audio_path", "")
            file_name = "Unbekannte Datei"
            
            if video_path:
                file_name = os.path.basename(video_path)
            elif audio_path:
                file_name = os.path.basename(audio_path)
            
            # Generate HTML report filename
            safe_name = file_name.replace(" ", "_").replace("-", "_")
            html_filename = f"{safe_name}_report.html"
            
            # Get file statistics
            duration = 0
            segments_count = 0
            screenshots_count = len(result.get("screenshots", []))
            pdfs_count = len(result.get("related_pdfs", []))
            
            if not has_error:
                transcription_data = result.get("transcription", {})
                if "transcription" in transcription_data:
                    transcription_data = transcription_data["transcription"]
                
                segments = transcription_data.get("segments", [])
                segments_count = len(segments)
                if segments:
                    duration = segments[-1].get("end", 0)
            
            # Create file card
            card_class = "file-card error" if has_error else "file-card"
            status_class = "status-error" if has_error else "status-success";
            status_text = "Fehler" if has_error else "✓ Erfolgreich";
            
            html_content += f'''
            <div class="{card_class}">
                <div class="file-header">
                    <a href="{html_filename}" class="file-title">{file_name}</a>
                    <span class="file-status {status_class}">{status_text}</span>
                </div>
                
                {'<div class="error-message">Fehler bei der Verarbeitung: ' + str(result.get("error", "Unbekannter Fehler")) + '</div>' if has_error else ''}
                
                <div class="file-details">
                    <div class="detail-item">
                        <span>🕒 Dauer:</span>
                        <span>{duration/60:.1f} min</span>
                    </div>
                    <div class="detail-item">
                        <span>📝 Segmente:</span>
                        <span>{segments_count}</span>
                    </div>
                    <div class="detail-item">
                        <span>🖼️ Screenshots:</span>
                        <span>{screenshots_count}</span>
                    </div>
                    <div class="detail-item">
                        <span>📄 PDFs:</span>
                        <span>{pdfs_count}</span>
                    </div>
                </div>
            </div>'''
        
        html_content += '''
        </div>
        
        <footer style="text-align: center; margin-top: 3rem; padding: 2rem; color: #666; border-top: 1px solid #e0e0e0;">
            <p>📚 Study Material Processor - Generiert am ''' + datetime.now().strftime("%d.%m.%Y um %H:%M:%S") + '''</p>
        </footer>
    </div>
</body>
</html>'''

        # Write the index page
        try:
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Index page generated successfully: {index_path}")
        except Exception as e:
            logger.error(f"Failed to write index page to {index_path}: {e}")