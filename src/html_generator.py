"""
HTML report generation module for creating interactive study material reports
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from .utils import format_timestamp_seconds

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
            results: Complete analysis results dictionary
            output_path: Path to save the HTML report
        """
        logger.info(f"Generating HTML report: {output_path}")
        
        html_content = self._generate_html_document(results)
        
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
        /* Add more styles as needed */
        '''

    def _generate_body_content(self, first_result_data: dict, report_title_base: str) -> str:
        """Generates the initial HTML structure for the report body."""
        # Use os.path.basename for cleaner file display name
        file_path = first_result_data.get("audio_file_path", "N/A")
        display_file_name = os.path.basename(file_path) if file_path != "N/A" else "N/A"
        
        # Initial values from first_result_data, JavaScript will update these
        processing_timestamp = first_result_data.get("processing_timestamp", new_datetime_string())
        duration_seconds = 0
        if first_result_data.get("transcription", {}).get("segments"):
            segments = first_result_data.get("transcription", {}).get("segments", [])
            if segments:
                duration_seconds = segments[-1].get("end", 0)
        duration_minutes_text = f"{(duration_seconds / 60):.1f} Minuten"
        
        language = first_result_data.get("transcription_config", {}).get("language") or first_result_data.get("transcription", {}).get("language", "N/A")
        screenshots_count = len(first_result_data.get("screenshots", []))
        pdfs_count = len(first_result_data.get("related_pdfs", []))

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
        if not transcription_data or not transcription_data.get("segments"):
            return '<div class="empty-state"><p>Kein Transkript verfügbar.</p></div>'
        
        segments_html = ""
        for segment in transcription_data.get("segments", []):
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
        full_text = transcription_data.get("text", "")
        word_count = len(full_text.split()) if full_text else 0
        return f'''
            <div class="section">
                <h2>📝 Transkript</h2>
                <div class="section-stats">
                    <span>Segmente: {len(transcription_data.get("segments", []))}</span>
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
        return r"""
        // JavaScript code for HTML report interactivity
        
        function showTab(tabId, containerId) {
            var container = document.getElementById(containerId);
            if (!container) {
                console.error("Container not found for tabs:", containerId);
                return;
            }
            var tabContents = container.querySelectorAll('.tab-content');
            tabContents.forEach(function(content) {
                content.classList.remove('active');
                content.style.display = 'none';
            });

            var tabs = container.querySelectorAll('.tabs .tab');
            tabs.forEach(function(tab) {
                tab.classList.remove('active');
            });

            var selectedTabContent = document.getElementById(tabId);
            if (selectedTabContent) {
                selectedTabContent.classList.add('active');
                selectedTabContent.style.display = 'block';
            }
            
            var selectedTabButton = container.querySelector('.tabs .tab[data-tab="' + tabId + '"]');
            if (selectedTabButton) {
                selectedTabButton.classList.add('active');
            }
            console.log("Showing tab: " + tabId + " in container: " + containerId);
        }

        let allAnalysisData = []; // Global store for all file data

        document.addEventListener('DOMContentLoaded', function() {
            const allDataScriptTag = document.getElementById('allData');
            if (allDataScriptTag) {
                try {
                    allAnalysisData = JSON.parse(allDataScriptTag.textContent);
                } catch (e) {
                    console.error('Error parsing embedded JSON data:', e);
                    return;
                }
            } else {
                console.error('Could not find embedded JSON data script tag.');
                return;
            }

            if (!allAnalysisData || allAnalysisData.length === 0) {
                console.warn('No analysis data found or data is empty.');
                const bodyContainer = document.querySelector('.container');
                if (bodyContainer) {
                    bodyContainer.innerHTML = '<div class="empty-state"><h1>Keine Analysedaten verfügbar</h1><p>Bitte stellen Sie sicher, dass die JSON-Daten korrekt eingebettet sind.</p></div>';
                }
                return;
            }

            populateFileSelector();
            if (allAnalysisData.length > 0) {
                updatePageContent(0); 
            }
            showTab('active_transcript', 'active_file_container'); 
        });

        function populateFileSelector() {
            const selector = document.getElementById('fileSelector');
            if (!selector) {
                console.error('File selector element not found.');
                return;
            }
            selector.innerHTML = ''; 

            allAnalysisData.forEach((fileData, index) => {
                let displayName = 'Unknown File';
                if (fileData.audio_file_path) {
                    const pathParts = fileData.audio_file_path.replace(/\\/g, '/').split('/');
                    displayName = pathParts.pop() || `Datei ${index + 1}`;
                } else if (fileData.error) {
                    displayName = `Fehler bei Datei ${index + 1}`;
                } else {
                    displayName = `Analyse ${index + 1}`;
                }
                
                const option = document.createElement('option');
                option.value = index; 
                option.textContent = displayName;
                selector.appendChild(option);
            });
        }

        function handleFileSelectionChange(selectedIndex) {
            console.log('File selection changed to index:', selectedIndex);
            updatePageContent(parseInt(selectedIndex, 10));
        }
        
        function getSafe(fn, defaultValue = 'N/A') {
            try {
                const value = fn();
                return (value === undefined || value === null) ? defaultValue : value;
            } catch (e) {
                return defaultValue;
            }
        }

        function updatePageContent(fileIndex) {
            if (fileIndex < 0 || fileIndex >= allAnalysisData.length) {
                console.error('Invalid file index:', fileIndex);
                return;
            }
            const selectedFileData = allAnalysisData[fileIndex];
            console.log('Updating page with data for:', getSafe(() => selectedFileData.audio_file_path, 'Unknown audio path'));

            const mainHeaderTitle = document.querySelector('.header h1');
            if (mainHeaderTitle) {
                let reportTitleBase = "Transcription Analysis Report";
                const audioPath = getSafe(() => selectedFileData.audio_file_path);
                if (audioPath !== 'N/A') {
                    reportTitleBase = audioPath.replace(/\\/g, '/').split('/').pop().split('.').slice(0, -1).join('.') || audioPath.replace(/\\/g, '/').split('/').pop();
                } else if (getSafe(() => selectedFileData.error) !== 'N/A') {
                    const errorFileName = getSafe(() => selectedFileData.audio_file_path, "Unknown File").replace(/\\/g, '/').split('/').pop();
                    reportTitleBase = `Error processing: ${errorFileName}`;
                }
                 mainHeaderTitle.textContent = `📚 ${reportTitleBase} - Analysis Report`;
            }
            
            document.getElementById('active_info_timestamp').textContent = getSafe(() => selectedFileData.processing_timestamp, new Date().toLocaleString());
            const filePathDisplay = getSafe(() => selectedFileData.audio_file_path, 'N/A');
            document.getElementById('active_info_filePath').textContent = filePathDisplay.replace(/\\/g, '/').split('/').pop();
            
            let durationMinutesText = '0.0 Minuten';
            const segmentsForDuration = getSafe(() => selectedFileData.transcription.segments, []);
            if (segmentsForDuration.length > 0) {
                const lastSegmentEnd = getSafe(() => segmentsForDuration[segmentsForDuration.length - 1].end, 0);
                durationMinutesText = (lastSegmentEnd / 60).toFixed(1) + ' Minuten';
            }
            document.getElementById('active_info_duration').textContent = durationMinutesText;
            
            const lang = getSafe(() => selectedFileData.transcription_config.language, getSafe(() => selectedFileData.transcription.language, 'N/A'));
            document.getElementById('active_info_language').textContent = lang;
            document.getElementById('active_info_screenshotsCount').textContent = getSafe(() => selectedFileData.screenshots.length, 0);
            document.getElementById('active_info_pdfsCount').textContent = getSafe(() => selectedFileData.related_pdfs.length, 0);

            const transcriptContentDiv = document.getElementById('active_transcript');
            if (transcriptContentDiv) {
                transcriptContentDiv.innerHTML = generateTranscriptSectionHTML(getSafe(() => selectedFileData.transcription, {}), "active_");
            }
            
            const statsContainer = document.getElementById('active_statistics'); 
            if (statsContainer) {
                statsContainer.innerHTML = generateStatisticsSectionHTML(selectedFileData, "active_");
            }
            
            const screenshotsContent = document.getElementById('active_screenshots');
            if (screenshotsContent) {
                screenshotsContent.innerHTML = generateScreenshotsSectionHTML(getSafe(() => selectedFileData.screenshots, []), "active_");
            }

            const pdfsContent = document.getElementById('active_pdfs');
            if (pdfsContent) {
                pdfsContent.innerHTML = generatePdfsSectionHTML(getSafe(() => selectedFileData.related_pdfs, []), "active_");
            }

            const mappingContent = document.getElementById('active_mapping');
            if (mappingContent) {
                mappingContent.innerHTML = generateMappingSectionHTML(getSafe(() => selectedFileData.screenshot_transcript_mapping, []), "active_");
            }
            
            const activeTabContainer = document.getElementById('active_file_container');
            let tabToReshowId = 'active_transcript'; 
            if (activeTabContainer) {
                const currentActiveTabButton = activeTabContainer.querySelector('.tabs .tab.active');
                if (currentActiveTabButton) {
                    tabToReshowId = currentActiveTabButton.getAttribute('data-tab');
                }
            }
            showTab(tabToReshowId, 'active_file_container');
        }

        function formatTimestampJS(totalSeconds) {
            if (typeof totalSeconds !== 'number' || isNaN(totalSeconds)) return '00:00';
            const hours = Math.floor(totalSeconds / 3600);
            const minutes = Math.floor((totalSeconds % 3600) / 60);
            const seconds = Math.floor(totalSeconds % 60);
            if (hours > 0) {
                return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
            return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }

        function getConfidenceClassJS(confidence) {
            if (typeof confidence !== 'number') return 'low-confidence';
            if (confidence >= 0.9) return 'high-confidence';
            if (confidence >= 0.7) return 'medium-confidence';
            return 'low-confidence';
        }
        
        function generateTranscriptSegmentsHTML(transcriptionData) {
            const segments = getSafe(() => transcriptionData.segments, []);
            if (segments.length === 0) {
                return '<div class="empty-state"><p>Kein Transkript verfügbar.</p></div>';
            }
            let segmentsHtml = '';
            segments.forEach(segment => {
                const startFormatted = formatTimestampJS(getSafe(() => segment.start, 0));
                const endFormatted = formatTimestampJS(getSafe(() => segment.end, 0));
                const text = getSafe(() => segment.text, "").trim();
                const confidence = getSafe(() => segment.confidence, 0);
                const confidenceClass = getConfidenceClassJS(confidence);

                if (!text) return; 

                segmentsHtml += `
                    <div class="transcript-segment" data-start="${getSafe(() => segment.start, 0) * 1000}" data-end="${getSafe(() => segment.end, 0) * 1000}">
                        <div class="segment-header">
                            <span class="timestamp">[${startFormatted} - ${endFormatted}]</span>
                            <span class="confidence ${confidenceClass}">Vertrauen: ${confidence.toFixed(2)}</span>
                        </div>
                        <p class="segment-text">${text.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</p>
                    </div>
                `;
            });
            return segmentsHtml;
        }

        function generateTranscriptSectionHTML(transcriptionData, file_id_prefix_unused) {
            const segments_html_content = generateTranscriptSegmentsHTML(transcriptionData); 
            const full_text = getSafe(() => transcriptionData.text, "");
            const word_count = full_text ? full_text.split(/\s+/).filter(Boolean).length : 0;
            const segments_count = getSafe(() => transcriptionData.segments.length, 0);

            return `
                <div class="section">
                    <h2>📝 Transkript</h2>
                    <div class="section-stats">
                        <span>Segmente: ${segments_count}</span>
                        <span>Wörter: ${word_count}</span>
                        <span>Zeichen: ${full_text.length}</span>
                    </div>
                    <div class="transcript-container">
                        ${segments_html_content}
                    </div>
                </div>
            `;
        }

        function generateStatisticsSectionHTML(fileData, file_id_prefix_unused) {
            const transcription_data = getSafe(() => fileData.transcription, {});
            const transcription_config = getSafe(() => fileData.transcription_config, {});
            const speech_pattern_analysis = getSafe(() => fileData.speech_pattern_analysis, {});

            const processing_time_seconds = getSafe(() => transcription_data.processing_time_seconds, getSafe(() => fileData.processing_time_seconds, 0));
            const segments_list = getSafe(() => transcription_data.segments, []);
            const segments_total = segments_list.length;
            
            const full_text = getSafe(() => transcription_data.text, "");
            const word_count = full_text ? full_text.split(/\\s+/).filter(Boolean).length : 0;
            const char_count = full_text.length;

            let params_html = '<div class="stats-card"><h3>⚙️ Transkriptionsparameter</h3><div class="stats-content">';
            const params_to_display = {
                "Modell": getSafe(() => transcription_config.model_name),
                "Gerät": getSafe(() => transcription_config.device),
                "Sprache": getSafe(() => transcription_config.language),
                "Segmentierungsmodus": getSafe(() => transcription_config.segmentation_mode),
            };
            for (const key in params_to_display) {
                const value = params_to_display[key];
                if (value !== 'N/A' && value !== undefined && value !== null) {
                    params_html += `<p><strong>${key}:</strong> ${value}</p>`;
                }
            }
            const config_params_details = getSafe(() => transcription_config.parameters, {});
            if (Object.keys(config_params_details).length > 0) {
                params_html += "<h4>Parameter Details:</h4>";
                for (const pk in config_params_details) {
                     params_html += `<p><strong>${pk.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}:</strong> ${config_params_details[pk]}</p>`;
                }
            }
            params_html += '</div></div>';
            
            let generalStatsHtml = `
                <div class="stats-card">
                    <h3>📊 Allgemeine Transkriptionsstatistiken</h3>
                    <div class="stats-content">
                        <p><strong>Verarbeitungszeit:</strong> ${processing_time_seconds.toFixed(2)} Sekunden</p>
                        <p><strong>Anzahl Segmente:</strong> ${segments_total}</p>
                        <p><strong>Gesamtwortzahl:</strong> ${word_count}</p>
                        <p><strong>Gesamtzeichenzahl:</strong> ${char_count}</p>
                    </div>
                </div>`;

            let speechPatternHtml = '<div class="stats-card"><h3>🗣️ Sprachmusteranalyse</h3><div class="stats-content">';
            const spa_to_display = {
                "Dauer (Sekunden)": getSafe(() => speech_pattern_analysis.duration_seconds),
                "Mittlere Lautstärke (dB)": getSafe(() => speech_pattern_analysis.mean_volume_db, 0).toFixed(2),
                "Lautstärke StdAbw (dB)": getSafe(() => speech_pattern_analysis.volume_std, 0).toFixed(2),
                "Anteil Stille": (getSafe(() => speech_pattern_analysis.quiet_ratio, 0) * 100).toFixed(1) + '%',
                "Anteil Sprache": (getSafe(() => speech_pattern_analysis.speech_ratio, 0) * 100).toFixed(1) + '%',
                "Sprechertyp": getSafe(() => speech_pattern_analysis.speaker_type),
                "Empf. Min. Stille (ms)": getSafe(() => speech_pattern_analysis.recommended_min_silence_len),
                "Empf. Padding (ms)": getSafe(() => speech_pattern_analysis.recommended_padding),
            };
            let spaContentFound = false;
            for (const key in spa_to_display) {
                const value = spa_to_display[key];
                if (value !== 'N/A' && value !== undefined && value !== null && !(typeof value === 'string' && value.includes('NaN'))) {
                    speechPatternHtml += `<p><strong>${key}:</strong> ${value}</p>`;
                    spaContentFound = true;
                }
            }
            if (!spaContentFound) {
                speechPatternHtml += '<p>Keine Sprachmusteranalyse-Daten verfügbar.</p>';
            }
            speechPatternHtml += '</div></div>';
            
            let errorHtml = '';
            const fileError = getSafe(() => fileData.error);
            if (fileError !== 'N/A') {
                errorHtml = `
                <div class="stats-card" style="background-color: #ffdddd; border-color: #ffaaaa;">
                    <h3>⚠️ Fehler bei der Verarbeitung</h3>
                    <p style="color: red; white-space: pre-wrap;">${fileError.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</p>
                </div>`;
            }

            return `
                <div class="section">
                    <h2>📊 Statistiken & Parameter</h2>
                    ${errorHtml}
                    ${params_html}
                    ${generalStatsHtml}
                    ${speechPatternHtml}
                </div>
            `;
        }

        function generateScreenshotsSectionHTML(screenshots, file_id_prefix_unused) {
            if (!screenshots || screenshots.length === 0) {
                return '<div class="section"><h2>🖼️ Screenshots</h2><div class="empty-state"><p>Keine Screenshots verfügbar.</p></div></div>';
            }
            let content = '<div class="section"><h2>🖼️ Screenshots</h2><div class="screenshots-container">';
            screenshots.forEach(ss => {
                const timestamp = getSafe(() => ss.timestamp, 0);
                const imagePath = getSafe(() => ss.image_path, '#');
                const notes = getSafe(() => ss.notes, '');
                content += `
                    <div class="screenshot-item stats-card">
                        <img src="${imagePath}" alt="Screenshot bei ${formatTimestampJS(timestamp)}" style="max-width: 100%; height: auto; border-radius: 4px; margin-bottom: 10px;">
                        <p><strong>Zeitstempel:</strong> ${formatTimestampJS(timestamp)}</p>
                        <p><strong>Pfad:</strong> ${imagePath}</p>
                        ${notes ? `<p><strong>Notizen:</strong> ${notes.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</p>` : ''}
                    </div>
                `;
            });
            content += '</div></div>';
            return content;
        }

        function generatePdfsSectionHTML(pdfs, file_id_prefix_unused) {
            if (!pdfs || pdfs.length === 0) {
                return '<div class="section"><h2>📄 PDFs</h2><div class="empty-state"><p>Keine PDFs verfügbar.</p></div></div>';
            }
            let content = '<div class="section"><h2>📄 PDFs</h2><div class="pdfs-container"><table><thead><tr><th>Dateiname</th><th>Relevanz</th><th>Notizen</th></tr></thead><tbody>';
            pdfs.forEach(pdf => {
                const fileName = getSafe(() => pdf.file_path, 'Unbekannte PDF').replace(/\\/g, '/').split('/').pop();
                const relevance = getSafe(() => pdf.relevance_score, 0).toFixed(2);
                const notes = getSafe(() => pdf.notes, '');
                content += `
                    <tr>
                        <td><a href="${getSafe(() => pdf.file_path, '#')}" target="_blank">${fileName}</a></td>
                        <td>${relevance}</td>
                        <td>${notes.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</td>
                    </tr>
                `;
            });
            content += '</tbody></table></div></div>';
            return content;
        }
        
        function generateMappingSectionHTML(mappingData, file_id_prefix_unused) {
            if (!mappingData || mappingData.length === 0) {
                return '<div class="section"><h2>🔗 Screenshot-Transkript-Mapping</h2><div class="empty-state"><p>Keine Mapping-Daten verfügbar.</p></div></div>';
            }
            let content = '<div class="section"><h2>🔗 Screenshot-Transkript-Mapping</h2><div class="mapping-container"><table><thead><tr><th>Screenshot Zeit</th><th>Screenshot Pfad</th><th>Transkriptsegment</th><th>Segment Zeit</th></tr></thead><tbody>';
            mappingData.forEach(mapItem => {
                const ssTime = formatTimestampJS(getSafe(() => mapItem.screenshot_timestamp, 0));
                const ssPath = getSafe(() => mapItem.screenshot_path, 'N/A').replace(/\\/g, '/').split('/').pop();
                const segmentText = getSafe(() => mapItem.transcript_segment_text, 'N/A');
                const segmentTime = `[${formatTimestampJS(getSafe(() => mapItem.segment_start_time, 0))} - ${formatTimestampJS(getSafe(() => mapItem.segment_end_time, 0))}]`;
                
                content += `
                    <tr>
                        <td>${ssTime}</td>
                        <td>${ssPath}</td>
                        <td>${segmentText.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</td>
                        <td>${segmentTime}</td>
                    </tr>
                `;
            });
            content += '</tbody></table></div></div>';
            return content;
        }
        
        function new_datetime_string() {
            const now = new Date();
            return now.getFullYear() + '-' + (now.getMonth() + 1).toString().padStart(2, '0') + '-' + now.getDate().toString().padStart(2, '0') + ' ' +
                   now.getHours().toString().padStart(2, '0') + ':' + now.getMinutes().toString().padStart(2, '0') + ':' + now.getSeconds().toString().padStart(2, '0');
        }
"""

    def _get_empty_result_structure(self) -> dict:
        """Returns a default structure for when no results are available or an error occurs."""
        return {
            "audio_file_path": "N/A",
            "processing_timestamp": new_datetime_string(), # Use Python's datetime for initial server-side render
            "transcription": {"text": "No data available.", "segments": []},
            "transcription_config": {},
            "speech_pattern_analysis": {},
            "error": "No data loaded"
        }

# Helper for Python side, if needed by _generate_body_content for initial render
def new_datetime_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")