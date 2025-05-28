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
    
    def _generate_html_document(self, results: Dict) -> str:
        """
        Generate complete HTML document.
        
        Args:
            results: Analysis results
            
        Returns:
            Complete HTML document as string
        """
        video_name = Path(results.get("video_path", "Unknown")).stem
        
        # Generate individual sections
        header_html = self._generate_header(results, video_name)
        navigation_html = self._generate_navigation()
        transcript_html = self._generate_transcript_section(results.get("transcription", {}))
        screenshots_html = self._generate_screenshots_section(results.get("screenshots", []))
        pdfs_html = self._generate_pdfs_section(results.get("related_pdfs", []))
        mapping_html = self._generate_mapping_section(results.get("screenshot_transcript_mapping", []))
        statistics_html = self._generate_statistics_section(results)
        
        # Generate CSS and JavaScript
        css = self._generate_css()
        javascript = self._generate_javascript()
        
        return f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Studienanalyse: {video_name}</title>
    <style>{css}</style>
</head>
<body>
    <div class="container">
        {header_html}
        {navigation_html}
        
        <div id="transcript" class="tab-content active">
            {transcript_html}
        </div>
        
        <div id="screenshots" class="tab-content">
            {screenshots_html}
        </div>
        
        <div id="pdfs" class="tab-content">
            {pdfs_html}
        </div>
        
        <div id="mapping" class="tab-content">
            {mapping_html}
        </div>
        
        <div id="statistics" class="tab-content">
            {statistics_html}
        </div>
    </div>
    
    <script>{javascript}</script>
</body>
</html>"""
    
    def _generate_header(self, results: Dict, video_name: str) -> str:
        """Generate header section with video information."""
        transcription = results.get("transcription", {})
        duration = transcription.get("total_duration", 0) / 1000 / 60  # Convert to minutes
        
        return f"""
        <div class="header">
            <h1>📚 Studienanalyse: {video_name}</h1>
            <div class="header-info">
                <div class="info-grid">
                    <div class="info-item">
                        <span class="info-label">📅 Verarbeitet:</span>
                        <span class="info-value">{results.get('processing_timestamp', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">🎥 Video:</span>
                        <span class="info-value">{Path(results.get('video_path', '')).name}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">⏱️ Dauer:</span>
                        <span class="info-value">{duration:.1f} Minuten</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">🔤 Sprache:</span>
                        <span class="info-value">{transcription.get('language', 'N/A')}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">📸 Screenshots:</span>
                        <span class="info-value">{len(results.get('screenshots', []))}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">📄 PDFs:</span>
                        <span class="info-value">{len(results.get('related_pdfs', []))}</span>
                    </div>
                </div>
            </div>
            
            <div class="search-container">
                <input type="text" id="searchInput" placeholder="🔍 Durchsuchen Sie Transkript, Screenshots und PDFs..." autocomplete="off">
                <div class="search-stats" id="searchStats"></div>
            </div>
        </div>
        """
    
    def _generate_navigation(self) -> str:
        """Generate navigation tabs."""
        return """
        <div class="tabs">
            <div class="tab active" onclick="showTab('transcript')" data-tab="transcript">
                📝 Transkript
            </div>
            <div class="tab" onclick="showTab('screenshots')" data-tab="screenshots">
                📸 Screenshots
            </div>
            <div class="tab" onclick="showTab('pdfs')" data-tab="pdfs">
                📄 PDFs
            </div>
            <div class="tab" onclick="showTab('mapping')" data-tab="mapping">
                🔗 Zuordnung
            </div>
            <div class="tab" onclick="showTab('statistics')" data-tab="statistics">
                📊 Statistiken
            </div>
        </div>
        """
    
    def _generate_transcript_section(self, transcription: Dict) -> str:
        """Generate transcript section HTML."""
        if not transcription or not transcription.get("segments"):
            return """
            <div class="section">
                <h2>📝 Transkript</h2>
                <div class="empty-state">
                    <p>Kein Transkript verfügbar.</p>
                </div>
            </div>
            """
        
        segments_html = []
        for segment in transcription["segments"]:
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            text = segment.get("text", "").strip()
            confidence = segment.get("confidence", 0)
            
            if not text:
                continue
            
            start_formatted = format_timestamp_seconds(start_time / 1000)
            end_formatted = format_timestamp_seconds(end_time / 1000)
            
            confidence_class = self._get_confidence_class(confidence)
            
            segments_html.append(f"""
            <div class="transcript-segment" data-start="{start_time}" data-end="{end_time}">
                <div class="segment-header">
                    <span class="timestamp">[{start_formatted} - {end_formatted}]</span>
                    <span class="confidence {confidence_class}">Vertrauen: {confidence:.2f}</span>
                </div>
                <p class="segment-text">{text}</p>
            </div>
            """)
        
        full_text = transcription.get("full_text", "")
        word_count = len(full_text.split()) if full_text else 0
        
        return f"""
        <div class="section">
            <h2>📝 Transkript</h2>
            <div class="section-stats">
                <span>Segmente: {len(segments_html)}</span>
                <span>Wörter: {word_count}</span>
                <span>Zeichen: {len(full_text)}</span>
            </div>
            <div class="transcript-container">
                {''.join(segments_html)}
            </div>
        </div>
        """
    
    def _generate_screenshots_section(self, screenshots: List[Dict]) -> str:
        """Generate screenshots section HTML."""
        if not screenshots:
            return """
            <div class="section">
                <h2>📸 Screenshots</h2>
                <div class="empty-state">
                    <p>Keine Screenshots verfügbar.</p>
                </div>
            </div>
            """
        
        screenshots_html = []
        for screenshot in screenshots:
            filename = screenshot.get("filename", "")
            timestamp = screenshot.get("timestamp_formatted", "")
            similarity = screenshot.get("similarity_score", 1.0)
            
            screenshots_html.append(f"""
            <div class="screenshot-item" data-timestamp="{screenshot.get('timestamp', 0)}">
                <div class="screenshot-container">
                    <img src="screenshots/{filename}" alt="Screenshot bei {timestamp}" loading="lazy">
                    <div class="screenshot-overlay">
                        <div class="screenshot-info">
                            <span class="screenshot-time">🕐 {timestamp}</span>
                            <span class="screenshot-similarity">Ähnlichkeit: {similarity:.3f}</span>
                        </div>
                    </div>
                </div>
            </div>
            """)
        
        return f"""
        <div class="section">
            <h2>📸 Screenshots</h2>
            <div class="section-stats">
                <span>Anzahl: {len(screenshots)}</span>
            </div>
            <div class="screenshots-grid">
                {''.join(screenshots_html)}
            </div>
        </div>
        """
    
    def _generate_pdfs_section(self, pdfs: List[Dict]) -> str:
        """Generate PDFs section HTML."""
        if not pdfs:
            return """
            <div class="section">
                <h2>📄 Verwandte PDFs</h2>
                <div class="empty-state">
                    <p>Keine verwandten PDFs gefunden.</p>
                </div>
            </div>
            """
        
        pdfs_html = []
        for pdf in pdfs:
            filename = pdf.get("filename", "")
            relevance = pdf.get("relevance_score", 0)
            preview = pdf.get("content_preview", "")
            page_count = pdf.get("page_count", 0)
            file_size = pdf.get("file_size_bytes", 0)
            
            # Format file size
            size_mb = file_size / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{file_size / 1024:.0f} KB"
            
            relevance_class = "high" if relevance >= 10 else "medium" if relevance >= 5 else "low"
            
            pdfs_html.append(f"""
            <div class="pdf-item" data-relevance="{relevance}">
                <div class="pdf-header">
                    <h3 class="pdf-title">📄 {filename}</h3>
                    <div class="pdf-metadata">
                        <span class="relevance-score {relevance_class}">Relevanz: {relevance}</span>
                        <span class="pdf-stats">{page_count} Seiten • {size_str}</span>
                    </div>
                </div>
                <div class="pdf-preview">
                    <h4>Vorschau:</h4>
                    <p class="preview-text">{preview[:500]}{'...' if len(preview) > 500 else ''}</p>
                </div>
            </div>
            """)
        
        return f"""
        <div class="section">
            <h2>📄 Verwandte PDFs</h2>
            <div class="section-stats">
                <span>Gefunden: {len(pdfs)}</span>
                <span>Höchste Relevanz: {max([p.get('relevance_score', 0) for p in pdfs], default=0)}</span>
            </div>
            <div class="pdfs-container">
                {''.join(pdfs_html)}
            </div>
        </div>
        """
    
    def _generate_mapping_section(self, mappings: List[Dict]) -> str:
        """Generate screenshot-transcript mapping section."""
        if not mappings:
            return """
            <div class="section">
                <h2>🔗 Screenshot-Transkript Zuordnung</h2>
                <div class="empty-state">
                    <p>Keine Zuordnungen verfügbar.</p>
                </div>
            </div>
            """
        
        mappings_html = []
        for i, mapping in enumerate(mappings):
            screenshot = mapping.get("screenshot", {})
            transcript = mapping.get("transcript_segment", {})
            time_diff = mapping.get("time_difference", 0)
            
            screenshot_time = screenshot.get("timestamp_formatted", "")
            screenshot_file = screenshot.get("filename", "")
            transcript_text = transcript.get("text", "")
            
            mappings_html.append(f"""
            <div class="mapping-item" data-index="{i}">
                <div class="mapping-content">
                    <div class="mapping-screenshot">
                        <img src="screenshots/{screenshot_file}" alt="Screenshot" loading="lazy">
                        <div class="mapping-screenshot-info">
                            <span class="mapping-time">🕐 {screenshot_time}</span>
                            <span class="time-difference">Δ {time_diff:.1f}s</span>
                        </div>
                    </div>
                    <div class="mapping-transcript">
                        <h4>📝 Entsprechender Text:</h4>
                        <p class="mapping-text">{transcript_text}</p>
                    </div>
                </div>
            </div>
            """)
        
        return f"""
        <div class="section">
            <h2>🔗 Screenshot-Transkript Zuordnung</h2>
            <div class="section-stats">
                <span>Zuordnungen: {len(mappings)}</span>
            </div>
            <div class="mappings-container">
                {''.join(mappings_html)}
            </div>
        </div>
        """
    
    def _generate_statistics_section(self, results: Dict) -> str:
        """Generate statistics section."""
        transcription = results.get("transcription", {})
        screenshots = results.get("screenshots", [])
        pdfs = results.get("related_pdfs", [])
        
        # Transcription stats
        processing_time = transcription.get("processing_time_seconds", 0)
        segments_total = transcription.get("segments_total", 0)
        segments_successful = transcription.get("segments_successful", 0)
        success_rate = (segments_successful / segments_total * 100) if segments_total > 0 else 0
        
        # Word and character count from full text
        full_text = transcription.get("full_text", "")
        word_count = len(full_text.split()) if full_text else 0
        char_count = len(full_text) if full_text else 0
        
        # Screenshot stats
        screenshot_times = [s.get("timestamp", 0) for s in screenshots]
        avg_interval = (max(screenshot_times) - min(screenshot_times)) / (len(screenshot_times) - 1) if len(screenshot_times) > 1 else 0
        
        return f"""
        <div class="section">
            <h2>📊 Verarbeitungsstatistiken</h2>
            
            <div class="stats-grid">
                <div class="stats-card">
                    <h3>🎙️ Transkription</h3>
                    <div class="stats-content">
                        <div class="stat-item">
                            <span class="stat-label">Verarbeitungszeit:</span>
                            <span class="stat-value">{processing_time:.1f}s</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Segmente gesamt:</span>
                            <span class="stat-value">{segments_total}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Erfolgreich:</span>
                            <span class="stat-value">{segments_successful}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Erfolgsrate:</span>
                            <span class="stat-value">{success_rate:.1f}%</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Wörter:</span>
                            <span class="stat-value">{word_count:,}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Zeichen:</span>
                            <span class="stat-value">{char_count:,}</span>
                        </div>
                    </div>
                </div>
                
                <div class="stats-card">
                    <h3>📸 Screenshots</h3>
                    <div class="stats-content">
                        <div class="stat-item">
                            <span class="stat-label">Anzahl:</span>
                            <span class="stat-value">{len(screenshots)}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Durchschnittlicher Abstand:</span>
                            <span class="stat-value">{avg_interval:.1f}s</span>
                        </div>
                    </div>
                </div>
                
                <div class="stats-card">
                    <h3>📄 PDFs</h3>
                    <div class="stats-content">
                        <div class="stat-item">
                            <span class="stat-label">Gefunden:</span>
                            <span class="stat-value">{len(pdfs)}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Mit hoher Relevanz:</span>
                            <span class="stat-value">{len([p for p in pdfs if p.get('relevance_score', 0) >= 10])}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _get_confidence_class(self, confidence: float) -> str:
        """Get CSS class based on confidence score."""
        if confidence >= 0.9:
            return "confidence-high"
        elif confidence >= 0.7:
            return "confidence-medium"
        else:
            return "confidence-low"
    
    def _generate_css(self) -> str:
        """Generate CSS styles for the HTML report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f8f9fa;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            margin-bottom: 20px;
            font-size: 2.5em;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }
        
        .info-item {
            display: flex;
            justify-content: space-between;
            background: rgba(255,255,255,0.1);
            padding: 10px 15px;
            border-radius: 6px;
        }
        
        .info-label {
            font-weight: 600;
        }
        
        .search-container {
            position: relative;
        }
        
        .search-container input {
            width: 100%;
            padding: 15px 20px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            background: rgba(255,255,255,0.9);
            backdrop-filter: blur(10px);
        }
        
        .search-stats {
            margin-top: 10px;
            font-size: 14px;
            color: rgba(255,255,255,0.8);
        }
        
        .tabs {
            display: flex;
            background: white;
            border-radius: 12px 12px 0 0;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 0;
        }
        
        .tab {
            flex: 1;
            padding: 15px 20px;
            cursor: pointer;
            background: #f8f9fa;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
            text-align: center;
            font-weight: 500;
        }
        
        .tab:hover {
            background: #e9ecef;
        }
        
        .tab.active {
            background: white;
            border-bottom-color: #667eea;
            color: #667eea;
        }
        
        .tab-content {
            display: none;
            background: white;
            border-radius: 0 0 12px 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .tab-content.active {
            display: block;
        }
        
        .section {
            padding: 30px;
        }
        
        .section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
            font-size: 1.8em;
        }
        
        .section-stats {
            margin-bottom: 20px;
            display: flex;
            gap: 20px;
            color: #666;
        }
        
        .section-stats span {
            padding: 5px 12px;
            background: #f8f9fa;
            border-radius: 20px;
            font-size: 14px;
        }
        
        .transcript-segment {
            margin-bottom: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            transition: background 0.2s ease;
        }
        
        .transcript-segment:hover {
            background: #e9ecef;
        }
        
        .segment-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .timestamp {
            color: #667eea;
            font-weight: bold;
            font-family: 'Courier New', monospace;
        }
        
        .confidence {
            font-size: 12px;
            padding: 2px 8px;
            border-radius: 12px;
            color: white;
        }
        
        .confidence-high { background: #28a745; }
        .confidence-medium { background: #ffc107; color: #333; }
        .confidence-low { background: #dc3545; }
        
        .segment-text {
            font-size: 16px;
            line-height: 1.6;
        }
        
        .screenshots-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .screenshot-item {
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        
        .screenshot-item:hover {
            transform: translateY(-2px);
        }
        
        .screenshot-container {
            position: relative;
        }
        
        .screenshot-container img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .screenshot-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(transparent, rgba(0,0,0,0.8));
            color: white;
            padding: 15px;
        }
        
        .screenshot-info {
            display: flex;
            justify-content: space-between;
            font-size: 14px;
        }
        
        .pdf-item {
            margin-bottom: 25px;
            padding: 25px;
            background: #fff8e1;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
        }
        
        .pdf-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }
        
        .pdf-title {
            color: #2c3e50;
            margin: 0;
        }
        
        .pdf-metadata {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .relevance-score {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }
        
        .relevance-score.high { background: #28a745; }
        .relevance-score.medium { background: #ffc107; color: #333; }
        .relevance-score.low { background: #6c757d; }
        
        .pdf-stats {
            color: #666;
            font-size: 14px;
        }
        
        .preview-text {
            color: #555;
            font-style: italic;
            max-height: 100px;
            overflow: hidden;
        }
        
        .mapping-item {
            margin-bottom: 30px;
            padding: 25px;
            background: #e3f2fd;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
        }
        
        .mapping-content {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 25px;
            align-items: start;
        }
        
        .mapping-screenshot img {
            width: 100%;
            border-radius: 6px;
        }
        
        .mapping-screenshot-info {
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            color: #666;
        }
        
        .mapping-transcript h4 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .mapping-text {
            font-size: 16px;
            line-height: 1.6;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
        }
        
        .stats-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 25px;
            border-left: 4px solid #667eea;
        }
        
        .stats-card h3 {
            color: #2c3e50;
            margin-bottom: 20px;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid #dee2e6;
        }
        
        .stat-label {
            color: #666;
        }
        
        .stat-value {
            font-weight: bold;
            color: #2c3e50;
        }
        
        .empty-state {
            text-align: center;
            color: #666;
            font-style: italic;
            padding: 50px;
        }
        
        .highlight {
            background-color: #ffeb3b;
            padding: 2px 4px;
            border-radius: 3px;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .tabs {
                flex-direction: column;
            }
            
            .info-grid {
                grid-template-columns: 1fr;
            }
            
            .mapping-content {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
        """
    
    def _generate_javascript(self) -> str:
        """Generate JavaScript for interactive functionality."""
        return """
        // Tab switching functionality
        function showTab(tabName) {
            // Hide all tab contents
            const contents = document.getElementsByClassName('tab-content');
            for (let content of contents) {
                content.classList.remove('active');
            }
            
            // Remove active class from all tabs
            const tabs = document.getElementsByClassName('tab');
            for (let tab of tabs) {
                tab.classList.remove('active');
            }
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            const activeTab = document.querySelector(`[data-tab="${tabName}"]`);
            if (activeTab) {
                activeTab.classList.add('active');
            }
        }
        
        // Search functionality
        document.addEventListener('DOMContentLoaded', function() {
            const searchInput = document.getElementById('searchInput');
            const searchStats = document.getElementById('searchStats');
            
            if (searchInput) {
                searchInput.addEventListener('input', function(e) {
                    performSearch(e.target.value, searchStats);
                });
            }
        });
        
        function performSearch(searchTerm, statsElement) {
            const term = searchTerm.toLowerCase().trim();
            
            // Clear previous highlights
            clearHighlights();
            
            if (!term) {
                showAllElements();
                updateSearchStats(0, 0, statsElement);
                return;
            }
            
            // Find all searchable elements
            const searchableElements = document.querySelectorAll(
                '.transcript-segment, .pdf-item, .mapping-item, .screenshot-item'
            );
            
            let visibleCount = 0;
            let highlightCount = 0;
            
            searchableElements.forEach(element => {
                const text = element.textContent.toLowerCase();
                const hasMatch = text.includes(term);
                
                if (hasMatch) {
                    element.style.display = '';
                    highlightText(element, term);
                    visibleCount++;
                    highlightCount += countOccurrences(text, term);
                } else {
                    element.style.display = 'none';
                }
            });
            
            updateSearchStats(visibleCount, highlightCount, statsElement);
        }
        
        function highlightText(element, term) {
            const walker = document.createTreeWalker(
                element,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );
            
            const textNodes = [];
            let node;
            
            while (node = walker.nextNode()) {
                if (node.nodeValue.toLowerCase().includes(term)) {
                    textNodes.push(node);
                }
            }
            
            textNodes.forEach(textNode => {
                const parent = textNode.parentNode;
                const text = textNode.nodeValue;
                const regex = new RegExp(`(${escapeRegExp(term)})`, 'gi');
                const highlightedText = text.replace(regex, '<span class="highlight">$1</span>');
                
                if (highlightedText !== text) {
                    const wrapper = document.createElement('span');
                    wrapper.innerHTML = highlightedText;
                    parent.replaceChild(wrapper, textNode);
                }
            });
        }
        
        function clearHighlights() {
            const highlights = document.querySelectorAll('.highlight');
            highlights.forEach(highlight => {
                const parent = highlight.parentNode;
                parent.replaceChild(document.createTextNode(highlight.textContent), highlight);
                parent.normalize();
            });
        }
        
        function showAllElements() {
            const elements = document.querySelectorAll(
                '.transcript-segment, .pdf-item, .mapping-item, .screenshot-item'
            );
            elements.forEach(element => {
                element.style.display = '';
            });
        }
        
        function countOccurrences(text, term) {
            return (text.match(new RegExp(escapeRegExp(term), 'gi')) || []).length;
        }
        
        function escapeRegExp(string) {
            return string.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
        }
        
        function updateSearchStats(visible, highlights, statsElement) {
            if (statsElement) {
                if (highlights > 0) {
                    statsElement.textContent = `${visible} Elemente gefunden, ${highlights} Treffer`;
                } else if (visible === 0) {
                    statsElement.textContent = 'Keine Ergebnisse gefunden';
                } else {
                    statsElement.textContent = '';
                }
            }
        }
        
        // Image lazy loading enhancement
        document.addEventListener('DOMContentLoaded', function() {
            const images = document.querySelectorAll('img[loading="lazy"]');
            
            if ('IntersectionObserver' in window) {
                const imageObserver = new IntersectionObserver((entries, observer) => {
                    entries.forEach(entry => {
                        if (entry.isIntersecting) {
                            const img = entry.target;
                            img.classList.add('loaded');
                            observer.unobserve(img);
                        }
                    });
                });
                
                images.forEach(img => imageObserver.observe(img));
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Ctrl/Cmd + F to focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
                e.preventDefault();
                const searchInput = document.getElementById('searchInput');
                if (searchInput) {
                    searchInput.focus();
                    searchInput.select();
                }
            }
            
            // Escape to clear search
            if (e.key === 'Escape') {
                const searchInput = document.getElementById('searchInput');
                if (searchInput && searchInput === document.activeElement) {
                    searchInput.value = '';
                    performSearch('', document.getElementById('searchStats'));
                }
            }
        });
        """
    
    def generate_index_page(self, all_results: List[Dict], output_path: str) -> None:
        """
        Generate an index page linking to all processed videos.
        
        Args:
            all_results: List of all processing results
            output_path: Path to save the index HTML
        """
        logger.info(f"Generating index page: {output_path}")
        
        videos_html = []
        for result in all_results:
            video_name = Path(result["video_path"]).stem
            transcription = result.get("transcription", {})
            screenshots_count = len(result.get("screenshots", []))
            pdfs_count = len(result.get("related_pdfs", []))
            
            duration = transcription.get("total_duration", 0) / 1000 / 60
            processing_time = result.get("processing_timestamp", "N/A")
            
            videos_html.append(f"""
            <div class="video-card">
                <h3><a href="{video_name}/{video_name}_report.html">📹 {video_name}</a></h3>
                <div class="video-stats">
                    <span>⏱️ {duration:.1f} min</span>
                    <span>📸 {screenshots_count} Screenshots</span>
                    <span>📄 {pdfs_count} PDFs</span>
                </div>
                <p class="processing-time">Verarbeitet: {processing_time}</p>
            </div>
            """)
        
        html_content = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📚 Studienanalyse - Übersicht</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; line-height: 1.6; background: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 30px; }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .summary {{ background: white; padding: 25px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .videos-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }}
        .video-card {{ background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); transition: transform 0.2s ease; }}
        .video-card:hover {{ transform: translateY(-2px); }}
        .video-card h3 {{ margin-top: 0; color: #2c3e50; }}
        .video-card a {{ text-decoration: none; color: inherit; }}
        .video-card a:hover {{ color: #667eea; }}
        .video-stats {{ display: flex; gap: 15px; margin: 15px 0; }}
        .video-stats span {{ background: #f8f9fa; padding: 5px 10px; border-radius: 15px; font-size: 14px; }}
        .processing-time {{ color: #666; font-size: 14px; margin: 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 Studienanalyse - Übersicht</h1>
            <p>Generiert am: {datetime.now().strftime("%d.%m.%Y %H:%M:%S")}</p>
        </div>
        
        <div class="summary">
            <h2>📊 Zusammenfassung</h2>
            <p><strong>{len(all_results)}</strong> Videos verarbeitet</p>
        </div>
        
        <h2>🎥 Verarbeitete Videos</h2>
        <div class="videos-grid">
            {''.join(videos_html)}
        </div>
    </div>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Index page saved to: {output_path}")