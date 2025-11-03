#!/usr/bin/env python3
"""
Enhanced Master Index Generator

Creates a comprehensive, interactive index page with:
- Central search functionality across all transcripts
- Tab-based navigation (Search, Overview, Report Viewer)
- Correct linking to Timeline reports
- Integrated report viewing capabilities

Usage:
    python generate_enhanced_index.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class EnhancedIndexGenerator:
    """Generates an enhanced master index with search and navigation capabilities"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.results_dir = Path(__file__).resolve().parent / 'results'
        self.all_transcripts = []
        self.reports_data = []
        
    def collect_transcript_data(self):
        """Collect all transcript data for search indexing"""
        self.logger.info("Collecting transcript data from all reports...")
        
        for analysis_file in self.results_dir.glob("*/analysis_result.json"):
            try:
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                folder_name = analysis_file.parent.name
                
                # Extract transcript segments
                segments = []
                if 'transcription' in data:
                    transcription = data['transcription']
                    if isinstance(transcription, dict) and 'transcription' in transcription:
                        segments = transcription['transcription'].get('segments', [])
                    elif isinstance(transcription, dict) and 'segments' in transcription:
                        segments = transcription['segments']
                    elif isinstance(transcription, list):
                        segments = transcription
                
                # Collect report metadata
                report_info = {
                    'folder': folder_name,
                    'filename': data.get('filename', folder_name),
                    'duration': data.get('duration', 0),
                    'segments_count': len(segments),
                    'screenshots_count': len(data.get('screenshots', [])),
                    'pdfs_count': len(data.get('pdfs', [])),
                    'has_timeline_report': self._check_timeline_report_exists(folder_name),
                    'timeline_path': f"{folder_name}/{folder_name}_report_TIMELINE.html",
                    'standard_path': f"{folder_name}/{folder_name}_report.html"
                }
                
                self.reports_data.append(report_info)
                
                # Add segments to search index with metadata
                for i, segment in enumerate(segments):
                    self.all_transcripts.append({
                        'report_folder': folder_name,
                        'report_name': report_info['filename'],
                        'segment_id': i,
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'text': segment.get('text', ''),
                        'confidence': segment.get('avg_logprob', 0),
                        'timestamp': self._format_timestamp(segment.get('start', 0))
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing {analysis_file}: {e}")
        
        self.logger.info(f"Collected {len(self.all_transcripts)} transcript segments from {len(self.reports_data)} reports")
    
    def _check_timeline_report_exists(self, folder_name: str) -> bool:
        """Check if timeline report exists for this folder"""
        timeline_path = self.results_dir / folder_name / f"{folder_name}_report_TIMELINE.html"
        return timeline_path.exists()
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS or HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def generate_html(self) -> str:
        """Generate the enhanced HTML index"""
        
        # Calculate summary statistics
        total_duration = sum(report['duration'] for report in self.reports_data)
        total_segments = sum(report['segments_count'] for report in self.reports_data)
        total_screenshots = sum(report['screenshots_count'] for report in self.reports_data)
        total_pdfs = sum(report['pdfs_count'] for report in self.reports_data)
        reports_with_timeline = sum(1 for report in self.reports_data if report['has_timeline_report'])
        
        # Convert transcripts to JSON for client-side search
        transcripts_json = json.dumps(self.all_transcripts, ensure_ascii=False, indent=2)
        reports_json = json.dumps(self.reports_data, ensure_ascii=False, indent=2)
        
        # Generate header stats HTML
        header_stats_html = f"""
                <div class="header-stat">
                    <span class="stat-number">{len(self.reports_data)}</span>
                    <span>Reports</span>
                </div>
                <div class="header-stat">
                    <span class="stat-number">{reports_with_timeline}</span>
                    <span>Timeline Reports</span>
                </div>
                <div class="header-stat">
                    <span class="stat-number">{self._format_timestamp(total_duration * 60)}</span>
                    <span>Gesamtdauer</span>
                </div>
                <div class="header-stat">
                    <span class="stat-number">{total_segments:,}</span>
                    <span>Transcript-Segmente</span>
                </div>
                <div class="header-stat">
                    <span class="stat-number">{total_screenshots}</span>
                    <span>Screenshots</span>
                </div>
                <div class="header-stat">
                    <span class="stat-number">{total_pdfs}</span>
                    <span>PDFs</span>
                </div>
        """
        
        # Generate report filter options
        report_filter_options = '\n'.join(
            f'<option value="{report["folder"]}">{report["filename"]}</option>' 
            for report in self.reports_data
        )
        
        # Generate report cards
        report_cards_html = '\n'.join(self._generate_report_card(report) for report in self.reports_data)
        
        # Generate report selector options
        report_selector_options = '\n'.join([
            '\n'.join(f'<option value="{report["timeline_path"]}">{report["filename"]} (Timeline)</option>' 
                     for report in self.reports_data if report["has_timeline_report"]),
            '\n'.join(f'<option value="{report["standard_path"]}">{report["filename"]} (Standard)</option>' 
                     for report in self.reports_data)
        ])
        
        # Read CSS template
        css_content = """
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; 
            background-color: #f8f9fa; 
            color: #333; 
            line-height: 1.6;
        }
        
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background-color: #fff; 
            min-height: 100vh;
            box-shadow: 0 0 20px rgba(0,0,0,0.1); 
        }
        
        .header { 
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white; 
            padding: 30px; 
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header h1 { 
            margin: 0 0 10px 0; 
            font-size: 2.2em; 
            font-weight: 600;
        }
        
        .header-stats { 
            display: flex; 
            flex-wrap: wrap; 
            gap: 25px; 
            margin-top: 15px; 
            font-size: 0.95em; 
        }
        
        .header-stat { 
            background-color: rgba(255,255,255,0.15); 
            padding: 12px 16px; 
            border-radius: 6px; 
            backdrop-filter: blur(10px);
        }
        
        .stat-number { 
            font-size: 1.4em; 
            font-weight: bold; 
            display: block; 
        }
        
        .tab-navigation { 
            background-color: #fff;
            border-bottom: 2px solid #e9ecef;
            position: sticky;
            top: 140px;
            z-index: 90;
        }
        
        .tab-list { 
            display: flex; 
            padding: 0 30px; 
            margin: 0;
            list-style: none;
        }
        
        .tab-button { 
            padding: 15px 25px; 
            cursor: pointer; 
            border: none;
            background: none;
            font-size: 16px;
            font-weight: 500;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
        }
        
        .tab-button:hover { 
            color: #007bff; 
            background-color: #f8f9fa;
        }
        
        .tab-button.active { 
            color: #007bff; 
            border-bottom-color: #007bff; 
            background-color: #f8f9fa;
        }
        
        .tab-content { 
            display: none; 
            padding: 30px; 
            min-height: 600px;
        }
        
        .tab-content.active { 
            display: block; 
        }
        
        .search-container { 
            max-width: 800px; 
            margin: 0 auto; 
        }
        
        .search-box { 
            position: relative; 
            margin-bottom: 30px; 
        }
        
        .search-input { 
            width: 100%; 
            padding: 18px 50px 18px 20px; 
            font-size: 16px; 
            border: 2px solid #e9ecef; 
            border-radius: 10px; 
            outline: none;
            transition: border-color 0.3s ease;
        }
        
        .search-input:focus { 
            border-color: #007bff; 
            box-shadow: 0 0 0 3px rgba(0,123,255,0.1);
        }
        
        .search-button { 
            position: absolute; 
            right: 10px; 
            top: 50%; 
            transform: translateY(-50%); 
            background: #007bff; 
            color: white; 
            border: none; 
            padding: 10px 15px; 
            border-radius: 6px; 
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        
        .search-button:hover { 
            background: #0056b3; 
        }
        
        .search-filters { 
            display: flex; 
            gap: 15px; 
            margin-bottom: 20px; 
            flex-wrap: wrap;
        }
        
        .filter-select { 
            padding: 8px 12px; 
            border: 1px solid #ddd; 
            border-radius: 6px; 
            font-size: 14px;
        }
        
        .search-results { 
            margin-top: 30px; 
        }
        
        .search-result { 
            border: 1px solid #e9ecef; 
            border-radius: 8px; 
            padding: 20px; 
            margin-bottom: 15px; 
            background: #fff;
            transition: box-shadow 0.3s ease;
        }
        
        .search-result:hover { 
            box-shadow: 0 4px 12px rgba(0,0,0,0.1); 
        }
        
        .result-header { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: 10px; 
        }
        
        .result-title { 
            font-weight: bold; 
            color: #007bff; 
            text-decoration: none;
            font-size: 1.1em;
        }
        
        .result-title:hover { 
            text-decoration: underline; 
        }
        
        .result-time { 
            color: #666; 
            font-size: 0.9em; 
        }
        
        .result-text { 
            line-height: 1.6; 
            margin-bottom: 10px; 
        }
        
        .result-text .highlight { 
            background-color: #fff3cd; 
            padding: 2px 4px; 
            border-radius: 3px; 
            font-weight: bold;
        }
        
        .result-meta { 
            display: flex; 
            gap: 15px; 
            font-size: 0.85em; 
            color: #666; 
        }
        
        .no-results { 
            text-align: center; 
            color: #666; 
            font-style: italic; 
            padding: 40px; 
        }
        
        .reports-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
            gap: 20px; 
        }
        
        .report-card { 
            border: 1px solid #e9ecef; 
            border-radius: 10px; 
            padding: 25px; 
            background: #fff;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .report-card:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 8px 25px rgba(0,0,0,0.1); 
        }
        
        .report-title { 
            font-size: 1.3em; 
            font-weight: bold; 
            margin-bottom: 15px; 
            color: #333;
        }
        
        .report-actions { 
            display: flex; 
            gap: 10px; 
            margin-bottom: 20px; 
        }
        
        .action-button { 
            padding: 10px 20px; 
            border: none; 
            border-radius: 6px; 
            text-decoration: none; 
            font-size: 14px; 
            font-weight: 500;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .timeline-button { 
            background: #007bff; 
            color: white; 
        }
        
        .timeline-button:hover { 
            background: #0056b3; 
        }
        
        .standard-button { 
            background: #6c757d; 
            color: white; 
        }
        
        .standard-button:hover { 
            background: #5a6268; 
        }
        
        .report-stats { 
            display: grid; 
            grid-template-columns: repeat(2, 1fr); 
            gap: 10px; 
            font-size: 0.9em; 
        }
        
        .stat-item { 
            display: flex; 
            justify-content: space-between; 
            padding: 8px 0;
            border-bottom: 1px solid #f8f9fa;
        }
        
        .stat-label { 
            color: #666; 
        }
        
        .stat-value { 
            font-weight: bold; 
            color: #007bff; 
        }
        
        .viewer-container { 
            display: flex; 
            gap: 20px; 
            height: calc(100vh - 300px);
        }
        
        .viewer-sidebar { 
            width: 300px; 
            border-right: 1px solid #e9ecef; 
            padding-right: 20px;
        }
        
        .report-selector { 
            width: 100%; 
            padding: 12px; 
            border: 1px solid #ddd; 
            border-radius: 6px; 
            font-size: 16px; 
            margin-bottom: 20px;
        }
        
        .viewer-content { 
            flex: 1; 
            border: 1px solid #e9ecef; 
            border-radius: 8px; 
            background: #f8f9fa;
        }
        
        .viewer-iframe { 
            width: 100%; 
            height: 100%; 
            border: none; 
            border-radius: 8px;
        }
        
        .viewer-placeholder { 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            height: 100%; 
            color: #666; 
            font-size: 1.2em;
        }
        
        @media (max-width: 768px) {
            .container { margin: 0; }
            .header { padding: 20px; }
            .header h1 { font-size: 1.8em; }
            .header-stats { gap: 15px; }
            .tab-content { padding: 20px; }
            .reports-grid { grid-template-columns: 1fr; }
            .viewer-container { flex-direction: column; height: auto; }
            .viewer-sidebar { width: 100%; border-right: none; border-bottom: 1px solid #e9ecef; padding-bottom: 20px; }
            .viewer-content { height: 500px; }
        }
        """
        
        html_content = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📚 Study Material Analysis - Enhanced Index</title>
    <style>
        {css_content}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>📚 Study Material Analysis Dashboard</h1>
            <p>Zentrale Suchfunktion und Navigation für alle Videoinhalte</p>
            <div class="header-stats">
                {header_stats_html}
            </div>
        </header>
        
        <nav class="tab-navigation">
            <ul class="tab-list">
                <li><button class="tab-button active" onclick="showTab('search')">🔍 Suche</button></li>
                <li><button class="tab-button" onclick="showTab('overview')">📊 Übersicht</button></li>
                <li><button class="tab-button" onclick="showTab('viewer')">👁️ Report Viewer</button></li>
            </ul>
        </nav>
        
        <div id="search-tab" class="tab-content active">
            <div class="search-container">
                <div class="search-box">
                    <input type="text" id="search-input" class="search-input" 
                           placeholder="Durchsuche alle Transkripte..." 
                           onkeypress="handleSearchKeypress(event)">
                    <button class="search-button" onclick="performSearch()">
                        🔍 Suchen
                    </button>
                </div>
                
                <div class="search-filters">
                    <select id="report-filter" class="filter-select">
                        <option value="">Alle Reports</option>
                        {report_filter_options}
                    </select>
                    <select id="sort-filter" class="filter-select">
                        <option value="relevance">Nach Relevanz</option>
                        <option value="time">Nach Zeit</option>
                        <option value="confidence">Nach Zuverlässigkeit</option>
                    </select>
                    <select id="confidence-filter" class="filter-select">
                        <option value="">Alle Zuverlässigkeiten</option>
                        <option value="high">Hoch (&gt; -0.5)</option>
                        <option value="medium">Mittel (-1.0 bis -0.5)</option>
                        <option value="low">Niedrig (&lt; -1.0)</option>
                    </select>
                </div>
                
                <div id="search-results" class="search-results">
                    <div class="no-results">
                        Geben Sie einen Suchbegriff ein, um durch alle Transkripte zu suchen.
                    </div>
                </div>
            </div>
        </div>
        
        <div id="overview-tab" class="tab-content">
            <h2>📁 Alle Reports</h2>
            <div class="reports-grid">
                {report_cards_html}
            </div>
        </div>
        
        <div id="viewer-tab" class="tab-content">
            <div class="viewer-container">
                <div class="viewer-sidebar">
                    <h3>Report auswählen</h3>
                    <select id="report-selector" class="report-selector" onchange="loadReport()">
                        <option value="">-- Report auswählen --</option>
                        {report_selector_options}
                    </select>
                    
                    <div id="report-info" style="margin-top: 20px; font-size: 0.9em; color: #666;">
                        Wählen Sie einen Report aus, um ihn anzuzeigen.
                    </div>
                </div>
                
                <div class="viewer-content">
                    <div id="viewer-placeholder" class="viewer-placeholder">
                        📄 Wählen Sie einen Report aus der Liste
                    </div>
                    <iframe id="report-iframe" class="viewer-iframe" style="display: none;"></iframe>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const transcriptData = {transcripts_json};
        const reportsData = {reports_json};
        
        let currentSearchResults = [];
        let currentQuery = '';
        
        function showTab(tabName) {{
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab-button').forEach(button => {{
                button.classList.remove('active');
            }});
            
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }}
        
        function handleSearchKeypress(event) {{
            if (event.key === 'Enter') {{
                performSearch();
            }}
        }}
        
        function performSearch() {{
            const query = document.getElementById('search-input').value.trim();
            const reportFilter = document.getElementById('report-filter').value;
            const sortFilter = document.getElementById('sort-filter').value;
            const confidenceFilter = document.getElementById('confidence-filter').value;
            
            if (!query) {{
                document.getElementById('search-results').innerHTML = 
                    '<div class="no-results">Bitte geben Sie einen Suchbegriff ein.</div>';
                return;
            }}
            
            currentQuery = query.toLowerCase();
            
            let results = transcriptData.filter(segment => {{
                const textMatch = segment.text.toLowerCase().includes(currentQuery);
                const reportMatch = !reportFilter || segment.report_folder === reportFilter;
                
                let confidenceMatch = true;
                if (confidenceFilter) {{
                    const conf = segment.confidence;
                    switch(confidenceFilter) {{
                        case 'high': confidenceMatch = conf > -0.5; break;
                        case 'medium': confidenceMatch = conf >= -1.0 && conf <= -0.5; break;
                        case 'low': confidenceMatch = conf < -1.0; break;
                    }}
                }}
                
                return textMatch && reportMatch && confidenceMatch;
            }});
            
            switch(sortFilter) {{
                case 'time':
                    results.sort((a, b) => a.start - b.start);
                    break;
                case 'confidence':
                    results.sort((a, b) => b.confidence - a.confidence);
                    break;
                case 'relevance':
                default:
                    results.forEach(result => {{
                        const text = result.text.toLowerCase();
                        const queryWords = currentQuery.split(' ');
                        let score = 0;
                        
                        queryWords.forEach(word => {{
                            const count = (text.match(new RegExp(word, 'g')) || []).length;
                            score += count;
                        }});
                        
                        result.relevanceScore = score;
                    }});
                    results.sort((a, b) => b.relevanceScore - a.relevanceScore);
                    break;
            }}
            
            currentSearchResults = results;
            displaySearchResults(results);
        }}
        
        function displaySearchResults(results) {{
            const container = document.getElementById('search-results');
            
            if (results.length === 0) {{
                container.innerHTML = '<div class="no-results">Keine Ergebnisse gefunden.</div>';
                return;
            }}
            
            const resultsHtml = results.slice(0, 50).map(result => {{
                const highlightedText = highlightSearchTerm(result.text, currentQuery);
                const confidenceClass = getConfidenceClass(result.confidence);
                const reportInfo = reportsData.find(r => r.folder === result.report_folder);
                const timelineLink = reportInfo ? reportInfo.timeline_path : '#';
                
                return `
                    <div class="search-result">
                        <div class="result-header">
                            <a href="${{timelineLink}}#segment-${{result.segment_id}}" class="result-title" target="_blank">
                                ${{result.report_name}}
                            </a>
                            <span class="result-time">${{result.timestamp}}</span>
                        </div>
                        <div class="result-text">${{highlightedText}}</div>
                        <div class="result-meta">
                            <span>Segment ${{result.segment_id + 1}}</span>
                            <span class="confidence ${{confidenceClass}}">
                                Zuverlässigkeit: ${{Math.round((result.confidence + 2) * 50)}}%
                            </span>
                        </div>
                    </div>
                `;
            }}).join('');
            
            const summary = results.length > 50 ? 
                `<p><strong>${{results.length}} Ergebnisse gefunden</strong> (Erste 50 angezeigt)</p>` :
                `<p><strong>${{results.length}} Ergebnisse gefunden</strong></p>`;
            
            container.innerHTML = summary + resultsHtml;
        }}
        
        function highlightSearchTerm(text, query) {{
            if (!query) return text;
            
            const words = query.split(' ').filter(word => word.length > 0);
            let highlightedText = text;
            
            words.forEach(word => {{
                const regex = new RegExp(`(${{word}})`, 'gi');
                highlightedText = highlightedText.replace(regex, '<span class="highlight">$1</span>');
            }});
            
            return highlightedText;
        }}
        
        function getConfidenceClass(confidence) {{
            if (confidence > -0.5) return 'high-confidence';
            if (confidence > -1.0) return 'medium-confidence';
            return 'low-confidence';
        }}
        
        function loadReport() {{
            const selector = document.getElementById('report-selector');
            const iframe = document.getElementById('report-iframe');
            const placeholder = document.getElementById('viewer-placeholder');
            const reportInfo = document.getElementById('report-info');
            
            if (!selector.value) {{
                iframe.style.display = 'none';
                placeholder.style.display = 'flex';
                reportInfo.innerHTML = 'Wählen Sie einen Report aus, um ihn anzuzeigen.';
                return;
            }}
            
            placeholder.style.display = 'none';
            iframe.style.display = 'block';
            iframe.src = selector.value;
            
            const selectedReport = reportsData.find(r => 
                selector.value.includes(r.folder)
            );
            
            if (selectedReport) {{
                reportInfo.innerHTML = `
                    <strong>${{selectedReport.filename}}</strong><br>
                    🕒 Dauer: ${{Math.round(selectedReport.duration)}} min<br>
                    📝 Segmente: ${{selectedReport.segments_count}}<br>
                    🖼️ Screenshots: ${{selectedReport.screenshots_count}}<br>
                    📄 PDFs: ${{selectedReport.pdfs_count}}
                `;
            }}
        }}
        
        document.addEventListener('DOMContentLoaded', function() {{
            document.getElementById('search-input').focus();
        }});
    </script>
</body>
</html>"""
        
        return html_content
    
    def _generate_report_card(self, report: Dict[str, Any]) -> str:
        """Generate HTML for a single report card"""
        timeline_link = report['timeline_path'] if report['has_timeline_report'] else '#'
        timeline_disabled = '' if report['has_timeline_report'] else 'style="opacity: 0.5; pointer-events: none;"'
        
        return f"""
        <div class="report-card">
            <div class="report-title">{report['filename']}</div>
            <div class="report-actions">
                <a href="{timeline_link}" class="action-button timeline-button" {timeline_disabled} target="_blank">
                    🎬 Timeline Report
                </a>
                <a href="{report['standard_path']}" class="action-button standard-button" target="_blank">
                    📄 Standard Report
                </a>
            </div>
            <div class="report-stats">
                <div class="stat-item">
                    <span class="stat-label">🕒 Dauer:</span>
                    <span class="stat-value">{self._format_timestamp(report['duration'] * 60)}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">📝 Segmente:</span>
                    <span class="stat-value">{report['segments_count']:,}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">🖼️ Screenshots:</span>
                    <span class="stat-value">{report['screenshots_count']}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">📄 PDFs:</span>
                    <span class="stat-value">{report['pdfs_count']}</span>
                </div>
            </div>
        </div>
        """
    
    def generate_index(self, output_path: Path = None):
        """Generate the enhanced index HTML file"""
        if not output_path:
            output_path = self.results_dir / 'enhanced_index.html'
        
        self.logger.info("Generating enhanced index...")
        
        self.collect_transcript_data()
        html_content = self.generate_html()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Enhanced index generated successfully: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing enhanced index: {e}")
            return False

def main():
    """Main function"""
    generator = EnhancedIndexGenerator()
    success = generator.generate_index()
    
    if success:
        print("✅ Enhanced index generated successfully!")
        print("📍 Location: results/enhanced_index.html")
        print("🔍 Features: Search, Overview, Report Viewer")
    else:
        print("❌ Failed to generate enhanced index")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
