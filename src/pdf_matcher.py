"""
PDF matching and content extraction module
"""

import os
import re
import logging
from typing import Dict, List, Optional
from pathlib import Path
from glob import glob
import PyPDF2
from datetime import datetime

from .config import DEFAULT_CONFIG
from .utils import sanitize_filename

logger = logging.getLogger(__name__)

class PDFMatcher:
    """
    Match and extract content from PDF files related to videos.
    """
    
    def __init__(self, studies_dir: str, config: Optional[Dict] = None):
        """
        Initialize the PDF matcher.
        
        Args:
            studies_dir: Directory containing PDF files
            config: Additional configuration options
        """
        self.studies_dir = Path(studies_dir)
        
        # Merge config with defaults
        self.config = DEFAULT_CONFIG['pdf_matching'].copy()
        if config:
            self.config.update(config)
        
        logger.info(f"Initialized PDFMatcher for directory: {studies_dir}")
    
    def find_related_pdfs(self, video_filename: str) -> List[Dict]:
        """
        Find PDFs related to video by various matching strategies.
        
        Args:
            video_filename: Name or path of the video file
            
        Returns:
            List of related PDF information dictionaries, sorted by relevance
        """
        video_path = Path(video_filename)
        video_name = video_path.stem
        
        logger.info(f"Finding PDFs related to video: {video_name}")
        
        # Extract various identifiers from video name
        identifiers = self._extract_identifiers(video_name)
        
        # Find all PDFs in the studies directory
        pdf_files = list(self.studies_dir.glob("*.pdf"))
        if not pdf_files:
            logger.info("No PDF files found in studies directory")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to analyze")
        
        # Score each PDF for relevance
        scored_pdfs = []
        for pdf_path in pdf_files:
            score = self._calculate_relevance_score(pdf_path, identifiers)
            if score > 0:
                pdf_info = self._create_pdf_info(pdf_path, score)
                scored_pdfs.append(pdf_info)
        
        # Sort by relevance score (highest first)
        scored_pdfs.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        logger.info(f"Found {len(scored_pdfs)} PDFs with relevance to {video_name}")
        return scored_pdfs
    
    def _extract_identifiers(self, video_name: str) -> Dict:
        """
        Extract various identifiers from video filename.
        
        Args:
            video_name: Video filename (without extension)
            
        Returns:
            Dictionary of extracted identifiers
        """
        identifiers = {
            "dates": [],
            "keywords": [],
            "numbers": [],
            "original_name": video_name.lower()
        }
        
        # Extract dates in various formats
        date_patterns = [
            r'(\d{2})\.(\d{2})\.(\d{4})',  # DD.MM.YYYY
            r'(\d{4})\.(\d{2})\.(\d{2})',  # YYYY.MM.DD
            r'(\d{4})(\d{2})(\d{2})',      # YYYYMMDD
            r'(\d{2})(\d{2})(\d{4})',      # DDMMYYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, video_name)
            for match in matches:
                # Normalize to YYYY-MM-DD format
                if len(match[0]) == 4:  # Year first
                    year, month, day = match
                else:  # Day first
                    day, month, year = match
                
                try:
                    # Validate date
                    datetime(int(year), int(month), int(day))
                    normalized_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    identifiers["dates"].append(normalized_date)
                    
                    # Also add alternative formats
                    identifiers["dates"].append(f"{year}{month.zfill(2)}{day.zfill(2)}")
                    identifiers["dates"].append(f"{day.zfill(2)}{month.zfill(2)}{year}")
                except ValueError:
                    continue
        
        # Extract keywords (alphanumeric sequences longer than 2 chars)
        keywords = re.findall(r'\b\w{3,}\b', video_name.lower())
        identifiers["keywords"] = list(set(keywords))
        
        # Extract numbers
        numbers = re.findall(r'\d+', video_name)
        identifiers["numbers"] = numbers
        
        logger.debug(f"Extracted identifiers: {identifiers}")
        return identifiers
    
    def _calculate_relevance_score(self, pdf_path: Path, identifiers: Dict) -> int:
        """
        Calculate relevance score between PDF and video identifiers.
        
        Args:
            pdf_path: Path to PDF file
            identifiers: Video identifiers dictionary
            
        Returns:
            Relevance score (higher = more relevant)
        """
        pdf_name = pdf_path.stem.lower()
        score = 0
        
        # Date matching (highest priority)
        for date in identifiers["dates"]:
            if date.replace("-", "") in pdf_name:
                score += 20
                logger.debug(f"Date match: {date} in {pdf_name}")
            elif date[:7].replace("-", "") in pdf_name:  # Year-month match
                score += 10
                logger.debug(f"Year-month match: {date[:7]} in {pdf_name}")
        
        # Keyword matching
        pdf_keywords = set(re.findall(r'\b\w{3,}\b', pdf_name))
        common_keywords = set(identifiers["keywords"]).intersection(pdf_keywords)
        score += len(common_keywords) * 3
        
        if common_keywords:
            logger.debug(f"Keyword matches: {common_keywords}")
        
        # Number matching
        pdf_numbers = set(re.findall(r'\d+', pdf_name))
        common_numbers = set(identifiers["numbers"]).intersection(pdf_numbers)
        score += len(common_numbers) * 2
        
        # Partial string matching
        video_parts = identifiers["original_name"].split()
        for part in video_parts:
            if len(part) > 3 and part in pdf_name:
                score += 1
        
        return score
    
    def _create_pdf_info(self, pdf_path: Path, relevance_score: int) -> Dict:
        """
        Create comprehensive PDF information dictionary.
        
        Args:
            pdf_path: Path to PDF file
            relevance_score: Calculated relevance score
            
        Returns:
            PDF information dictionary
        """
        pdf_info = {
            "filepath": str(pdf_path),
            "filename": pdf_path.name,
            "relevance_score": relevance_score,
            "file_size_bytes": pdf_path.stat().st_size,
            "modification_time": pdf_path.stat().st_mtime,
            "content_preview": "",
            "page_count": 0,
            "extraction_error": None
        }
        
        # Extract content preview and metadata
        try:
            preview_info = self.extract_pdf_preview(str(pdf_path))
            pdf_info.update(preview_info)
        except Exception as e:
            pdf_info["extraction_error"] = str(e)
            logger.warning(f"Could not extract content from {pdf_path.name}: {e}")
        
        return pdf_info
    
    def extract_pdf_preview(self, pdf_path: str) -> Dict:
        """
        Extract preview text and metadata from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted content and metadata
        """
        result = {
            "content_preview": "Preview not available",
            "page_count": 0,
            "title": "",
            "author": "",
            "subject": "",
            "creator": "",
            "creation_date": None
        }
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                result["page_count"] = len(pdf_reader.pages)
                
                # Extract metadata
                if pdf_reader.metadata:
                    result["title"] = pdf_reader.metadata.get('/Title', '')
                    result["author"] = pdf_reader.metadata.get('/Author', '')
                    result["subject"] = pdf_reader.metadata.get('/Subject', '')
                    result["creator"] = pdf_reader.metadata.get('/Creator', '')
                    
                    # Handle creation date
                    creation_date = pdf_reader.metadata.get('/CreationDate')
                    if creation_date:
                        try:
                            # PDF dates are in format D:YYYYMMDDHHmmSSOHH'mm'
                            if creation_date.startswith('D:'):
                                date_str = creation_date[2:16]  # Extract YYYYMMDDHHMMSS
                                result["creation_date"] = datetime.strptime(
                                    date_str, '%Y%m%d%H%M%S'
                                ).isoformat()
                        except (ValueError, TypeError):
                            pass
                
                # Extract text from first few pages
                text_parts = []
                max_pages = min(self.config['max_pages_preview'], len(pdf_reader.pages))
                
                for page_num in range(max_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        if text.strip():
                            text_parts.append(text.strip())
                    except Exception as e:
                        logger.debug(f"Could not extract text from page {page_num}: {e}")
                        continue
                
                # Combine and truncate text
                full_text = '\n\n'.join(text_parts)
                if len(full_text) > self.config['max_preview_chars']:
                    result["content_preview"] = full_text[:self.config['max_preview_chars']] + "..."
                else:
                    result["content_preview"] = full_text
                
                # Clean up whitespace
                result["content_preview"] = re.sub(r'\s+', ' ', result["content_preview"]).strip()
                
        except Exception as e:
            logger.error(f"Error extracting PDF content from {pdf_path}: {e}")
            raise
        
        return result
    
    def search_pdf_content(self, pdf_path: str, search_terms: List[str]) -> Dict:
        """
        Search for specific terms within PDF content.
        
        Args:
            pdf_path: Path to PDF file
            search_terms: List of terms to search for
            
        Returns:
            Search results dictionary
        """
        results = {
            "pdf_path": pdf_path,
            "search_terms": search_terms,
            "matches": [],
            "total_matches": 0
        }
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text().lower()
                        
                        for term in search_terms:
                            term_lower = term.lower()
                            if term_lower in text:
                                # Find all occurrences with context
                                start = 0
                                while True:
                                    index = text.find(term_lower, start)
                                    if index == -1:
                                        break
                                    
                                    # Extract context (50 chars before and after)
                                    context_start = max(0, index - 50)
                                    context_end = min(len(text), index + len(term_lower) + 50)
                                    context = text[context_start:context_end]
                                    
                                    results["matches"].append({
                                        "term": term,
                                        "page": page_num + 1,
                                        "context": context.strip(),
                                        "position": index
                                    })
                                    
                                    start = index + 1
                        
                    except Exception as e:
                        logger.debug(f"Could not search page {page_num}: {e}")
                        continue
                
                results["total_matches"] = len(results["matches"])
                
        except Exception as e:
            logger.error(f"Error searching PDF {pdf_path}: {e}")
            results["error"] = str(e)
        
        return results
    
    def get_pdf_statistics(self) -> Dict:
        """
        Get statistics about PDFs in the studies directory.
        
        Returns:
            Statistics dictionary
        """
        pdf_files = list(self.studies_dir.glob("*.pdf"))
        
        stats = {
            "total_pdfs": len(pdf_files),
            "total_size_bytes": 0,
            "accessible_pdfs": 0,
            "page_count_total": 0,
            "creation_dates": [],
            "file_sizes": []
        }
        
        for pdf_path in pdf_files:
            stats["total_size_bytes"] += pdf_path.stat().st_size
            stats["file_sizes"].append(pdf_path.stat().st_size)
            
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    stats["accessible_pdfs"] += 1
                    stats["page_count_total"] += len(pdf_reader.pages)
                    
                    # Extract creation date
                    if pdf_reader.metadata and '/CreationDate' in pdf_reader.metadata:
                        creation_date = pdf_reader.metadata['/CreationDate']
                        if creation_date.startswith('D:'):
                            try:
                                date_str = creation_date[2:16]
                                date_obj = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                                stats["creation_dates"].append(date_obj.isoformat())
                            except ValueError:
                                pass
                            
            except Exception as e:
                logger.debug(f"Could not read PDF {pdf_path.name}: {e}")
                continue
        
        # Calculate averages
        if stats["accessible_pdfs"] > 0:
            stats["average_pages"] = stats["page_count_total"] / stats["accessible_pdfs"]
            stats["average_size_bytes"] = sum(stats["file_sizes"]) / len(stats["file_sizes"])
        else:
            stats["average_pages"] = 0
            stats["average_size_bytes"] = 0
        
        return stats