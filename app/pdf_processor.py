"""
PDF Processing Module for GovInsight

This module handles PDF loading, text extraction, and cleaning
for Indian government budget documents.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import fitz  # PyMuPDF


class PDFProcessor:
    """
    Handles PDF text extraction and cleaning for budget documents.
    """
    
    def __init__(self, pdf_directory: str = "data/raw_pdfs"):
        """
        Initialize PDF processor.
        
        Args:
            pdf_directory: Path to directory containing PDF files
        """
        self.pdf_directory = Path(pdf_directory)
        if not self.pdf_directory.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")
    
    def load_pdfs(self) -> List[Path]:
        """
        Load all PDF files from the specified directory.
        
        Returns:
            List of PDF file paths
        """
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.pdf_directory}")
        return sorted(pdf_files)
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, any]]:
        """
        Extract text from a PDF file page by page.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries containing page text and metadata:
            [
                {
                    'page_number': int,
                    'text': str,
                    'document_name': str
                }
            ]
        """
        pages_data = []
        document_name = pdf_path.stem
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                
                # Clean the extracted text
                cleaned_text = self._clean_text(text)
                
                if cleaned_text.strip():  # Only include pages with content
                    pages_data.append({
                        'page_number': page_num + 1,
                        'text': cleaned_text,
                        'document_name': document_name
                    })
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return []
        
        return pages_data
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing headers, footers, and noise.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common header/footer patterns
        text = self._remove_headers_footers(text)
        
        # Remove page numbers (standalone numbers)
        text = re.sub(r'\b\d{1,3}\b\s*$', '', text)
        text = re.sub(r'^\s*\d{1,3}\b', '', text)
        
        # Remove common noise patterns
        text = re.sub(r'www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'\|{2,}', '', text)  # Remove multiple pipes
        text = re.sub(r'-{3,}', '', text)  # Remove multiple dashes
        text = re.sub(r'_{3,}', '', text)  # Remove multiple underscores
        
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _remove_headers_footers(self, text: str) -> str:
        """
        Remove common header and footer patterns from budget documents.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with headers/footers removed
        """
        # Common header/footer patterns in Indian budget documents
        patterns = [
            r'Ministry of Finance.*?Department of Economic Affairs',
            r'Government of India.*?Budget',
            r'Union Budget \d{4}-\d{2,4}',
            r'Printed by.*?Controller of Publications',
            r'Page \d+ of \d+',
            r'^\s*\d+\s*$',  # Standalone page numbers
            r'Source:.*?(?=\n|$)',
            r'Note:.*?(?=\n|$)',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """
        Extract metadata (year, ministry, type) from PDF filename.
        
        Expected filename patterns:
        - Budget_2023-24_Expenditure.pdf
        - MoRTH_Annual_Report_2022-23.pdf
        - Demands_for_Grants_2024-25_Ministry_Name.pdf
        
        Args:
            filename: PDF filename (without path)
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            'year': None,
            'ministry': None,
            'document_type': None
        }
        
        # Extract year (format: 2023-24 or 2023)
        year_match = re.search(r'(\d{4})-?(\d{2,4})?', filename)
        if year_match:
            metadata['year'] = year_match.group(1) + '-' + year_match.group(2) if year_match.group(2) else year_match.group(1)
        
        # Extract ministry
        ministry_patterns = [
            r'MoRTH', r'Ministry.*?Transport', r'Road.*?Transport',
            r'Housing', r'Urban.*?Affairs', r'Finance',
            r'NITI.*?Aayog', r'Planning.*?Commission'
        ]
        for pattern in ministry_patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                metadata['ministry'] = match.group(0)
                break
        
        # Extract document type
        if 'expenditure' in filename.lower():
            metadata['document_type'] = 'Expenditure Budget'
        elif 'demand' in filename.lower():
            metadata['document_type'] = 'Demands for Grants'
        elif 'glance' in filename.lower():
            metadata['document_type'] = 'Budget at a Glance'
        elif 'annual' in filename.lower():
            metadata['document_type'] = 'Annual Report'
        
        return metadata
    
    def process_all_pdfs(self) -> List[Dict[str, any]]:
        """
        Process all PDFs in the directory and extract text with metadata.
        
        Returns:
            List of dictionaries containing text and metadata for all pages:
            [
                {
                    'text': str,
                    'page_number': int,
                    'document_name': str,
                    'year': str,
                    'ministry': str,
                    'document_type': str
                }
            ]
        """
        all_pages = []
        pdf_files = self.load_pdfs()
        
        print(f"Processing {len(pdf_files)} PDF files...")
        
        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path.name}")
            
            # Extract text from PDF
            pages_data = self.extract_text_from_pdf(pdf_path)
            
            # Extract metadata from filename
            file_metadata = self.extract_metadata_from_filename(pdf_path.name)
            
            # Combine text and metadata
            for page_data in pages_data:
                page_data.update(file_metadata)
                all_pages.append(page_data)
        
        print(f"Extracted {len(all_pages)} pages from {len(pdf_files)} documents")
        
        return all_pages


def extract_text_from_single_pdf(pdf_path: str) -> List[Dict[str, any]]:
    """
    Utility function to extract text from a single PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of page dictionaries with text and metadata
    """
    processor = PDFProcessor()
    return processor.extract_text_from_pdf(Path(pdf_path))

