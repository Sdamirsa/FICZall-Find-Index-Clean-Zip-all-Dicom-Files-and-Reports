# FICZall Source Utilities
"""
src - Utility modules for FICZall DICOM Processing Pipeline

Modules:
- document_extractor: Extract text from DOCX, PDF with OCR fallback
"""

from .document_extractor import DocumentExtractor, extract_text_from_file

__all__ = ['DocumentExtractor', 'extract_text_from_file']
