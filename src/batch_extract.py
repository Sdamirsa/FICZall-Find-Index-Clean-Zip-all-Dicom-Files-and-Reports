#!/usr/bin/env python3
"""
batch_extract.py - Batch extract text from documents with OCR fallback

This script processes multiple documents from a folder or Excel file,
extracting text using native methods and falling back to OCR when needed.

Usage:
    # Process all files in a folder
    python batch_extract.py --input-folder /path/to/files --output output.xlsx

    # Process files from S7 manual extraction package
    python batch_extract.py --excel /path/to/reports_for_extraction.xlsx

    # Process with GPU acceleration
    python batch_extract.py --input-folder /path/to/files --gpu

Author: FICZall Pipeline
Version: 1.0
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_extractor import DocumentExtractor, ExtractionResult, ExtractionStatus

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchExtractor:
    """Batch process documents for text extraction."""

    def __init__(
        self,
        languages: List[str] = None,
        use_gpu: bool = False,
        max_workers: int = 1,
        progress_callback=None
    ):
        """
        Initialize batch extractor.

        Args:
            languages: OCR languages (default: ['en', 'fa'])
            use_gpu: Use GPU for OCR
            max_workers: Number of parallel workers (1 for OCR due to memory)
            progress_callback: Function called with (current, total, filename)
        """
        self.languages = languages or ['en', 'fa']
        self.use_gpu = use_gpu
        self.max_workers = max_workers
        self.progress_callback = progress_callback

        # Create extractor (shared for single-threaded, per-thread for multi)
        self._extractor = None
        self._lock = threading.Lock()

    def get_extractor(self) -> DocumentExtractor:
        """Get or create document extractor."""
        if self._extractor is None:
            with self._lock:
                if self._extractor is None:
                    self._extractor = DocumentExtractor(
                        languages=self.languages,
                        use_gpu=self.use_gpu
                    )
        return self._extractor

    def process_folder(
        self,
        folder_path: Path,
        output_path: Path = None,
        file_extensions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Process all documents in a folder.

        Args:
            folder_path: Path to folder containing documents
            output_path: Path to save results (Excel or JSON)
            file_extensions: Filter by extensions (default: all supported)

        Returns:
            Dictionary with results and statistics
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")

        # Get list of files
        all_extensions = []
        for exts in DocumentExtractor.SUPPORTED_EXTENSIONS.values():
            all_extensions.extend(exts)

        if file_extensions:
            all_extensions = [e.lower() for e in file_extensions]

        files = []
        for ext in all_extensions:
            files.extend(folder_path.glob(f"*{ext}"))
            files.extend(folder_path.glob(f"*{ext.upper()}"))

        files = sorted(set(files))
        logger.info(f"Found {len(files)} files to process")

        # Process files
        results = self._process_files(files)

        # Save results
        if output_path:
            self._save_results(results, output_path)

        return results

    def process_excel(
        self,
        excel_path: Path,
        file_column: str = '_copied_filename',
        files_folder: str = 'files_to_extract',
        output_path: Path = None
    ) -> Dict[str, Any]:
        """
        Process files listed in Excel from S7 manual extraction package.

        Args:
            excel_path: Path to Excel file
            file_column: Column containing filenames
            files_folder: Subfolder containing the files
            output_path: Path to save updated Excel

        Returns:
            Dictionary with results and statistics
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required. Install with: pip install pandas openpyxl")

        excel_path = Path(excel_path)
        if not excel_path.exists():
            raise ValueError(f"Excel file not found: {excel_path}")

        # Determine files folder location
        files_dir = excel_path.parent / files_folder
        if not files_dir.exists():
            # Try current directory
            files_dir = Path(files_folder)
            if not files_dir.exists():
                raise ValueError(f"Files folder not found: {files_folder}")

        # Read Excel
        df = pd.read_excel(excel_path)
        logger.info(f"Loaded {len(df)} rows from Excel")

        if file_column not in df.columns:
            raise ValueError(f"Column '{file_column}' not found in Excel")

        # Get list of files
        files = []
        row_mapping = {}  # Map file path to row index

        for idx, row in df.iterrows():
            filename = row[file_column]
            if pd.notna(filename) and str(filename).strip():
                file_path = files_dir / str(filename).strip()
                if file_path.exists():
                    files.append(file_path)
                    row_mapping[str(file_path)] = idx
                else:
                    logger.warning(f"File not found: {file_path}")

        logger.info(f"Found {len(files)} files to process")

        # Process files
        results = self._process_files(files)

        # Update DataFrame with results
        if '_extracted_text' not in df.columns:
            df['_extracted_text'] = ''
        if '_extraction_status' not in df.columns:
            df['_extraction_status'] = ''
        if '_extraction_confidence' not in df.columns:
            df['_extraction_confidence'] = ''
        if '_extraction_method' not in df.columns:
            df['_extraction_method'] = ''

        for file_result in results['files']:
            file_path = file_result['file_path']
            if file_path in row_mapping:
                idx = row_mapping[file_path]
                df.loc[idx, '_extracted_text'] = file_result['text']
                df.loc[idx, '_extraction_status'] = file_result['status']
                df.loc[idx, '_extraction_method'] = file_result['method']
                df.loc[idx, '_extraction_confidence'] = file_result['confidence']

        # Save updated Excel
        if output_path is None:
            # Create new filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = excel_path.parent / f"extracted_{timestamp}.xlsx"

        df.to_excel(output_path, index=False)
        logger.info(f"Saved results to: {output_path}")

        results['output_file'] = str(output_path)
        return results

    def _process_files(self, files: List[Path]) -> Dict[str, Any]:
        """Process a list of files."""
        results = {
            'total_files': len(files),
            'successful': 0,
            'failed': 0,
            'ocr_used': 0,
            'empty': 0,
            'files': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }

        extractor = self.get_extractor()

        for i, file_path in enumerate(files):
            if self.progress_callback:
                self.progress_callback(i + 1, len(files), file_path.name)

            logger.info(f"Processing [{i+1}/{len(files)}]: {file_path.name}")

            try:
                result = extractor.extract(file_path)

                file_result = {
                    'file_path': str(file_path),
                    'filename': file_path.name,
                    'text': result.text,
                    'status': result.status.value,
                    'method': result.method.value,
                    'confidence': result.confidence,
                    'error': result.error_message
                }

                results['files'].append(file_result)

                if result.status in [ExtractionStatus.SUCCESS, ExtractionStatus.OCR_FALLBACK]:
                    results['successful'] += 1
                    if result.status == ExtractionStatus.OCR_FALLBACK:
                        results['ocr_used'] += 1
                elif result.status == ExtractionStatus.EMPTY_DOCUMENT:
                    results['empty'] += 1
                else:
                    results['failed'] += 1

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results['files'].append({
                    'file_path': str(file_path),
                    'filename': file_path.name,
                    'text': '',
                    'status': 'error',
                    'method': 'failed',
                    'confidence': 0,
                    'error': str(e)
                })
                results['failed'] += 1

        results['end_time'] = datetime.now().isoformat()
        return results

    def _save_results(self, results: Dict[str, Any], output_path: Path):
        """Save results to file."""
        output_path = Path(output_path)

        if output_path.suffix.lower() in ['.xlsx', '.xls']:
            self._save_excel(results, output_path)
        elif output_path.suffix.lower() == '.json':
            self._save_json(results, output_path)
        else:
            # Default to JSON
            self._save_json(results, output_path.with_suffix('.json'))

    def _save_excel(self, results: Dict[str, Any], output_path: Path):
        """Save results to Excel."""
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not installed, saving as JSON instead")
            self._save_json(results, output_path.with_suffix('.json'))
            return

        df = pd.DataFrame(results['files'])
        df.to_excel(output_path, index=False)
        logger.info(f"Saved Excel results to: {output_path}")

    def _save_json(self, results: Dict[str, Any], output_path: Path):
        """Save results to JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON results to: {output_path}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Batch extract text from documents with OCR fallback',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process folder
    python batch_extract.py --input-folder ./documents --output results.xlsx

    # Process S7 manual extraction package
    python batch_extract.py --excel ./reports_for_extraction.xlsx

    # Use GPU for faster OCR
    python batch_extract.py --input-folder ./documents --gpu

    # Specify languages
    python batch_extract.py --input-folder ./documents --languages en fa ar
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-folder', '-i',
        type=str,
        help='Folder containing documents to process'
    )
    input_group.add_argument(
        '--excel', '-e',
        type=str,
        help='Excel file from S7 manual extraction package'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (Excel or JSON)'
    )

    parser.add_argument(
        '--languages', '-l',
        nargs='+',
        default=['en', 'fa'],
        help='OCR languages (default: en fa for English and Persian)'
    )

    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for OCR (requires CUDA)'
    )

    parser.add_argument(
        '--file-column',
        type=str,
        default='_copied_filename',
        help='Column name containing filenames in Excel'
    )

    parser.add_argument(
        '--files-folder',
        type=str,
        default='files_to_extract',
        help='Subfolder containing files (for Excel mode)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create batch extractor
    def progress_callback(current, total, filename):
        print(f"\rProgress: {current}/{total} - {filename[:50]}...", end='', flush=True)

    extractor = BatchExtractor(
        languages=args.languages,
        use_gpu=args.gpu,
        progress_callback=progress_callback
    )

    try:
        if args.input_folder:
            output_path = args.output or 'extraction_results.xlsx'
            results = extractor.process_folder(
                Path(args.input_folder),
                Path(output_path)
            )
        else:
            results = extractor.process_excel(
                Path(args.excel),
                file_column=args.file_column,
                files_folder=args.files_folder,
                output_path=Path(args.output) if args.output else None
            )

        print()  # New line after progress
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Total files: {results['total_files']}")
        print(f"Successful: {results['successful']}")
        print(f"OCR used: {results['ocr_used']}")
        print(f"Empty documents: {results['empty']}")
        print(f"Failed: {results['failed']}")
        if 'output_file' in results:
            print(f"Output: {results['output_file']}")
        print("=" * 60)

        return 0 if results['failed'] == 0 else 1

    except Exception as e:
        logger.error(f"Batch extraction failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
