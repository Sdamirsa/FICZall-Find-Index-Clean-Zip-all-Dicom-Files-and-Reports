#!/usr/bin/env python3
"""
S0_zipExtract.py - DICOM Pipeline Stage 0: Fast ZIP File Extraction

This script extracts ZIP files found in a directory tree, with smart logic to avoid
re-extracting already processed files. Designed to be fast and efficient for large
numbers of ZIP files.

Features:
- Recursive ZIP file discovery
- Smart extraction checking (avoids re-extraction)
- Fast concurrent processing
- Progress tracking and resume capability
- Comprehensive logging
- Safe extraction with validation
- Preserves directory structure

Author: Claude Code Assistant  
Version: 1.0
"""

import os
import sys
import json
import logging
import argparse
import zipfile
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import traceback

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not found. Progress bars will be disabled.")
    tqdm = None


@dataclass
class ExtractionStats:
    """Statistics for ZIP extraction process"""
    total_zip_files: int = 0
    processed_files: int = 0
    extracted_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class ZipFileInfo:
    """Information about a ZIP file"""
    zip_path: Path
    extract_path: Path
    size_bytes: int
    modification_time: float
    checksum: Optional[str] = None
    is_extracted: bool = False
    extraction_time: Optional[float] = None


class S0ZipExtractor:
    """Main ZIP file extractor with smart logic and fast processing"""
    
    def __init__(self, 
                 root_dir: str, 
                 max_workers: int = 4,
                 check_integrity: bool = True,
                 overwrite: bool = False):
        self.root_dir = Path(root_dir).resolve()
        self.max_workers = max_workers
        self.check_integrity = check_integrity
        self.overwrite = overwrite
        self.stats = ExtractionStats()
        
        # Setup logging
        self.setup_logging()
        
        # Thread lock for progress updates
        self.progress_lock = threading.Lock()
        
        # Progress tracking
        self.progress_file = self.root_dir / "S0_zipExtract_progress.json"
        self.extracted_record = self.root_dir / "S0_extracted_files.json"
        
        # Load existing progress
        self.processed_files: Set[str] = set()
        self.extraction_records: Dict[str, dict] = {}
        self.load_progress()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.root_dir / "S0_zipExtract.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger('S0_zipExtract')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def load_progress(self):
        """Load processing progress from previous runs"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    self.processed_files = set(progress_data.get('processed_files', []))
                    self.logger.info(f"Loaded progress: {len(self.processed_files)} files already processed")
            
            if self.extracted_record.exists():
                with open(self.extracted_record, 'r', encoding='utf-8') as f:
                    self.extraction_records = json.load(f)
                    
        except Exception as e:
            self.logger.warning(f"Could not load progress files: {e}")
            self.processed_files = set()
            self.extraction_records = {}
    
    def save_progress(self):
        """Save current processing progress"""
        try:
            # Save processing progress
            progress_data = {
                'processed_files': list(self.processed_files),
                'last_updated': time.time(),
                'stats': {
                    'total_zip_files': self.stats.total_zip_files,
                    'processed_files': self.stats.processed_files,
                    'extracted_files': self.stats.extracted_files,
                    'skipped_files': self.stats.skipped_files,
                    'failed_files': self.stats.failed_files
                }
            }
            
            # Atomic write
            temp_file = self.progress_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
            temp_file.rename(self.progress_file)
            
            # Save extraction records
            temp_file = self.extracted_record.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.extraction_records, f, indent=2, ensure_ascii=False)
            temp_file.rename(self.extracted_record)
            
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")
    
    def calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file for integrity checking"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""
    
    def find_zip_files(self) -> List[ZipFileInfo]:
        """Find all ZIP files in the directory tree"""
        zip_files = []
        
        self.logger.info(f"Scanning for ZIP files in {self.root_dir}...")
        
        for zip_path in self.root_dir.rglob("*.zip"):
            if zip_path.is_file():
                try:
                    # Determine extraction path (remove .zip extension)
                    extract_path = zip_path.parent / zip_path.stem
                    
                    # Get file info
                    stat = zip_path.stat()
                    
                    zip_info = ZipFileInfo(
                        zip_path=zip_path,
                        extract_path=extract_path,
                        size_bytes=stat.st_size,
                        modification_time=stat.st_mtime
                    )
                    
                    zip_files.append(zip_info)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {zip_path}: {e}")
        
        self.logger.info(f"Found {len(zip_files)} ZIP files")
        return zip_files
    
    def is_already_extracted(self, zip_info: ZipFileInfo) -> bool:
        """Smart logic to check if ZIP file is already extracted"""
        zip_path_str = str(zip_info.zip_path)
        
        # Check if we have a record of this extraction
        if zip_path_str in self.extraction_records:
            record = self.extraction_records[zip_path_str]
            
            # Check if extraction directory exists
            if not zip_info.extract_path.exists():
                self.logger.debug(f"Extraction directory missing for {zip_path_str}")
                return False
            
            # Check if the ZIP file has been modified since extraction
            if record.get('zip_modification_time', 0) < zip_info.modification_time:
                self.logger.debug(f"ZIP file modified since extraction: {zip_path_str}")
                return False
            
            # Check if extraction was successful
            if not record.get('extraction_successful', False):
                self.logger.debug(f"Previous extraction failed: {zip_path_str}")
                return False
            
            # If integrity checking is enabled, verify checksum
            if self.check_integrity and record.get('zip_checksum'):
                current_checksum = self.calculate_file_checksum(zip_info.zip_path)
                if current_checksum != record.get('zip_checksum'):
                    self.logger.debug(f"Checksum mismatch for {zip_path_str}")
                    return False
            
            # Check if extracted files still exist
            extracted_files = record.get('extracted_files', [])
            if extracted_files:
                # Sample check - verify a few files exist
                sample_files = extracted_files[:min(5, len(extracted_files))]
                for rel_path in sample_files:
                    full_path = zip_info.extract_path / rel_path
                    if not full_path.exists():
                        self.logger.debug(f"Extracted file missing: {full_path}")
                        return False
            
            self.logger.debug(f"ZIP already properly extracted: {zip_path_str}")
            return True
        
        # No record found, check if directory exists and has content
        if zip_info.extract_path.exists() and any(zip_info.extract_path.iterdir()):
            # Directory exists with content but no record - could be from manual extraction
            if not self.overwrite:
                self.logger.info(f"Directory exists but no extraction record: {zip_path_str}")
                return True
        
        return False
    
    def extract_zip_file(self, zip_info: ZipFileInfo) -> bool:
        """Extract a single ZIP file safely"""
        zip_path_str = str(zip_info.zip_path)
        
        try:
            # Create extraction directory
            zip_info.extract_path.mkdir(parents=True, exist_ok=True)
            
            # Calculate checksum if integrity checking is enabled
            if self.check_integrity:
                zip_info.checksum = self.calculate_file_checksum(zip_info.zip_path)
            
            extracted_files = []
            start_time = time.time()
            
            # Extract the ZIP file
            with zipfile.ZipFile(zip_info.zip_path, 'r') as zip_ref:
                # Get list of files to extract
                file_list = zip_ref.namelist()
                
                # Validate file paths for security
                for file_path in file_list:
                    if os.path.isabs(file_path) or ".." in file_path:
                        self.logger.warning(f"Suspicious file path in {zip_path_str}: {file_path}")
                        continue
                
                # Extract all files
                zip_ref.extractall(zip_info.extract_path)
                extracted_files = file_list
            
            extraction_time = time.time() - start_time
            
            # Record successful extraction
            self.extraction_records[zip_path_str] = {
                'extraction_timestamp': time.time(),
                'extraction_time_seconds': extraction_time,
                'zip_modification_time': zip_info.modification_time,
                'zip_size_bytes': zip_info.size_bytes,
                'zip_checksum': zip_info.checksum,
                'extract_path': str(zip_info.extract_path),
                'extracted_files': extracted_files,
                'file_count': len(extracted_files),
                'extraction_successful': True
            }
            
            self.logger.debug(f"Successfully extracted {zip_path_str} ({len(extracted_files)} files)")
            return True
            
        except zipfile.BadZipFile:
            self.logger.error(f"Bad ZIP file: {zip_path_str}")
            return False
        except PermissionError:
            self.logger.error(f"Permission denied: {zip_path_str}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to extract {zip_path_str}: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def process_zip_file(self, zip_info: ZipFileInfo) -> Tuple[str, bool, str]:
        """Process a single ZIP file (wrapper for threading)"""
        zip_path_str = str(zip_info.zip_path)
        
        try:
            # Skip if already processed and not overwriting
            if not self.overwrite and zip_path_str in self.processed_files:
                return zip_path_str, True, "already_processed"
            
            # Check if already extracted
            if not self.overwrite and self.is_already_extracted(zip_info):
                with self.progress_lock:
                    self.processed_files.add(zip_path_str)
                    self.stats.skipped_files += 1
                    self.stats.processed_files += 1
                return zip_path_str, True, "already_extracted"
            
            # Extract the ZIP file
            success = self.extract_zip_file(zip_info)
            
            with self.progress_lock:
                self.processed_files.add(zip_path_str)
                self.stats.processed_files += 1
                
                if success:
                    self.stats.extracted_files += 1
                    return zip_path_str, True, "extracted"
                else:
                    self.stats.failed_files += 1
                    return zip_path_str, False, "failed"
                    
        except Exception as e:
            self.logger.error(f"Error processing {zip_path_str}: {e}")
            with self.progress_lock:
                self.stats.failed_files += 1
                self.stats.processed_files += 1
            return zip_path_str, False, f"error: {str(e)}"
    
    def extract_all_zip_files(self, zip_files: List[ZipFileInfo]):
        """Extract all ZIP files using concurrent processing"""
        self.stats.total_zip_files = len(zip_files)
        self.stats.start_time = time.time()
        
        if not zip_files:
            self.logger.info("No ZIP files found to extract")
            return
        
        self.logger.info(f"Starting extraction of {len(zip_files)} ZIP files with {self.max_workers} workers")
        
        # Setup progress bar if available
        if tqdm:
            progress_bar = tqdm(
                total=len(zip_files),
                desc="Extracting ZIP files",
                unit="file"
            )
        else:
            progress_bar = None
        
        try:
            # Use ThreadPoolExecutor for concurrent processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_zip = {
                    executor.submit(self.process_zip_file, zip_info): zip_info 
                    for zip_info in zip_files
                }
                
                # Process completed tasks
                for future in as_completed(future_to_zip):
                    zip_info = future_to_zip[future]
                    try:
                        zip_path_str, success, status = future.result()
                        
                        if progress_bar:
                            progress_bar.update(1)
                            progress_bar.set_postfix({
                                'Extracted': self.stats.extracted_files,
                                'Skipped': self.stats.skipped_files,
                                'Failed': self.stats.failed_files
                            })
                        
                        # Save progress periodically
                        if self.stats.processed_files % 10 == 0:
                            self.save_progress()
                            
                    except Exception as e:
                        self.logger.error(f"Future failed for {zip_info.zip_path}: {e}")
                        
        finally:
            if progress_bar:
                progress_bar.close()
        
        self.stats.end_time = time.time()
    
    def run(self):
        """Main extraction process"""
        self.logger.info("Starting S0 ZIP file extraction")
        self.logger.info(f"Root directory: {self.root_dir}")
        self.logger.info(f"Max workers: {self.max_workers}")
        self.logger.info(f"Integrity checking: {self.check_integrity}")
        self.logger.info(f"Overwrite mode: {self.overwrite}")
        
        try:
            # Find all ZIP files
            zip_files = self.find_zip_files()
            
            if not zip_files:
                self.logger.info("No ZIP files found")
                return
            
            # Extract all ZIP files
            self.extract_all_zip_files(zip_files)
            
            # Save final progress
            self.save_progress()
            
            # Print summary
            self.print_summary()
            
            self.logger.info("S0 ZIP extraction completed successfully")
            
        except Exception as e:
            self.logger.error(f"S0 extraction failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def print_summary(self):
        """Print extraction summary"""
        processing_time = (self.stats.end_time or time.time()) - (self.stats.start_time or time.time())
        
        print("\n" + "="*50)
        print("S0 ZIP EXTRACTION SUMMARY")
        print("="*50)
        print(f"Total ZIP files found: {self.stats.total_zip_files}")
        print(f"Files processed: {self.stats.processed_files}")
        print(f"Files extracted: {self.stats.extracted_files}")
        print(f"Files skipped (already extracted): {self.stats.skipped_files}")
        print(f"Files failed: {self.stats.failed_files}")
        
        if self.stats.processed_files > 0:
            success_rate = (self.stats.extracted_files + self.stats.skipped_files) / self.stats.processed_files * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        print(f"Processing time: {processing_time:.1f} seconds")
        
        if self.stats.processed_files > 0:
            avg_time = processing_time / self.stats.processed_files
            print(f"Average time per file: {avg_time:.2f} seconds")
        
        print("="*50)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="S0 ZIP Extractor - Fast extraction of ZIP files with smart logic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python S0_zipExtract.py --root-dir /path/to/zip/files
  python S0_zipExtract.py --root-dir ./data --workers 8 --overwrite
  python S0_zipExtract.py --root-dir ./data --no-integrity-check
        """
    )
    
    parser.add_argument(
        '--root-dir', '-r',
        required=True,
        help='Root directory to search for ZIP files'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of worker threads for parallel processing (default: 4)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='Overwrite existing extractions (default: False)'
    )
    
    parser.add_argument(
        '--no-integrity-check',
        action='store_true',
        default=False,
        help='Disable integrity checking with checksums (faster but less safe)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    root_path = Path(args.root_dir)
    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_path}")
        sys.exit(1)
    
    if not root_path.is_dir():
        print(f"Error: Root path is not a directory: {root_path}")
        sys.exit(1)
    
    try:
        # Create and run extractor
        extractor = S0ZipExtractor(
            root_dir=str(root_path),
            max_workers=args.workers,
            check_integrity=not args.no_integrity_check,
            overwrite=args.overwrite
        )
        
        extractor.run()
        
    except KeyboardInterrupt:
        print("\nExtraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()