#!/usr/bin/env python3
"""
DICOM Study Zip Creator with Resume Capability
==============================================

This script processes JSONL files from S3_processForStore directory and creates
ZIP archives for each DICOM study. Each JSONL file represents one study and 
contains multiple DICOM file paths that will be compressed into a single ZIP file.

Features:
- Concurrent processing with configurable workers
- Resume capability - skips already processed files
- Progress tracking and statistics
- Comprehensive error handling
- Configurable compression level
- Maximum ZIP file size checking

Environment Variables:
- ZIP_COMPRESSION_LEVEL: Compression level 0-9 (default: 6)
- MAX_ZIP_SIZE_GB: Maximum ZIP file size in GB (default: 10)
- S4_PROGRESS_FILE: Progress tracking filename
- SKIP_FILES: Comma-separated list of files to skip

Author: AI Assistant
Date: 2025-06-21
"""

import os
import sys
import json
import zipfile
import argparse
import logging
import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from tqdm import tqdm

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Default list of JSONL files to skip during processing
DEFAULT_SKIP_FILES = [
    '.DS_Store',
    'Thumbs.db',
    'desktop.ini',
    '.gitkeep',
    'README.md',
    'log.txt'
]

# Get skip files from environment variable or use default
SKIP_FILES = os.getenv('SKIP_FILES', ','.join(DEFAULT_SKIP_FILES)).split(',')
SKIP_FILES = [f.strip() for f in SKIP_FILES if f.strip()]

# Progress tracking file name
PROGRESS_FILE = os.getenv('S4_PROGRESS_FILE', 'S4_zipStore_processing_progress.json')

# Compression settings
DEFAULT_COMPRESSION_LEVEL = int(os.getenv('ZIP_COMPRESSION_LEVEL', '6'))  # 0-9, where 9 is max compression
MAX_ZIP_SIZE_GB = float(os.getenv('MAX_ZIP_SIZE_GB', '10'))  # Maximum ZIP file size in GB

# =============================================================================
# STATISTICS TRACKING
# =============================================================================

@dataclass
class ZipCreatorStats:
    """Class to track processing statistics"""
    total_jsonl_files: int = 0
    processed_studies: int = 0
    skipped_files: int = 0
    failed_studies: int = 0
    already_processed: int = 0  # New: count of already existing ZIP files
    total_dicom_files: int = 0
    total_compressed_size: int = 0
    total_original_size: int = 0
    missing_dicom_files: int = 0
    errors: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    
    def add_error(self, error: str):
        """Add an error to the error list"""
        self.errors.append(error)
        
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio as percentage"""
        if self.total_original_size == 0:
            return 0.0
        return (1 - self.total_compressed_size / self.total_original_size) * 100
    
    def print_summary(self):
        """Print comprehensive statistics summary"""
        print("\n" + "="*60)
        print("DICOM STUDY ZIP CREATION SUMMARY")
        print("="*60)
        print(f"Total JSONL files found: {self.total_jsonl_files}")
        print(f"Successfully processed studies: {self.processed_studies}")
        print(f"Already processed (skipped): {self.already_processed}")
        print(f"Failed studies: {self.failed_studies}")
        print(f"Skipped files: {self.skipped_files}")
        print(f"Total DICOM files processed: {self.total_dicom_files}")
        print(f"Missing DICOM files: {self.missing_dicom_files}")
        print(f"Total original size: {self._format_size(self.total_original_size)}")
        print(f"Total compressed size: {self._format_size(self.total_compressed_size)}")
        print(f"Compression ratio: {self.get_compression_ratio():.1f}%")
        print(f"Processing time: {self.processing_time:.2f} seconds")
        
        if self.errors:
            print(f"\nErrors encountered: {len(self.errors)}")
            for i, error in enumerate(self.errors[:10], 1):  # Show first 10 errors
                print(f"  {i}. {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
        
        print("="*60)
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        size = float(size_bytes)
        unit_index = 0
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        return f"{size:.2f} {units[unit_index]}"

# =============================================================================
# PROGRESS TRACKING
# =============================================================================

@dataclass
class ProcessingProgress:
    """Class to track processing progress for resume functionality"""
    completed_studies: Set[str] = field(default_factory=set)
    failed_studies: Set[str] = field(default_factory=set)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    total_processed: int = 0
    
    def save(self, output_dir: str):
        """Save progress to file"""
        progress_file = os.path.join(output_dir, PROGRESS_FILE)
        progress_data = {
            'completed_studies': list(self.completed_studies),
            'failed_studies': list(self.failed_studies),
            'last_updated': datetime.now().isoformat(),
            'total_processed': self.total_processed
        }
        
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save progress file: {e}")
    
    @classmethod
    def load(cls, output_dir: str) -> 'ProcessingProgress':
        """Load progress from file"""
        progress_file = os.path.join(output_dir, PROGRESS_FILE)
        
        if not os.path.exists(progress_file):
            return cls()
        
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            progress = cls()
            progress.completed_studies = set(data.get('completed_studies', []))
            progress.failed_studies = set(data.get('failed_studies', []))
            progress.last_updated = data.get('last_updated', '')
            progress.total_processed = data.get('total_processed', 0)
            
            return progress
            
        except Exception as e:
            print(f"Warning: Could not load progress file: {e}")
            return cls()
    
    def is_completed(self, study_name: str) -> bool:
        """Check if a study has been completed"""
        return study_name in self.completed_studies
    
    def is_failed(self, study_name: str) -> bool:
        """Check if a study has failed"""
        return study_name in self.failed_studies
    
    def mark_completed(self, study_name: str):
        """Mark a study as completed"""
        self.completed_studies.add(study_name)
        self.failed_studies.discard(study_name)  # Remove from failed if present
        self.total_processed += 1
    
    def mark_failed(self, study_name: str):
        """Mark a study as failed"""
        self.failed_studies.add(study_name)
        self.completed_studies.discard(study_name)  # Remove from completed if present

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to both console and file"""
    
    # Create logs directory
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"S4_zipStore_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file created: {log_file}")
    
    return logger

# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================

def get_study_name_from_jsonl(jsonl_file: str) -> str:
    """Extract study name from JSONL filename"""
    study_name = os.path.splitext(os.path.basename(jsonl_file))[0]
    if study_name.startswith('S3_processForStore_'):
        study_name = study_name[19:]  # Remove prefix
    return study_name

def check_existing_zip(study_name: str, output_dir: str) -> bool:
    """Check if ZIP file already exists for the study"""
    zip_path = os.path.join(output_dir, f"{study_name}.zip")
    return os.path.exists(zip_path) and os.path.getsize(zip_path) > 0

def validate_existing_zip(study_name: str, output_dir: str, logger: logging.Logger) -> bool:
    """Validate that existing ZIP file is not corrupted"""
    zip_path = os.path.join(output_dir, f"{study_name}.zip")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            # Test the ZIP file integrity
            zipf.testzip()
            # Check if it has any files
            return len(zipf.namelist()) > 0
    except Exception as e:
        logger.warning(f"Existing ZIP file {zip_path} appears corrupted: {e}")
        return False

def validate_file_paths(jsonl_file: str, logger: logging.Logger) -> Tuple[List[Dict], List[str]]:
    """
    Read and validate DICOM file paths from JSONL file
    
    Args:
        jsonl_file: Path to JSONL file
        logger: Logger instance
        
    Returns:
        Tuple of (valid_records, missing_files)
    """
    valid_records = []
    missing_files = []
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    record = json.loads(line)
                    file_path = record.get('file_path', '')
                    
                    if not file_path:
                        logger.warning(f"Empty file_path in {jsonl_file} line {line_num}")
                        continue
                    
                    if os.path.exists(file_path):
                        valid_records.append(record)
                    else:
                        missing_files.append(file_path)
                        logger.warning(f"Missing file: {file_path}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in {jsonl_file} line {line_num}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error reading {jsonl_file}: {e}")
        
    return valid_records, missing_files

def create_study_zip_sync(jsonl_file: str, output_dir: str, logger: logging.Logger, 
                         progress: ProcessingProgress, force_reprocess: bool = False) -> Tuple[bool, Dict]:
    """
    Create ZIP archive for a single DICOM study (synchronous version for thread execution)
    
    Args:
        jsonl_file: Path to processed JSONL file containing study information
        output_dir: Output directory for ZIP files
        logger: Logger instance
        progress: Progress tracking object
        force_reprocess: Force reprocessing even if ZIP exists
        
    Returns:
        Tuple of (success_flag, statistics_dict)
    """
    stats_dict = {
        'dicom_files': 0,
        'original_size': 0,
        'compressed_size': 0,
        'missing_files': 0,
        'errors': [],
        'already_existed': False
    }
    
    try:
        # Get study name from JSONL filename
        study_name = get_study_name_from_jsonl(jsonl_file)
        
        # Check if already processed and valid (unless force_reprocess is True)
        if not force_reprocess:
            if progress.is_completed(study_name) and check_existing_zip(study_name, output_dir):
                if validate_existing_zip(study_name, output_dir, logger):
                    logger.info(f"Skipping {study_name} - already processed and ZIP exists")
                    stats_dict['already_existed'] = True
                    # Get existing ZIP size for statistics
                    zip_path = os.path.join(output_dir, f"{study_name}.zip")
                    stats_dict['compressed_size'] = os.path.getsize(zip_path)
                    return True, stats_dict
                else:
                    logger.info(f"Re-processing {study_name} - existing ZIP is corrupted")
                    progress.failed_studies.discard(study_name)
                    progress.completed_studies.discard(study_name)
        
        # Validate and get file paths
        valid_records, missing_files = validate_file_paths(jsonl_file, logger)
        
        if not valid_records:
            error_msg = f"No valid DICOM files found in {jsonl_file}"
            logger.error(error_msg)
            stats_dict['errors'].append(f"No valid files in {study_name}")
            progress.mark_failed(study_name)
            return False, stats_dict
        
        # Update statistics
        stats_dict['missing_files'] = len(missing_files)
        
        # Create ZIP file
        zip_filename = f"{study_name}.zip"
        zip_path = os.path.join(output_dir, zip_filename)
        
        # Create temporary ZIP file first
        temp_zip_path = f"{zip_path}.tmp"
        
        original_size = 0
        
        try:
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=DEFAULT_COMPRESSION_LEVEL) as zipf:
                for record in valid_records:
                    file_path = record['file_path']
                    
                    try:
                        # Get file size for statistics
                        file_size = os.path.getsize(file_path)
                        original_size += file_size
                        
                        # Create archive name (preserve directory structure from original path)
                        # Use the original filename to maintain DICOM file names
                        archive_name = os.path.basename(file_path)
                        
                        # Add file to ZIP
                        zipf.write(file_path, archive_name)
                        
                        # Also add metadata as a companion JSON file
                        metadata_name = f"{os.path.splitext(archive_name)[0]}_metadata.json"
                        metadata_content = json.dumps(record, ensure_ascii=False, indent=2)
                        zipf.writestr(metadata_name, metadata_content.encode('utf-8'))
                        
                    except Exception as e:
                        error_msg = f"Error adding {file_path} to ZIP: {e}"
                        logger.error(error_msg)
                        stats_dict['errors'].append(f"Failed to add {os.path.basename(file_path)} to {study_name}")
                        continue
            
            # Move temporary file to final location (atomic operation)
            os.replace(temp_zip_path, zip_path)
            
        except Exception as e:
            # Clean up temporary file if it exists
            if os.path.exists(temp_zip_path):
                try:
                    os.remove(temp_zip_path)
                except:
                    pass
            raise e
        
        # Get compressed size
        compressed_size = os.path.getsize(zip_path)
        
        # Update statistics
        stats_dict['dicom_files'] = len(valid_records)
        stats_dict['original_size'] = original_size
        stats_dict['compressed_size'] = compressed_size
        
        compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        
        logger.info(f"Created {zip_filename}: {len(valid_records)} files, "
                   f"{ZipCreatorStats._format_size(original_size)} → {ZipCreatorStats._format_size(compressed_size)} "
                   f"({compression_ratio:.1f}% compression)")
        
        # Mark as completed in progress
        progress.mark_completed(study_name)
        
        return True, stats_dict
        
    except Exception as e:
        error_msg = f"Failed to create ZIP for {jsonl_file}: {e}"
        logger.error(error_msg)
        stats_dict['errors'].append(f"Failed to create ZIP for {os.path.basename(jsonl_file)}: {str(e)}")
        progress.mark_failed(study_name)
        return False, stats_dict

async def process_jsonl_files_concurrent(input_dir: str, output_dir: str, logger: logging.Logger, 
                                       concurrency: int = 4, force_reprocess: bool = False) -> ZipCreatorStats:
    """
    Process all JSONL files in the input directory with concurrency and resume capability
    
    Args:
        input_dir: Directory containing JSONL files
        output_dir: Output directory for ZIP files
        logger: Logger instance
        concurrency: Number of concurrent workers
        force_reprocess: Force reprocessing of all files
        
    Returns:
        Statistics object with processing results
    """
    stats = ZipCreatorStats()
    start_time = datetime.now()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load or initialize progress tracking
    progress = ProcessingProgress.load(output_dir)
    if force_reprocess:
        logger.info("Force reprocessing enabled - clearing previous progress")
        progress = ProcessingProgress()
    elif progress.completed_studies or progress.failed_studies:
        logger.info(f"Resuming previous session - {len(progress.completed_studies)} completed, "
                   f"{len(progress.failed_studies)} failed")
    
    # Find all JSONL files
    jsonl_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.jsonl') and filename not in SKIP_FILES:
            jsonl_files.append(os.path.join(input_dir, filename))
        elif filename in SKIP_FILES:
            stats.skipped_files += 1
            logger.info(f"Skipped file: {filename}")
    
    stats.total_jsonl_files = len(jsonl_files)
    
    if not jsonl_files:
        logger.warning("No JSONL files found to process")
        return stats
    
    # Filter files that need processing
    files_to_process = []
    for jsonl_file in jsonl_files:
        study_name = get_study_name_from_jsonl(jsonl_file)
        
        if force_reprocess:
            files_to_process.append(jsonl_file)
        elif not progress.is_completed(study_name) or not check_existing_zip(study_name, output_dir):
            files_to_process.append(jsonl_file)
        else:
            # File already processed and ZIP exists
            stats.already_processed += 1
            logger.debug(f"Skipping {study_name} - already processed")
    
    logger.info(f"Found {len(jsonl_files)} JSONL files total")
    logger.info(f"Processing {len(files_to_process)} files with {concurrency} workers")
    logger.info(f"Skipping {stats.already_processed} already processed files")
    
    if not files_to_process:
        logger.info("All files have been processed. Use --force-reprocess to reprocess all files.")
        stats.processing_time = (datetime.now() - start_time).total_seconds()
        return stats
    
    # Setup concurrent processing
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=concurrency)
    
    # Save progress periodically
    progress_save_interval = max(1, len(files_to_process) // 10)  # Save every 10% of progress
    processed_count = 0
    
    async def process_one_jsonl(jsonl_file: str) -> None:
        """Process a single JSONL file asynchronously"""
        nonlocal processed_count
        
        try:
            success, file_stats = await loop.run_in_executor(
                executor, create_study_zip_sync, jsonl_file, output_dir, logger, progress, force_reprocess
            )
            
            # Update global statistics (thread-safe since we're in the main event loop)
            if file_stats.get('already_existed', False):
                stats.already_processed += 1
            elif success:
                stats.processed_studies += 1
            else:
                stats.failed_studies += 1
                
            # Aggregate statistics
            stats.total_dicom_files += file_stats['dicom_files']
            stats.total_original_size += file_stats['original_size']
            stats.total_compressed_size += file_stats['compressed_size']
            stats.missing_dicom_files += file_stats['missing_files']
            
            # Add errors to main stats
            for error in file_stats['errors']:
                stats.add_error(error)
            
            # Save progress periodically
            processed_count += 1
            if processed_count % progress_save_interval == 0:
                progress.save(output_dir)
                
        except Exception as e:
            logger.error(f"Unexpected error processing {jsonl_file}: {e}")
            stats.failed_studies += 1
            stats.add_error(f"Unexpected error processing {os.path.basename(jsonl_file)}: {str(e)}")
            study_name = get_study_name_from_jsonl(jsonl_file)
            progress.mark_failed(study_name)
    
    # Process files with progress bar
    tasks = []
    for jsonl_file in files_to_process:
        task = process_one_jsonl(jsonl_file)
        tasks.append(task)
    
    # Use tqdm for progress tracking
    with tqdm(total=len(tasks), desc="Processing JSONL files") as pbar:
        for task in asyncio.as_completed(tasks):
            await task
            pbar.update(1)
    
    # Final progress save
    progress.save(output_dir)
    
    # Clean up executor
    executor.shutdown(wait=True)
    
    # Calculate processing time
    end_time = datetime.now()
    stats.processing_time = (end_time - start_time).total_seconds()
    
    return stats

def process_jsonl_files(input_dir: str, output_dir: str, logger: logging.Logger, 
                       concurrency: int = 4, force_reprocess: bool = False) -> ZipCreatorStats:
    """
    Process all JSONL files in the input directory (wrapper for concurrent processing)
    
    Args:
        input_dir: Directory containing JSONL files
        output_dir: Output directory for ZIP files
        logger: Logger instance
        concurrency: Number of concurrent workers
        force_reprocess: Force reprocessing of all files
        
    Returns:
        Statistics object with processing results
    """
    return asyncio.run(process_jsonl_files_concurrent(input_dir, output_dir, logger, concurrency, force_reprocess))

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def get_valid_directory(prompt: str, check_exists: bool = True) -> str:
    """Get a valid directory path from user input"""
    while True:
        path = input(prompt).strip()
        
        # Remove quotes if present
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        elif path.startswith("'") and path.endswith("'"):
            path = path[1:-1]
        
        if not path:
            print("Please enter a valid path.")
            continue
            
        if check_exists and not os.path.exists(path):
            print(f"Directory does not exist: {path}")
            print("Please enter a valid directory path.")
            continue
            
        if check_exists and not os.path.isdir(path):
            print(f"Path is not a directory: {path}")
            continue
            
        return os.path.abspath(path)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Create ZIP archives for DICOM studies from JSONL files with resume capability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input /path/to/S3_processForStore --output /path/to/S4_zipStore
  %(prog)s --input /path/to/jsonl/files --output ./zip_output --overwrite
  %(prog)s --show-skip-files
  %(prog)s --input /path/to/files --concurrency 8
  %(prog)s --input /path/to/files --force-reprocess  # Reprocess all files
  %(prog)s --input /path/to/files --show-progress    # Show current progress
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input directory containing JSONL files (from S3_processForStore)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for ZIP files (default: S4_zipStore)'
    )
    
    parser.add_argument(
        '--concurrency', '-c',
        type=int,
        default=4,
        help='Number of concurrent workers (default: 4)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output directory'
    )
    
    parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help='Force reprocessing of all files, ignoring previous progress'
    )
    
    parser.add_argument(
        '--show-skip-files',
        action='store_true',
        help='Show the list of files to skip and exit'
    )
    
    parser.add_argument(
        '--show-progress',
        action='store_true',
        help='Show current processing progress and exit'
    )
    
    return parser.parse_args()

def show_current_progress(output_dir: str):
    """Show current processing progress"""
    if not os.path.exists(output_dir):
        print(f"Output directory does not exist: {output_dir}")
        return
    
    progress = ProcessingProgress.load(output_dir)
    
    print("="*60)
    print("CURRENT PROCESSING PROGRESS")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Last updated: {progress.last_updated}")
    print(f"Completed studies: {len(progress.completed_studies)}")
    print(f"Failed studies: {len(progress.failed_studies)}")
    print(f"Total processed: {progress.total_processed}")
    
    if progress.completed_studies:
        print(f"\nRecent completed studies (last 10):")
        for study in list(progress.completed_studies)[-10:]:
            print(f"  ✓ {study}")
    
    if progress.failed_studies:
        print(f"\nFailed studies:")
        for study in progress.failed_studies:
            print(f"  ✗ {study}")
    
    print("="*60)

def main():
    """Main function"""
    args = parse_arguments()
    
    # Show skip files if requested
    if args.show_skip_files:
        print("Files configured to be skipped:")
        for skip_file in SKIP_FILES:
            print(f"  - {skip_file}")
        print(f"\nTotal: {len(SKIP_FILES)} files")
        print("\nTo modify this list, set the SKIP_FILES environment variable:")
        print("export SKIP_FILES='file1.txt,file2.log,file3.tmp'")
        return
    
    # Show progress if requested
    if args.show_progress:
        output_dir = os.path.abspath(args.output)
        show_current_progress(output_dir)
        return
    
    # Validate concurrency parameter
    cpu_count = multiprocessing.cpu_count()
    if args.concurrency < 1:
        print("Error: Concurrency must be at least 1")
        sys.exit(1)
    elif args.concurrency > cpu_count:
        print(f"Warning: Concurrency ({args.concurrency}) exceeds CPU count ({cpu_count})")
        print(f"Consider using a lower value for better performance.")
    
    # Get input directory
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input directory does not exist: {args.input}")
            sys.exit(1)
        input_dir = os.path.abspath(args.input)
    else:
        print("Input directory not provided.")
        input_dir = get_valid_directory(
            "Please enter the path to directory containing (processed) JSONL files: "
        )
        
    # Get output directory
    if args.output:
        if not os.path.exists(args.output):
            print(f"Error: Output directory does not exist: {args.output}")
            sys.exit(1)
        output_dir = os.path.abspath(args.output)
    else:
        print("Output directory not provided.")
        output_dir = get_valid_directory(
            "Please enter the path to output directory to save zip files: "
        )
    
    # Check if output directory exists
    if os.path.exists(output_dir):
        if not args.overwrite:
            response = input(f"Output directory '{output_dir}' already exists. Continue/Resume? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Operation cancelled.")
                sys.exit(0)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    # Print configuration
    print("="*60)
    print("DICOM STUDY ZIP CREATOR WITH RESUME")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Concurrency: {args.concurrency} workers")
    print(f"Force reprocess: {args.force_reprocess}")
    print(f"Skip files: {', '.join(SKIP_FILES)}")
    print("="*60)
    
    logger.info("Starting DICOM study ZIP creation process")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Concurrency: {args.concurrency} workers")
    logger.info(f"Force reprocess: {args.force_reprocess}")
    logger.info(f"Skip files: {SKIP_FILES}")
    
    try:
        # Process files with concurrency and resume capability
        stats = process_jsonl_files(input_dir, output_dir, logger, args.concurrency, args.force_reprocess)
        
        # Print and log summary
        stats.print_summary()
        logger.info("Processing completed successfully")
        
        # Save statistics to file
        stats_file = os.path.join(output_dir, 'processing_statistics.json')
        stats_data = {
            'timestamp': datetime.now().isoformat(),
            'input_directory': input_dir,
            'output_directory': output_dir,
            'concurrency_workers': args.concurrency,
            'force_reprocess': args.force_reprocess,
            'total_jsonl_files': stats.total_jsonl_files,
            'processed_studies': stats.processed_studies,
            'already_processed': stats.already_processed,
            'failed_studies': stats.failed_studies,
            'skipped_files': stats.skipped_files,
            'total_dicom_files': stats.total_dicom_files,
            'missing_dicom_files': stats.missing_dicom_files,
            'total_original_size_bytes': stats.total_original_size,
            'total_compressed_size_bytes': stats.total_compressed_size,
            'compression_ratio_percent': stats.get_compression_ratio(),
            'processing_time_seconds': stats.processing_time,
            'errors': stats.errors
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nStatistics saved to: {stats_file}")
        print(f"Progress tracking saved to: {os.path.join(output_dir, PROGRESS_FILE)}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\nProcess interrupted by user.")
        print("Progress has been saved. You can resume processing later.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


