import os
import json
import argparse
import logging
from typing import List, Callable, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict

# =============================================================================
# CONFIGURATION - Files to skip during processing
# =============================================================================
# Environment variable for files to skip (comma-separated list)
# Developers can modify this list or set the SKIP_FILES environment variable
DEFAULT_SKIP_FILES = [
    "ALL_PATIENTS_inCT.json",
    "ALL_PATIENTS_inCT.jsonl", 
    "summary.jsonl",
    "CONCAT_ALL.jsonl"
    "processed_files.jsonl",
    "temp.jsonl",
    "backup.jsonl",
    "test.jsonl",
    "S3_processForStore.json"
]

# Get skip files from environment variable or use default
SKIP_FILES_ENV = os.getenv('SKIP_FILES', ','.join(DEFAULT_SKIP_FILES))
SKIP_FILES = [filename.strip() for filename in SKIP_FILES_ENV.split(',') if filename.strip()]

print(f"Files configured to skip: {SKIP_FILES}")
print("-" * 70)

# =============================================================================
# FILTER FUNCTIONS
# =============================================================================

def minimum_files_filter(records: List[Dict[str, Any]], min_files: int) -> Tuple[bool, str]:
    """
    Filter function to check if study has minimum number of files.
    
    Args:
        records: List of records from JSONL file
        min_files: Minimum number of files required
        
    Returns:
        Tuple of (passed_filter, reason)
    """
    num_files = len(records)
    if num_files >= min_files:
        return True, f"Passed: {num_files} files >= {min_files} minimum"
    else:
        return False, f"Failed: {num_files} files < {min_files} minimum"

def duplicate_removal_filter(records: List[Dict[str, Any]], _) -> Tuple[List[Dict[str, Any]], str]:
    """
    Remove duplicate records from the list.
    
    Args:
        records: List of records from JSONL file
        _: Unused parameter for consistency
        
    Returns:
        Tuple of (filtered_records, reason)
    """
    original_count = len(records)
    
    # Convert records to JSON strings for comparison
    seen = set()
    unique_records = []
    
    for record in records:
        # Create a canonical JSON string for comparison
        record_str = json.dumps(record, sort_keys=True, ensure_ascii=False)
        if record_str not in seen:
            seen.add(record_str)
            unique_records.append(record)
    
    duplicates_removed = original_count - len(unique_records)
    reason = f"Removed {duplicates_removed} duplicates ({original_count} -> {len(unique_records)})"
    
    return unique_records, reason

# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

class S3ProcessorStats:
    """Class to track processing statistics."""
    
    def __init__(self):
        self.total_files_found = 0
        self.files_skipped = 0
        self.files_processed = 0
        self.files_passed_filters = 0
        self.files_failed_filters = 0
        self.total_records_original = 0
        self.total_records_after_dedup = 0
        self.total_records_final = 0
        self.skipped_files_list = []
        self.failed_files_list = []
        self.processing_details = []
        
    def add_file_result(self, filename: str, status: str, details: Dict[str, Any]):
        """Add processing result for a file."""
        self.processing_details.append({
            'filename': filename,
            'status': status,
            'details': details
        })
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_files_found': self.total_files_found,
            'files_skipped': self.files_skipped,
            'files_processed': self.files_processed,
            'files_passed_filters': self.files_passed_filters,
            'files_failed_filters': self.files_failed_filters,
            'total_records_original': self.total_records_original,
            'total_records_after_dedup': self.total_records_after_dedup,
            'total_records_final': self.total_records_final,
            'skipped_files': self.skipped_files_list,
            'failed_files': self.failed_files_list
        }

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, f"S3_process_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Create logger
    logger = logging.getLogger('S3Processor')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def process_jsonl_files(
    input_dir: str, 
    output_dir: str, 
    min_files: int,
    filter_functions: List[Callable] = None
) -> S3ProcessorStats:
    """
    Process JSONL files with filtering and deduplication.
    
    Args:
        input_dir: Input directory containing JSONL files
        output_dir: Output directory for processed files
        min_files: Minimum number of files required per study
        filter_functions: List of filter functions to apply
        
    Returns:
        S3ProcessorStats object with processing statistics
    """
    
    # Setup logging
    logger = setup_logging(output_dir)
    stats = S3ProcessorStats()
    
    # Default filter functions
    if filter_functions is None:
        filter_functions = [
            lambda records, min_f=min_files: minimum_files_filter(records, min_f)
        ]
    
    logger.info(f"Starting S3 processing for directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Minimum files per study: {min_files}")
    logger.info(f"Number of filter functions: {len(filter_functions)}")
    logger.info(f"Files to skip: {SKIP_FILES}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSONL files
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    stats.total_files_found = len(all_files)
    
    logger.info(f"Found {stats.total_files_found} JSONL files")
    
    for filename in all_files:
        file_path = os.path.join(input_dir, filename)
        
        # Check if file should be skipped
        if filename in SKIP_FILES:
            stats.files_skipped += 1
            stats.skipped_files_list.append(filename)
            logger.info(f"SKIPPED: {filename} (in skip list)")
            stats.add_file_result(filename, "SKIPPED", {"reason": "In skip list"})
            continue
            
        try:
            # Read JSONL file
            records = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num} in {filename}: {e}")
            
            original_count = len(records)
            stats.total_records_original += original_count
            stats.files_processed += 1
            
            logger.info(f"PROCESSING: {filename} ({original_count} records)")
            
            # Step 1: Remove duplicates
            records, dedup_reason = duplicate_removal_filter(records, None)
            after_dedup_count = len(records)
            stats.total_records_after_dedup += after_dedup_count
            
            logger.info(f"  Deduplication: {dedup_reason}")
            
            # Step 2: Apply filter functions
            passed_all_filters = True
            filter_results = []
            
            for i, filter_func in enumerate(filter_functions):
                try:
                    passed, reason = filter_func(records, min_files)
                    filter_results.append(f"Filter {i+1}: {reason}")
                    logger.info(f"  Filter {i+1}: {reason}")
                    
                    if not passed:
                        passed_all_filters = False
                        break
                        
                except Exception as e:
                    logger.error(f"  Filter {i+1} error: {e}")
                    passed_all_filters = False
                    filter_results.append(f"Filter {i+1}: ERROR - {e}")
                    break
            
            # Process based on filter results
            if passed_all_filters:
                # Save processed file
                output_filename = f"S3_processForStore_{filename}"
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, 'w', encoding='utf-8') as out_f:
                    for record in records:
                        json.dump(record, out_f, ensure_ascii=False)
                        out_f.write('\n')
                
                stats.files_passed_filters += 1
                stats.total_records_final += len(records)
                
                logger.info(f"  PASSED: Saved as {output_filename}")
                
                stats.add_file_result(filename, "PASSED", {
                    "original_records": original_count,
                    "after_dedup": after_dedup_count,
                    "final_records": len(records),
                    "output_file": output_filename,
                    "filter_results": filter_results
                })
                
            else:
                stats.files_failed_filters += 1
                stats.failed_files_list.append(filename)
                
                logger.info(f"  FAILED: Did not pass all filters")
                
                stats.add_file_result(filename, "FAILED", {
                    "original_records": original_count,
                    "after_dedup": after_dedup_count,
                    "filter_results": filter_results
                })
                
        except Exception as e:
            logger.error(f"ERROR processing {filename}: {e}")
            stats.files_failed_filters += 1
            stats.failed_files_list.append(filename)
            stats.add_file_result(filename, "ERROR", {"error": str(e)})
    
    # Save processing summary
    summary = stats.get_summary()
    summary_file = os.path.join(output_dir, "S3_processForStore.json")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "processing_timestamp": datetime.now().isoformat(),
            "input_directory": input_dir,
            "output_directory": output_dir,
            "parameters": {
                "min_files_per_study": min_files,
                "skip_files": SKIP_FILES
            },
            "summary": summary,
            "detailed_results": stats.processing_details
        }, f, ensure_ascii=False, indent=2)
    
    # Print final summary
    logger.info("=" * 70)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total files found: {stats.total_files_found}")
    logger.info(f"Files skipped: {stats.files_skipped}")
    logger.info(f"Files processed: {stats.files_processed}")
    logger.info(f"Files passed filters: {stats.files_passed_filters}")
    logger.info(f"Files failed filters: {stats.files_failed_filters}")
    logger.info(f"Total records (original): {stats.total_records_original}")
    logger.info(f"Total records (after dedup): {stats.total_records_after_dedup}")
    logger.info(f"Total records (final): {stats.total_records_final}")
    logger.info(f"Summary saved to: {summary_file}")
    
    return stats

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process JSONL files for S3 storage preparation with filtering and deduplication",
        epilog="""
Environment Variables:
  SKIP_FILES    Comma-separated list of filenames to skip during processing
                (default: ALL_PATIENTS_inCT.json,ALL_PATIENTS_inCT.jsonl,summary.jsonl,processed_files.jsonl,temp.jsonl,backup.jsonl,test.jsonl,S3_processForStore.json)
        
Examples:
  python script.py
  python script.py --input_dir "C:\\Data\\CT_Files" --min_files 10
  python script.py --input_dir "C:\\Data\\CT_Files" --output_dir "C:\\Output\\S3_Ready" --min_files 5 --overwrite
  
  # Set custom skip files via environment variable
  set SKIP_FILES=output.jsonl,temp.jsonl,backup.jsonl
  python script.py --input_dir "C:\\Data\\CT_Files"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input_dir", 
                       help="Input directory containing JSONL files to process")
    parser.add_argument("--output_dir", 
                       help="Output directory for processed files (default: S3_processForStore/ in input_dir)")
    parser.add_argument("--min_files", 
                       type=int,
                       help="Minimum number of files required per study (default: 1)")
    parser.add_argument("--overwrite", 
                       action="store_true", 
                       help="Whether to overwrite existing output directory")
    parser.add_argument("--show-skip-files", 
                       action="store_true", 
                       help="Show the list of files configured to be skipped and exit")
    
    args = parser.parse_args()

    # Show skip files and exit if requested
    if args.show_skip_files:
        print("Files configured to be skipped:")
        for i, filename in enumerate(SKIP_FILES, 1):
            print(f"  {i}. {filename}")
        print(f"\nTo modify this list, set the SKIP_FILES environment variable:")
        print(f"Example: set SKIP_FILES=file1.jsonl,file2.jsonl,file3.jsonl")
        exit(0)

    # Get input directory - either from args or ask user
    input_dir = args.input_dir
    if not input_dir:
        print("Please provide the input directory containing JSONL files.")
        input_dir = input("Enter input directory path: ").strip()
        
        # Remove quotes if user copied path with quotes
        if input_dir.startswith('"') and input_dir.endswith('"'):
            input_dir = input_dir[1:-1]
        if input_dir.startswith("'") and input_dir.endswith("'"):
            input_dir = input_dir[1:-1]
        
        # Keep asking until valid directory is provided
        while not os.path.isdir(input_dir):
            print(f"Directory '{input_dir}' does not exist.")
            input_dir = input("Enter valid input directory path containing JSONL files: ").strip()
            
            # Remove quotes if user copied path with quotes
            if input_dir.startswith('"') and input_dir.endswith('"'):
                input_dir = input_dir[1:-1]
            if input_dir.startswith("'") and input_dir.endswith("'"):
                input_dir = input_dir[1:-1]
    
    
    # Validate input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        exit(1)

    # Get minimum files parameter
    min_files = args.min_files
    if not min_files:
        try:
            min_files = int(input(f"Enter minimum number of files per study (current: {min_files}): ") or min_files)
        except ValueError:
            print("Invalid number, using default value of 1")
            min_files = 1

    # Set output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(input_dir, "S3_processForStore")

    # Check if output directory exists and handle overwrite
    if os.path.exists(output_dir) and os.listdir(output_dir):
        if not args.overwrite:
            response = input(f"Output directory {output_dir} already exists and is not empty. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                exit(0)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("S3 PROCESS FOR STORE - JSONL FILE PROCESSOR")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Minimum files per study: {min_files}")
    print("=" * 70)

    # Process the files
    stats = process_jsonl_files(input_dir, output_dir, min_files)
    
    print("\nProcessing completed successfully!")
    print(f"Check the log file in {output_dir} for detailed information.")
