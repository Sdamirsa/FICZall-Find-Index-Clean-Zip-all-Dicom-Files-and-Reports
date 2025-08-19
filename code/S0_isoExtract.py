#!/usr/bin/env python3
"""
ISO File Extractor - A tool to find and extract ISO files using PowerISO.

This script recursively finds ISO files in a directory and extracts them to a flattened
directory structure, keeping track of progress in a JSON file.
"""

import os
import json
import logging
import subprocess
import argparse
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from functools import wraps
from time import sleep

# Configuration constants with environment variable support
POWERISO_PATH = os.getenv(
    'POWERISO_PATH',
    r"C:\Users\LEGION\Downloads\PowerISO.9.0.Portable\PowerISO.9.0.Portable\App\PowerISO\piso.exe"
)
ALL_JSONS_FILE = os.getenv('ISO_PROGRESS_FILE', "All_jsons_in_dir.json")
EXTRACTION_FILE_THRESHOLD = int(os.getenv('EXTRACTION_THRESHOLD', '100'))
LOG_FILE = os.getenv('ISO_LOG_FILE', "iso_extractor.log")
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', '5'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def retry_on_failure(max_retries: int = MAX_RETRIES, delay: int = RETRY_DELAY):
    """Decorator to retry failed operations with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
                        sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed: {str(e)}")
            raise last_exception
        return wrapper
    return decorator


def run_shell_command(command: List[str], input_str: Optional[str] = None) -> Tuple[int, str, str]:
    """
    Run a command synchronously and optionally pipe 'input_str' into stdin.
    
    Args:
        command: List of command arguments
        input_str: Optional string to pipe to stdin
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    try:
        # Create the subprocess
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if input_str else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'  # Handle encoding errors gracefully
        )

        if input_str:
            # Write the input and then close
            proc.stdin.write(input_str)
            proc.stdin.close()

        stdout, stderr = proc.communicate(timeout=300)  # 5-minute timeout
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return -1, stdout, f"Command timed out after 300 seconds: {' '.join(command)}"
    except Exception as e:
        return -1, "", f"Error executing command: {str(e)}"


def find_iso_files(search_dir: str) -> List[str]:
    """
    Recursively find all .iso files in 'search_dir'.
    
    Args:
        search_dir: Directory to search for ISO files
        
    Returns:
        List of absolute paths to the ISO files
    """
    logger.info(f"Searching for ISO files in {search_dir}")
    iso_files = []
    try:
        for root, _, files in os.walk(search_dir):
            for file in files:
                if file.lower().endswith(".iso"):
                    full_path = os.path.join(root, file)
                    iso_files.append(os.path.abspath(full_path))
                    
        logger.info(f"Found {len(iso_files)} ISO files")
        return iso_files
    except Exception as e:
        logger.error(f"Error finding ISO files: {str(e)}")
        return []


def flatten_path(iso_path: str, base_output_dir: str) -> str:
    """
    Convert a path like F:/12412/1.iso into a flattened output folder
    like F:/flat_iso_extract/12412_._1
    
    Args:
        iso_path: Path to the ISO file
        base_output_dir: Base directory for extraction
        
    Returns:
        Flattened output path
    """
    p = Path(iso_path)
    drive = p.drive
    parts_no_drive = p.parts[1:]
    last_part = p.stem

    # Create a safe, filesystem-friendly name
    if len(parts_no_drive) > 1:
        flattened = "_._".join(list(parts_no_drive[:-1]) + [last_part])
    else:
        flattened = last_part
        
    # Limit length to avoid path too long errors
    if len(flattened) > 100:
        flattened = flattened[:97] + "..."

    # Build final path
    output_path = Path(drive) / base_output_dir / flattened
    return str(output_path)


def count_files_in_dir(directory: str) -> Tuple[int, int, int]:
    """
    Count total files in directory and categorize them.
    
    Args:
        directory: Directory to count files in
        
    Returns:
        Tuple of (total_files, dcm_count, no_ext_count)
    """
    total = 0
    dcm_count = 0
    no_ext_count = 0

    try:
        for root, _, files in os.walk(directory):
            for file in files:
                total += 1
                fp = Path(file)
                suffix = fp.suffix.lower()
                if suffix == ".dcm":
                    dcm_count += 1
                elif suffix == "":
                    no_ext_count += 1
    except Exception as e:
        logger.error(f"Error counting files in {directory}: {str(e)}")

    return total, dcm_count, no_ext_count


@retry_on_failure(max_retries=2, delay=3)
def extract_iso(iso_record, base_output_dir, index, total_count):
    """
    Extract one ISO using PowerISO.
    
    Args:
        iso_record: Dictionary with ISO information
        base_output_dir: Base directory for extraction
        index: Current index for progress reporting
        total_count: Total number of ISOs for progress reporting
        
    Returns:
        Updated iso_record with extraction results
    """
    iso_path = iso_record["iso_path"]
    start_time = time.time()

    # Skip if already processed
    if iso_record.get("status") in ("extracted", "skipped", "error"):
        print(f"[{index}/{total_count}] Skipping already processed: {iso_path}")
        return iso_record

    print(f"[{index}/{total_count}] Starting extraction: {iso_path}")

    # Build flattened output directory
    flat_output = flatten_path(iso_path, base_output_dir)

    # Check if the folder already has enough files -> treat as extracted
    if os.path.exists(flat_output):
        existing_total, existing_dcm, existing_no_ext = count_files_in_dir(flat_output)
        if existing_total > EXTRACTION_FILE_THRESHOLD:
            iso_record["status"] = "extracted"
            iso_record["file_count"] = existing_total
            iso_record["dcm_count"] = existing_dcm
            iso_record["no_ext_count"] = existing_no_ext
            print(f"[{index}/{total_count}] Already extracted (found {existing_total} files): {iso_path}")
            return iso_record

    # Verify ISO file exists and path is safe
    if not os.path.exists(iso_path):
        iso_record["status"] = "error"
        iso_record["error_message"] = f"ISO file not found: {iso_path}"
        print(f"[{index}/{total_count}] {iso_record['error_message']}")
        return iso_record
    
    # Basic path validation for security
    if ".." in iso_path or not os.path.abspath(iso_path).startswith(os.path.abspath(".")):
        iso_record["status"] = "error"
        iso_record["error_message"] = f"Suspicious ISO file path detected: {iso_path}"
        logger.warning(f"Blocked suspicious path: {iso_path}")
        print(f"[{index}/{total_count}] {iso_record['error_message']}")
        return iso_record

    # Verify PowerISO exists
    if not os.path.exists(POWERISO_PATH):
        iso_record["status"] = "error"
        iso_record["error_message"] = f"PowerISO not found at: {POWERISO_PATH}. Set POWERISO_PATH environment variable."
        logger.error(iso_record['error_message'])
        print(f"[{index}/{total_count}] {iso_record['error_message']}")
        return iso_record

    # Create output directory
    try:
        os.makedirs(flat_output, exist_ok=True)
    except Exception as e:
        iso_record["status"] = "error"
        iso_record["error_message"] = f"Failed to create output directory: {str(e)}"
        print(f"[{index}/{total_count}] {iso_record['error_message']}")
        return iso_record

    # Attempt extraction
    cmd = [
        POWERISO_PATH,
        "extract",
        iso_path,
        "/",
        "-od",
        flat_output
    ]
    
    # Try piping "IgnoreAll" so it auto-continues on error
    returncode, out, err = run_shell_command(cmd, input_str="IgnoreAll\nIgnoreAll\n")

    if returncode != 0 or "Extraction failed" in out or "Extraction failed" in err:
        # Mark as error, store the error message
        iso_record["status"] = "error"
        error_msg = (err.strip() or out.strip())[:500]  # Limit length
        iso_record["error_message"] = error_msg
        print(f"[{index}/{total_count}] ERROR extracting {iso_path}\n  --> {error_msg}")
    else:
        # Check how many files we extracted
        total, dcm_count, no_ext_count = count_files_in_dir(flat_output)
        iso_record["file_count"] = total
        iso_record["dcm_count"] = dcm_count
        iso_record["no_ext_count"] = no_ext_count
        iso_record["extraction_time"] = round(time.time() - start_time, 2)

        if total > EXTRACTION_FILE_THRESHOLD:
            iso_record["status"] = "extracted"
        else:
            # Possibly partial, but let's call it extracted anyway
            iso_record["status"] = "extracted"
            
        print(f"[{index}/{total_count}] Extraction complete for {iso_path}\n"
              f"  --> total files: {total}, .dcm: {dcm_count}, no_ext: {no_ext_count}, "
              f"time: {iso_record['extraction_time']}s")

    return iso_record

def save_progress(iso_list: List[Dict[str, Any]]) -> bool:
    """
    Save current progress to JSON file.
    
    Args:
        iso_list: List of ISO records
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create a backup of the previous file if it exists
        if os.path.exists(ALL_JSONS_FILE):
            backup_file = f"{ALL_JSONS_FILE}.bak"
            try:
                os.replace(ALL_JSONS_FILE, backup_file)
            except Exception as e:
                logger.warning(f"Failed to create backup file: {str(e)}")
        
        # Write the new data
        with open(ALL_JSONS_FILE, "w", encoding="utf-8") as f:
            json.dump({"isos": iso_list}, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving progress: {str(e)}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract ISO files using PowerISO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Environment Variables:
  POWERISO_PATH       Path to PowerISO executable
  ISO_PROGRESS_FILE   Progress tracking file (default: All_jsons_in_dir.json)
  EXTRACTION_THRESHOLD Minimum files for successful extraction (default: 100)
  ISO_LOG_FILE        Log file path (default: iso_extractor.log)
  MAX_RETRIES         Maximum retry attempts (default: 3)
  RETRY_DELAY         Initial retry delay in seconds (default: 5)
        """
    )
    parser.add_argument("--search-dir", help="Directory to search for ISO files")
    parser.add_argument("--output-dir", default="flat_iso_extract", 
                        help="Subfolder name for extractions (default: flat_iso_extract)")
    parser.add_argument("--poweriso-path", help="Override PowerISO executable path")
    return parser.parse_args()


def main() -> None:
    """
    Main program flow:
     - Get search and output directories
     - Load or create the JSON with ISO records
     - Process each ISO in a for-loop
     - After each ISO, save progress to JSON
    """
    args = parse_args()
    
    # Override PowerISO path if provided
    global POWERISO_PATH
    if args.poweriso_path:
        POWERISO_PATH = args.poweriso_path
        logger.info(f"Using PowerISO path: {POWERISO_PATH}")
    
    # Get search directory
    search_dir = args.search_dir
    if not search_dir:
        search_dir = input("Enter the directory to search for .iso files: ").strip()
    
    if not search_dir:
        logger.error("No search directory provided. Exiting.")
        return
    
    if not os.path.isdir(search_dir):
        logger.error(f"Search directory does not exist: {search_dir}")
        return
    
    # Get output directory
    output_dir = args.output_dir
    if not args.search_dir:  # Only prompt if not provided via command line
        user_output = input(f"Enter the subfolder name to flatten ISO extractions (default: {output_dir}): ").strip()
        if user_output:
            output_dir = user_output

    # Load or build the iso list
    if os.path.exists(ALL_JSONS_FILE):
        try:
            with open(ALL_JSONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            iso_list = data.get("isos", [])
            logger.info(f"Loaded {len(iso_list)} ISOs from {ALL_JSONS_FILE}.")
        except Exception as e:
            logger.error(f"Error loading JSON file: {str(e)}")
            logger.info("Creating new ISO list.")
            iso_list = []
    else:
        iso_list = []
    
    # If we have no ISOs yet, search for them
    if not iso_list:
        found_isos = find_iso_files(search_dir)
        logger.info(f"Found {len(found_isos)} ISOs in {search_dir}.")
        
        for iso_path in found_isos:
            iso_list.append({
                "iso_path": iso_path,
                "status": "pending"
            })
        
        # Save initial list
        if save_progress(iso_list):
            logger.info(f"Saved initial ISO list to {ALL_JSONS_FILE}.")
        else:
            logger.error("Failed to save initial ISO list.")
            return

    # Filter to find any that are still pending
    pending_records = [iso for iso in iso_list 
                      if iso.get("status") not in ("extracted", "skipped", "error")]
    
    total_pending = len(pending_records)
    if total_pending == 0:
        logger.info("No pending ISOs to extract.")
    else:
        logger.info(f"Preparing to extract {total_pending} ISOs...")

    # Process each pending ISO
    for i, iso_record in enumerate(pending_records, start=1):
        # Extract the ISO
        extract_iso(iso_record, output_dir, i, total_pending)
        
        # Save progress after each extraction
        if i % 5 == 0 or i == total_pending:  # Save every 5 ISOs or at the end
            if save_progress(iso_list):
                logger.info(f"Saved progress ({i}/{total_pending} complete)")
            else:
                logger.warning(f"Failed to save progress at item {i}")

    # Final summary
    extracted = sum(1 for iso in iso_list if iso.get("status") == "extracted")
    errors = sum(1 for iso in iso_list if iso.get("status") == "error")
    skipped = sum(1 for iso in iso_list if iso.get("status") == "skipped")
    pending = sum(1 for iso in iso_list if iso.get("status") == "pending")
    
    logger.info("Extraction process complete!")
    logger.info(f"Summary: {extracted} extracted, {errors} errors, {skipped} skipped, {pending} pending")
    logger.info(f"Results saved to {ALL_JSONS_FILE}")
    logger.info(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
        print("\nProcess interrupted. Exiting gracefully...")
    except Exception as e:
        logger.exception("Unexpected error occurred")
        print(f"An unexpected error occurred: {str(e)}")