# S0_zipExtract.py

## Purpose
Fast ZIP file extraction with smart logic to avoid re-extracting already processed files.

## Code Logic
- Recursively discovers ZIP files in directory tree
- Smart extraction checking using checksums and modification times
- Concurrent processing with ThreadPoolExecutor
- Progress tracking and resume capability
- Security validation for file paths

## Inputs
- Root directory containing ZIP files
- Number of worker threads (default: 4)
- Integrity checking enabled/disabled
- Overwrite mode (force re-extraction)

## Outputs
- Extracted ZIP contents (preserving filenames)
- Progress tracking: "S0_zipExtract_progress.json"
- Extraction records: "S0_extracted_files.json"
- Log file: "S0_zipExtract.log"

## User Changeable Settings
- `--workers`: Number of concurrent workers (default: 4)
- `--overwrite`: Force overwrite existing extractions
- `--no-integrity-check`: Disable checksum validation
- Progress save interval and logging levels in code