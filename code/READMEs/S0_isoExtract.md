# S0_isoExtract.py

## Purpose
Extracts ISO files using PowerISO with progress tracking and resume capability.

## Code Logic
- Recursively finds ISO files in a directory
- Uses PowerISO to extract each ISO file to a flattened directory structure
- Implements retry mechanism with exponential backoff
- Tracks progress in JSON file for resume capability
- Validates extracted file counts against threshold

## Inputs
- Search directory containing ISO files
- Output directory name (default: "flat_iso_extract")
- PowerISO executable path

## Outputs
- Extracted ISO contents in flattened directory structure
- Progress tracking file: "All_jsons_in_dir.json"
- Log file: "iso_extractor.log"

## User Changeable Settings
- `POWERISO_PATH`: Path to PowerISO executable
- `EXTRACTION_FILE_THRESHOLD`: Minimum files for successful extraction (default: 100)
- `MAX_RETRIES`: Maximum retry attempts (default: 3)
- `RETRY_DELAY`: Initial retry delay in seconds (default: 5)

Set via environment variables or modify constants in code.