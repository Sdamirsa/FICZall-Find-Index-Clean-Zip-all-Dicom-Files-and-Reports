# S4_path1_zipStore.py

## Purpose
Creates ZIP archives for DICOM studies from processed JSONL files with resume capability.

## Code Logic
- Processes JSONL files from S3_processForStore directory
- Creates individual ZIP archives for each study
- Includes DICOM files plus metadata JSON companions
- Concurrent processing with configurable workers
- Resume capability skips already processed files
- Validates ZIP integrity and handles corrupted files

## Inputs
- Input directory containing processed JSONL files
- Output directory for ZIP archives
- Concurrency and compression settings

## Outputs
- Individual ZIP files named by study
- Progress tracking: "S4_zipStore_processing_progress.json"
- Processing statistics and error logs
- Comprehensive processing logs in logs/ subdirectory

## User Changeable Settings
- `--concurrency`: Number of concurrent workers (default: 4)
- `ZIP_COMPRESSION_LEVEL`: Compression level 0-9 (default: 6)
- `MAX_ZIP_SIZE_GB`: Maximum ZIP file size (default: 10GB)
- `SKIP_FILES`: Files to skip during processing
- `--force-reprocess`: Ignore previous progress and reprocess all