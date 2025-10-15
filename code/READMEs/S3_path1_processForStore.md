# S3_path1_processForStore.py

## Purpose
Filters and prepares JSONL files for storage by applying quality filters and removing duplicates.

## Code Logic
- Applies minimum file count filter to studies
- Removes duplicate records within each JSONL file
- Processes files that pass all filter criteria
- Comprehensive logging and statistics tracking
- Skips configured files and already processed studies

## Inputs
- Input directory containing JSONL files
- Minimum files per study threshold
- Optional custom filter functions

## Outputs
- Filtered JSONL files with "S3_processForStore_" prefix
- "S3_processForStore.json": Processing summary and statistics
- Detailed log files with timestamps

## User Changeable Settings
- `--min_files`: Minimum number of files per study (default: 1)
- `SKIP_FILES`: Environment variable for files to skip
- Filter functions can be customized in code
- Progress save intervals and logging levels
- Output directory can be specified