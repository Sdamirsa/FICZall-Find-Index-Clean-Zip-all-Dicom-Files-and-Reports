# S3_path2_consolidateSources.py

## Purpose
Consolidates patient summaries from multiple processed folders into unified Excel and JSON files.

## Code Logic
- Finds S2_patient_summary.json files in processed directories
- Detects duplicate patients using ID matching and name similarity
- Identifies patients with multiple studies across different dates
- Provides both GUI and CLI interfaces for operation
- Creates comprehensive consolidated reports

## Inputs
- Processed directory containing S2_concatenated_summaries folders
- OR specific list of folders to process
- Output directory for consolidated results

## Outputs
- Timestamped consolidated JSON and Excel files
- Statistics file with folder processing details
- Multi-sheet Excel with patients, statistics, and summary data

## User Changeable Settings
- Duplicate detection criteria (80% name similarity threshold)
- Patient ID validation length (>3 characters)
- GUI/CLI mode selection
- Output directory paths
- Duplicate matching algorithm parameters in code