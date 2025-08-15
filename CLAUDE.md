# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DICOM file processing pipeline (FICZall - Find, Index, Clean, Zip all DICOM Files and Reports) that extracts ISO files, indexes DICOM files, processes metadata, and creates compressed archives for storage.

## Pipeline Architecture

The pipeline consists of 5 sequential stages:

1. **S0_isoExtract.py**: Extracts ISO files using PowerISO
   - Finds all ISO files in a directory tree
   - Extracts them to a flattened directory structure
   - Tracks progress in `All_jsons_in_dir.json`
   - Requires PowerISO executable at configured path

2. **S1_indexingFiles_E2.py**: Indexes and extracts DICOM metadata
   - Discovers all folders and files in the extracted data
   - Identifies DICOM files (.dcm or no extension)
   - Extracts DICOM metadata using pydicom
   - Creates JSONL files with metadata for each patient/study

3. **S2_concatIndex_E2.py**: Aggregates patient data
   - Processes JSONL files from S1
   - Extracts first-line patient data from each file
   - Counts CT objects per study
   - Outputs consolidated JSON and Excel files

4. **S3_processForStore.py**: Filters and prepares studies for storage
   - Applies configurable filters (minimum file count, duplicate removal)
   - Processes studies meeting criteria
   - Prepares data for compression

5. **S4_zipStore.py**: Creates ZIP archives for each study
   - Reads processed JSONL files from S3
   - Creates one ZIP file per DICOM study
   - Supports resume capability and concurrent processing
   - Tracks compression statistics

## Common Development Commands

```bash
# Run individual stages
python code/S0_isoExtract.py --search-dir [DIR] --output-dir flat_iso_extract
python code/S1_indexingFiles_E2.py --root [ROOT_DIR] --output_dir [OUTPUT_DIR]
python code/S2_concatIndex_E2.py --folder_path [PATH] --output_json [JSON] --output_excel [EXCEL]
python code/S3_processForStore.py --input-dir [DIR] --output-dir [DIR] --min-files [N]
python code/S4_zipStore.py --input-dir [DIR] --output-dir [DIR] --workers [N]

# Run multiple instances in parallel
python code/Run_multiple.py  # Configured for parallel execution of S1 and S2
```

## Key Dependencies

- **pydicom**: DICOM file reading and metadata extraction
- **pandas**: Data manipulation for Excel output
- **python-docx**: Document processing
- **PyPDF2**: PDF processing
- **tqdm**: Progress bars
- **asyncio/concurrent.futures**: Parallel processing
- **PowerISO**: External tool for ISO extraction (Windows-specific path configured in S0)

## Important Configuration

- **Virtual environment path**: Configured in Run_multiple.py for parallel execution
- **PowerISO path**: Hardcoded in S0_isoExtract.py, needs adjustment per system
- **Skip files**: Configurable via SKIP_FILES environment variable or DEFAULT_SKIP_FILES lists in S2, S3, S4
- **Concurrency settings**: Adjustable worker counts in S1, S4, and Run_multiple.py

## Data Flow

1. ISO files → Extracted DICOM files (S0)
2. DICOM files → JSONL metadata files per folder (S1)
3. JSONL files → Aggregated patient JSON/Excel (S2)
4. JSONL files → Filtered/processed JSONL (S3)
5. Processed JSONL → ZIP archives per study (S4)

## Error Handling

- All stages implement atomic file writes using temporary files
- Progress tracking with resume capability (S0, S1, S4)
- Comprehensive logging to file and console
- Error records maintained in output JSON files

## Parallel Processing Support

- Run_multiple.py orchestrates parallel execution of multiple S1/S2 instances
- Configurable worker pools for concurrent processing
- ProcessPoolExecutor for CPU-bound tasks
- ThreadPoolExecutor for I/O-bound operations