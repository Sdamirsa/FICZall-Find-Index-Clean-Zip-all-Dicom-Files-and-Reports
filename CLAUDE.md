# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DICOM file processing pipeline (FICZall - Find, Index, Clean, Zip all DICOM Files and Reports) that extracts ISO files, indexes DICOM files, processes metadata, and creates compressed archives for storage.

**Key Features:**
- Zero-dependency startup (only requires Python 3.6+)
- Automatic virtual environment management
- Interactive configuration wizard
- Resume capability for interrupted processing
- Parallel processing support
- Comprehensive error handling and logging

## Pipeline Architecture

The pipeline consists of 7 stages (2 optional pre-processing + 4 core + 1 optional AI stage):

1. **S0_zipExtract.py**: Fast extraction of ZIP files (NEW - Optional)
   - Recursively finds all ZIP files in directory tree
   - Smart extraction logic avoids re-processing
   - Multi-threaded concurrent processing
   - Integrity checking with MD5 checksums
   - Resume capability with progress tracking

2. **S0_isoExtract.py**: Extracts ISO files using PowerISO
   - Finds all ISO files in a directory tree
   - Extracts them to a flattened directory structure
   - Tracks progress in `All_jsons_in_dir.json`
   - Requires PowerISO executable at configured path

3. **S1_indexingFiles_E2.py**: Indexes and extracts DICOM metadata
   - Discovers all folders and files in the extracted data
   - Identifies DICOM files (.dcm or no extension)
   - Extracts DICOM metadata using pydicom
   - Creates JSONL files with metadata for each patient/study
   - Also creates S1_indexingFiles_allDocuments.json for document processing

4. **S2_concatIndex_E2.py**: Aggregates patient data
   - Processes JSONL files from S1
   - Extracts first-line patient data from each file
   - Counts CT objects per study
   - Outputs consolidated JSON and Excel files

5. **S3_processForStore.py**: Filters and prepares studies for storage
   - Applies configurable filters (minimum file count, duplicate removal)
   - Processes studies meeting criteria
   - Prepares data for compression

6. **S4_zipStore.py**: Creates ZIP archives for each study
   - Reads processed JSONL files from S3
   - Creates one ZIP file per DICOM study
   - Supports resume capability and concurrent processing
   - Tracks compression statistics

7. **S5_llmExtract.py**: AI-powered medical document extraction (NEW - Optional)
   - Processes documents from S1_indexingFiles_allDocuments.json
   - Uses OpenAI API with structured outputs (Pydantic models)
   - Extracts patient info, imaging details, findings, medical history
   - Handles large datasets with chunking system
   - Resume capability and overwrite protection
   - Supports multiple LLM backends (OpenAI, Ollama)

## Main Entry Point

**IMPORTANT**: Always use `run.py` as the main entry point:

```bash
# Interactive mode (recommended)
python run.py

# Setup only
python run.py --setup

# Run specific stage
python run.py --stage S1

# Non-interactive with existing config
python run.py --non-interactive

# Show configuration
python run.py --config
```

The `run.py` script:
- Works with basic Python installation (no pip packages required)
- Creates and manages virtual environment automatically
- Installs all dependencies in isolated environment
- Provides interactive configuration
- Runs pipeline stages through the virtual environment

## Development Commands

For development/debugging individual stages (run from within venv):

```bash
# Activate virtual environment first
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run individual stages
python code/S0_zipExtract.py --root-dir [DIR] --workers 8 --overwrite
python code/S0_isoExtract.py --search-dir [DIR] --output-dir flat_iso_extract
python code/S1_indexingFiles_E2.py --root [ROOT_DIR] --output_dir [OUTPUT_DIR]
python code/S2_concatIndex_E2.py --folder_path [PATH] --output_json [JSON] --output_excel [EXCEL]
python code/S3_processForStore.py --input-dir [DIR] --output-dir [DIR] --min-files [N]
python code/S4_zipStore.py --input-dir [DIR] --output-dir [DIR] --workers [N]
python code/S5_llmExtract.py --input [JSON] --output [DIR] --overwrite --chunk-size 500

# Note: Multi-runner functionality has been removed for simplicity
```

## Key Dependencies

### Python Packages (auto-installed by run.py)
- **pydicom**: DICOM file reading and metadata extraction
- **pandas**: Data manipulation for Excel output (optional)
- **openpyxl**: Excel file support (optional)
- **python-docx**: Document processing (optional)
- **PyPDF2**: PDF processing (optional)
- **tqdm**: Progress bars
- **numpy**: Array operations (optional, improves pydicom performance)
- **Pillow**: Image processing (optional)
- **openai**: AI-powered document extraction (S5 stage)
- **pydantic**: Data validation for structured outputs (S5 stage)

### External Tools
- **PowerISO**: Required only for S0 stage (ISO extraction on Windows)
  - Download from: https://www.poweriso.com/
  - Path configured in `run_config.py`

## Configuration System

### Central Configuration (`run_config.py`)
All settings are centralized in `run_config.py` with environment variable overrides:
- **Paths**: All paths are dynamic, relative to repository root
- **PowerISO**: Example path provided, users update as needed
- **Processing parameters**: Workers, compression, thresholds
- **Stage control**: Which stages to run
- **Skip files**: Files to ignore during processing

### Environment Variables
Every setting can be overridden via environment variables:
```bash
export S1_ROOT_DIR=/path/to/dicom
export ZIP_COMPRESSION_LEVEL=9
export S3_MIN_FILES=20
export MAX_WORKERS=8
```

### Key Configuration Files
- `run_config.py`: Central configuration (created on first run)
- `S5_llmExtract_config.py`: AI extraction configuration (Pydantic models, LLM settings)
- `requirements.txt`: Python dependencies
- `All_jsons_in_dir.json`: S0 ISO extraction progress tracking
- `S0_zipExtract_progress.json`: S0 ZIP extraction progress tracking
- `S4_zipStore_processing_progress.json`: S4 progress tracking
- `S5_llmExtract_progress.json`: S5 AI extraction progress tracking

## Data Flow

```
┌─────────────┐
│  ZIP Files  │ (Optional - S0 ZIP Extract)
└──────┬──────┘
       ↓ Concurrent extraction
┌─────────────┐
│  ISO Files  │ (Optional - S0 ISO Extract)
└──────┬──────┘
       ↓ PowerISO extraction
┌─────────────┐
│ DICOM Files │ (Required - S1 starts here)
└──────┬──────┘
       ↓ Metadata extraction
┌─────────────┐
│ JSONL Files │ (Per folder metadata)
│ + Documents │ (S1_indexingFiles_allDocuments.json)
└──────┬──────┘
       ├────────────────┬─────────────┐
       ↓ S2             ↓ S3          ↓ S5 (Optional)
┌──────────────┐  ┌────────────┐  ┌────────────┐
│ Patient      │  │ Filtered   │  │ AI Extract │
│ Summary      │  │ Studies    │  │ Medical    │
│ (JSON/Excel) │  └─────┬──────┘  │ Data       │
└──────────────┘        ↓ S4      └────────────┘
                  ┌────────────┐
                  │ ZIP Files  │
                  │ (Archives) │
                  └────────────┘
```

### File Formats

- **Input**: ISO files (S0) or DICOM files (.dcm or no extension)
- **Intermediate**: JSONL files (one JSON object per line)
- **Output**: 
  - JSON summaries (S2)
  - Excel reports (S2, optional)
  - ZIP archives with metadata (S4)

## Error Handling & Resilience

### Automatic Recovery
- **Virtual environment**: Auto-created if missing
- **Dependencies**: Graceful degradation for optional packages
- **Configuration**: Works even if run_config.py doesn't exist
- **Paths**: All paths validated and created as needed

### Progress Tracking
- **Resume capability**: S0, S1, S4 can resume from interruption
- **Atomic writes**: Using temporary files to prevent corruption
- **Progress files**: JSON files track completed work
- **Comprehensive logging**: Both file and console output

### Error Recovery
- **Retry mechanisms**: Automatic retries with exponential backoff (S0)
- **Validation**: File paths and data validated before processing
- **Error records**: Maintained in output JSON files
- **Skip lists**: Problematic files can be skipped

## Parallel Processing Support

- **Automatic worker sizing**: Based on CPU count
- **Configurable concurrency**: Via MAX_WORKERS, S1_DICOM_CONCURRENCY, etc.
- **Multiple execution modes**:
  - Sequential (default for safety)
  - Parallel within stages (ThreadPoolExecutor)
- **Resource management**: Automatic cleanup and resource limits

## Testing & Validation

When testing changes:
1. Always test with `python run.py` first (not individual scripts)
2. Test with `--setup` to ensure clean environment
3. Verify paths are relative (no hardcoded absolute paths)
4. Check that all stages can resume from interruption
5. Validate output formats match expected schemas

## Code Style Guidelines

- **Type hints**: Use for function parameters and returns
- **Docstrings**: Required for all public functions
- **Error messages**: Include context and recovery suggestions
- **Logging**: Use appropriate levels (INFO, WARNING, ERROR)
- **Path handling**: Always use pathlib.Path for cross-platform compatibility