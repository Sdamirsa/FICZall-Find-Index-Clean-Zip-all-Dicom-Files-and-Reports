#!/usr/bin/env python3
"""
Configuration file for DICOM Processing Pipeline
=================================================

This file contains all configurable parameters for the pipeline.
Users should review and modify these settings before running the pipeline.

IMPORTANT CONFIGURATION GUIDE:
==============================

1. PATHS:
   - Use forward slashes (/) or double backslashes (\\) in paths
   - Examples:
     * Windows: r"C:\\Users\\YourName\\Documents\\Data"
     * Linux/Mac: "/home/username/data"
     * Relative: "./my_data" (relative to this repository)

2. REQUIRED STAGES: S1, S2, S3
   - S1: Must have S1_ROOT_DIR pointing to your DICOM files
   - S2: Will use S1 output automatically
   - S3: Will use S1 output automatically

3. OPTIONAL STAGES:
   - S0: Only needed if you have ISO files to extract
   - S4: Only needed if you want ZIP archives for storage

4. COMMON CONFIGURATIONS:
   - Small dataset: Set MAX_WORKERS='2', S3_MIN_FILES='5'
   - Large dataset: Set MAX_WORKERS='8', S3_MIN_FILES='20'
   - Fast processing: Set ZIP_COMPRESSION_LEVEL='1'
   - Max compression: Set ZIP_COMPRESSION_LEVEL='9'

5. TROUBLESHOOTING:
   - If paths have spaces, wrap in quotes: r"C:\\My Folder\\Data"
   - If PowerISO not found, set full path to piso.exe
   - If memory issues, reduce MAX_WORKERS and batch sizes

All paths can be:
1. Absolute paths (e.g., C:\\Data\\MyFiles)
2. Relative paths (e.g., ./data)
3. Environment variables (will be loaded if set)
"""

import os
from pathlib import Path


# =============================================================================
# ⚠️  ESSENTIAL USER CONFIGURATION - MUST BE SET BY USER ⚠️
# =============================================================================

# 📍 STEP 1: Set the location of your DICOM data (REQUIRED)
# This is the full path to the folder containing your DICOM files
# Examples:
#   Windows: r"C:\Medical_Data\Hospital_Archive\CT_Scans"
#   Linux: "/home/user/medical_data/dicom_files"
#   Network: r"\\server\medical_data\dicom"
Location_of_your_data = r"C:\Users\LEGION\Documents\GIT\FICZall-Find-Index-Clean-Zip-all-Dicom-Files-and-Reports\data\raw\test"  # ⚠️  CHANGE THIS PTH! ⚠️A

# 📍 STEP 2: Choose a project name (OPTIONAL - will auto-generate if not set)
# This name will be used to create a folder in the 'data' directory for all outputs
# Examples: "hospital_a_ct_study", "cardiac_research_2024", "pilot_study_batch1"  
# If left empty, will use the last two folder names from Location_of_your_data
desired_name_of_project = ""  # Leave empty for auto-naming: flatten two parent directory names


# =============================================================================
# VALIDATION AND PROJECT SETUP - DO NOT MODIFY THIS SECTION
# =============================================================================

# Get the base directory (repository root)
BASE_DIR = Path(__file__).parent.resolve()
CODE_DIR = BASE_DIR / "code"
DATA_DIR = BASE_DIR / "data" / "processed"

# Validate essential settings
if not Location_of_your_data:
    raise ValueError(
        "⚠️  ERROR: You MUST set 'Location_of_your_data' to point to your DICOM files!\n"
        "Please edit run_config.py and set Location_of_your_data to your actual DICOM directory.\n"
        "Example: Location_of_your_data = r'C:\\MyData\\DICOM_Files'"
    )

if not os.path.exists(Location_of_your_data):
    raise ValueError(
        f"⚠️  ERROR: The specified data location does not exist: {Location_of_your_data}\n"
        f"Please check the path and make sure the directory exists."
    )

# Create project name from path if not provided
if not desired_name_of_project or desired_name_of_project.strip() == "":
    # Get the last two folder names and flatten them
    path_parts = Path(Location_of_your_data).parts
    if len(path_parts) >= 2:
        # Take last two parts and join with _._
        desired_name_of_project = f"{path_parts[-2]}_._{path_parts[-1]}"
    else:
        # Fallback to just the last part
        desired_name_of_project = path_parts[-1] if path_parts else "dicom_project"
    
    # Clean the name to be filesystem-friendly (preserve _._ delimiter)
    import re
    # First protect the _._ delimiter
    temp_name = desired_name_of_project.replace("_._", "|||DELIMITER|||")
    # Clean other characters
    temp_name = re.sub(r'[^\w\-_]', '_', temp_name)
    # Restore the _._ delimiter
    desired_name_of_project = temp_name.replace("|||DELIMITER|||", "_._")
    print(f"Auto-generated project name: {desired_name_of_project}")

# Create the project directory in data folder
PROJECT_DIR = DATA_DIR / desired_name_of_project
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Project directory: {PROJECT_DIR}")

# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================

# Virtual environment path (None to use system Python)
# This will be created automatically if it doesn't exist
VENV_PATH = os.getenv('VENV_PATH', str(BASE_DIR / "venv"))

# Maximum number of parallel workers for processing
MAX_WORKERS = os.getenv('MAX_WORKERS', str(min(4, os.cpu_count() or 1)))

# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# =============================================================================
# S0 - ZIP EXTRACTION CONFIGURATION (NEW STAGE)
# =============================================================================

# Root directory containing ZIP files to extract
S0_ZIP_ROOT_DIR = os.getenv('S0_ZIP_ROOT_DIR', Location_of_your_data)  # Uses same location as your data by default

# Number of worker threads for parallel extraction
S0_ZIP_WORKERS = os.getenv('S0_ZIP_WORKERS', '4')

# Enable integrity checking with MD5 checksums
S0_ZIP_INTEGRITY_CHECK = os.getenv('S0_ZIP_INTEGRITY_CHECK', 'yes')  # 'yes' or 'no'

# Overwrite existing extractions
S0_ZIP_OVERWRITE = os.getenv('S0_ZIP_OVERWRITE', 'no')  # 'yes' or 'no'

# =============================================================================
# S0 - ISO EXTRACTION CONFIGURATION
# =============================================================================

# Path to PowerISO executable (Windows only)
# Download from: https://www.poweriso.com/download.php
# Example: r"C:\Users\LEGION\Downloads\PowerISO.9.0.Portable\PowerISO.9.0.Portable\App\PowerISO\piso.exe"
POWERISO_PATH = os.getenv('POWERISO_PATH', r"C:\Users\LEGION\Downloads\PowerISO.9.0.Portable\PowerISO.9.0.Portable\App\PowerISO\piso.exe")

# ISO extraction settings
ISO_SEARCH_DIR = os.getenv('ISO_SEARCH_DIR', '')  # Will prompt if empty
ISO_OUTPUT_DIR = os.getenv('ISO_OUTPUT_DIR', 'flat_iso_extract')
ISO_PROGRESS_FILE = os.getenv('ISO_PROGRESS_FILE', 'All_jsons_in_dir.json')
EXTRACTION_THRESHOLD = os.getenv('EXTRACTION_THRESHOLD', '100')
ISO_LOG_FILE = os.getenv('ISO_LOG_FILE', 'iso_extractor.log')
MAX_RETRIES = os.getenv('MAX_RETRIES', '3')
RETRY_DELAY = os.getenv('RETRY_DELAY', '5')

# =============================================================================
# S1 - INDEXING FILES CONFIGURATION
# =============================================================================

# ⚠️  MOST IMPORTANT SETTING: Root directory containing your DICOM files ⚠️
# CHANGE THIS PATH to point to your DICOM data:
# Examples:
#   Windows: r"C:\\Medical_Data\\DICOM_Files" 
#   Linux: "/home/user/medical_data/dicom"
#   Network: r"\\\\server\\medical_data"
# Root directory containing your DICOM files (automatically set from Location_of_your_data)
S1_ROOT_DIR = os.getenv('S1_ROOT_DIR', Location_of_your_data)
S1_OUTPUT_DIR = os.getenv('S1_OUTPUT_DIR', str(PROJECT_DIR / 'S1_indexed_metadata'))

# Concurrency settings for S1
S1_DICOM_CONCURRENCY = os.getenv('S1_DICOM_CONCURRENCY', '3') # Number of documents to process concurrently --> monitor CPU usage and adjust if needed
S1_DOC_CONCURRENCY = os.getenv('S1_DOC_CONCURRENCY', '2') # Number of documents to process concurrently --> monitor CPU usage and adjust if needed
S1_BATCH_SIZE = os.getenv('S1_BATCH_SIZE', '50') # number of files to process in each batch --> monitor memory usage and adjust if needed

# =============================================================================
# S2 - CONCATENATE INDEX CONFIGURATION
# =============================================================================

# Input folder containing JSONL files from S1 (automatically uses S1 output)
S2_INPUT_DIR = os.getenv('S2_INPUT_DIR', str(PROJECT_DIR / 'S1_indexed_metadata'))
S2_OUTPUT_DIR = os.getenv('S2_OUTPUT_DIR', str(PROJECT_DIR / 'S2_concatenated_summaries'))
S2_OUTPUT_JSON = os.getenv('S2_OUTPUT_JSON', str(Path(S2_OUTPUT_DIR) / 'S2_patient_summary.json'))
S2_OUTPUT_EXCEL = os.getenv('S2_OUTPUT_EXCEL', str(Path(S2_OUTPUT_DIR) / 'S2_patient_summary.xlsx'))

# =============================================================================
# S3 - PROCESS FOR STORE CONFIGURATION
# =============================================================================

# Input directory for S3 (automatically uses S1 output)
S3_INPUT_DIR = os.getenv('S3_INPUT_DIR', str(PROJECT_DIR / 'S1_indexed_metadata'))
S3_OUTPUT_DIR = os.getenv('S3_OUTPUT_DIR', str(PROJECT_DIR / 'S3_filtered_studies'))

# Minimum number of files required per study
S3_MIN_FILES = os.getenv('S3_MIN_FILES', '10') # ⚠️ Review this setting based on your dataset size. In my case I don't have any dicom with less than 10 data as I am working wiht CT scans ⚠️

# =============================================================================
# S4 - ZIP STORE CONFIGURATION
# =============================================================================

# Input directory for S4 (automatically uses S3 output)
S4_INPUT_DIR = os.getenv('S4_INPUT_DIR', str(PROJECT_DIR / 'S3_filtered_studies'))
S4_OUTPUT_DIR = os.getenv('S4_OUTPUT_DIR', str(PROJECT_DIR / 'S4_zip_archives'))

# ZIP compression settings
ZIP_COMPRESSION_LEVEL = os.getenv('ZIP_COMPRESSION_LEVEL', '6')  # 0-9
MAX_ZIP_SIZE_GB = os.getenv('MAX_ZIP_SIZE_GB', '10')
S4_PROGRESS_FILE = os.getenv('S4_PROGRESS_FILE', 'S4_zipStore_processing_progress.json')
S4_WORKERS = os.getenv('S4_WORKERS', '2')

# =============================================================================
# S5 - LLM DOCUMENT EXTRACTION CONFIGURATION (NEW STAGE)
# =============================================================================

# NOTE: Core LLM API settings (BASE_URL, API_KEY, MODEL_NAME, etc.) are in S5_llmExtract_config.py
# This section contains path, processing, and basic client type settings

# Input file path (S1_indexingFiles_allDocuments.json from S1)
S5_INPUT_FILE = os.getenv('S5_INPUT_FILE', str(Path(S1_OUTPUT_DIR) / "S1_indexingFiles_allDocuments.json"))

# Output directory for extracted medical data
S5_OUTPUT_DIR = os.getenv('S5_OUTPUT_DIR', str(PROJECT_DIR / 'S5_llm_extractions'))

# Batch size for progress saving
S5_BATCH_SIZE = os.getenv('S5_BATCH_SIZE', '10')

# Chunk size for handling large datasets
S5_CHUNK_SIZE = os.getenv('S5_CHUNK_SIZE', '1000')

# Overwrite existing extractions
S5_OVERWRITE = os.getenv('S5_OVERWRITE', 'false')  # 'true' or 'false'

# ⚠️ PRIVACY WARNING FOR LLM USAGE ⚠️
# =================================
# When using OpenAI or other external APIs, patient data will be sent to external servers.
# For HIPAA compliance, consider using local Ollama models.
#
# Client type selection (auto, openai, ollama)
S5_LLM_CLIENT_TYPE = os.getenv('S5_LLM_CLIENT_TYPE', 'auto')  # auto-detects based on configuration
#
# Ollama Configuration (only used when client type is 'ollama' or auto-detected)
S5_OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
S5_OLLAMA_MODEL = os.getenv('S5_LLM_MODEL', 'gpt-oss:20b')  # Model name for Ollama
#
# Quick setup examples:
# For LOCAL processing (private, HIPAA-friendly):
#   - Install Ollama: https://ollama.ai/
#   - Run: ollama pull gpt-oss:20b
#   - Run: ollama serve
#   - Set S5_LLM_CLIENT_TYPE='ollama'
#
# For OPENAI API:
#   - Set OPENAI_API_KEY environment variable
#   - Set S5_LLM_CLIENT_TYPE='openai'
#   - ⚠️ Patient data will be sent to OpenAI servers

# =============================================================================
# FILES TO SKIP (COMMON ACROSS ALL STAGES)
# =============================================================================

# Comma-separated list of files to skip during processing
SKIP_FILES = os.getenv('SKIP_FILES', ','.join([
    'ALL_PATIENTS_inCT.json',
    'ALL_PATIENTS_inCT.jsonl',
    'summary.jsonl',
    'CONCAT_ALL.jsonl',
    'processed_files.jsonl',
    'temp.jsonl',
    'backup.jsonl',
    'test.jsonl',
    'S2_ALL_PATIENTS_inCT.json',
    'S2_ALL_PATIENTS_inCT.xlsx',
    'S3_processForStore.json',
    '.DS_Store',
    'Thumbs.db',
    'desktop.ini'
]))

# =============================================================================
# PIPELINE EXECUTION CONFIGURATION
# =============================================================================

# Which stages to run (set to 'true' or 'false')
# 
# TYPICAL WORKFLOWS:
# ==================
# Most users (DICOM files in folders): S1=true, S2=true, S3=true, S4=false
# ZIP extraction needed: S0_ZIP=true, S1=true, S2=true, S3=true
# ISO extraction needed: S0_ISO=true, S1=true, S2=true, S3=true, S4=false  
# Full archival workflow: S1=true, S2=true, S3=true, S4=true
# AI-powered extraction: S1=true, S2=true, S3=true, S5=true
#
# REQUIRED STAGES (recommended to keep as 'true'):
RUN_S1_INDEXING = os.getenv('RUN_S1_INDEXING', 'true')  # REQUIRED: Index DICOM files
RUN_S2_CONCAT = os.getenv('RUN_S2_CONCAT', 'true')      # REQUIRED: Create patient summaries
RUN_S3_PROCESS = os.getenv('RUN_S3_PROCESS', 'true')    # REQUIRED: Filter and process studies

# OPTIONAL STAGES (set to 'true' only if needed):
RUN_S0_ZIP_EXTRACT = os.getenv('RUN_S0_ZIP_EXTRACT', 'false')  # OPTIONAL: Only if you have ZIP files to extract
RUN_S0_ISO_EXTRACT = os.getenv('RUN_S0_ISO_EXTRACT', 'false')  # OPTIONAL: Only if you have ISO files
RUN_S4_ZIP = os.getenv('RUN_S4_ZIP', 'false')                  # OPTIONAL: Only if you want ZIP archives
RUN_S5_LLM_EXTRACT = os.getenv('RUN_S5_LLM_EXTRACT', 'false')  # OPTIONAL: AI-powered document extraction

# Run stages sequentially or in parallel where possible
RUN_SEQUENTIAL = os.getenv('RUN_SEQUENTIAL', 'false') # Don't change this, as I haven't completed integrating sequential processing yet.

# =============================================================================
# MULTI-RUNNER CONFIGURATION (for parallel processing)
# =============================================================================

# Configuration file for multi-runner
# Note: Multi-runner functionality has been removed for simplicity

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config_dict():
    """Get all configuration as a dictionary."""
    return {
        # General
        'BASE_DIR': str(BASE_DIR),
        'CODE_DIR': str(CODE_DIR),
        'DATA_DIR': str(DATA_DIR),
        'VENV_PATH': VENV_PATH,
        'MAX_WORKERS': MAX_WORKERS,
        'LOG_LEVEL': LOG_LEVEL,
        
        # S0_ZIP
        'S0_ZIP_ROOT_DIR': S0_ZIP_ROOT_DIR,
        'S0_ZIP_WORKERS': S0_ZIP_WORKERS,
        'S0_ZIP_INTEGRITY_CHECK': S0_ZIP_INTEGRITY_CHECK,
        'S0_ZIP_OVERWRITE': S0_ZIP_OVERWRITE,
        
        # S0_ISO
        'POWERISO_PATH': POWERISO_PATH,
        'ISO_SEARCH_DIR': ISO_SEARCH_DIR,
        'ISO_OUTPUT_DIR': ISO_OUTPUT_DIR,
        'ISO_PROGRESS_FILE': ISO_PROGRESS_FILE,
        'EXTRACTION_THRESHOLD': EXTRACTION_THRESHOLD,
        'ISO_LOG_FILE': ISO_LOG_FILE,
        'MAX_RETRIES': MAX_RETRIES,
        'RETRY_DELAY': RETRY_DELAY,
        
        # S1
        'S1_ROOT_DIR': S1_ROOT_DIR,
        'S1_OUTPUT_DIR': S1_OUTPUT_DIR,
        'S1_DICOM_CONCURRENCY': S1_DICOM_CONCURRENCY,
        'S1_DOC_CONCURRENCY': S1_DOC_CONCURRENCY,
        'S1_BATCH_SIZE': S1_BATCH_SIZE,
        
        # S2
        'S2_INPUT_DIR': S2_INPUT_DIR,
        'S2_OUTPUT_DIR': S2_OUTPUT_DIR,
        'S2_OUTPUT_JSON': S2_OUTPUT_JSON,
        'S2_OUTPUT_EXCEL': S2_OUTPUT_EXCEL,
        
        # S3
        'S3_INPUT_DIR': S3_INPUT_DIR,
        'S3_OUTPUT_DIR': S3_OUTPUT_DIR,
        'S3_MIN_FILES': S3_MIN_FILES,
        
        # S4
        'S4_INPUT_DIR': S4_INPUT_DIR,
        'S4_OUTPUT_DIR': S4_OUTPUT_DIR,
        'ZIP_COMPRESSION_LEVEL': ZIP_COMPRESSION_LEVEL,
        'MAX_ZIP_SIZE_GB': MAX_ZIP_SIZE_GB,
        'S4_PROGRESS_FILE': S4_PROGRESS_FILE,
        'S4_WORKERS': S4_WORKERS,
        
        # S5
        'S5_INPUT_FILE': S5_INPUT_FILE,
        'S5_OUTPUT_DIR': S5_OUTPUT_DIR,
        'S5_BATCH_SIZE': S5_BATCH_SIZE,
        'S5_CHUNK_SIZE': S5_CHUNK_SIZE,
        'S5_OVERWRITE': S5_OVERWRITE,
        'S5_LLM_CLIENT_TYPE': S5_LLM_CLIENT_TYPE,
        'S5_OLLAMA_HOST': S5_OLLAMA_HOST,
        'S5_OLLAMA_MODEL': S5_OLLAMA_MODEL,
        
        # Common
        'SKIP_FILES': SKIP_FILES,
        
        # Pipeline
        'RUN_S0_ZIP_EXTRACT': RUN_S0_ZIP_EXTRACT,
        'RUN_S0_ISO_EXTRACT': RUN_S0_ISO_EXTRACT,
        'RUN_S1_INDEXING': RUN_S1_INDEXING,
        'RUN_S2_CONCAT': RUN_S2_CONCAT,
        'RUN_S3_PROCESS': RUN_S3_PROCESS,
        'RUN_S4_ZIP': RUN_S4_ZIP,
        'RUN_S5_LLM_EXTRACT': RUN_S5_LLM_EXTRACT,
        'RUN_SEQUENTIAL': RUN_SEQUENTIAL,
        # Project settings
        'Location_of_your_data': Location_of_your_data,
        'desired_name_of_project': desired_name_of_project,
        'PROJECT_DIR': str(PROJECT_DIR)
    }

def set_environment_variables():
    """Set all configuration as environment variables."""
    config = get_config_dict()
    for key, value in config.items():
        if value:  # Only set non-empty values
            os.environ[key] = str(value)

def print_config():
    """Print current configuration."""
    config = get_config_dict()
    print("="*60)
    print("DICOM PIPELINE CONFIGURATION")
    print("="*60)
    
    sections = {
        'General': ['BASE_DIR', 'CODE_DIR', 'DATA_DIR', 'VENV_PATH', 'MAX_WORKERS', 'LOG_LEVEL'],
        'S0 - ZIP Extract': ['S0_ZIP_ROOT_DIR', 'S0_ZIP_WORKERS', 'S0_ZIP_INTEGRITY_CHECK'],
        'S0 - ISO Extract': ['POWERISO_PATH', 'ISO_SEARCH_DIR', 'ISO_OUTPUT_DIR'],
        'S1 - Indexing': ['S1_ROOT_DIR', 'S1_OUTPUT_DIR'],
        'S2 - Concatenate': ['S2_INPUT_DIR', 'S2_OUTPUT_DIR', 'S2_OUTPUT_JSON', 'S2_OUTPUT_EXCEL'],
        'S3 - Process': ['S3_INPUT_DIR', 'S3_OUTPUT_DIR', 'S3_MIN_FILES'],
        'S4 - ZIP': ['S4_INPUT_DIR', 'S4_OUTPUT_DIR', 'ZIP_COMPRESSION_LEVEL'],
        'S5 - LLM Extract': ['S5_INPUT_FILE', 'S5_OUTPUT_DIR', 'S5_BATCH_SIZE', 'S5_CHUNK_SIZE', 'S5_LLM_CLIENT_TYPE', 'S5_OLLAMA_HOST', 'S5_OLLAMA_MODEL'],
        'Pipeline Control': ['RUN_S0_ZIP_EXTRACT', 'RUN_S0_ISO_EXTRACT', 'RUN_S1_INDEXING', 
                           'RUN_S2_CONCAT', 'RUN_S3_PROCESS', 'RUN_S4_ZIP', 'RUN_S5_LLM_EXTRACT']
    }
    
    for section, keys in sections.items():
        print(f"\n{section}:")
        print("-"*40)
        for key in keys:
            if key in config:
                value = config[key]
                if len(str(value)) > 50:
                    value = str(value)[:47] + "..."
                print(f"  {key}: {value}")

if __name__ == "__main__":
    print_config()