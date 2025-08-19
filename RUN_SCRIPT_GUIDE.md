# FICZall - Command Line Guide (run.py)

**Complete guide for using the single-location DICOM processing script**

This guide is for users who prefer command-line interfaces or need to integrate FICZall into automated workflows.

---

## 🎯 When to Use run.py

### **Perfect for:**
- **Technical users** comfortable with command line
- **Single DICOM datasets** (one hospital, one study, one project)
- **Automated workflows** and scripting
- **Custom configurations** with specific parameters
- **Development and testing** of individual pipeline stages
- **Integration** with other medical data processing tools

### **Not ideal for:**
- **Multiple locations** - use `runMulti.py` instead
- **Non-technical users** - use the GUI (`python launch_gui.py`) instead
- **First-time users** - try the GUI first to understand the workflow

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7+ installed
- FICZall project downloaded and extracted

### 1. Initial Setup (One-time)
```bash
# Navigate to FICZall directory
cd FICZall-Find-Index-Clean-Zip-all-Dicom-Files-and-Reports

# Setup virtual environment and install dependencies
python run.py --setup
```

### 2. First Run (Interactive)
```bash
# Windows
python run.py

# macOS/Linux
python3 run.py
```

The script will guide you through:
- Selecting which pipeline stages to run
- Specifying your DICOM data location
- Configuring processing parameters
- Setting up output directories

### 3. Subsequent Runs (Non-interactive)
```bash
# Use saved configuration
python run.py --non-interactive
```

---

## 📋 Command Reference

### Basic Commands

| Command | Description | Example |
|---------|-------------|---------|
| `python run.py` | Interactive setup and run | `python run.py` |
| `python run.py --setup` | Setup environment only | `python run.py --setup` |
| `python run.py --config` | Show current configuration | `python run.py --config` |
| `python run.py --non-interactive` | Run with saved settings | `python run.py --non-interactive` |
| `python run.py --stage S1` | Run specific stage only | `python run.py --stage S1` |
| `python run.py --skip-setup` | Skip environment check | `python run.py --skip-setup` |

### Stage-Specific Commands

```bash
# Run individual stages
python run.py --stage S0_ZIP    # ZIP extraction only
python run.py --stage S0_ISO    # ISO extraction only  
python run.py --stage S1        # DICOM indexing only
python run.py --stage S2        # Create summaries only
python run.py --stage S3        # Filter studies only
python run.py --stage S4        # Create archives only
python run.py --stage S5        # AI extraction only
```

---

## ⚙️ Configuration

### Understanding run_config.py

After your first run, FICZall creates `run_config.py` with your settings:

```python
# Essential settings you'll want to modify
Location_of_your_data = r"C:\Your\Path\To\DICOM\Files"
desired_name_of_project = "MyHospital_CT_Study_2024"

# Pipeline control (which stages to run)
RUN_S0_ZIP_EXTRACT = 'false'
RUN_S0_ISO_EXTRACT = 'false'  
RUN_S1_INDEXING = 'true'
RUN_S2_CONCAT = 'true'
RUN_S3_PROCESS = 'true'
RUN_S4_ZIP = 'false'
RUN_S5_LLM_EXTRACT = 'false'

# Performance settings
MAX_WORKERS = '4'
S3_MIN_FILES = '10'
ZIP_COMPRESSION_LEVEL = '6'
```

### Environment Variable Override

You can override any setting without editing files:

**Windows (Command Prompt):**
```cmd
set S1_ROOT_DIR=C:\Medical\Data\DICOM
set MAX_WORKERS=8
set S3_MIN_FILES=20
python run.py --non-interactive
```

**Windows (PowerShell):**
```powershell
$env:S1_ROOT_DIR="C:\Medical\Data\DICOM"
$env:MAX_WORKERS="8"
$env:S3_MIN_FILES="20"
python run.py --non-interactive
```

**macOS/Linux:**
```bash
export S1_ROOT_DIR="/path/to/medical/data"
export MAX_WORKERS=8
export S3_MIN_FILES=20
python3 run.py --non-interactive
```

---

## 🔄 Common Workflows

### Workflow 1: Basic DICOM Processing
**For DICOM files already extracted from archives**

```bash
# Setup (first time only)
python run.py --setup

# Interactive configuration
python run.py
# Choose: S1, S2, S3 (minimum required stages)
# Enter: C:\Medical_Data\CT_Scans (your DICOM path)
# Accept defaults for other settings

# Subsequent runs with same settings
python run.py --non-interactive
```

**Result:** Organized DICOM index, patient summaries, filtered studies

---

### Workflow 2: Processing ZIP Archives
**For DICOM data stored in ZIP files**

```bash
# Interactive setup
python run.py
# Choose: S0_ZIP, S1, S2, S3
# Enter: C:\Medical_Data\ZIP_Files (folder containing ZIPs)

# Or use environment variables
set S0_ZIP_ROOT_DIR=C:\Medical_Data\ZIP_Files
set RUN_S0_ZIP_EXTRACT=true
set RUN_S1_INDEXING=true
set RUN_S2_CONCAT=true  
set RUN_S3_PROCESS=true
python run.py --non-interactive
```

**Result:** ZIPs extracted, DICOM indexed, organized results

---

### Workflow 3: Creating Long-term Archives
**Process and compress for storage**

```bash
# Full pipeline with archiving
python run.py
# Choose: S1, S2, S3, S4
# Set high compression: ZIP_COMPRESSION_LEVEL=9

# Or direct configuration
set RUN_S1_INDEXING=true
set RUN_S2_CONCAT=true
set RUN_S3_PROCESS=true
set RUN_S4_ZIP=true
set ZIP_COMPRESSION_LEVEL=9
python run.py --non-interactive
```

**Result:** Complete processing + compressed ZIP archives

---

### Workflow 4: AI-Powered Document Analysis
**Extract structured medical data using AI**

**Option A: Local AI (Privacy-Safe)**
```bash
# Setup Ollama (one-time)
# Download from: https://ollama.ai/
ollama pull llama3-groq-tool-use:8b-q8_0
ollama serve

# Configure and run
set S5_LLM_CLIENT_TYPE=ollama
set RUN_S5_LLM_EXTRACT=true
python run.py --stage S5
```

**Option B: Cloud AI (OpenAI)**
```bash
# Get API key from: https://platform.openai.com/api-keys
set OPENAI_API_KEY=your_api_key_here
set S5_LLM_CLIENT_TYPE=openai
set RUN_S5_LLM_EXTRACT=true
python run.py --stage S5
```

**Result:** Structured medical data extracted from documents

---

### Workflow 5: Re-processing Existing Data
**Run specific stages on previously processed data**

```bash
# Re-run filtering with different criteria
set S3_MIN_FILES=5
python run.py --stage S3

# Create new archives from existing processed data  
set ZIP_COMPRESSION_LEVEL=1
python run.py --stage S4

# Add AI analysis to existing project
python run.py --stage S5
```

---

### Workflow 6: Batch Processing Multiple Datasets
**Process several datasets with run.py**

```bash
# Windows batch script example
@echo off
for %%d in (
    "C:\Hospital_A\CT_2024"
    "C:\Hospital_B\MRI_2024"  
    "C:\Research_Project_Data"
) do (
    set S1_ROOT_DIR=%%d
    set desired_name_of_project=%%~nd
    python run.py --non-interactive
)
```

```bash
# Linux/Mac shell script example
#!/bin/bash
for dataset in \
    "/data/Hospital_A/CT_2024" \
    "/data/Hospital_B/MRI_2024" \
    "/data/Research_Project_Data"
do
    export S1_ROOT_DIR="$dataset"
    export desired_name_of_project=$(basename "$dataset")
    python3 run.py --non-interactive
done
```

**Note:** For complex multi-location processing, consider using `runMulti.py` instead.

---

## 🎛️ Advanced Configuration

### Performance Tuning

**Small Dataset (< 1,000 files):**
```python
MAX_WORKERS = '2'
S1_DICOM_CONCURRENCY = '2'
S3_MIN_FILES = '5'
S4_WORKERS = '2'
ZIP_COMPRESSION_LEVEL = '6'
```

**Large Dataset (> 10,000 files):**
```python
MAX_WORKERS = '8'
S1_DICOM_CONCURRENCY = '6'
S3_MIN_FILES = '20'
S4_WORKERS = '4'
ZIP_COMPRESSION_LEVEL = '1'  # Faster compression
```

**High-Quality Research:**
```python
S3_MIN_FILES = '1'  # Include all studies
ZIP_COMPRESSION_LEVEL = '9'  # Maximum compression
RUN_S5_LLM_EXTRACT = 'true'  # Enable AI analysis
```

### Output Directory Customization

```python
# Custom output structure
S1_OUTPUT_DIR = r"C:\Results\Indexed_Metadata"
S2_OUTPUT_JSON = r"C:\Results\Patient_Summary.json"
S2_OUTPUT_EXCEL = r"C:\Results\Patient_Summary.xlsx"
S3_OUTPUT_DIR = r"C:\Results\Filtered_Studies"
S4_OUTPUT_DIR = r"C:\Results\ZIP_Archives"
S5_OUTPUT_DIR = r"C:\Results\AI_Extractions"
```

### AI Configuration (S5 Stage)

```python
# Local AI settings
S5_LLM_CLIENT_TYPE = 'ollama'
S5_OLLAMA_HOST = 'http://localhost:11434'
S5_OLLAMA_MODEL = 'llama3-groq-tool-use:8b-q8_0'

# Processing settings
S5_BATCH_SIZE = '10'  # Documents per batch
S5_CHUNK_SIZE = '1000'  # Words per chunk
S5_OVERWRITE = 'false'  # Don't overwrite existing results
```

---

## 📊 Understanding Output

### Output Directory Structure

After running `python run.py`, you'll find:

```
data/processed/your_project_name/
├── S1_indexed_metadata/           # Stage 1 results
│   ├── Patient_001_Study.jsonl   # Individual patient metadata
│   ├── Patient_002_Study.jsonl
│   ├── ...
│   └── S1_indexingFiles_allDocuments.json  # Document index
├── S2_concatenated_summaries/     # Stage 2 results  
│   ├── S2_patient_summary.json   # Machine-readable summary
│   └── S2_patient_summary.xlsx   # Human-readable Excel
├── S3_filtered_studies/           # Stage 3 results
│   └── S3_filtered_studies.json  # Studies meeting criteria
├── S4_zip_archives/               # Stage 4 results (if enabled)
│   ├── Patient_001_Study.zip     # Compressed studies
│   ├── Patient_002_Study.zip
│   └── S4_compression_stats.json # Compression statistics
└── S5_llm_extractions/           # Stage 5 results (if enabled)
    ├── S5_extracted_data.json    # AI-extracted medical data
    ├── S5_processing_results.json # Processing statistics
    └── S5_failed_extractions.json # Failed documents
```

### Key Output Files

**S1_indexed_metadata/**
- `*.jsonl`: One file per patient/study with complete DICOM metadata
- `S1_indexingFiles_allDocuments.json`: Index of all documents found

**S2_concatenated_summaries/**
- `S2_patient_summary.json`: Aggregated patient data (machine-readable)
- `S2_patient_summary.xlsx`: Excel spreadsheet for human review

**S3_filtered_studies/**
- `S3_filtered_studies.json`: Studies that passed filtering criteria

**S4_zip_archives/** (if S4 enabled)
- Individual ZIP files for each study with embedded metadata
- Compression statistics and integrity information

**S5_llm_extractions/** (if S5 enabled)
- Structured medical data extracted from documents
- Patient information, findings, diagnoses, treatments

---

## 🔧 Integration with Other Tools

### Using FICZall in Scripts

**Python Integration:**
```python
import subprocess
import os

# Configure environment
os.environ['S1_ROOT_DIR'] = '/path/to/dicom'
os.environ['desired_name_of_project'] = 'my_project'
os.environ['RUN_S1_INDEXING'] = 'true'
os.environ['RUN_S2_CONCAT'] = 'true'
os.environ['RUN_S3_PROCESS'] = 'true'

# Run FICZall
result = subprocess.run(['python', 'run.py', '--non-interactive'], 
                       capture_output=True, text=True)

if result.returncode == 0:
    print("FICZall processing completed successfully")
    # Process the results...
else:
    print(f"FICZall failed: {result.stderr}")
```

**Docker Integration:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN python run.py --setup

# Set environment variables
ENV S1_ROOT_DIR=/data/input
ENV S1_OUTPUT_DIR=/data/output

CMD ["python", "run.py", "--non-interactive"]
```

### Data Analysis Pipeline

```python
# After FICZall processing, analyze results
import json
import pandas as pd

# Load patient summary
with open('data/processed/project/S2_concatenated_summaries/S2_patient_summary.json') as f:
    patient_data = json.load(f)

# Convert to DataFrame for analysis
df = pd.DataFrame(patient_data)

# Analyze study statistics
print(f"Total patients: {len(df)}")
print(f"Modalities: {df['Modality'].value_counts()}")
print(f"Study dates: {df['StudyDate'].min()} to {df['StudyDate'].max()}")

# Export for R, SPSS, etc.
df.to_csv('analysis_ready_data.csv', index=False)
```

---

## 🚨 Troubleshooting

### Common Issues

**"Python not found"**
```bash
# Check Python installation
python --version
# or
python3 --version

# Windows: Add Python to PATH
# macOS: brew install python3
# Linux: sudo apt install python3
```

**"Virtual environment issues"**
```bash
# Reset virtual environment
rm -rf venv  # Linux/Mac
rmdir /s venv  # Windows

# Re-run setup
python run.py --setup
```

**"DICOM files not found"**
```bash
# Check path in configuration
python run.py --config

# Fix path format (use forward slashes)
set S1_ROOT_DIR=C:/Medical/Data/DICOM
```

**"Out of memory errors"**
```bash
# Reduce worker count
set MAX_WORKERS=2
set S1_DICOM_CONCURRENCY=1
```

**"Permission denied"**
```bash
# Windows: Run as Administrator
# Linux/Mac: Check file permissions
chmod +x run.py
```

### Debug Individual Stages

```bash
# Test stage components directly (from venv)
venv/Scripts/activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run individual stage scripts
python code/S1_indexingFiles_E2.py --root /path/to/data --output_dir /output
python code/S2_concatIndex_E2.py --folder_path /path --output_json /out.json
python code/S3_processForStore.py --input_dir /in --output_dir /out
```

### Getting Help

1. **Check logs**: Look in output directories for `.log` files
2. **Test configuration**: `python run.py --config`
3. **Verify setup**: `python run.py --setup`
4. **Check dependencies**: Look at `requirements.txt`
5. **GitHub Issues**: Report bugs with log files attached

---

## 📈 Performance Guidelines

### Processing Time Estimates

| Dataset Size | S1 (Index) | S2+S3 (Summary) | S4 (Archive) | S5 (AI) |
|--------------|------------|-----------------|--------------|---------|
| 1,000 files | 5-15 min | 1-2 min | 5-10 min | 2-5 min |
| 10,000 files | 30-90 min | 5-10 min | 30-60 min | 10-30 min |
| 100,000 files | 2-6 hours | 15-30 min | 2-4 hours | 1-3 hours |

### Resource Requirements

- **CPU**: More cores = better performance (adjust MAX_WORKERS)
- **RAM**: 4GB minimum, 8GB+ recommended for large datasets
- **Storage**: 2-3x input data size for temporary files and outputs
- **Network**: Only needed for AI cloud processing (OpenAI API)

### Optimization Tips

```python
# For speed (less compression, more workers)
MAX_WORKERS = '8'
ZIP_COMPRESSION_LEVEL = '1'
S1_DICOM_CONCURRENCY = '6'

# For quality (maximum compression, thorough processing)  
ZIP_COMPRESSION_LEVEL = '9'
S3_MIN_FILES = '1'
S5_LLM_EXTRACT = 'true'

# For large datasets (balanced settings)
MAX_WORKERS = '6'
S3_MIN_FILES = '15'
ZIP_COMPRESSION_LEVEL = '6'
```

---

## 🎯 Best Practices

### Before Processing
1. **Backup your data** - FICZall doesn't modify original files, but always be safe
2. **Test with small dataset** - Run on a subset first to validate settings
3. **Check disk space** - Ensure 2-3x your data size is available
4. **Review configuration** - Use `python run.py --config` to verify settings

### During Processing
1. **Monitor progress** - Check console output for any warnings
2. **Don't interrupt stages** - Let individual stages complete for resume capability
3. **Check system resources** - Monitor CPU, RAM, and disk usage

### After Processing  
1. **Verify outputs** - Check that expected files were created
2. **Review logs** - Look for any errors or warnings in output directories
3. **Backup results** - Save processed data before making changes
4. **Document settings** - Keep track of which settings worked for your data

---

**Happy DICOM processing with run.py!** 🎉

For GUI users, see `GUI_USER_GUIDE.md`. For batch processing, see the main `README.md` section on `runMulti.py`.