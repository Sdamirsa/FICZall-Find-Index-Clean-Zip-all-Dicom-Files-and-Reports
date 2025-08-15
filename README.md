# FICZall - DICOM Processing Pipeline

**Find, Index, Clean, Zip all DICOM Files and Reports**

A comprehensive, automated pipeline for processing DICOM medical imaging data, from extraction through archival.

## 🚀 Quick Start

### Prerequisites
- **Python 3.6+** (that's it!)
- No additional packages needed - the script handles everything

### Installation & Usage

1. **Install Python** (if not already installed):
   - Download from [python.org](https://www.python.org/downloads/)
   - ✅ Make sure to check "Add Python to PATH" during installation

2. **Download/Clone this repository**:
   ```bash
   git clone <repository-url>
   cd FICZall-Find-Index-Clean-Zip-all-Dicom-Files-and-Reports
   ```

3. **Run the pipeline**:
   
   **Single project (interactive):**
   ```bash
   python run.py
   ```
   
   **Multiple projects (batch mode - NEW!):**
   ```bash
   python runs.py
   ```

**That's it!** The script will:
- ✅ Create a virtual environment automatically
- ✅ Install all required dependencies
- ✅ Guide you through configuration
- ✅ Run the pipeline stages you select

## 📋 Pipeline Overview

The pipeline consists of 7 stages:

| Stage | Name | Description | Required |
|-------|------|-------------|----------|
| **S0_ZIP** | ZIP Extraction | Extract ZIP files in directory tree with smart logic | Optional |
| **S0_ISO** | ISO Extraction | Extract DICOM files from ISO archives | Optional |
| **S1** | DICOM Indexing | Scan directories and extract DICOM metadata | ✅ Required |
| **S2** | Concatenate Index | Aggregate patient information into summaries | ✅ Required |
| **S3** | Process for Storage | Filter studies and remove duplicates | ✅ Required |
| **S4** | Create ZIP Archives | Compress DICOM studies for long-term storage | Optional |
| **S5** | LLM Extract | Extract structured medical data from documents using AI | Optional |

## 🎯 Use Cases

### Scenario 1: DICOM Files in Folders
**Most common use case**
```bash
python run.py
# Select: S1, S2, S3, S4 (skip S0)
# Provide root directory containing DICOM files
# Follow the interactive prompts
```

### Scenario 2: DICOM Files in ZIP Archives
**For compressed medical data**
```bash
python run.py
# Select: S0_ZIP, S1, S2, S3 (ZIP extraction + core stages)
# Provide directory containing ZIP files
# Follow the interactive prompts
```

### Scenario 3: DICOM Files in ISO Archives
**For archived medical data**
```bash
python run.py
# Select: S0_ISO, S1, S2, S3, S4 (all stages except ZIP)
# Provide directory containing ISO files
# Provide PowerISO path (Windows)
# Follow the interactive prompts
```

### Scenario 4: Re-process Existing Data
**Process already indexed files**
```bash
python run.py --stage S3  # Filter and process
# or
python run.py --stage S4  # Create archives only
```

### Scenario 5: Extract ZIP Files Only
**Extract all ZIP files in directory tree**
```bash
python code/S0_zipExtract.py --root-dir /path/to/zips --workers 8
```

### Scenario 6: AI-Powered Document Processing (NEW!)
**🤖 Extract structured medical data from reports using AI - PRIVACY SAFE options available!**

#### Option A: Use Local AI (Recommended for Medical Data - HIPAA Compliant)
```bash
# 1. Install Ollama (local AI - keeps data on your computer)
# Download from: https://ollama.ai/
# Follow installation instructions for your operating system

# 2. Download a medical AI model (do this once)
ollama pull llama3-groq-tool-use:8b-q8_0

# 3. Start Ollama service
ollama serve

# 4. Configure for local processing (data never leaves your computer)
export S5_LLM_CLIENT_TYPE=ollama

# 5. Run AI extraction
python run.py --stage S5
```

#### Option B: Use OpenAI (Requires API Key - Data sent to OpenAI)
```bash
# 1. Get API key from https://platform.openai.com/api-keys
export OPENAI_API_KEY=your_api_key_here

# 2. Configure for OpenAI
export S5_LLM_CLIENT_TYPE=openai

# 3. Run AI extraction
python run.py --stage S5
```

#### 🔒 Privacy Information:
- **Local AI (Ollama)**: ✅ Patient data stays on your computer - HIPAA friendly
- **OpenAI**: ⚠️ Patient data sent to OpenAI servers - ensure compliance agreements

### Scenario 7: Batch Processing Multiple Locations (NEW!)
**🏥 Process multiple hospitals, studies, or projects with the same settings**
```bash
# Interactive multi-location setup
python runs.py

# The script will ask for:
# 1. Multiple data locations (hospitals, studies, projects)
# 2. Single configuration for all locations
# 3. Processing settings that apply to all

# Example locations:
# - C:/Hospital_A/CT_Studies_2024
# - C:/Hospital_B/MRI_Studies_2024  
# - C:/Research_Project_1/Data
# - C:/Research_Project_2/Data

# Resume interrupted batch processing
python runs.py --resume

# Use existing batch configuration
python runs.py --non-interactive
```

#### ✨ Batch Processing Features:
- **Shared Configuration**: Set up once, apply to all locations
- **Progress Tracking**: Resume from interruption points
- **Individual Projects**: Each location gets its own project folder
- **Error Recovery**: Continue processing even if one location fails
- **Flexible Naming**: Auto-generate or custom project names

## 🛠️ Command Options

### Single Location Processing
```bash
# Interactive mode (recommended for first-time users)
python run.py

# Show current configuration
python run.py --config

# Setup only (create virtual environment and install dependencies)
python run.py --setup

# Run a specific stage only
python run.py --stage S1  # Options: S0_ZIP, S0_ISO, S1, S2, S3, S4, S5

# Run without prompts (use existing configuration)
python run.py --non-interactive
```

### Multi-Location Batch Processing
```bash
# Interactive batch setup (recommended)
python runs.py

# Show batch configuration
python runs.py --config

# Setup environment for batch processing
python runs.py --setup

# Resume interrupted batch processing
python runs.py --resume

# Run batch with existing configuration
python runs.py --non-interactive
```

## 📁 Output Structure

After running, your directory will look like:

```
FICZall-Find-Index-Clean-Zip-all-Dicom-Files-and-Reports/
├── run.py                          # 🎯 Single location processing
├── runs.py                         # 🏥 Multi-location batch processing (NEW!)
├── run_config.py                   # ⚙️ Configuration (created after first run)
├── batch_processing_config.json    # 📋 Batch configuration (created by runs.py)
├── batch_processing_progress.json  # 📊 Batch progress tracking (created by runs.py)
├── S5_llmExtract_config.py         # 🤖 AI extraction configuration
├── venv/                          # 📦 Virtual environment (auto-created)
├── data/
│   ├── processed/
│   │   ├── project_name/                 # 📁 Single location results (run.py)
│   │   │   ├── S1_indexed_metadata/      # 📋 Indexed DICOM metadata
│   │   │   ├── S2_concatenated_summaries/ # 📊 Patient summaries
│   │   │   ├── S3_filtered_studies/      # 🔄 Filtered studies
│   │   │   ├── S4_zip_archives/          # 📦 ZIP archives (if enabled)
│   │   │   └── S5_llm_extractions/       # 🤖 AI extractions (if enabled)
│   │   ├── hospital_a_ct_2024/           # 🏥 Batch location 1 (runs.py)
│   │   │   ├── S1_indexed_metadata/
│   │   │   ├── S2_concatenated_summaries/
│   │   │   ├── S3_filtered_studies/
│   │   │   ├── S4_zip_archives/
│   │   │   └── S5_llm_extractions/
│   │   ├── hospital_b_mri_2024/          # 🏥 Batch location 2 (runs.py)
│   │   │   ├── S1_indexed_metadata/
│   │   │   ├── S2_concatenated_summaries/
│   │   │   ├── S3_filtered_studies/
│   │   │   ├── S4_zip_archives/
│   │   │   └── S5_llm_extractions/
│   │   └── research_project_001/         # 🧪 Batch location 3 (runs.py)
│   │       ├── S1_indexed_metadata/
│   │       ├── S2_concatenated_summaries/
│   │       ├── S3_filtered_studies/
│   │       ├── S4_zip_archives/
│   │       └── S5_llm_extractions/
├── code/                          # 🔧 Pipeline scripts (don't run directly)
└── requirements.txt               # 📋 Python dependencies
```

## ⚙️ Configuration

### First Run Configuration
On first run, the script will interactively ask for:

1. **Which stages to run**
   - S0_ZIP: Only needed if you have ZIP files to extract
   - S0_ISO: Only needed if you have ISO files
   - S1-S3: Required for complete processing
   - S4: Optional for ZIP archives
   - S5: Optional for AI document extraction

2. **Input/Output directories**
   - Where your DICOM files are located
   - Where to save processed results

3. **Processing parameters**
   - Compression level for ZIP files
   - Minimum files per study
   - Number of parallel workers

### Persistent Configuration
Settings are saved in `run_config.py` and can be edited directly:

```python
# Example configuration
S1_ROOT_DIR = '/path/to/dicom/files'
ZIP_COMPRESSION_LEVEL = '6'  # 0-9, where 9 is maximum compression
S3_MIN_FILES = '10'          # Minimum files required per study
MAX_WORKERS = '4'            # Number of parallel workers
```

### Environment Variables
Advanced users can override any setting:

```bash
# Examples
export S1_ROOT_DIR=/path/to/dicom/files
export ZIP_COMPRESSION_LEVEL=9
export S3_MIN_FILES=20
export MAX_WORKERS=8
```

## 🔧 Advanced Features

### Resume Capability
- Pipeline stages can be interrupted and resumed
- Progress is automatically saved
- No data loss on unexpected shutdown

### Parallel Processing
- Automatic CPU detection for optimal performance
- Configurable worker pools
- Memory-efficient streaming for large datasets

### Error Handling
- Graceful degradation for missing dependencies
- Automatic retries for transient failures
- Comprehensive logging for troubleshooting

### Flexible Input Formats
- DICOM files (.dcm or no extension)
- ISO archives containing DICOM data
- Mixed directory structures

## 🩺 DICOM Data Processing

### Metadata Extraction
The pipeline extracts comprehensive DICOM metadata including:
- Patient information (name, ID, age)
- Study details (date, modality, body part)
- Series information (description, protocol)
- Technical parameters (slice thickness, orientation)
- Institution and equipment details

### Quality Control
- Duplicate detection and removal
- Study validation (minimum file requirements)
- Corrupted file detection
- Missing metadata handling

### Output Formats
- **JSONL**: One JSON object per DICOM file for detailed analysis
- **JSON**: Aggregated patient summaries
- **Excel**: Human-readable reports with statistics
- **ZIP**: Compressed archives with embedded metadata

## 🛠️ External Dependencies

### Required for S0_ZIP (ZIP Extraction)
- **No additional tools required** - uses Python's built-in zipfile module
- **Multi-threaded extraction** with integrity checking
- **Smart resume logic** avoids re-extraction

### Required for S0_ISO (ISO Extraction)
- **PowerISO** (Windows only): Download from [poweriso.com](https://www.poweriso.com/)
  - Used to extract DICOM files from ISO archives
  - Configure path in `run_config.py`
  - Optional - skip S0 if DICOM files are already extracted

### Python Packages (Auto-installed)
- **pydicom**: DICOM file reading and metadata extraction
- **pandas**: Data manipulation and Excel output
- **openpyxl**: Excel file support
- **python-docx**: Document processing (for reports)
- **PyPDF2**: PDF processing (for reports)
- **tqdm**: Progress bars
- **numpy**: Array operations (improves performance)
- **Pillow**: Image processing support
- **openai**: AI-powered document extraction (S5 stage)
- **pydantic**: Data validation for structured outputs

## 🚨 Troubleshooting

### Common Issues

#### "Python not found"
```bash
# Make sure Python is installed and added to PATH
# Try alternative command:
python3 run.py
```

#### "PowerISO not found" (S0 stage only)
- S0 is optional - you can skip it if DICOM files are already extracted
- Download PowerISO from [poweriso.com](https://www.poweriso.com/)
- Update path in `run_config.py` or when prompted

#### "Permission denied"
```bash
# Linux/Mac: Make script executable
chmod +x run.py
./run.py

# Windows: Run as administrator if needed
```

#### Dependencies installation fails
- The script will continue with available packages
- Some packages are optional (e.g., Excel output needs pandas)
- Manual installation: `venv/bin/pip install <package>`

### Getting Help

1. **Check log files** in each output directory for detailed error information
2. **Run setup again**: `python run.py --setup`
3. **Verify configuration**: `python run.py --config`
4. **Check paths** in `run_config.py` for accuracy
5. **Open an issue** on GitHub with error messages and log files

## 🔬 Technical Details

### Performance Optimization
- **Streaming processing**: Handles large datasets without loading everything into memory
- **Parallel processing**: Automatic CPU detection and worker pool management
- **Resume capability**: Progress tracking prevents reprocessing completed work
- **Efficient I/O**: Atomic file operations prevent corruption

### Data Integrity
- **Checksum validation**: Ensures data integrity during processing
- **Atomic operations**: Temporary files prevent partial writes
- **Backup creation**: Original data is never modified
- **Error logging**: Comprehensive audit trail

### Security Considerations
- **Isolated environment**: Virtual environment prevents system contamination
- **Path validation**: Prevents directory traversal attacks
- **Input sanitization**: Handles malformed DICOM files safely
- **No network access**: Pipeline operates entirely offline

## 📈 Performance Guidelines

### System Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB+ RAM, 4+ CPU cores
- **Storage**: 2-3x the size of input data for temporary files

### Optimization Tips
```python
# In run_config.py
MAX_WORKERS = '8'           # Increase for more CPU cores
S1_DICOM_CONCURRENCY = '4'  # Parallel DICOM processing
S4_WORKERS = '4'            # Parallel ZIP creation
ZIP_COMPRESSION_LEVEL = '1'  # Lower for speed, higher for compression
```

### Batch Processing
For multiple datasets:
```bash
# Process dataset 1
python run.py --non-interactive
# Edit S1_ROOT_DIR in run_config.py
# Process dataset 2
python run.py --non-interactive
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Test your changes with `python run.py --setup`
4. Submit a pull request

### Development Setup
```bash
git clone <repository-url>
cd FICZall-Find-Index-Clean-Zip-all-Dicom-Files-and-Reports
python run.py --setup  # Creates development environment
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏥 Medical Data Disclaimer

This software is designed for research and archival purposes. Ensure compliance with:
- HIPAA (Health Insurance Portability and Accountability Act)
- GDPR (General Data Protection Regulation)
- Local healthcare data protection regulations

Always verify that processed data maintains patient privacy and institutional compliance requirements.

---

**Need help?** 
- 📖 See [README_USAGE.md](README_USAGE.md) for detailed usage instructions
- 🐛 Report issues on GitHub
- 💬 Join discussions in GitHub Discussions
