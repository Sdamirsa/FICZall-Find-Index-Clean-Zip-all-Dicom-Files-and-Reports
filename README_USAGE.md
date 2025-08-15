# FICZall DICOM Processing Pipeline - Complete User Guide

## 🚀 For Medical Professionals & Researchers

### What is FICZall?
FICZall is a comprehensive medical imaging data processing pipeline designed for hospitals, research institutions, and medical professionals. It helps you:

- 📁 **Extract & Organize**: Process ZIP/ISO archives containing thousands of DICOM files
- 🔍 **Index & Search**: Create searchable databases of patient studies and metadata  
- 🤖 **AI Analysis**: Extract structured medical information from reports using advanced AI
- 📊 **Generate Reports**: Create patient summaries and study statistics
- 💾 **Archive**: Prepare data for long-term storage with proper compression

**🆕 NEW in 2024**: AI-powered medical document processing with privacy-safe local options!

---

## 📋 Requirements

### Software Requirements
- **Python 3.6 or higher** (that's it - everything else is installed automatically!)
- **Operating System**: Windows, Mac, or Linux
- **Disk Space**: 2-5 GB for dependencies + space for your medical data
- **Memory**: 4GB RAM minimum (8GB+ recommended for large datasets)

### For AI Features (S5 Stage)
- **Local AI (Recommended)**: Ollama software (free, keeps data private)
- **Cloud AI (Alternative)**: OpenAI API key (data sent to OpenAI servers)

---

## 🛠️ Installation

### Step 1: Install Python
If you don't have Python installed:
1. Go to [python.org](https://www.python.org/downloads/)
2. Download Python 3.8 or newer
3. **Important**: During installation, check "Add Python to PATH"

### Step 2: Download FICZall
```bash
# Option A: Download ZIP from GitHub
# Go to GitHub page → Click "Code" → "Download ZIP" → Extract

# Option B: Use Git (if you have it)
git clone <repository-url>
cd FICZall-Find-Index-Clean-Zip-all-Dicom-Files-and-Reports
```

### Step 3: Initial Setup
Open command line in the FICZall folder and run:
```bash
python run.py --setup
```

This will:
- ✅ Create a safe virtual environment
- ✅ Install all medical processing libraries
- ✅ Verify everything works properly
- ✅ Only needs to be done once (takes 2-5 minutes)

---

## 📊 Pipeline Overview

The FICZall pipeline has 7 stages. You can run all stages or just the ones you need:

| Stage | Name | Purpose | Required? | Time |
|-------|------|---------|-----------|------|
| **S0_ZIP** | ZIP Extraction | Extract ZIP archives | Optional | 5-30 min |
| **S0_ISO** | ISO Extraction | Extract ISO files (Windows only) | Optional | 10-60 min |
| **S1** | DICOM Indexing | Scan & index DICOM files | **Required** | 10-120 min |
| **S2** | Summary Creation | Create patient summaries | **Required** | 1-10 min |
| **S3** | Study Filtering | Filter & prepare studies | **Required** | 1-15 min |
| **S4** | ZIP Archiving | Create storage archives | Optional | 15-180 min |
| **S5** | 🤖 AI Extraction | Extract medical data with AI | Optional | 5-60 min |

---

## 🎯 Quick Start Guide

### Scenario 1: I have DICOM files already extracted
```bash
# 1. Run setup (first time only)
python run.py --setup

# 2. Start interactive mode
python run.py

# 3. Follow prompts:
# - Point to your DICOM folder
# - Choose stages: S1, S2, S3 (minimum required)
# - Optionally add S4 (archiving) or S5 (AI)
```

### Scenario 2: I have ZIP files with DICOM data
```bash
# 1. Run setup (first time only)
python run.py --setup

# 2. Start with ZIP extraction
python run.py

# 3. Follow prompts:
# - Point to your ZIP folder
# - Choose stages: S0_ZIP, S1, S2, S3
# - Optionally add S4 or S5
```

### Scenario 3: I want AI-powered document analysis
```bash
# Option A: Local AI (Privacy-Safe - Recommended for Medical Data)

# 1. Install Ollama (one-time setup)
# Download from: https://ollama.ai/
# Follow installation guide for your OS

# 2. Download medical AI model (one-time, ~4GB download)
ollama pull llama3-groq-tool-use:8b-q8_0

# 3. Start Ollama service
ollama serve

# 4. Configure FICZall for local AI
export S5_LLM_CLIENT_TYPE=ollama  # Linux/Mac
# OR for Windows PowerShell:
$env:S5_LLM_CLIENT_TYPE="ollama"

# 5. Run pipeline with AI
python run.py --stage S5

# Option B: OpenAI API (Data leaves your computer)

# 1. Get API key from https://platform.openai.com/api-keys
export OPENAI_API_KEY=your_api_key_here  # Linux/Mac
# OR for Windows PowerShell:
$env:OPENAI_API_KEY="your_api_key_here"

# 2. Configure for OpenAI
export S5_LLM_CLIENT_TYPE=openai

# 3. Run pipeline with AI
python run.py --stage S5
```

### Scenario 4: I have multiple hospitals/projects to process (NEW!)
**🏥 Process multiple data locations with the same settings**
```bash
# 1. Run setup (first time only)
python runs.py --setup

# 2. Start interactive batch mode
python runs.py

# 3. The script will ask for:
# - Multiple data locations (e.g., different hospitals, studies, projects)
# - Single configuration that applies to all locations  
# - Processing stages for all locations

# Example: Processing multiple hospitals
# Location 1: C:/Hospital_A/CT_Studies_2024
# Location 2: C:/Hospital_B/MRI_Studies_2024
# Location 3: C:/Research_Project_Data

# 4. FICZall will process each location sequentially:
# - Location 1 → Complete ALL stages → Individual project folder
# - Location 2 → Complete ALL stages → Individual project folder  
# - Location 3 → Complete ALL stages → Individual project folder
# IMPORTANT: Each location finishes completely before the next one starts

# Resume if interrupted
python runs.py --resume

# Use existing batch configuration
python runs.py --non-interactive
```

#### ✨ Batch Processing Benefits:
- **Sequential processing**: Each location completed fully before moving to next (no parallel conflicts)
- **One-time setup**: Configure once, process many locations
- **Individual projects**: Each location gets its own organized output folder
- **Progress tracking**: Resume from any interruption point
- **Error recovery**: Continue even if one location fails
- **Time efficient**: Unattended processing of large datasets

---

## 🤖 AI-Powered Medical Document Extraction (S5)

### What does S5 do?
The AI stage (S5) processes medical documents and extracts structured information such as:

- **Patient Information**: Names, ages, IDs, demographics
- **Study Details**: Imaging modality (CT, MRI, etc.), body parts examined
- **Medical Findings**: Key findings from radiology reports
- **Diagnoses**: Current and previous medical conditions
- **Treatments**: Previous interventions, surgeries, medications
- **Clinical Context**: Reasons for imaging, medical history

### Privacy & Security Options

#### 🔒 Local AI (Ollama) - HIPAA Friendly
- ✅ **Patient data stays on your computer**
- ✅ **No internet required for processing**
- ✅ **HIPAA compliant when properly configured**
- ✅ **Free to use (no API costs)**
- ❓ **Requires ~8GB disk space for AI model**
- ❓ **Slower than cloud APIs**

#### ☁️ Cloud AI (OpenAI) - Faster but Data Leaves
- ✅ **Very fast processing**
- ✅ **High accuracy**
- ✅ **No local storage requirements**
- ⚠️ **Patient data sent to OpenAI servers**
- ⚠️ **Requires API key (costs money)**
- ⚠️ **Need data processing agreements for HIPAA**

### Setting Up Local AI (Recommended)

1. **Install Ollama** (one-time setup):
   - Go to [ollama.ai](https://ollama.ai/)
   - Download for your operating system
   - Follow installation instructions

2. **Download AI Model** (one-time, ~4GB):
   ```bash
   ollama pull llama3-groq-tool-use:8b-q8_0
   ```

3. **Start Ollama Service**:
   ```bash
   ollama serve
   ```
   Keep this running when using S5.

4. **Test AI Setup**:
   ```bash
   python code/S5_llmExtract_test.py --client-type ollama
   ```

### AI Configuration Options

You can customize AI processing in `run_config.py`:

```python
# AI Client Type
S5_LLM_CLIENT_TYPE = "auto"  # Options: auto, ollama, openai

# Local Ollama Settings  
S5_OLLAMA_HOST = "http://localhost:11434"
S5_OLLAMA_MODEL = "llama3-groq-tool-use:8b-q8_0"

# Processing Settings
S5_BATCH_SIZE = "10"  # Process 10 documents at a time
S5_CHUNK_SIZE = "1000"  # Split large documents
```

---

## 📂 Understanding Your Data

### Input Data
FICZall can process:
- **DICOM files** (.dcm or no extension)
- **ZIP archives** containing DICOM data
- **ISO files** with medical imaging data
- **Medical documents** (reports, text files) for AI processing

### Output Structure
After processing, you'll find organized data in:

```
data/processed/your_project_name/
├── S1_indexed_metadata/           # 📋 DICOM file index
│   ├── Patient_001_Study.jsonl   # Individual patient data
│   ├── Patient_002_Study.jsonl
│   └── S1_indexingFiles_allDocuments.json  # Complete document index
│
├── S2_concatenated_summaries/     # 📊 Patient summaries  
│   ├── S2_patient_summary.json   # Machine-readable summary
│   └── S2_patient_summary.xlsx   # Excel spreadsheet
│
├── S3_filtered_studies/           # 🔄 Filtered studies
│   └── S3_filtered_studies.json  # Studies meeting your criteria
│
├── S4_zip_archives/               # 💾 Compressed archives
│   ├── Patient_001_Study.zip     # Individual study archives
│   └── S4_compression_stats.json # Compression statistics
│
└── S5_llm_extractions/           # 🤖 AI-extracted medical data
    ├── S5_extracted_data.json    # Structured medical information
    ├── S5_processing_results.json # Processing statistics
    └── S5_failed_extractions.json # Documents that couldn't be processed
```

---

## ⚙️ Configuration Guide

### Essential Configuration (run_config.py)

The most important setting to change:

```python
# STEP 1: Point to your DICOM data (CHANGE THIS!)
Location_of_your_data = r"C:\Your\Path\To\DICOM\Files"

# STEP 2: Project name (optional - auto-generated if empty)
desired_name_of_project = "MyHospital_CT_Study_2024"
```

### Common Settings for Different Use Cases

#### Small Dataset (< 1,000 files)
```python
MAX_WORKERS = '2'
S3_MIN_FILES = '5'
ZIP_COMPRESSION_LEVEL = '6'
```

#### Large Dataset (> 10,000 files)
```python
MAX_WORKERS = '8'
S3_MIN_FILES = '20'
ZIP_COMPRESSION_LEVEL = '1'  # Faster compression
```

#### Research/Quality (need maximum detail)
```python
S3_MIN_FILES = '1'  # Include all studies
ZIP_COMPRESSION_LEVEL = '9'  # Maximum compression
RUN_S5_LLM_EXTRACT = 'true'  # Enable AI extraction
```

#### Batch Processing Multiple Locations
```python
# Batch configuration is stored in batch_processing_config.json
# These settings apply to ALL locations in the batch:

MAX_WORKERS = '4'               # Workers for all stages
S3_MIN_FILES = '10'            # Filter studies across all locations  
ZIP_COMPRESSION_LEVEL = '6'     # Compression for all ZIP archives
NAMING_STRATEGY = 'auto'        # 'auto' uses folder names, 'custom' uses base name + numbers
BASE_PROJECT_NAME = 'hospital_study'  # Used with 'custom' naming strategy

# Locations are processed sequentially (one completes before next starts):
# Location 1: /path/to/hospital_a → hospital_a/ project folder (ALL stages complete)
# Location 2: /path/to/hospital_b → hospital_b/ project folder (ALL stages complete)
# Location 3: /path/to/research   → research/ project folder (ALL stages complete)
```

---

## 🚀 Command Reference

### Single Location Processing
```bash
# Interactive setup and run (recommended for beginners)
python run.py

# Show current configuration without running
python run.py --config

# Set up environment only (do this first)
python run.py --setup

# Run specific stage only
python run.py --stage S1
python run.py --stage S5

# Run without any prompts (use saved settings)
python run.py --non-interactive

# Run but skip the setup check
python run.py --skip-setup
```

### Multi-Location Batch Processing
```bash
# Interactive batch setup (recommended for first-time users)
python runs.py

# Show current batch configuration and progress
python runs.py --config

# Set up environment for batch processing
python runs.py --setup

# Resume interrupted batch processing
python runs.py --resume

# Run batch with existing configuration (no prompts)
python runs.py --non-interactive

# Skip setup and run batch directly
python runs.py --skip-setup
```

### Advanced Commands
```bash
# Test AI functionality (works for both single and batch)
python code/S5_llmExtract_test.py --client-type ollama

# Check configuration files
python run.py --config                    # Single location config
python runs.py --config                   # Batch processing config and progress

# Debug individual stages (single location)
python code/S1_indexingFiles_E2.py --root /path/to/data --output_dir /output

# Check virtual environment status
python run.py --config | grep -i venv     # On Linux/Mac
python runs.py --config                   # Shows all config including venv
```

---

## 🛠️ Troubleshooting

### Common Issues

#### "Python not found"
**Solution**: Install Python and add to PATH
- Windows: Reinstall Python, check "Add to PATH"
- Mac: Install via Homebrew: `brew install python3`
- Linux: `sudo apt install python3 python3-pip`

#### "Permission denied" errors
**Solution**: Run as administrator or fix permissions
```bash
# Windows (run PowerShell as Administrator)
python run.py

# Linux/Mac
sudo python run.py
```

#### "Out of memory" during processing
**Solution**: Reduce worker count
```python
# In run_config.py
MAX_WORKERS = '2'  # Reduce from 4-8 to 2
```

#### "Ollama not found" for AI features
**Solution**: Install and start Ollama
```bash
# Download from https://ollama.ai/
# Then:
ollama serve
```

#### "DICOM files not found"
**Solution**: Check your path in run_config.py
```python
# Use forward slashes or double backslashes
Location_of_your_data = r"C:\\Medical\\Data\\DICOM"
# OR
Location_of_your_data = "C:/Medical/Data/DICOM"
```

### Getting Help

1. **Check logs**: Look in the `data/processed/project_name/` folder for `.log` files
2. **Test components**: Use `python run.py --config` to verify settings
3. **AI testing**: Run `python code/S5_llmExtract_test.py` to test AI features
4. **GitHub Issues**: Report bugs at the project's GitHub page

---

## 📊 Performance Guidelines

### Processing Time Estimates

| Dataset Size | S1 (Indexing) | S2+S3 (Summary) | S4 (Archive) | S5 (AI) |
|--------------|---------------|-----------------|--------------|---------|
| 1,000 files | 5-15 min | 1-2 min | 5-10 min | 2-5 min |
| 10,000 files | 30-90 min | 5-10 min | 30-60 min | 10-30 min |
| 100,000 files | 2-6 hours | 15-30 min | 2-4 hours | 1-3 hours |

### Resource Requirements

- **CPU**: More cores = faster processing (use MAX_WORKERS setting)
- **RAM**: 4GB minimum, 8GB+ recommended for large datasets
- **Storage**: 2x your data size (for outputs and temporary files)
- **Network**: Only needed for OpenAI API or downloading models

---

## 🔐 Security & Compliance

### For Medical Data (HIPAA/PHI)

#### ✅ Recommended Setup
- Use **local AI (Ollama)** for document processing
- Store data on **encrypted drives**
- Use **private networks** (no cloud storage during processing)
- **Audit logs** are automatically created

#### ⚠️ Important Considerations
- **OpenAI option** sends data to external servers
- **Network transfers** should use encryption
- **User access** should be controlled and logged
- **Data retention** policies should be defined

### Data Privacy Features
- **Local processing**: All stages can run completely offline
- **No telemetry**: FICZall doesn't send usage data anywhere
- **Open source**: Code can be audited and modified
- **Audit trails**: Complete processing logs are maintained

---

## 📈 Advanced Usage

### Batch Processing Multiple Projects
```bash
# Process multiple datasets
for folder in /data/Hospital_A /data/Hospital_B /data/Hospital_C; do
    export Location_of_your_data="$folder"
    python run.py --non-interactive
done
```

### Custom AI Models
```python
# In S5_llmExtract_config.py, change model:
MODEL_NAME = "your-custom-model-name"

# For Ollama, first download:
# ollama pull your-custom-model-name
```

### Integration with Other Tools
```python
# Export data for analysis tools
import json
with open('data/processed/project/S5_llm_extractions/S5_extracted_data.json') as f:
    medical_data = json.load(f)
    # Use with pandas, R, SPSS, etc.
```

---

This guide should get you started with FICZall! For technical details and development information, see `CLAUDE.md`.