# FICZall - DICOM Processing Pipeline

**Find, Index, Clean, Zip all DICOM Files and Reports**

A comprehensive, automated pipeline for processing DICOM medical imaging data, from extraction through archival, with AI-powered document analysis and batch processing capabilities.

---

## 🎯 What Problem Does FICZall Solve?

### For Medical Professionals & Researchers

Medical imaging generates **massive amounts of unorganized data**. FICZall transforms this chaos into structured, searchable, and archival-ready information:

#### **Common Medical Data Challenges:**
- 🗂️ **Scattered DICOM files** across thousands of folders with cryptic names
- 📁 **ZIP/ISO archives** containing medical data that need bulk extraction  
- 🔍 **No searchable index** - finding specific patients or studies is time-consuming
- 📊 **Missing metadata** - difficult to generate reports or statistics
- 💾 **Storage inefficiency** - duplicate studies and poor compression
- 📄 **Unstructured reports** - medical documents with valuable data locked in text format
- 🏥 **Multiple locations** - different hospitals/projects need the same processing

#### **What FICZall Provides:**
- ✅ **Automated DICOM indexing** with comprehensive metadata extraction
- ✅ **Bulk ZIP/ISO extraction** with smart resume capability and integrity checking
- ✅ **Patient summaries** in Excel and JSON format for easy analysis
- ✅ **Study filtering** to remove duplicates and incomplete studies
- ✅ **Compressed archives** optimized for long-term storage
- ✅ **AI-powered extraction** of structured medical data from reports (HIPAA-compliant local processing available)
- ✅ **Batch processing** for multiple hospitals/projects with one configuration
- ✅ **User-friendly GUI** for non-technical users
- ✅ **Zero-dependency startup** - only requires Python 3.7+

#### **Typical Use Cases:**
- **Hospitals**: Organize imaging archives, create patient databases, prepare data for research
- **Research Institutions**: Process multi-site studies, extract standardized datasets
- **Medical AI Companies**: Prepare training datasets with structured metadata
- **Radiology Departments**: Archive studies efficiently, generate summary reports
- **PACS Administrators**: Migrate data between systems, create searchable indexes

---

## 🚀 Quick Setup Guide

### Prerequisites
- **Python 3.7+** (Download from [python.org](https://python.org))
- **Operating System**: Windows, macOS, or Linux
- **Disk Space**: 2-5 GB for dependencies + space for your medical data
- **Memory**: 4GB RAM minimum (8GB+ recommended for large datasets)

### Installation Steps

#### 1. Clone the Repository
```bash
# Option A: Using Git
git clone https://github.com/your-username/FICZall-Find-Index-Clean-Zip-all-Dicom-Files-and-Reports.git
cd FICZall-Find-Index-Clean-Zip-all-Dicom-Files-and-Reports

# Option B: Download ZIP
# Go to GitHub → Click "Code" → "Download ZIP" → Extract
# Then navigate to the extracted folder
```

#### 2. Initial Setup (One-time)
```bash
# This creates virtual environment and installs all dependencies
python run.py --setup
```

**That's it!** The script handles everything automatically:
- ✅ Creates isolated virtual environment
- ✅ Installs all medical processing libraries
- ✅ Verifies everything works properly
- ✅ Takes 2-5 minutes depending on internet speed

---

## 📋 Pipeline Overview

FICZall consists of **7 stages** that can be run individually or together:

| Stage | Name | Purpose | Required | Time* |
|-------|------|---------|----------|--------|
| **S0_ZIP** | ZIP Extraction | Extract ZIP files with smart logic | Optional | 5-30 min |
| **S0_ISO** | ISO Extraction | Extract DICOM from ISO archives | Optional | 10-60 min |
| **S1** | DICOM Indexing | Scan & extract metadata | **✅ Required** | 10-120 min |
| **S2** | Summary Creation | Aggregate patient data | **✅ Required** | 1-10 min |
| **S3** | Study Filtering | Remove duplicates & filter | **✅ Required** | 1-15 min |
| **S4** | ZIP Archiving | Compress for storage | Optional | 15-180 min |
| **S5** | 🤖 AI Extraction | Extract medical data with AI | Optional | 5-60 min |

*Processing times for ~10,000 files on modern hardware

---

## 💻 Three Ways to Use FICZall

Choose the method that best fits your technical comfort level:

### 🖱️ Option 1: Graphical User Interface (Recommended for Non-Technical Users)
**Simple point-and-click interface with real-time progress monitoring**

**Universal Launch Method (Works on All Systems):**

1. **Download and extract FICZall** (as shown above)

2. **Launch the GUI:**
   
   **Windows (Command Prompt or PowerShell):**
   ```cmd
   python launch_gui.py
   ```
   
   **macOS/Linux (Terminal):**
   ```bash
   python3 launch_gui.py
   ```

3. **Using the GUI:**
   - **Single Location Tab**: Process one DICOM folder
   - **Multi-Location Tab**: Process multiple folders with same settings
   - **Browse buttons**: Click to select your DICOM folders
   - **Memory system**: Remembers your last 3 inputs for each field
   - **Real-time logs**: Watch processing progress live
   - **Stop/Resume**: Control processing as needed
   - **Helpful hints**: Tooltips guide you through settings

4. **Basic workflow:**
   - Select your DICOM data folder using "Browse"
   - Choose which pipeline stages to run (S1-S3 minimum required)
   - Adjust settings with helpful hints (min files: 5-20, max workers: 2-8)
   - Click "Start Processing"
   - Monitor progress in the log window
   - Find results in `data/processed/[project_name]/`

**Note:** The GUI launcher automatically checks for Python, tkinter, and all required files before starting.

---

### 🔧 Option 2: Single Location Processing (run.py)
**Command-line interface for processing one DICOM dataset at a time**

<details>
<summary><strong>🖥️ Windows Setup</strong></summary>

1. **Open Command Prompt or PowerShell** in the FICZall folder

2. **Interactive mode (recommended for first-time users):**
   ```cmd
   python run.py
   ```
   The script will ask you:
   - Where your DICOM files are located
   - Which stages to run
   - Project name (optional)
   - Processing settings

3. **Common workflows:**

   **Basic DICOM processing:**
   ```cmd
   python run.py
   # Choose: S1, S2, S3 (required stages)
   # Enter path like: C:\Medical_Data\CT_Scans
   ```

   **Process ZIP files first:**
   ```cmd
   python run.py  
   # Choose: S0_ZIP, S1, S2, S3
   # Enter path to folder containing ZIP files
   ```

   **Add AI analysis:**
   ```cmd
   python run.py
   # Choose: S1, S2, S3, S5
   # Make sure to setup AI first (see AI section below)
   ```

4. **Non-interactive mode (after first setup):**
   ```cmd
   python run.py --non-interactive
   ```

5. **Other useful commands:**
   ```cmd
   python run.py --config          # Show current configuration
   python run.py --stage S1        # Run only specific stage
   python run.py --setup           # Setup virtual environment only
   ```

</details>

<details>
<summary><strong>🍎 macOS Setup</strong></summary>

1. **Open Terminal** and navigate to FICZall folder:
   ```bash
   cd /path/to/FICZall-Find-Index-Clean-Zip-all-Dicom-Files-and-Reports
   ```

2. **Interactive mode (recommended for first-time users):**
   ```bash
   python3 run.py
   ```
   The script will ask you:
   - Where your DICOM files are located
   - Which stages to run
   - Project name (optional)
   - Processing settings

3. **Common workflows:**

   **Basic DICOM processing:**
   ```bash
   python3 run.py
   # Choose: S1, S2, S3 (required stages)
   # Enter path like: /Users/yourname/Medical_Data/CT_Scans
   ```

   **Process ZIP files first:**
   ```bash
   python3 run.py
   # Choose: S0_ZIP, S1, S2, S3  
   # Enter path to folder containing ZIP files
   ```

   **Add AI analysis:**
   ```bash
   python3 run.py
   # Choose: S1, S2, S3, S5
   # Make sure to setup AI first (see AI section below)
   ```

4. **Non-interactive mode (after first setup):**
   ```bash
   python3 run.py --non-interactive
   ```

5. **Other useful commands:**
   ```bash
   python3 run.py --config          # Show current configuration
   python3 run.py --stage S1        # Run only specific stage  
   python3 run.py --setup           # Setup virtual environment only
   ```

</details>

---

### 🏥 Option 3: Multi-Location Batch Processing (runMulti.py)
**Process multiple hospitals, studies, or projects with the same configuration**

<details>
<summary><strong>🖥️ Windows Setup</strong></summary>

1. **Open Command Prompt or PowerShell** in the FICZall folder

2. **Interactive batch setup (recommended):**
   ```cmd
   python runMulti.py
   ```
   The script will ask you:
   - Multiple data locations (hospitals, studies, projects)
   - Single configuration that applies to ALL locations
   - Processing stages for all locations

3. **Example workflow:**
   ```cmd
   python runMulti.py
   
   # When prompted, enter locations like:
   # Location 1: C:\Hospital_A\CT_Studies_2024
   # Location 2: C:\Hospital_B\MRI_Studies_2024
   # Location 3: C:\Research_Project_Data
   # 
   # Then choose stages: S1, S2, S3 (or add S4, S5 as needed)
   # 
   # FICZall will process each location completely before moving to the next:
   # Hospital_A → ALL stages complete → Individual project folder
   # Hospital_B → ALL stages complete → Individual project folder  
   # Research_Project → ALL stages complete → Individual project folder
   ```

4. **Resume interrupted processing:**
   ```cmd
   python runMulti.py --resume
   ```

5. **Use existing batch configuration:**
   ```cmd
   python runMulti.py --non-interactive
   ```

6. **Other useful commands:**
   ```cmd
   python runMulti.py --config      # Show batch configuration and progress
   python runMulti.py --setup       # Setup virtual environment for batch processing
   ```

**Batch Processing Benefits:**
- **Sequential processing**: Each location completes fully before next starts (no conflicts)
- **One-time setup**: Configure once, process many locations
- **Individual projects**: Each location gets its own organized output folder
- **Progress tracking**: Resume from any interruption point
- **Error recovery**: Continue processing even if one location fails

</details>

<details>
<summary><strong>🍎 macOS Setup</strong></summary>

1. **Open Terminal** and navigate to FICZall folder:
   ```bash
   cd /path/to/FICZall-Find-Index-Clean-Zip-all-Dicom-Files-and-Reports
   ```

2. **Interactive batch setup (recommended):**
   ```bash
   python3 runMulti.py
   ```
   The script will ask you:
   - Multiple data locations (hospitals, studies, projects)
   - Single configuration that applies to ALL locations
   - Processing stages for all locations

3. **Example workflow:**
   ```bash
   python3 runMulti.py
   
   # When prompted, enter locations like:
   # Location 1: /Users/yourname/Hospital_A/CT_Studies_2024
   # Location 2: /Users/yourname/Hospital_B/MRI_Studies_2024
   # Location 3: /Users/yourname/Research_Project_Data
   #
   # Then choose stages: S1, S2, S3 (or add S4, S5 as needed)
   #
   # FICZall will process each location completely before moving to the next:
   # Hospital_A → ALL stages complete → Individual project folder
   # Hospital_B → ALL stages complete → Individual project folder
   # Research_Project → ALL stages complete → Individual project folder
   ```

4. **Resume interrupted processing:**
   ```bash
   python3 runMulti.py --resume
   ```

5. **Use existing batch configuration:**
   ```bash
   python3 runMulti.py --non-interactive
   ```

6. **Other useful commands:**
   ```bash
   python3 runMulti.py --config      # Show batch configuration and progress
   python3 runMulti.py --setup       # Setup virtual environment for batch processing
   ```

**Batch Processing Benefits:**
- **Sequential processing**: Each location completes fully before next starts (no conflicts)
- **One-time setup**: Configure once, process many locations
- **Individual projects**: Each location gets its own organized output folder
- **Progress tracking**: Resume from any interruption point
- **Error recovery**: Continue processing even if one location fails

</details>

---

## 🤖 AI-Powered Medical Document Extraction (S5)

### What Does AI Extraction Do?

The AI stage (S5) processes medical documents and extracts structured information:

- **Patient Information**: Names, ages, IDs, demographics
- **Study Details**: Imaging modality (CT, MRI, etc.), body parts examined
- **Medical Findings**: Key findings from radiology reports
- **Diagnoses**: Current and previous medical conditions
- **Treatments**: Previous interventions, surgeries, medications
- **Clinical Context**: Reasons for imaging, medical history

### Privacy & Security Options

#### 🔒 Local AI (Ollama) - HIPAA Compliant
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

### Setting Up Local AI (Recommended for Medical Data)

<details>
<summary><strong>🖥️ Windows Setup</strong></summary>

1. **Install Ollama** (one-time setup):
   - Go to [ollama.ai](https://ollama.ai/)
   - Download "Ollama for Windows"
   - Run the installer

2. **Download AI Model** (one-time, ~4GB download):
   ```cmd
   ollama pull llama3-groq-tool-use:8b-q8_0
   ```

3. **Start Ollama Service** (keep running when using S5):
   ```cmd
   ollama serve
   ```

4. **Configure FICZall for Local AI:**
   ```cmd
   set S5_LLM_CLIENT_TYPE=ollama
   ```

5. **Test AI Setup:**
   ```cmd
   python code/S5_llmExtract_test.py --client-type ollama
   ```

6. **Use AI in Pipeline:**
   ```cmd
   python run.py --stage S5
   # OR include S5 when running full pipeline
   python run.py
   # Choose stages: S1, S2, S3, S5
   ```

</details>

<details>
<summary><strong>🍎 macOS Setup</strong></summary>

1. **Install Ollama** (one-time setup):
   - Go to [ollama.ai](https://ollama.ai/)
   - Download "Ollama for macOS"
   - Open the downloaded file and follow installation instructions

2. **Download AI Model** (one-time, ~4GB download):
   ```bash
   ollama pull llama3-groq-tool-use:8b-q8_0
   ```

3. **Start Ollama Service** (keep running when using S5):
   ```bash
   ollama serve
   ```

4. **Configure FICZall for Local AI:**
   ```bash
   export S5_LLM_CLIENT_TYPE=ollama
   ```

5. **Test AI Setup:**
   ```bash
   python3 code/S5_llmExtract_test.py --client-type ollama
   ```

6. **Use AI in Pipeline:**
   ```bash
   python3 run.py --stage S5
   # OR include S5 when running full pipeline
   python3 run.py
   # Choose stages: S1, S2, S3, S5
   ```

</details>

### Setting Up Cloud AI (OpenAI)

<details>
<summary><strong>🖥️ Windows Setup</strong></summary>

1. **Get OpenAI API Key:**
   - Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Create account and generate API key

2. **Set Environment Variable:**
   ```cmd
   set OPENAI_API_KEY=your_api_key_here
   set S5_LLM_CLIENT_TYPE=openai
   ```

3. **Use AI in Pipeline:**
   ```cmd
   python run.py --stage S5
   ```

</details>

<details>
<summary><strong>🍎 macOS Setup</strong></summary>

1. **Get OpenAI API Key:**
   - Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Create account and generate API key

2. **Set Environment Variable:**
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   export S5_LLM_CLIENT_TYPE=openai
   ```

3. **Use AI in Pipeline:**
   ```bash
   python3 run.py --stage S5
   ```

</details>

---

## 📁 Output Structure

After processing, your data will be organized as follows:

### Single Location Processing (run.py)
```
data/processed/your_project_name/
├── S1_indexed_metadata/           # 📋 DICOM metadata index
│   ├── Patient_001_Study.jsonl   # Individual patient data
│   ├── Patient_002_Study.jsonl
│   └── S1_indexingFiles_allDocuments.json
├── S2_concatenated_summaries/     # 📊 Patient summaries
│   ├── S2_patient_summary.json   # Machine-readable
│   └── S2_patient_summary.xlsx   # Human-readable Excel
├── S3_filtered_studies/           # 🔄 Filtered studies
│   └── S3_filtered_studies.json
├── S4_zip_archives/               # 💾 Compressed archives (optional)
│   └── Patient_Studies.zip
└── S5_llm_extractions/           # 🤖 AI-extracted data (optional)
    ├── S5_extracted_data.json
    └── S5_processing_results.json
```

### Multi-Location Batch Processing (runMulti.py)
```
data/processed/
├── hospital_a_ct_2024/           # 🏥 Location 1 results
│   ├── S1_indexed_metadata/
│   ├── S2_concatenated_summaries/
│   ├── S3_filtered_studies/
│   ├── S4_zip_archives/
│   └── S5_llm_extractions/
├── hospital_b_mri_2024/          # 🏥 Location 2 results
│   ├── S1_indexed_metadata/
│   ├── S2_concatenated_summaries/
│   ├── S3_filtered_studies/
│   ├── S4_zip_archives/
│   └── S5_llm_extractions/
└── research_project_001/         # 🧪 Location 3 results
    ├── S1_indexed_metadata/
    ├── S2_concatenated_summaries/
    ├── S3_filtered_studies/
    ├── S4_zip_archives/
    └── S5_llm_extractions/
```

---

## ⚙️ Configuration

### Essential Settings

The most important configuration is in `run_config.py`:

```python
# STEP 1: Point to your DICOM data (CHANGE THIS!)
Location_of_your_data = r"C:\Your\Path\To\DICOM\Files"

# STEP 2: Project name (optional - auto-generated if empty)
desired_name_of_project = "MyHospital_CT_Study_2024"
```

### Performance Tuning

For different dataset sizes:

**Small Dataset (< 1,000 files):**
```python
MAX_WORKERS = '2'
S3_MIN_FILES = '5'
ZIP_COMPRESSION_LEVEL = '6'
```

**Large Dataset (> 10,000 files):**
```python
MAX_WORKERS = '8'
S3_MIN_FILES = '20'
ZIP_COMPRESSION_LEVEL = '1'  # Faster compression
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

---

## 🛠️ External Dependencies

### For S0_ISO (ISO Extraction) - Windows Only
- **PowerISO**: Download from [poweriso.com](https://www.poweriso.com/)
- Configure path in `run_config.py`
- Optional - skip S0 if DICOM files are already extracted

### For S5 (AI Processing) - Optional
- **Local AI**: Ollama software (free, privacy-safe)
- **Cloud AI**: OpenAI API key (paid, data sent to OpenAI)

### Python Packages (Auto-installed)
All required packages are installed automatically:
- **pydicom**: DICOM file reading and metadata extraction
- **pandas**: Data manipulation and Excel output
- **numpy**: Array operations for performance
- **tqdm**: Progress bars
- **openai**: AI-powered document extraction
- **pydantic**: Data validation for AI outputs

---

## 🚨 Troubleshooting

### Common Issues

**"Python not found"**
- **Windows**: Reinstall Python, check "Add to PATH" during installation
- **macOS**: Install Python 3.7+ from python.org or use `brew install python3`

**"Permission denied"**
- **Windows**: Run Command Prompt as Administrator
- **macOS**: Use `sudo` before the python command

**"DICOM files not found"**
- Check your path uses forward slashes: `C:/Medical/Data/DICOM`
- Verify the directory exists and contains DICOM files

**"Out of memory"**
- Reduce worker count in `run_config.py`: `MAX_WORKERS = '2'`

**"Ollama not found" (for AI features)**
- Install Ollama from [ollama.ai](https://ollama.ai/)
- Start the service: `ollama serve`
- Test with: `python code/S5_llmExtract_test.py --client-type ollama`

### Getting Help

1. **Check logs**: Look in `data/processed/project_name/` for detailed error information
2. **Test setup**: Run `python run.py --config` to verify configuration
3. **AI testing**: Run `python code/S5_llmExtract_test.py` to test AI features
4. **GUI logs**: Use "Save Log" button to save processing logs for troubleshooting
5. **GitHub Issues**: Report bugs at the project's GitHub repository

---

## 📊 Performance Guidelines

### Processing Time Estimates

| Dataset Size | S1 (Indexing) | S2+S3 (Summary) | S4 (Archive) | S5 (AI) |
|--------------|---------------|-----------------|--------------|---------|
| 1,000 files | 5-15 min | 1-2 min | 5-10 min | 2-5 min |
| 10,000 files | 30-90 min | 5-10 min | 30-60 min | 10-30 min |
| 100,000 files | 2-6 hours | 15-30 min | 2-4 hours | 1-3 hours |

### System Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB+ RAM, 4+ CPU cores
- **Storage**: 2-3x the size of input data for temporary files

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

## 🏥 Medical Data Disclaimer

This software is designed for research and archival purposes. Ensure compliance with:
- **HIPAA** (Health Insurance Portability and Accountability Act)
- **GDPR** (General Data Protection Regulation)
- **Local healthcare data protection regulations**

Always verify that processed data maintains patient privacy and institutional compliance requirements.

---

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

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Need help?** 
- 🖱️ **Try the GUI**: Run `python launch_gui.py` for a user-friendly interface
- 📖 **Command-line users**: See `RUN_SCRIPT_GUIDE.md` for detailed run.py instructions
- 🖥️ **GUI users**: See `GUI_USER_GUIDE.md` for complete GUI documentation
- 🐛 **Report issues**: Use GitHub Issues with error logs
- 💬 **Discussions**: Join GitHub Discussions for usage questions