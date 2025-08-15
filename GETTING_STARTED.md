# 🚀 FICZall Quick Start Guide

## For Busy Medical Professionals - 5 Minute Setup

### What You Need
1. **Computer**: Windows, Mac, or Linux
2. **Python**: Download from [python.org](https://python.org) (check "Add to PATH" during install)
3. **Your DICOM data**: Know where your medical imaging files are stored

---

## ⚡ Super Quick Start (3 Steps)

### Step 1: Download & Setup (2 minutes)
```bash
# Download FICZall from GitHub (or get the ZIP file)
cd FICZall-Find-Index-Clean-Zip-all-Dicom-Files-and-Reports

# One-time setup (installs everything needed)
python run.py --setup
```

### Step 2: Point to Your Data (30 seconds)

**Single Location:**
```bash
# Start the interactive setup
python run.py
```
When prompted, enter the path to your DICOM files, like:
- Windows: `C:\Medical_Data\CT_Scans`
- Mac/Linux: `/Users/yourname/Medical_Data/CT_Scans`

**Multiple Locations (NEW!):**
```bash
# Process multiple hospitals/projects at once
python runs.py
```
When prompted, enter multiple paths like:
- Location 1: `C:\Hospital_A\CT_2024`
- Location 2: `C:\Hospital_B\MRI_2024`  
- Location 3: `C:\Research_Project_Data`

### Step 3: Choose What to Run (30 seconds)
Select stages based on your needs:
- **S1, S2, S3**: Basic processing (always needed)
- **S4**: Add if you want compressed archives  
- **S5**: Add if you want AI analysis

**Done!** FICZall will process your data automatically.

---

## 🤖 Want AI Analysis? (Extra 5 minutes)

### For Privacy-Safe AI (Recommended for Medical Data)
```bash
# 1. Install Ollama (download from ollama.ai)
# 2. Download AI model (one-time, ~4GB)
ollama pull llama3-groq-tool-use:8b-q8_0

# 3. Start Ollama
ollama serve

# 4. Run AI analysis
python run.py --stage S5
```

### For Fast Cloud AI (Data leaves your computer)
```bash
# 1. Get OpenAI API key from platform.openai.com
# 2. Set environment variable
export OPENAI_API_KEY=your_key_here

# 3. Run AI analysis  
python run.py --stage S5
```

---

## 📁 What You'll Get

**Single Location Processing:**
```
data/processed/your_project_name/
├── S1_indexed_metadata/     # 📋 DICOM index files
├── S2_concatenated_summaries/ # 📊 Patient summaries (Excel/JSON)
├── S4_zip_archives/         # 💾 Compressed studies (if S4 run)
└── S5_llm_extractions/      # 🤖 AI-extracted medical data (if S5 run)
```

**Multi-Location Batch Processing:**
```
data/processed/
├── hospital_a_ct_2024/      # 🏥 First location results
│   ├── S1_indexed_metadata/
│   ├── S2_concatenated_summaries/
│   ├── S4_zip_archives/
│   └── S5_llm_extractions/
├── hospital_b_mri_2024/     # 🏥 Second location results
│   ├── S1_indexed_metadata/
│   ├── S2_concatenated_summaries/
│   ├── S4_zip_archives/
│   └── S5_llm_extractions/
└── research_project_001/    # 🧪 Third location results
    ├── S1_indexed_metadata/
    ├── S2_concatenated_summaries/
    ├── S4_zip_archives/
    └── S5_llm_extractions/
```

---

## ❓ Need Help?

### Quick Fixes
- **"Python not found"**: Install Python, check "Add to PATH"
- **"Permission denied"**: Run as Administrator (Windows) or with `sudo` (Mac/Linux)
- **"Files not found"**: Check your path has forward slashes: `C:/Medical/Data`

### More Help
- **Detailed Guide**: See `README_USAGE.md` for complete instructions
- **Technical Info**: See `CLAUDE.md` for development details
- **Test AI**: Run `python code/S5_llmExtract_test.py --client-type ollama`
- **Batch Status**: Run `python runs.py --config` to see batch progress
- **Resume Batch**: Run `python runs.py --resume` if batch processing was interrupted

---

## 🔒 Privacy Notes

- **Local processing** (default): All your data stays on your computer
- **Local AI** (Ollama): Patient data never leaves your machine - HIPAA friendly
- **Cloud AI** (OpenAI): Faster but data sent to OpenAI servers
- **No telemetry**: FICZall doesn't send any usage data anywhere

---

**That's it! You're ready to process medical imaging data with FICZall.**

🎯 **Choose Your Approach:**
- **Single Project**: Use `python run.py` for one dataset at a time
- **Multiple Projects**: Use `python runs.py` for batch processing multiple locations

For advanced features and detailed configuration, see the full documentation in `README_USAGE.md`.