# DICOM Processing Pipeline - GUI User Guide

## Quick Start

### Universal Method (All Operating Systems)
1. **Navigate** to the FICZall project directory
2. **Launch the GUI:**
   
   **Windows (Command Prompt or PowerShell):**
   ```cmd
   python launch_gui.py
   ```
   
   **macOS/Linux (Terminal):**
   ```bash
   python3 launch_gui.py
   ```

3. The GUI will open automatically with error checking and helpful messages

---

## Application Overview

The GUI provides a simple, user-friendly interface for running DICOM processing pipelines without needing to use command-line tools.

### Main Features
- ✅ **Two Processing Modes**: Single location or multiple locations (batch)
- ✅ **Memory System**: Remembers your last 3 inputs for each field
- ✅ **Real-time Progress**: See processing progress and logs in real-time
- ✅ **Virtual Environment Integration**: Automatically uses the project's virtual environment
- ✅ **Log Management**: View, clear, and save processing logs

---

## Tab 1: Single Location Processing

Use this tab to process DICOM files from **one data location**.

### Configuration Steps

1. **DICOM Data Location**
   - Click "Browse" to select your DICOM data folder
   - Or type the path directly
   - The dropdown shows your 3 most recent locations

2. **Project Name** (Optional)
   - Enter a custom name for your project
   - If left empty, the app auto-generates a name from folder structure
   - The dropdown shows your 3 most recent project names

3. **Pipeline Stages**
   - ✅ **Required stages** (S1, S2, S3): Always needed
   - ⚙️ **Optional stages**: Check only what you need
     - **S0 ZIP**: Extract ZIP files first
     - **S0 ISO**: Extract ISO files first  
     - **S4**: Create ZIP archives for storage
     - **S5**: AI-powered document extraction

4. **Advanced Settings**
   - **Min files per study**: Minimum files required (default: 10)
   - **Max workers**: Number of parallel processes (default: 4)

### Action Buttons
- **Start Processing**: Begin the full pipeline
- **Setup Only**: Just setup the virtual environment
- **Show Config**: Display current configuration

---

## Tab 2: Multi-Location (Batch) Processing

Use this tab to process **multiple DICOM locations** with the same settings.

### Configuration Steps

1. **Add Data Locations**
   - Enter or browse for each DICOM data location
   - Click "Add" to add it to the list
   - Use "Remove Selected" or "Clear All" to manage the list
   - The dropdown remembers your recent locations

2. **Batch Settings**
   - **Project Naming**: 
     - **Auto**: Uses folder names (recommended)
     - **Custom**: You provide a base name (gets numbered: name_001, name_002...)
   - **Max Workers**: Parallel processes for batch (default: 4)
   - **Min Files per Study**: Same as single location (default: 10)

### Action Buttons
- **Start Batch Processing**: Process all locations sequentially
- **Resume Processing**: Continue interrupted batch processing
- **Show Batch Config**: Display current batch configuration

---

## Processing Output & Logs

### Real-time Display
- All processing output appears in the bottom section
- Shows commands being run, progress, and any errors
- Auto-scrolls to show latest output

### Control Buttons
- **Stop Process**: Immediately stop the current processing
- **Clear Output**: Clear the log display
- **Save Log**: Save the current log to a file

---

## Memory System

The application remembers your **3 most recent inputs** for each field:

### What's Remembered
- ✅ DICOM data locations (single mode)
- ✅ Project names (single mode)  
- ✅ Multi-location paths (batch mode)

### How It Works
- Input history saved automatically in `gui_memory.pkl`
- Dropdown lists show recent entries
- Most recent entries appear first
- Memory persists between sessions

---

## Tips for Non-Technical Users

### Getting Started
1. **Start with Single Location** if you're new to the tool
2. **Use Browse buttons** instead of typing paths manually
3. **Keep default settings** for most cases
4. **Check the logs** if something goes wrong

### Common Workflows

**Basic DICOM Processing:**
1. Select your DICOM folder
2. Keep S1, S2, S3 checked (required stages)
3. Click "Start Processing"
4. Wait and watch the progress logs

**Processing ZIP Files:**
1. Check "S0 ZIP - ZIP File Extraction" 
2. Select the folder containing ZIP files
3. The tool will extract ZIPs first, then process DICOM files

**Batch Processing Multiple Hospitals:**
1. Go to "Multi Location (Batch)" tab
2. Add each hospital's data folder
3. Click "Start Batch Processing"
4. Each location processes completely before moving to the next

### Troubleshooting

**"Process is already running"**
- Wait for current process to finish, or click "Stop Process"

**"Data location does not exist"**
- Double-check your folder path
- Make sure the folder contains DICOM files

**Application won't start:**
- Make sure you have Python 3.7+ installed
- Run `start_gui.bat` (Windows) or `python launch_gui.py`

**Virtual environment issues:**
- Click "Setup Only" to reinstall dependencies
- Check that `venv` folder exists in the project directory

---

## File Structure

```
📁 Project Directory/
├── 🚀 launch_gui.py          # Universal GUI launcher (all systems)
├── 🖥️ gui_app.py              # Main GUI application
├── 💾 gui_memory.pkl         # Your input history (auto-created)
├── ⚙️ run.py                  # Single location processor
├── ⚙️ runMulti.py             # Multi location processor
├── 📋 runMulti_config.json   # Batch configuration (auto-created)
└── 📁 data/                  # Output directory (auto-created)
    └── processed/            # Processed results appear here
```

---

## Getting Help

### If You Need Support
1. **Check the logs** in the bottom panel for error messages
2. **Save the log** using "Save Log" button before asking for help
3. **Include your log file** when reporting issues

### Common Log Messages
- `[STEP]`: Normal processing step
- `[OK]`: Successfully completed action
- `[WARNING]`: Warning (usually safe to ignore)  
- `[ERROR]`: Error occurred (processing may fail)
- `[COMPLETED]`: Processing finished successfully

---

## Advanced Features

### Resume Capability
- Single location: Stages can resume if interrupted
- Batch processing: Use "Resume Processing" to continue from where it stopped

### Configuration Files
- Single location: Uses `run_config.py`
- Batch processing: Creates `runMulti_config.json`
- All output goes to `data/processed/[project_name]/`

### Virtual Environment
- The app automatically uses the project's Python virtual environment
- All required packages are installed automatically
- No need to manually manage Python dependencies

---

**Enjoy using the DICOM Processing Pipeline GUI!** 🎉