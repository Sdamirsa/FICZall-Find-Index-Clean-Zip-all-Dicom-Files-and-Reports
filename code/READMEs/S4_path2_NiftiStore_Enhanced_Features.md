# S4_path2_NiftiStore.py - Enhanced Features

## Overview
The S4_path2_NiftiStore.py script has been enhanced with two major new functionalities:

1. **Smart Image3Ddata File Detection** - Skip analysis step if processed file exists
2. **Resume Functionality** - Progress tracking and resume capability for long-running conversions

## New Features

### 1. Image3Ddata File Detection and Skip Logic

**Purpose**: Automatically detect existing processed Image3Ddata files and allow users to skip the analysis step.

**How it works**:
- When starting the pipeline, checks for `{excel_name}_Image3Ddata.xlsx` in the same directory
- If found, prompts user with file information (size, modification date)
- User can choose to:
  - Use existing file (skip analysis step)
  - Regenerate analysis file

**User Interface**:
```
📊 Found existing processed file: consolidated_studies_Image3Ddata.xlsx
📅 File last modified: 2024-10-15 14:30:22
📏 File size: 2.45 MB

🤔 Do you want to use this existing file and skip the analysis step? (y/n):
```

### 2. Resume Functionality with Progress Tracking

**Purpose**: Save processing progress every 30 series and allow resuming from where left off.

**Key Components**:

#### Progress Tracking File
- **Location**: `{output_dir}/{excel_name}_nifti_progress.json`
- **Contains**: 
  - Processed series with success/failure status
  - Timestamps and error messages
  - Total series count and batch information
  - Source Excel file path for validation

#### NIfTI Processing Excel Tracker
- **Location**: `{output_dir}/{excel_name}_nifti_results.xlsx`
- **Created at start** with all series to be processed
- **Updated every 30 series** with current progress
- **Two sheets**:
  - `NIfTI_Processing`: Detailed series information
  - `Summary`: Processing statistics

#### Batch Processing
- Default: Save progress every 30 series (configurable with `--batch-size`)
- Automatic progress file creation and updates
- Excel tracker updated in sync with JSON progress

**User Interface for Resume**:
```
🔄 Found existing progress file:
   📊 Processed series: 45
   📅 Last update: 2024-10-15T14:25:33
   📂 Progress file: consolidated_studies_nifti_progress.json

🤔 Do you want to resume from where you left off? (y/n):
```

## Implementation Details

### New Methods Added

#### Progress Management
- `setup_progress_tracking(excel_path)` - Initialize progress tracking
- `ask_user_resume_or_restart()` - User prompt for resume/restart
- `save_progress()` - Save current progress to JSON
- `is_series_already_processed(hash_id)` - Check if series processed
- `mark_series_processed(hash_id, success, error_msg)` - Mark series complete

#### Image3D File Management
- `check_existing_image3d_file(excel_path)` - Find existing Image3Ddata file
- `ask_user_skip_first_step(image3d_file)` - User prompt to skip analysis

#### NIfTI Excel Tracking
- `create_nifti_excel_at_start(excel_path, studies_df)` - Create tracking Excel
- `update_nifti_excel()` - Update Excel with current progress

### Modified Methods

#### Core Processing
- `__init__()` - Added progress tracking variables
- `process_single_series()` - Integrated progress tracking calls
- `run_pipeline()` - Added all new functionality integration

#### Error Handling
- All failure points now update progress tracking
- Resume capability handles both successful and failed series
- Backward compatibility with existing processed files

## Usage Examples

### Basic Usage (Default 30-series batches)
```bash
python S4_path2_NiftiStore.py --excel-file /path/to/studies.xlsx
```

### Custom Batch Size
```bash
python S4_path2_NiftiStore.py --excel-file /path/to/studies.xlsx --batch-size 50
```

### Force Restart (Ignore Progress)
```bash
python S4_path2_NiftiStore.py --excel-file /path/to/studies.xlsx --force-reprocess
```

### Analysis Only Mode
```bash
python S4_path2_NiftiStore.py --excel-file /path/to/studies.xlsx --analysis-only
```

## File Outputs

### New Files Created
1. `{excel_name}_nifti_progress.json` - Progress tracking
2. `{excel_name}_nifti_results.xlsx` - Processing tracker Excel

### Enhanced Workflow
1. **Start**: Check for existing Image3Ddata file
2. **Setup**: Create progress tracking and NIfTI Excel
3. **Process**: Save progress every N series (default 30)
4. **Resume**: Skip already processed series on restart
5. **Complete**: Final progress save and Excel update

## Benefits

### Efficiency
- Skip time-consuming analysis if already done
- Resume long-running conversions without losing progress
- Avoid reprocessing already completed series

### Reliability
- Comprehensive error tracking with timestamps
- Progress preserved across interruptions
- User control over restart vs resume decisions

### Monitoring
- Real-time Excel tracker for progress monitoring
- Detailed statistics and processing rates
- Clear success/failure tracking per series

## Backward Compatibility

- All existing functionality preserved
- New features are additive and optional
- Existing progress checking methods still supported
- Command-line arguments are all optional with sensible defaults