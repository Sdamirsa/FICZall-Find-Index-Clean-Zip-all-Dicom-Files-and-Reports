# S5_NLPseriesLabel.py

## Purpose
Applies multi-layer NLP logic to extract and standardize medical imaging labels from consolidated DICOM metadata Excel files. Provides comprehensive series-level classification for contrast enhancement, anatomical regions, reconstruction types, and temporal phases.

## Code Logic
- **Multi-layer pattern matching**: Each label uses 2-4 layers of extraction logic with confidence scoring
- **Medical terminology recognition**: Comprehensive patterns for contrast phases, anatomical regions, and technical parameters
- **Confidence-based filtering**: Only assigns labels with confidence ≥ 0.7 to avoid uncertain classifications
- **Hierarchical logic**: Study-level patterns influence series-level classifications appropriately
- **Regex and keyword matching**: Sophisticated pattern detection for complex medical abbreviations

## Core Label Categories
1. **Image Origin**: ORIGINAL vs DERIVED (from image_type)
2. **Reconstruction Type**: MPR, MIP, LOCALIZER, VOLUME, HELICAL, REFORMATTED, AVERAGE, AXIAL
3. **Slice Thickness**: Numerical value in mm (from parsed_slice_thickness)
4. **Study Contrast Enhanced**: YES/NO/UNKNOWN (study-level contrast usage)
5. **Series Contrast Enhanced**: YES/NO/UNKNOWN (series-level contrast presence)
6. **Image Orientation**: AXIAL, CORONAL, SAGITTAL, OBLIQUE
7. **Contrast Phase**: NON_CONTRAST, ARTERIAL, VENOUS, PORTAL, DELAY, EQUILIBRIUM
8. **Standardized Body Part**: HEAD, NECK, CHEST, ABDOMEN, PELVIS, combinations, etc.

## Inputs
- **Primary**: Excel files from `data/consolidated_summaries/` directory
- **Key columns processed**:
  - `study_description`: Study-level contrast and protocol information
  - `study_body_part`: Anatomical region classification
  - `study_protocol`: Technical protocol details
  - `image_type`: Image origin and reconstruction type
  - `study_series_description`: Series-level phase and contrast information
  - `parsed_slice_thickness`: Slice thickness measurements

## Outputs
- **Default location**: `data/consolidated_summaries/{filename}_NLPlabel.xlsx`
- **New columns added** (each with confidence score):
  - `image_origin` + `image_origin_confidence`
  - `reconstruction_type` + `reconstruction_confidence`
  - `slice_thickness_mm` + `thickness_confidence`
  - `study_contrast_enhanced` + `study_contrast_confidence`
  - `series_contrast_enhanced` + `series_contrast_confidence`
  - `image_orientation` + `orientation_confidence`
  - `contrast_phase` + `phase_confidence`
  - `standardized_body_part` + `body_part_confidence`

## User Changeable Settings
- `--input`: Specify input Excel file path
- `--output`: Override default output location
- `--test`: Process only first 100 rows for testing
- **Confidence threshold**: Currently hardcoded to 0.7 (can be modified in code)
- **Pattern libraries**: Extensive medical terminology patterns in LabelingConfig class

## Usage Examples

### Interactive Mode (Default)
```bash
python code/S5_NLPseriesLabel.py
```
- Presents file selection interface
- Lists available Excel files in default directory
- Guides user through selection process

### Direct File Processing
```bash
python code/S5_NLPseriesLabel.py --input "data/consolidated_summaries/input_file.xlsx"
```

### Test Mode
```bash
python code/S5_NLPseriesLabel.py --test --input "path/to/file.xlsx"
```

### Custom Output
```bash
python code/S5_NLPseriesLabel.py --input "input.xlsx" --output "custom_output.xlsx"
```

## Medical Pattern Recognition

### Study Contrast Patterns
- **With contrast**: `WITH`, `CONTRAST`, `3PHASE`, `3PHASIC`, `3P`, `3PH`, `+C`, `+CM`, `WOWITH`, `WITHOUT_WITH`, `WW`
- **Without contrast**: `WITHOUT`, `WO`, `-C`, `PLAIN`, `NATIVE`
- **Complex patterns**: `Private^thxabdpelWOWITH_TALEGHANI` → Study has contrast

### Series Contrast & Phase Patterns
- **Non-contrast**: `PRE`, `PLAIN`, `W/O`, `WO`, `PREMONITORING`, `ABD WO`
- **Arterial**: `ARTERIAL`, `ART`, `EARLY`, `THIN ARTERIAL`
- **Venous**: `VENOUS`, `VEN`, `VENOUS PHASE`
- **Portal**: `PORTAL`, `PORTOVENOUS`, `COR-PORTAL`
- **Delay**: `DELAY`, `DELAY PHASE`, `3M DELAY`, `15 MIN`, `7MIN`
- **Enhancement**: `CE` (with numbers), `Body 5.000 CE`, `CORONAL C`

### Timing-Based Detection
- **Arterial phase**: ≤30 seconds timing
- **Portal phase**: 60-90 seconds timing  
- **Delay phase**: >120 seconds or >3 minutes

### Anatomical Region Patterns
- **Chest**: `CHEST`, `THORAX`, `THX`, `PULMONARY`, `ROUTINE CHEST`
- **Abdomen**: `ABDOMEN`, `ABD`, `LIVER`, `PANCREAS`
- **Combined regions**: `ABD.*PEL`, `THX.*ABD.*PEL`, `THXABDPEL`

## Performance Characteristics
- **Processing speed**: ~1,000 rows per 150ms
- **Memory usage**: Efficient row-by-row processing
- **Accuracy**: High confidence thresholds ensure reliable classifications
- **Scalability**: Tested on 37,000+ row datasets

## Error Handling
- **Missing data**: Graceful handling of null/NaN values
- **Pattern conflicts**: Hierarchical logic resolves ambiguities
- **Low confidence**: Returns "UNKNOWN" rather than uncertain labels
- **File access**: Comprehensive error messages for file I/O issues

## Dependencies
- `pandas`: Excel file processing and data manipulation
- `numpy`: Numerical operations and data handling
- `openpyxl`: Excel file read/write operations
- `tkinter`: GUI file selection interface
- `pathlib`: Cross-platform path handling
- `re`: Regular expression pattern matching

## Integration with Pipeline
- **Input dependency**: Uses consolidated summary files from S2/S3 stages
- **Standalone operation**: Can process any Excel file with required columns
- **Output compatibility**: Maintains all original columns plus new NLP labels
- **Resume capability**: Overwrites existing files (no built-in resume for this stage)

## Configuration Details
The `LabelingConfig` class contains extensive medical terminology libraries:
- **470+ unique protocols** analyzed and incorporated
- **Complex timing patterns** for phase detection
- **Anatomical abbreviations** and medical shorthand
- **Reconstruction indicators** from various imaging systems
- **Multi-language support** for common medical terms

## Quality Assurance
- **Confidence scoring**: Each label includes reliability metric
- **Multi-layer validation**: Cross-references multiple data sources
- **Medical accuracy**: Patterns based on real-world medical imaging data
- **False positive prevention**: High confidence thresholds prevent uncertain labeling