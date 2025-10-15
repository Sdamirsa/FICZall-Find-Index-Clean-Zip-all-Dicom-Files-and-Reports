# S1_indexingFiles_E2.py

## Purpose
Indexes and extracts metadata from DICOM files and documents in a directory structure.

## Code Logic
- Scans directory structure and builds folder/file inventory
- Extracts DICOM metadata using pydicom (patient info, study details, imaging parameters)
- Processes documents (PDF, DOC, DOCX) and extracts text content
- Groups DICOM files by study (date, patient ID, institution)
- Uses async processing for concurrent file handling

## Inputs
- Root directory to scan
- Output directory for results
- Concurrency settings (auto-detected from CPU count)

## Outputs
- "S1_indexingFiles_allFolders.json": Directory structure
- "S1_indexingFiles_allFiles.json": File inventory with metadata
- "S1_indexingFiles_allDocuments.json": Extracted document text
- JSONL files: "{study_date}_._{patient_id}_._{center}_._Study.jsonl"

## User Changeable Settings
- `concurrency`: Override CPU-based worker count
- `batch_size`: DICOM processing batch size (default: 50)
- `write_interval`: Progress save frequency (default: 20)
- Supported file extensions in code (DCM, PDF, DOC, DOCX, PNG, JPG)