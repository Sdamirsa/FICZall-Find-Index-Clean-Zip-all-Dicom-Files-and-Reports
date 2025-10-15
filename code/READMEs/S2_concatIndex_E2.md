# S2_concatIndex_E2.py

## Purpose
Processes JSONL files from S1 to extract patient summaries with CT object counts.

## Code Logic
- Scans folder for JSONL files (skips configured files)
- Reads first line of each JSONL file for patient data
- Counts total lines in file as CT object count
- Extracts patient demographics and study information
- Generates both JSON and Excel outputs with statistics

## Inputs
- Folder containing JSONL files from S1 output
- Optional output file paths

## Outputs
- "S2_ALL_PATIENTS_inCT.json": Patient summaries with metadata
- "S2_ALL_PATIENTS_inCT.xlsx": Excel spreadsheet with same data
- Statistics and processing summary

## User Changeable Settings
- `SKIP_FILES`: Files to ignore during processing (environment variable)
- Default skip list includes system files and previous outputs
- Output file names can be specified via command line
- Fields extracted: patient_id, patient_name, study_date, patient_age, study_modality, study_body_part, institution_name