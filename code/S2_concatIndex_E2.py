import os
import json
import argparse
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Warning: pandas not installed. Excel output will be unavailable.")
    print("Install with: pip install pandas openpyxl")
    pd = None

# =============================================================================
# CONFIGURATION - Files to skip during processing
# =============================================================================
# Environment variable for files to skip (comma-separated list)
# Developers can modify this list or set the SKIP_FILES environment variable
DEFAULT_SKIP_FILES = [
    "ALL_PATIENTS_inCT.json",
    "ALL_PATIENTS_inCT.jsonl", 
    "summary.jsonl",
    "CONCAT_ALL.jsonl",  # Fixed missing comma
    "processed_files.jsonl",
    "temp.jsonl",
    "backup.jsonl",
    "test.jsonl",
    "S2_ALL_PATIENTS_inCT.json",  # Added S2 output file
    "S2_ALL_PATIENTS_inCT.xlsx"   # Added S2 output file
]

# Get skip files from environment variable or use default
SKIP_FILES_ENV = os.getenv('SKIP_FILES', ','.join(DEFAULT_SKIP_FILES))
SKIP_FILES = [filename.strip() for filename in SKIP_FILES_ENV.split(',') if filename.strip()]

print(f"Files configured to skip: {SKIP_FILES}")
print("-" * 50)

# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def S2_process_jsonl_files(folder_path: str, output_json: str, output_excel: Optional[str] = None) -> int:
    """Process JSONL files and extract patient data with CT object counts.
    
    Args:
        folder_path: Directory containing JSONL files
        output_json: Path for output JSON file
        output_excel: Optional path for output Excel file
        
    Returns:
        Number of records processed
    """
    
    # Fields to extract
    # Fields to extract from DICOM data for patient summaries
    fields = [
        "patient_id",           # NEW: Patient ID as requested
        "patient_name", 
        "study_date", 
        "patient_age", 
        "study_modality", 
        "study_body_part", 
        "institution_name"
    ]

    # Container for collected data
    extracted_data = []
    skipped_files = []
    processed_files = []

    # Loop through all .jsonl files
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            # Check if file should be skipped
            if filename in SKIP_FILES or filename == os.path.basename(output_json):
                skipped_files.append(filename)
                continue
                
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # More memory-efficient: count lines without loading all into memory
                    total_lines = sum(1 for _ in f)
                    
                    # Reset to beginning and read first line only
                    f.seek(0)
                    first_line = f.readline()
                    
                    # Process the first line for patient data
                    if first_line:
                        record = json.loads(first_line)

                        # Extract desired fields, filling missing with None
                        filtered_record = {key: record.get(key, None) for key in fields}
                        filtered_record["source_file"] = filename  # Add source filename
                        filtered_record["ct_objects_count"] = total_lines  # Add line count
                        
                        # Extract folder path from the file_path field
                        if "file_path" in record and record["file_path"]:
                            # Get the directory part of the file path
                            dicom_folder_path = os.path.dirname(record["file_path"])
                            filtered_record["dicom_folder_path"] = dicom_folder_path
                        else:
                            filtered_record["dicom_folder_path"] = None
                        
                        extracted_data.append(filtered_record)
                        processed_files.append(filename)
                    else:
                        # Handle empty files
                        print(f"Warning: Empty file {filename}")
                        filtered_record = {key: None for key in fields}
                        filtered_record["source_file"] = filename
                        filtered_record["ct_objects_count"] = 0
                        filtered_record["dicom_folder_path"] = None
                        extracted_data.append(filtered_record)
                        processed_files.append(filename)
                        
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                # Add error record with zero count and no folder path
                filtered_record = {key: None for key in fields}
                filtered_record["source_file"] = filename
                filtered_record["ct_objects_count"] = 0
                filtered_record["dicom_folder_path"] = None
                extracted_data.append(filtered_record)
                processed_files.append(filename)

    # Write to JSON (pretty formatted)
    with open(output_json, 'w', encoding='utf-8') as out_json:
        json.dump(extracted_data, out_json, ensure_ascii=False, indent=2)

    # Write to Excel if pandas is available and output_excel is specified
    if extracted_data and output_excel:
        if pd:
            df = pd.DataFrame(extracted_data)
            try:
                df.to_excel(output_excel, index=False, engine='openpyxl')
            except ImportError:
                print("Warning: openpyxl not installed. Skipping Excel output.")
                print("Install with: pip install openpyxl")
        else:
            print("Warning: pandas not available. Skipping Excel output.")
        print(f"Saved {len(extracted_data)} records to:")
        print(f"  JSON: {output_json}")
        if output_excel and pd:
            print(f"  Excel: {output_excel}")
        print(f"Added 'ct_objects_count' column showing number of CT objects per file")
        print(f"Added 'dicom_folder_path' column showing the folder containing DICOM files")
        
        # Display file processing statistics
        print(f"\nFile Processing Summary:")
        print(f"Total JSONL files found: {len(processed_files) + len(skipped_files)}")
        print(f"Files processed: {len(processed_files)}")
        print(f"Files skipped: {len(skipped_files)}")
        
        if skipped_files:
            print(f"Skipped files: {', '.join(skipped_files)}")
        
        # Display summary statistics
        print(f"\nData Summary:")
        print(f"Total patients: {len(extracted_data)}")
        if pd and 'df' in locals():
            print(f"Total CT objects: {df['ct_objects_count'].sum()}")
            print(f"Average CT objects per patient: {df['ct_objects_count'].mean():.1f}")
        else:
            total_ct = sum(rec.get('ct_objects_count', 0) for rec in extracted_data)
            avg_ct = total_ct / len(extracted_data) if extracted_data else 0
            print(f"Total CT objects: {total_ct}")
            print(f"Average CT objects per patient: {avg_ct:.1f}")
        
        # Display column information
        print(f"\nColumns in output files:")
        if pd and 'df' in locals():
            for i, col in enumerate(df.columns, 1):
                print(f"  {i}. {col}")
        elif extracted_data:
            for i, col in enumerate(extracted_data[0].keys(), 1):
                print(f"  {i}. {col}")
        
    else:
        print("No valid data extracted.")
        if skipped_files:
            print(f"Files skipped: {', '.join(skipped_files)}")
    
    return len(extracted_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process JSONL files containing CT scan data and extract patient information with object counts",
        epilog="""
Environment Variables:
  SKIP_FILES    Comma-separated list of filenames to skip during processing
                (default: ALL_PATIENTS_inCT.json,ALL_PATIENTS_inCT.jsonl,summary.jsonl,processed_files.jsonl,temp.jsonl,backup.jsonl,test.jsonl)
        
Examples:
  python script.py
  python script.py --folder_path "C:\\Data\\CT_Files"
  
  # Set custom skip files via environment variable
  set SKIP_FILES=output.jsonl,temp.jsonl,backup.jsonl
  python script.py --folder_path "C:\\Data\\CT_Files"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--folder_path", 
                       help="Folder path containing JSONL files to process")
    parser.add_argument("--output_json", 
                       help="Output JSON file path (default: ALL_PATIENTS_inCT.json in folder_path)")
    parser.add_argument("--output_excel", 
                       help="Output Excel file path (default: ALL_PATIENTS_inCT.xlsx in folder_path)")
    parser.add_argument("--overwrite", 
                       action="store_true", 
                       help="Whether to overwrite existing output files")
    parser.add_argument("--show-skip-files", 
                       action="store_true", 
                       help="Show the list of files configured to be skipped and exit")
    
    args = parser.parse_args()

    # Show skip files and exit if requested
    if args.show_skip_files:
        print("Files configured to be skipped:")
        for i, filename in enumerate(SKIP_FILES, 1):
            print(f"  {i}. {filename}")
        print(f"\nTo modify this list, set the SKIP_FILES environment variable:")
        print(f"Example: set SKIP_FILES=file1.jsonl,file2.jsonl,file3.jsonl")
        exit(0)

    # Get folder path - either from args or ask user
    folder_path = args.folder_path
    if not folder_path:
        print("Please provide the folder path containing JSONL files.")
        folder_path = input("Enter folder path: ").strip()
        
        # Remove quotes if user copied path with quotes
        if folder_path.startswith('"') and folder_path.endswith('"'):
            folder_path = folder_path[1:-1]
        if folder_path.startswith("'") and folder_path.endswith("'"):
            folder_path = folder_path[1:-1]
        
        # Keep asking until valid directory is provided
        while not os.path.isdir(folder_path):
            print(f"Directory '{folder_path}' does not exist.")
            folder_path = input("Enter valid folder path containing JSONL files: ").strip()
            
            # Remove quotes if user copied path with quotes
            if folder_path.startswith('"') and folder_path.endswith('"'):
                folder_path = folder_path[1:-1]
            if folder_path.startswith("'") and folder_path.endswith("'"):
                folder_path = folder_path[1:-1]
    
    # Validate folder path exists
    if not os.path.isdir(folder_path):
        print(f"Error: Directory '{folder_path}' does not exist.")
        exit(1)

    # Set output file paths
    output_json = args.output_json
    if not output_json:
        output_json = os.path.join(folder_path, "S2_ALL_PATIENTS_inCT.json")
    
    output_excel = args.output_excel
    if not output_excel:
        output_excel = os.path.join(folder_path, "S2_ALL_PATIENTS_inCT.xlsx")

    # Check if output files exist and handle overwrite
    if not args.overwrite:
        if os.path.exists(output_json):
            response = input(f"Output file {output_json} already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                exit(0)
        
        if os.path.exists(output_excel):
            response = input(f"Output file {output_excel} already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                exit(0)

    # Create output directory if it doesn't exist
    output_json_dir = os.path.dirname(output_json)
    if output_json_dir:
        os.makedirs(output_json_dir, exist_ok=True)
    
    if output_excel:
        output_excel_dir = os.path.dirname(output_excel)
        if output_excel_dir:
            os.makedirs(output_excel_dir, exist_ok=True)

    print(f"Processing JSONL files in: {folder_path}")
    print(f"Output JSON: {output_json}")
    print(f"Output Excel: {output_excel}")
    print("-" * 50)

    # Process the files
    num_processed = S2_process_jsonl_files(folder_path, output_json, output_excel)
