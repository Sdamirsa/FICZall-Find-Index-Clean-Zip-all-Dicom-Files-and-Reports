#!/usr/bin/env python3
"""
S7_prepare_manual_extraction.py - Prepare files for manual text extraction

This script processes consolidated reports Excel files and prepares a package
for manual text extraction by team members.

Features:
- Filters rows where 'need_manual_check' column has a value (non-empty)
- Copies the original document files to a timestamped folder
- Creates a filtered Excel file with only the rows needing manual check
- Generates a README.txt with instructions for team members

Usage:
    python S7_prepare_manual_extraction.py
    python S7_prepare_manual_extraction.py --input [EXCEL_PATH]
    python S7_prepare_manual_extraction.py --source-drive E --target-drive D

Author: FICZall Pipeline
Version: 1.0
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    script_dir = Path(__file__).parent
    return script_dir.parent


def change_drive_letter(file_path: str, target_drive: str) -> str:
    """
    Change the drive letter of a Windows file path.

    Args:
        file_path: Original file path (e.g., 'E:/Data/file.pdf')
        target_drive: Target drive letter without colon (e.g., 'D')

    Returns:
        Modified file path with new drive letter
    """
    if not file_path:
        return file_path

    # Handle both forward and backward slashes
    path_str = str(file_path)

    # Check if it's a Windows path with drive letter
    if len(path_str) >= 2 and path_str[1] == ':':
        return f"{target_drive}:{path_str[2:]}"

    return path_str


def prepare_manual_extraction_package(
    input_excel: Path,
    output_base_dir: Path,
    source_drive: str = None,
    target_drive: str = 'D',
    file_path_column: str = 'file_path',
    check_column: str = 'need_manual_check'
) -> Tuple[bool, str, Optional[Path]]:
    """
    Prepare a package for manual text extraction.

    Args:
        input_excel: Path to the input Excel file
        output_base_dir: Base directory for output
        source_drive: Original drive letter to replace (auto-detect if None)
        target_drive: Target drive letter for file access
        file_path_column: Name of the column containing file paths
        check_column: Name of the column indicating manual check needed

    Returns:
        Tuple of (success, message, output_folder_path)
    """
    try:
        import pandas as pd
    except ImportError:
        return False, "pandas is required. Install with: pip install pandas openpyxl", None

    # Validate input file
    if not input_excel.exists():
        return False, f"Input file not found: {input_excel}", None

    logger.info(f"Reading Excel file: {input_excel}")

    # Read the Excel file
    try:
        df = pd.read_excel(input_excel)
    except Exception as e:
        return False, f"Failed to read Excel file: {e}", None

    logger.info(f"Total rows in file: {len(df)}")

    # Check if required columns exist
    if check_column not in df.columns:
        available_cols = ', '.join(df.columns.tolist())
        return False, f"Column '{check_column}' not found. Available columns: {available_cols}", None

    if file_path_column not in df.columns:
        available_cols = ', '.join(df.columns.tolist())
        return False, f"Column '{file_path_column}' not found. Available columns: {available_cols}", None

    # Filter rows where need_manual_check is NOT empty
    # Handle various empty values: NaN, None, empty string, whitespace
    df_filtered = df[df[check_column].notna() & (df[check_column].astype(str).str.strip() != '')]

    if len(df_filtered) == 0:
        return False, f"No rows found with non-empty '{check_column}' values", None

    logger.info(f"Rows needing manual check: {len(df_filtered)}")

    # Create timestamped output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = output_base_dir / f"manual_extraction_{timestamp}"
    files_folder = output_folder / "files_to_extract"

    try:
        output_folder.mkdir(parents=True, exist_ok=True)
        files_folder.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Failed to create output folder: {e}", None

    logger.info(f"Created output folder: {output_folder}")

    # Process each row and copy files
    copied_files = []
    failed_files = []
    file_mapping = []  # Track original path -> new filename mapping

    for idx, row in df_filtered.iterrows():
        original_path = row[file_path_column]

        if pd.isna(original_path) or not str(original_path).strip():
            failed_files.append({
                'row_index': idx,
                'original_path': str(original_path),
                'reason': 'Empty file path'
            })
            continue

        # Change drive letter
        modified_path = change_drive_letter(str(original_path), target_drive)
        source_file = Path(modified_path)

        if not source_file.exists():
            # Try with original path as fallback
            source_file = Path(str(original_path))
            if not source_file.exists():
                failed_files.append({
                    'row_index': idx,
                    'original_path': str(original_path),
                    'modified_path': modified_path,
                    'reason': 'File not found'
                })
                continue

        # Create a unique filename with row index prefix for easy matching
        original_name = source_file.name
        # Clean the filename and add row index for easy reference
        new_filename = f"ROW_{idx:04d}_{original_name}"
        dest_file = files_folder / new_filename

        try:
            shutil.copy2(source_file, dest_file)
            copied_files.append({
                'row_index': idx,
                'original_path': str(original_path),
                'new_filename': new_filename
            })
            file_mapping.append({
                'row_index': idx,
                'original_filename': original_name,
                'new_filename': new_filename,
                'original_path': str(original_path)
            })
            logger.info(f"Copied: {original_name} -> {new_filename}")
        except Exception as e:
            failed_files.append({
                'row_index': idx,
                'original_path': str(original_path),
                'reason': str(e)
            })
            logger.warning(f"Failed to copy {original_path}: {e}")

    # Add new_filename column to the filtered DataFrame for reference
    df_filtered = df_filtered.copy()
    df_filtered['_copied_filename'] = ''
    df_filtered['_extraction_status'] = ''
    df_filtered['_extracted_text'] = ''

    for mapping in file_mapping:
        df_filtered.loc[mapping['row_index'], '_copied_filename'] = mapping['new_filename']

    # Save filtered Excel
    output_excel = output_folder / f"reports_for_extraction_{timestamp}.xlsx"
    try:
        df_filtered.to_excel(output_excel, index=False)
        logger.info(f"Saved filtered Excel: {output_excel}")
    except Exception as e:
        return False, f"Failed to save Excel file: {e}", None

    # Create README.txt
    readme_content = create_readme_content(
        timestamp=timestamp,
        total_files=len(df_filtered),
        copied_files=len(copied_files),
        failed_files=failed_files,
        file_path_column=file_path_column,
        check_column=check_column,
        file_mapping=file_mapping
    )

    readme_path = output_folder / "README.txt"
    try:
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        logger.info(f"Created README: {readme_path}")
    except Exception as e:
        logger.warning(f"Failed to create README: {e}")

    # Create file mapping JSON for programmatic access
    try:
        import json
        mapping_path = output_folder / "file_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'total_rows': len(df_filtered),
                'files_copied': copied_files,
                'files_failed': failed_files
            }, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to create mapping JSON: {e}")

    # Summary
    summary = f"""
Package created successfully!
=============================
Output folder: {output_folder}
Total rows needing manual check: {len(df_filtered)}
Files copied successfully: {len(copied_files)}
Files failed to copy: {len(failed_files)}

Contents:
- README.txt: Instructions for team members
- reports_for_extraction_{timestamp}.xlsx: Filtered data with extraction columns
- files_to_extract/: Folder containing all document files
- file_mapping.json: Programmatic mapping of files to rows
"""

    if failed_files:
        summary += f"\nWARNING: {len(failed_files)} files could not be copied. Check README.txt for details."

    return True, summary, output_folder


def create_readme_content(
    timestamp: str,
    total_files: int,
    copied_files: int,
    failed_files: List[dict],
    file_path_column: str,
    check_column: str,
    file_mapping: List[dict]
) -> str:
    """Create README content with instructions for team members."""

    readme = f"""
================================================================================
MANUAL TEXT EXTRACTION PACKAGE
================================================================================
Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Package ID: {timestamp}

================================================================================
OVERVIEW
================================================================================
This folder contains documents that require MANUAL text extraction.
These files could not be automatically processed (corrupted, scanned PDFs, etc.)

Total files requiring extraction: {total_files}
Files included in this package: {copied_files}
Files that could not be copied: {len(failed_files)}

================================================================================
INSTRUCTIONS FOR TEAM MEMBERS
================================================================================

1. OPEN THE EXCEL FILE:
   - File: reports_for_extraction_{timestamp}.xlsx
   - This contains all rows that need manual text extraction

2. FOR EACH ROW IN THE EXCEL:
   a. Find the corresponding file in the 'files_to_extract' folder
      - Look at the '_copied_filename' column to find the exact filename
      - Files are named: ROW_XXXX_originalname.ext (XXXX = row number)

   b. Open the document file and manually extract the text
      - For PDFs: Open and copy the text content
      - For images: Use OCR or type the visible text
      - For Word docs: Open and copy the text content

   c. Paste the extracted text into the '_extracted_text' column

   d. Update the '_extraction_status' column:
      - "DONE" = Text extracted successfully
      - "EMPTY" = Document has no extractable text
      - "UNREADABLE" = Cannot read/understand the content
      - "PASSWORD" = Document is password protected
      - "CORRUPT" = File is corrupted and cannot be opened

3. SAVE THE EXCEL FILE when you're done with your assigned rows

4. RETURN THE COMPLETED EXCEL FILE to the data processing team

================================================================================
FILE MAPPING REFERENCE
================================================================================
The files are renamed with a ROW_ prefix to help you match them to Excel rows:

Format: ROW_XXXX_originalfilename.ext
        ^^^^
        This number matches the row in the Excel file

Example:
  - ROW_0001_report.pdf -> This is the file for row 1 in Excel
  - ROW_0025_scan.jpg   -> This is the file for row 25 in Excel

"""

    # Add file mapping table
    if file_mapping:
        readme += """
--------------------------------------------------------------------------------
FILES IN THIS PACKAGE:
--------------------------------------------------------------------------------
"""
        for item in file_mapping[:50]:  # Limit to first 50 for readability
            readme += f"  Row {item['row_index']:4d}: {item['new_filename']}\n"

        if len(file_mapping) > 50:
            readme += f"\n  ... and {len(file_mapping) - 50} more files (see file_mapping.json for complete list)\n"

    # Add failed files section if any
    if failed_files:
        readme += f"""

================================================================================
FILES THAT COULD NOT BE COPIED ({len(failed_files)} files)
================================================================================
These files were not found or could not be accessed. You may need to locate
them manually or mark them as unavailable in the Excel file.

"""
        for item in failed_files:
            readme += f"  Row {item.get('row_index', '?')}: {item.get('original_path', 'Unknown')}\n"
            readme += f"      Reason: {item.get('reason', 'Unknown error')}\n\n"

    readme += """
================================================================================
TIPS FOR TEXT EXTRACTION
================================================================================
- For scanned PDFs: Use Adobe Acrobat's OCR feature or online OCR tools
- For Persian/Arabic text: Make sure your text editor supports RTL languages
- For images: Use Google Lens, Adobe Acrobat, or similar OCR tools
- If a file is corrupted: Try opening with different applications
- Save your work frequently!

================================================================================
QUESTIONS OR ISSUES?
================================================================================
Contact the data processing team if you encounter any problems or have questions
about specific files.

Thank you for your help with manual extraction!
================================================================================
"""

    return readme


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Prepare files for manual text extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python S7_prepare_manual_extraction.py
  python S7_prepare_manual_extraction.py --input path/to/file.xlsx
  python S7_prepare_manual_extraction.py --source-drive E --target-drive D
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input Excel file (default: consolidated_reports_v1.xlsx)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory (default: data/consolidated_summaries/reports_for_manual_text_extract)'
    )

    parser.add_argument(
        '--target-drive', '-d',
        type=str,
        default='D',
        help='Target drive letter for file paths (default: D)'
    )

    parser.add_argument(
        '--file-column',
        type=str,
        default='file_path',
        help='Name of column containing file paths (default: file_path)'
    )

    parser.add_argument(
        '--check-column',
        type=str,
        default='need_manual_check',
        help='Name of column indicating manual check needed (default: need_manual_check)'
    )

    args = parser.parse_args()

    # Get project root
    project_root = get_project_root()

    # Set default input path
    if args.input:
        input_excel = Path(args.input)
    else:
        input_excel = project_root / "data" / "consolidated_summaries" / "consolidated_reports_v1.xlsx"

    # Set default output path
    if args.output:
        output_base_dir = Path(args.output)
    else:
        output_base_dir = project_root / "data" / "consolidated_summaries" / "reports_for_manual_text_extract"

    print("=" * 70)
    print("MANUAL EXTRACTION PACKAGE PREPARATION")
    print("=" * 70)
    print(f"Input Excel: {input_excel}")
    print(f"Output Directory: {output_base_dir}")
    print(f"Target Drive: {args.target_drive}")
    print(f"File Path Column: {args.file_column}")
    print(f"Check Column: {args.check_column}")
    print("=" * 70)

    # Run the preparation
    success, message, output_folder = prepare_manual_extraction_package(
        input_excel=input_excel,
        output_base_dir=output_base_dir,
        target_drive=args.target_drive,
        file_path_column=args.file_column,
        check_column=args.check_column
    )

    print(message)

    if success and output_folder:
        print(f"\nPackage location: {output_folder}")
        print("\nYou can now share this folder with your team for manual text extraction.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
