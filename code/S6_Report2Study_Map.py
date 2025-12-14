#!/usr/bin/env python3
"""
S6 Report to Study Mapping

This script consolidates all document reports from S1 processing across multiple
processed directories into a unified long-format JSON and Excel file.

Input: data/processed/*/S1_indexed_metadata/S1_indexingFiles_allDocuments.json
Output: data/consolidated_summaries/consolidated_reports.(json|xlsx)

Each JSON contains objects where keys are file paths and values are report texts.
This script converts them to long format: [{file_path, report_text, source_info}]
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import os


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to the project root
    """
    # Assume script is in 'code' folder
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    return project_root


def find_all_document_json_files(processed_dir: Path) -> List[Path]:
    """
    Find all S1_indexingFiles_allDocuments.json files in processed directories.

    Args:
        processed_dir: Path to the data/processed directory

    Returns:
        List of paths to all document JSON files
    """
    if not processed_dir.exists():
        print(f"Error: Processed directory not found: {processed_dir}")
        return []

    json_files = []

    print("=" * 70)
    print("SEARCHING FOR DOCUMENT JSON FILES")
    print("=" * 70)
    print(f"Scanning: {processed_dir}\n")

    # Look for all subdirectories in processed/
    subdirs = [d for d in processed_dir.iterdir() if d.is_dir()]

    if not subdirs:
        print("Warning: No subdirectories found in processed directory")
        return []

    print(f"Found {len(subdirs)} processed directories:")

    for subdir in sorted(subdirs):
        # Expected path: processed/{name}/S1_indexed_metadata/S1_indexingFiles_allDocuments.json
        json_path = subdir / "S1_indexed_metadata" / "S1_indexingFiles_allDocuments.json"

        if json_path.exists():
            json_files.append(json_path)
            print(f"  ✓ {subdir.name}: {json_path.name}")
        else:
            print(f"  ✗ {subdir.name}: No document JSON found")

    print(f"\nTotal document JSON files found: {len(json_files)}")
    return json_files


def load_and_convert_json_file(json_path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSON file and convert it from {file_path: report_text} format
    to long format: [{file_path, report_text, ...}]

    Args:
        json_path: Path to the JSON file

    Returns:
        List of document records in long format, or empty list on error
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert dictionary format to long format list
        if isinstance(data, dict):
            long_format = []
            for file_path, report_text in data.items():
                long_format.append({
                    'file_path': file_path,
                    'report_text': report_text
                })
            return long_format
        elif isinstance(data, list):
            # Already in list format, return as is
            return data
        else:
            print(f"  Warning: Unexpected data format in {json_path.name}")
            return []

    except json.JSONDecodeError as e:
        print(f"  Error: Invalid JSON in {json_path}: {e}")
        return []
    except Exception as e:
        print(f"  Error reading {json_path}: {e}")
        return []


def add_source_metadata(records: List[Dict[str, Any]], source_path: Path) -> List[Dict[str, Any]]:
    """
    Add source metadata and quality check flags to each record.

    Args:
        records: List of document records
        source_path: Path to the source JSON file

    Returns:
        Updated records with source metadata and need_manual_check flag
    """
    # Extract processed directory name
    # Path structure: .../processed/{name}/S1_indexed_metadata/...
    processed_dir_name = source_path.parent.parent.name

    for record in records:
        record['source_processed_dir'] = processed_dir_name
        record['source_json_file'] = str(source_path)
        
        # Add need_manual_check flag
        report_text = record.get('report_text', '')
        need_check = []
        
        # Check 1: No value or empty
        if not report_text or report_text.strip() == '':
            need_check.append('empty')
        
        # Check 2: Less than 50 characters
        elif len(report_text.strip()) < 50:
            need_check.append('too_short')
        
        # Check 3: Contains ".doc not implemented" message
        if 'Extraction for .doc is not implemented.' in report_text:
            need_check.append('.doc')
        
        # Set the flag
        if need_check:
            record['need_manual_check'] = ', '.join(need_check)
        else:
            record['need_manual_check'] = ''

    return records


def concatenate_all_documents(json_files: List[Path]) -> List[Dict[str, Any]]:
    """
    Load and concatenate all document JSON files in long format.

    Args:
        json_files: List of paths to document JSON files

    Returns:
        Consolidated list of all document records in long format
    """
    all_documents = []

    print("\n" + "=" * 70)
    print("LOADING AND CONCATENATING DOCUMENTS")
    print("=" * 70)

    for i, json_path in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] Processing: {json_path.parent.parent.name}")

        # Load and convert documents from this file
        documents = load_and_convert_json_file(json_path)

        if documents:
            # Add source metadata
            documents = add_source_metadata(documents, json_path)
            all_documents.extend(documents)
            print(f"  Loaded {len(documents)} reports")
        else:
            print(f"  No documents loaded")

    print("\n" + "=" * 70)
    print(f"Total reports concatenated: {len(all_documents)}")
    print("=" * 70)

    return all_documents


def save_consolidated_json(documents: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save consolidated documents to JSON file.

    Args:
        documents: List of document records
        output_path: Path to save the JSON file
    """
    try:
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)

        print(f"\n✓ JSON file saved: {output_path}")
        print(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"  Format: Long format with columns: file_path, report_text, need_manual_check, source_processed_dir, source_json_file")

    except Exception as e:
        print(f"\nError saving JSON file: {e}")
        sys.exit(1)


def save_consolidated_excel(documents: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save consolidated documents to Excel file.

    Args:
        documents: List of document records
        output_path: Path to save the Excel file
    """
    try:
        import pandas as pd
    except ImportError:
        print("\nError: pandas is required. Install it: pip install pandas openpyxl")
        sys.exit(1)

    if not documents:
        print("\nWarning: No documents to save to Excel")
        return

    try:
        # Convert to DataFrame
        print("\nCreating DataFrame...")
        df = pd.DataFrame(documents)

        # Reorder columns for better readability
        column_order = ['file_path', 'report_text', 'need_manual_check', 'source_processed_dir', 'source_json_file']
        # Keep only columns that exist
        column_order = [col for col in column_order if col in df.columns]
        df = df[column_order]

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to Excel
        print("Saving to Excel...")
        df.to_excel(output_path, index=False, engine='openpyxl')

        print(f"\n✓ Excel file saved: {output_path}")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Count records needing manual check
        needs_check = df[df['need_manual_check'] != ''].shape[0]
        if needs_check > 0:
            print(f"  ⚠️  Records needing manual check: {needs_check:,}")

    except Exception as e:
        print(f"\nError saving Excel file: {e}")
        sys.exit(1)


def print_statistics(documents: List[Dict[str, Any]]) -> None:
    """
    Print statistics about the consolidated documents.

    Args:
        documents: List of document records
    """
    if not documents:
        return

    print("\n" + "=" * 70)
    print("CONSOLIDATION STATISTICS")
    print("=" * 70)

    # Count by source directory
    source_counts = {}
    for doc in documents:
        source = doc.get('source_processed_dir', 'Unknown')
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"\nTotal reports: {len(documents):,}")
    print(f"\nReports by source directory:")
    for source in sorted(source_counts.keys()):
        print(f"  {source}: {source_counts[source]:,}")

    # Calculate average report length
    if documents:
        total_length = sum(len(doc.get('report_text', '')) for doc in documents)
        avg_length = total_length / len(documents)
        print(f"\nAverage report length: {avg_length:.0f} characters")

    # Count records needing manual check
    needs_check_counts = {
        'empty': 0,
        'too_short': 0,
        '.doc': 0,
        'total_flagged': 0
    }
    
    for doc in documents:
        check_flag = doc.get('need_manual_check', '')
        if check_flag:
            needs_check_counts['total_flagged'] += 1
            if 'empty' in check_flag:
                needs_check_counts['empty'] += 1
            if 'too_short' in check_flag:
                needs_check_counts['too_short'] += 1
            if '.doc' in check_flag:
                needs_check_counts['.doc'] += 1
    
    print(f"\n" + "-" * 70)
    print("QUALITY CHECK RESULTS")
    print("-" * 70)
    print(f"Total records needing manual check: {needs_check_counts['total_flagged']:,} ({needs_check_counts['total_flagged']/len(documents)*100:.1f}%)")
    print(f"\nBreakdown by issue:")
    print(f"  Empty reports: {needs_check_counts['empty']:,}")
    print(f"  Too short (<50 chars): {needs_check_counts['too_short']:,}")
    print(f"  .doc extraction not implemented: {needs_check_counts['.doc']:,}")
    print(f"  Clean reports: {len(documents) - needs_check_counts['total_flagged']:,}")

    print("=" * 70)


def main():
    """Main function to orchestrate the consolidation process."""
    print("=" * 70)
    print("S6 REPORT TO STUDY MAPPING - CONSOLIDATION")
    print("=" * 70)

    # Get project root and paths
    project_root = get_project_root()
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "data" / "consolidated_summaries"

    print(f"\nProject root: {project_root}")
    print(f"Processed directory: {processed_dir}")
    print(f"Output directory: {output_dir}\n")

    # Check if processed directory exists
    if not processed_dir.exists():
        print(f"Error: Processed directory not found: {processed_dir}")
        print("\nPlease ensure you have run the S1 indexing step first.")
        sys.exit(1)

    # Find all document JSON files
    json_files = find_all_document_json_files(processed_dir)

    if not json_files:
        print("\nNo document JSON files found!")
        print("Please ensure S1 processing has been completed.")
        sys.exit(1)

    # Load and concatenate all documents
    all_documents = concatenate_all_documents(json_files)

    if not all_documents:
        print("\nNo documents loaded!")
        sys.exit(1)

    # Define output paths
    json_output = output_dir / "consolidated_reports.json"
    excel_output = output_dir / "consolidated_reports.xlsx"

    # Save consolidated data
    print("\n" + "=" * 70)
    print("SAVING CONSOLIDATED DATA")
    print("=" * 70)

    save_consolidated_json(all_documents, json_output)
    save_consolidated_excel(all_documents, excel_output)

    # Print statistics
    print_statistics(all_documents)

    print("\n" + "=" * 70)
    print("CONSOLIDATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  JSON: {json_output}")
    print(f"  Excel: {excel_output}")
    print("\nData format: Long format with columns:")
    print("  - file_path: Original file path")
    print("  - report_text: Full report content")
    print("  - need_manual_check: Quality flag (empty|too_short|.doc)")
    print("  - source_processed_dir: Source hospital/directory")
    print("  - source_json_file: Source JSON file path")
    print("\nQuality check flags:")
    print("  - empty: Report text is empty or null")
    print("  - too_short: Report has less than 50 characters")
    print("  - .doc: Contains 'Extraction for .doc is not implemented.'")
    print("=" * 70)


if __name__ == "__main__":
    main()
