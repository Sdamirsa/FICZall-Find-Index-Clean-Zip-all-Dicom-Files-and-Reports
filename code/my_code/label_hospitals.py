#!/usr/bin/env python3
"""
Hospital Labeling Script

This script maps source file paths to hospital names in an Excel file.
It creates a mapping JSON file on first run, then applies the mappings on subsequent runs.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Map source_folder column values to hospital names in an Excel file.'
    )
    parser.add_argument(
        '--excel',
        type=str,
        help='Path to the Excel file'
    )
    return parser.parse_args()


def get_excel_path(args_excel: str = None) -> Path:
    """
    Get the Excel file path from arguments or user input.

    Args:
        args_excel: Excel path from command line arguments

    Returns:
        Path object for the Excel file

    Raises:
        SystemExit: If the file doesn't exist or is invalid
    """
    if args_excel:
        excel_path = Path(args_excel)
    else:
        excel_input = input("Enter the path to the Excel file: ").strip()
        excel_path = Path(excel_input)

    if not excel_path.exists():
        print(f"Error: Excel file not found: {excel_path}")
        sys.exit(1)

    if excel_path.suffix.lower() not in ['.xlsx', '.xls']:
        print(f"Error: File must be an Excel file (.xlsx or .xls): {excel_path}")
        sys.exit(1)

    return excel_path


def load_excel(excel_path: Path):
    """
    Load the Excel file using pandas.

    Args:
        excel_path: Path to the Excel file

    Returns:
        DataFrame containing the Excel data
    """
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required. Please install it: pip install pandas openpyxl")
        sys.exit(1)

    try:
        df = pd.read_excel(excel_path)
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        sys.exit(1)


def extract_unique_sources(df) -> Set[str]:
    """
    Extract unique values from the source_folder column.

    Args:
        df: DataFrame containing the Excel data

    Returns:
        Set of unique source file values

    Raises:
        SystemExit: If source_folder column doesn't exist
    """
    if 'source_folder' not in df.columns:
        print("Error: Excel file must contain a 'source_folder' column")
        print(f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)

    # Get unique values, excluding NaN/None
    unique_sources = set()
    for value in df['source_folder'].dropna().unique():
        if value and str(value).strip():
            unique_sources.add(str(value))

    if not unique_sources:
        print("Error: No valid values found in 'source_folder' column")
        sys.exit(1)

    return unique_sources


def get_mapping_path(excel_path: Path) -> Path:
    """
    Get the path for the mapping JSON file.

    Args:
        excel_path: Path to the Excel file

    Returns:
        Path to the mapping JSON file (same directory as Excel)
    """
    return excel_path.parent / "label_hospitals_map.json"


def create_mapping_file(mapping_path: Path, unique_sources: Set[str]) -> None:
    """
    Create a mapping JSON file with empty values for user to fill.

    Args:
        mapping_path: Path where the JSON file should be created
        unique_sources: Set of unique source file values
    """
    # Create mapping dictionary with empty values
    mapping = {source: "" for source in sorted(unique_sources)}

    try:
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        print(f"\nMapping file created: {mapping_path}")
        print(f"Found {len(unique_sources)} unique source file values.")
    except Exception as e:
        print(f"Error creating mapping file: {e}")
        sys.exit(1)


def load_mapping_file(mapping_path: Path) -> Dict[str, str]:
    """
    Load the mapping JSON file.

    Args:
        mapping_path: Path to the mapping JSON file

    Returns:
        Dictionary mapping source files to hospital names
    """
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        return mapping
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        sys.exit(1)


def validate_mapping(mapping: Dict[str, str], unique_sources: Set[str]) -> bool:
    """
    Validate that all source files have hospital mappings.

    Args:
        mapping: Dictionary mapping source files to hospital names
        unique_sources: Set of unique source file values from Excel

    Returns:
        True if all mappings are complete, False otherwise
    """
    # Check if all sources are in the mapping
    missing_sources = unique_sources - set(mapping.keys())
    if missing_sources:
        print("\nError: The following source files are not in the mapping file:")
        for source in sorted(missing_sources):
            print(f"  - {source}")
        return False

    # Check if any mappings are empty
    empty_mappings = [source for source, hospital in mapping.items()
                     if source in unique_sources and (not hospital or not str(hospital).strip())]

    if empty_mappings:
        print("\nError: The following source files have empty hospital mappings:")
        for source in sorted(empty_mappings):
            print(f"  - {source}")
        return False

    return True


def apply_hospital_mapping(df, mapping: Dict[str, str]) -> None:
    """
    Apply hospital mapping to the DataFrame by adding a Hospital column.

    Args:
        df: DataFrame containing the Excel data
        mapping: Dictionary mapping source files to hospital names
    """
    import pandas as pd

    # Create Hospital column by mapping source_folder values
    df['Hospital'] = df['source_folder'].apply(
        lambda x: mapping.get(str(x), "") if pd.notna(x) else ""
    )


def save_excel(df, excel_path: Path) -> None:
    """
    Save the updated DataFrame back to the Excel file.

    Args:
        df: DataFrame to save
        excel_path: Path to save the Excel file
    """
    try:
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"\nSuccess! Hospital column added to: {excel_path}")
        print(f"Total rows updated: {len(df)}")

        # Show summary of mappings
        import pandas as pd
        hospital_counts = df['Hospital'].value_counts()
        print("\nHospital distribution:")
        for hospital, count in hospital_counts.items():
            if hospital:  # Skip empty values
                print(f"  {hospital}: {count} rows")
    except Exception as e:
        print(f"Error saving Excel file: {e}")
        sys.exit(1)


def main():
    """Main function to orchestrate the hospital labeling process."""
    print("=" * 60)
    print("Hospital Labeling Script")
    print("=" * 60)

    # Parse arguments and get Excel path
    args = parse_arguments()
    excel_path = get_excel_path(args.excel)
    print(f"\nProcessing Excel file: {excel_path}")

    # Load Excel file
    df = load_excel(excel_path)
    print(f"Loaded {len(df)} rows")

    # Extract unique source files
    unique_sources = extract_unique_sources(df)
    print(f"Found {len(unique_sources)} unique source file values")

    # Get mapping file path
    mapping_path = get_mapping_path(excel_path)

    # Check if mapping file exists
    if not mapping_path.exists():
        # First run: create mapping file
        create_mapping_file(mapping_path, unique_sources)
        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("1. Open the mapping file and fill in the hospital names")
        print(f"   File: {mapping_path}")
        print("2. Run this script again to apply the mappings")
        print("=" * 60)
        return

    # Load existing mapping file
    print(f"\nMapping file found: {mapping_path}")
    mapping = load_mapping_file(mapping_path)

    # Validate that all mappings are complete
    if not validate_mapping(mapping, unique_sources):
        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("1. Update the mapping file with missing/empty values")
        print(f"   File: {mapping_path}")
        print("2. Run this script again to apply the mappings")
        print("=" * 60)
        return

    # Check if Hospital column already exists
    if 'Hospital' in df.columns:
        response = input("\nWarning: 'Hospital' column already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return

    # Apply mappings and save
    print("\nApplying hospital mappings...")
    apply_hospital_mapping(df, mapping)
    save_excel(df, excel_path)
    print("\n" + "=" * 60)
    print("Process completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
