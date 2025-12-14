#!/usr/bin/env python3
"""
S5_NLPseriesLabel.py - DICOM Pipeline Stage 5: NLP-Based Series Labeling

This script processes consolidated DICOM metadata Excel files and applies
multi-layer NLP logic to extract and standardize medical imaging labels.

Features:
- Processes Excel files from data/consolidated_summaries directory
- Interactive file selection interface
- Multi-layer logic for robust label extraction
- Comprehensive medical terminology recognition
- Outputs labeled data with confidence scores

Author: SAA Safavi-Naini
Version: 1.0
"""

import pandas as pd
import numpy as np
import os
import sys
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import tkinter as tk
from tkinter import filedialog, messagebox

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LabelingConfig:
    """Configuration for NLP labeling logic"""
    
    # Image origin patterns
    ORIGINAL_PATTERNS = ["ORIGINAL"]
    DERIVED_PATTERNS = ["DERIVED"]
    
    # Reconstruction type patterns
    RECONSTRUCTION_PATTERNS = {
        'MPR': ['MPR'],
        'MIP': ['MIP'],
        'LOCALIZER': ['LOCALIZER'],
        'VOLUME': ['VOLUME'],
        'HELICAL': ['HELICAL'],
        'REFORMATTED': ['REFORMATTED'],
        'AVERAGE': ['AVERAGE']
    }
    
    # Multi-phase study indicators (for study-level contrast detection)
    MULTI_PHASE_INDICATORS = ['3PHASE', '3PHASIC', '3P', '3PH', '3Ph']
    
    # Enhanced contrast patterns for studies
    CONTRAST_YES_PATTERNS = [
        'WITH', 'CONTRAST', 'DYNAMIC', 'ENHANCED', '+-C', '+C', '+CM', 'WOWITH', 
        'WITHOUT_WITH', 'WW', 'WITHWO', 'WITH_WITHOUT', 'AP+', 'ABP+CM', 'T-AP+',
        'C+', '+&-C', 'AP+&-C'
    ] + MULTI_PHASE_INDICATORS
    
    CONTRAST_NO_PATTERNS = ['WITHOUT', 'WO', '-C', 'PLAIN', 'NATIVE', 'CHEST-CM']
    
    # Enhanced series contrast patterns
    CONTRAST_SERIES_YES_PATTERNS = [
        'CE', 'CONTRAST', 'ENHANCED', 'POST', 'CORONAL C', 'COR C', '+C', '+CM',
        'WITH', 'ENHANCEMENT'
    ]
    
    CONTRAST_SERIES_NO_PATTERNS = [
        'PRE', 'PLAIN', 'NATIVE', 'BASELINE', 'W/O', 'WO', '-C', 'WITHOUT'
    ]
    
    # Enhanced phase patterns (for individual series)
    PHASE_PATTERNS = {
        'NON_CONTRAST': [
            'PRE', 'PLAIN', 'NATIVE', 'BASELINE', 'W/O', 'WO', '-C', 'WITHOUT',
            'PREMONITORING', 'CHEST WO', 'ABD WO', 'NECK -', 'ABDPLV -', 'AP -C'
        ],
        'ARTERIAL': [
            'ARTERIAL', 'ART', 'EARLY', 'THIN ARTERIAL', 'COR ART'
        ],
        'VENOUS': [
            'VENOUS', 'VEN', 'VENOUS PHASE', 'LUNG.VEIN'
        ],
        'PORTAL': [
            'PORTAL', 'PORTOVENOUS', 'THIN PORTOVENOUS', 'PORTAL AXIAL',
            'COR-PORTAL', 'SAG PORTAL', 'PORTAL, IDOSE'
        ],
        'DELAY': [
            'DELAY', 'DELAYED', 'LATE', 'DELAY PHASE', 'DELAY THIN CUT',
            'DELAY--', 'DELAY 3 MIN', 'DELAY 7MIN', 'DELAY 15 MIN',
            'COR?DELAY', 'SAG DELAY', 'AX 3M DELAY', 'COR 3M DELAY'
        ],
        'EQUILIBRIUM': ['EQUILIBRIUM', 'EQUIL']
    }
    
    # Enhanced orientation patterns
    ORIENTATION_PATTERNS = {
        'CORONAL': [
            'COR', 'CORONAL', 'CORO', 'COR-', 'COR.', 'COR/', 'MPR.*COR'
        ],
        'SAGITTAL': [
            'SAG', 'SAGITTAL', 'SAG.*', 'SAGITTAL.REF', 'MPR.*SAG'
        ],
        'AXIAL': [
            'AXIAL', 'BODY.*AXIAL', 'AX', 'AXIAL.*TRUE'
        ]
    }
    
    # Enhanced body part patterns
    BODY_PART_PATTERNS = {
        'HEAD': ['HEAD', 'BRAIN', 'SKULL', 'CRANIAL'],
        'NECK': ['NECK', 'CERVICAL', 'COR NECK'],
        'CHEST': [
            'CHEST', 'THORAX', 'THX', 'PULMONARY', 'LUNG', 'CARDIAC',
            'ROUTINE CHEST', 'CHEST NEW', 'THORAXROUTIN'
        ],
        'ABDOMEN': [
            'ABDOMEN', 'ABD', 'ABDOMINAL', 'LIVER', 'PANCREAS',
            'ABD_MILAD', 'ABDOMENROUTINE'
        ],
        'PELVIS': ['PELVIS', 'PEL', 'PELVIC'],
        'ABDOMEN_PELVIS': [
            'ABD.*PEL', 'ABDPEL', 'ABDOMEN.*PELVIS', 'ABP', 'ABDOMENPELVIS',
            'ABD_PEL', 'YGHABDPEL', 'ABDPLV'
        ],
        'CHEST_ABDOMEN': [
            'THX.*ABD', 'CHEST.*ABDOMEN', 'CHE.*ABD', 'CHEST&'
        ],
        'CHEST_ABDOMEN_PELVIS': [
            'THX.*ABD.*PEL', 'CHEST.*ABDOMEN.*PELVIS', 'CH.*ABD.*PLV',
            'THXABDPEL', 'NECKTTHXABDPEL'
        ],
        'LIVER': ['LIVER', 'HEPATIC'],
        'EXTREMITIES': ['EXTREMITY', 'ARM', 'LEG', 'HAND', 'FOOT'],
        'SPINE': ['SPINE', 'SPINAL', 'VERTEBRA', 'LUMBAR', 'CERVICAL']
    }

class DICOMSeriesLabeler:
    """Main class for DICOM series NLP labeling"""
    
    def __init__(self):
        self.config = LabelingConfig()
        
    def extract_image_origin(self, image_type: str) -> Tuple[str, float]:
        """Extract image origin from image_type field"""
        if pd.isna(image_type):
            return "UNKNOWN", 0.0
            
        image_type_upper = str(image_type).upper()
        
        if any(pattern in image_type_upper for pattern in self.config.ORIGINAL_PATTERNS):
            return "ORIGINAL", 1.0
        elif any(pattern in image_type_upper for pattern in self.config.DERIVED_PATTERNS):
            return "DERIVED", 1.0
        else:
            return "UNKNOWN", 0.0
    
    def extract_reconstruction_type(self, image_type: str, series_desc: str) -> Tuple[str, float]:
        """Extract reconstruction type using multi-layer logic"""
        if pd.isna(image_type):
            return "UNKNOWN", 0.0
            
        image_type_upper = str(image_type).upper()
        
        # Layer 1: Check image_type for reconstruction patterns
        for recon_type, patterns in self.config.RECONSTRUCTION_PATTERNS.items():
            if any(pattern in image_type_upper for pattern in patterns):
                return recon_type, 1.0
        
        # Layer 2: Check series description for reconstruction hints
        if not pd.isna(series_desc):
            series_upper = str(series_desc).upper()
            
            # Check for explicit MPR indicators
            if 'MPR' in series_upper:
                return "MPR", 0.9
            
            # Check for orientation indicators (suggest MPR)
            if any(pattern in series_upper for pattern in ['COR', 'CORONAL', 'SAG', 'SAGITTAL']):
                return "MPR", 0.8
                
            # Check for MIP indicators
            if 'MIP' in series_upper:
                return "MIP", 0.9
                
            # Check for 3D indicators
            if '3D' in series_upper or 'VOLUME' in series_upper:
                return "VOLUME", 0.8
        
        # Layer 3: Default to AXIAL if image_type contains AXIAL
        if "AXIAL" in image_type_upper:
            return "AXIAL", 0.6
        
        return "UNKNOWN", 0.0
    
    def extract_slice_thickness(self, parsed_thickness: float, series_desc: str) -> Tuple[Optional[float], float]:
        """Extract slice thickness value"""
        # Layer 1: Use parsed thickness directly
        if not pd.isna(parsed_thickness) and parsed_thickness > 0:
            return parsed_thickness, 1.0
        
        # Layer 2: Extract from series description
        if not pd.isna(series_desc):
            # Look for patterns like "2.0", "1.5", etc. in series description
            thickness_match = re.search(r'(\d+\.?\d*)\s*(?:mm|MM)?', str(series_desc))
            if thickness_match:
                try:
                    thickness = float(thickness_match.group(1))
                    return thickness, 0.7
                except ValueError:
                    pass
        
        return None, 0.0
    
    def extract_study_contrast(self, study_desc: str, study_protocol: str) -> Tuple[str, float]:
        """Extract study-level contrast enhancement"""
        
        # Layer 1: Analyze study description for explicit patterns
        if not pd.isna(study_desc):
            study_upper = str(study_desc).upper()
            
            # Check for specific combined patterns first (highest priority)
            if 'WOWITH' in study_upper or 'WITHOUT_WITH' in study_upper or 'WW' in study_upper:
                return "YES", 0.98
            
            # Check for explicit WITHOUT patterns (high confidence) - but not if combined with WITH
            if any(pattern in study_upper for pattern in self.config.CONTRAST_NO_PATTERNS) and not any(with_pattern in study_upper for with_pattern in ['WITH', 'WOWITH']):
                return "NO", 0.95
            
            # Check for explicit WITH patterns (high confidence)  
            if any(pattern in study_upper for pattern in self.config.CONTRAST_YES_PATTERNS):
                return "YES", 0.95
        
        # Layer 2: Analyze study protocol
        if not pd.isna(study_protocol):
            protocol_upper = str(study_protocol).upper()
            
            if any(pattern in protocol_upper for pattern in self.config.CONTRAST_NO_PATTERNS):
                return "NO", 0.9
            elif any(pattern in protocol_upper for pattern in self.config.CONTRAST_YES_PATTERNS):
                return "YES", 0.9
        
        # Only return UNKNOWN if confidence is very low
        return "UNKNOWN", 0.0
    
    def extract_series_contrast(self, series_desc: str, study_contrast: str) -> Tuple[str, float]:
        """Extract series-level contrast enhancement"""
        
        # If study is definitively NO contrast, series must be NO
        if study_contrast == "NO":
            return "NO", 0.95
        
        if pd.isna(series_desc):
            # Inherit from study if series unclear and study has high confidence
            if study_contrast in ["YES", "NO"]:
                return study_contrast, 0.5
            return "UNKNOWN", 0.0
        
        series_upper = str(series_desc).upper()
        
        # Layer 1: Direct contrast indicators in series description
        # Check for specific high-priority patterns first
        if 'W/O' in series_upper or series_upper.endswith(' WO') or 'WITHOUT' in series_upper:
            return "NO", 0.95
        
        # Check for contrast enhancement indicators
        if ('CE' in series_upper and any(num in series_upper for num in ['1.', '2.', '3.', '4.', '5.', '10.'])):
            return "YES", 0.95
        
        if 'CORONAL C' in series_upper or 'COR C' in series_upper:
            return "YES", 0.9
            
        # Check for timing-based contrast indicators    
        if '+C' in series_upper or '+CM' in series_upper:
            return "YES", 0.9
            
        if any(pattern in series_upper for pattern in self.config.CONTRAST_SERIES_NO_PATTERNS):
            return "NO", 0.9
        
        if any(pattern in series_upper for pattern in self.config.CONTRAST_SERIES_YES_PATTERNS):
            return "YES", 0.9
        
        # Layer 2: Phase indicators
        if any(pattern in series_upper for pattern in self.config.PHASE_PATTERNS['NON_CONTRAST']):
            return "NO", 0.85
        
        if any(pattern in series_upper for pattern in 
               self.config.PHASE_PATTERNS['ARTERIAL'] + 
               self.config.PHASE_PATTERNS['VENOUS'] + 
               self.config.PHASE_PATTERNS['PORTAL'] + 
               self.config.PHASE_PATTERNS['DELAY']):
            return "YES", 0.85
        
        # Layer 3: Check for timing patterns
        timing_match = re.search(r'(\d+)\s*[sm]', series_upper)
        if timing_match or 'PHASE' in series_upper:
            return "YES", 0.7
        
        # Layer 4: Inherit from study only if high confidence
        if study_contrast == "YES":
            return "YES", 0.6
        elif study_contrast == "NO":
            return "NO", 0.6
        
        return "UNKNOWN", 0.0
    
    def extract_image_orientation(self, image_type: str, series_desc: str) -> Tuple[str, float]:
        """Extract image orientation"""
        # Layer 1: Parse image_type
        if not pd.isna(image_type):
            image_upper = str(image_type).upper()
            if "AXIAL" in image_upper:
                return "AXIAL", 0.8
        
        # Layer 2: Analyze series description
        if not pd.isna(series_desc):
            series_upper = str(series_desc).upper()
            for orientation, patterns in self.config.ORIENTATION_PATTERNS.items():
                if any(re.search(pattern, series_upper) for pattern in patterns):
                    return orientation, 0.9
        
        # Layer 3: Default to AXIAL for CT studies (only if confident)
        return "UNKNOWN", 0.0
    
    def extract_contrast_phase(self, series_desc: str, study_desc: str) -> Tuple[str, float]:
        """Extract contrast phase using timing and keyword analysis"""
        if pd.isna(series_desc):
            return "UNKNOWN", 0.0
        
        series_upper = str(series_desc).upper()
        
        # Layer 1: Direct phase identification
        for phase, patterns in self.config.PHASE_PATTERNS.items():
            if any(pattern in series_upper for pattern in patterns):
                return phase, 0.9
        
        # Layer 2: Enhanced timing-based detection
        timing_match = re.search(r'(\d+)\s*[sm]', series_upper)
        if timing_match:
            timing = int(timing_match.group(1))
            if timing <= 30:
                return "ARTERIAL", 0.8
            elif 60 <= timing <= 90:
                return "PORTAL", 0.8
            elif timing > 120:
                return "DELAY", 0.8
        
        # Check for minute-based timing patterns
        minute_match = re.search(r'(\d+)\s*MIN', series_upper)
        if minute_match:
            minutes = int(minute_match.group(1))
            if minutes >= 3:
                return "DELAY", 0.8
        
        # Check for specific timing descriptions
        if '3M' in series_upper or '3 MM' in series_upper:
            return "DELAY", 0.7
        if '15 MIN' in series_upper or '7MIN' in series_upper:
            return "DELAY", 0.8
        
        # No study-level inference for series phases
        return "UNKNOWN", 0.0
    
    def extract_body_part(self, study_body_part: str, study_desc: str, study_protocol: str) -> Tuple[str, float]:
        """Extract and standardize body part"""
        # Layer 1: Direct mapping from study_body_part
        if not pd.isna(study_body_part):
            body_part_upper = str(study_body_part).upper()
            
            direct_mapping = {
                'ABDOMEN': 'ABDOMEN',
                'CHEST': 'CHEST',
                'CHEST_TO_PELVIS': 'CHEST_ABDOMEN_PELVIS',
                'WHOLEBODY': 'CHEST_ABDOMEN_PELVIS',
                'LIVER': 'LIVER',
                'HEAD': 'HEAD',
                'NECK': 'NECK'
            }
            
            if body_part_upper in direct_mapping:
                return direct_mapping[body_part_upper], 1.0
        
        # Layer 2: Parse study description and protocol (higher confidence required)
        text_sources = [study_desc, study_protocol]
        for text in text_sources:
            if not pd.isna(text):
                text_upper = str(text).upper()
                for body_part, patterns in self.config.BODY_PART_PATTERNS.items():
                    if any(re.search(pattern, text_upper) for pattern in patterns):
                        return body_part, 0.85
        
        return "UNKNOWN", 0.0
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process entire dataframe with NLP labeling"""
        logger.info(f"Processing {len(df)} rows for NLP labeling...")
        
        # Initialize new columns
        df['image_origin'] = None
        df['image_origin_confidence'] = None
        df['reconstruction_type'] = None
        df['reconstruction_confidence'] = None
        df['slice_thickness_mm'] = None
        df['thickness_confidence'] = None
        df['study_contrast_enhanced'] = None
        df['study_contrast_confidence'] = None
        df['series_contrast_enhanced'] = None
        df['series_contrast_confidence'] = None
        df['image_orientation'] = None
        df['orientation_confidence'] = None
        df['contrast_phase'] = None
        df['phase_confidence'] = None
        df['standardized_body_part'] = None
        df['body_part_confidence'] = None
        
        for idx, row in df.iterrows():
            # Extract all labels
            origin, origin_conf = self.extract_image_origin(row.get('image_type'))
            recon, recon_conf = self.extract_reconstruction_type(
                row.get('image_type'), row.get('study_series_description'))
            thickness, thick_conf = self.extract_slice_thickness(
                row.get('parsed_slice_thickness'), row.get('study_series_description'))
            study_contrast, study_contrast_conf = self.extract_study_contrast(
                row.get('study_description'), row.get('study_protocol'))
            series_contrast, series_contrast_conf = self.extract_series_contrast(
                row.get('study_series_description'), study_contrast)
            orientation, orient_conf = self.extract_image_orientation(
                row.get('image_type'), row.get('study_series_description'))
            phase, phase_conf = self.extract_contrast_phase(
                row.get('study_series_description'), row.get('study_description'))
            body_part, body_conf = self.extract_body_part(
                row.get('study_body_part'), row.get('study_description'), row.get('study_protocol'))
            
            # Apply confidence filtering - only assign if confidence >= 0.7
            confidence_threshold = 0.7
            
            df.at[idx, 'image_origin'] = origin if origin_conf >= confidence_threshold else "UNKNOWN"
            df.at[idx, 'image_origin_confidence'] = origin_conf
            df.at[idx, 'reconstruction_type'] = recon if recon_conf >= confidence_threshold else "UNKNOWN"
            df.at[idx, 'reconstruction_confidence'] = recon_conf
            df.at[idx, 'slice_thickness_mm'] = thickness if thick_conf >= confidence_threshold else None
            df.at[idx, 'thickness_confidence'] = thick_conf
            df.at[idx, 'study_contrast_enhanced'] = study_contrast if study_contrast_conf >= confidence_threshold else "UNKNOWN"
            df.at[idx, 'study_contrast_confidence'] = study_contrast_conf
            df.at[idx, 'series_contrast_enhanced'] = series_contrast if series_contrast_conf >= confidence_threshold else "UNKNOWN"
            df.at[idx, 'series_contrast_confidence'] = series_contrast_conf
            df.at[idx, 'image_orientation'] = orientation if orient_conf >= confidence_threshold else "UNKNOWN"
            df.at[idx, 'orientation_confidence'] = orient_conf
            df.at[idx, 'contrast_phase'] = phase if phase_conf >= confidence_threshold else "UNKNOWN"
            df.at[idx, 'phase_confidence'] = phase_conf
            df.at[idx, 'standardized_body_part'] = body_part if body_conf >= confidence_threshold else "UNKNOWN"
            df.at[idx, 'body_part_confidence'] = body_conf
            
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1} rows...")
        
        logger.info("NLP labeling completed!")
        return df

def select_input_file() -> Optional[str]:
    """Interactive file selection interface"""
    default_dir = Path("data/consolidated_summaries")
    
    print("\n=== DICOM Series NLP Labeling Tool ===")
    print("Please select input Excel file:")
    print("1. Browse for file")
    print("2. Use default directory (data/consolidated_summaries)")
    print("3. Enter custom path")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        # GUI file selection
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Excel file",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        root.destroy()
        return file_path if file_path else None
        
    elif choice == "2":
        # List files in default directory
        if default_dir.exists():
            excel_files = list(default_dir.glob("*.xlsx"))
            if not excel_files:
                print("No Excel files found in default directory")
                return None
            
            print("\nAvailable Excel files:")
            for i, file_path in enumerate(excel_files, 1):
                print(f"{i}. {file_path.name}")
            
            try:
                selection = int(input(f"Select file (1-{len(excel_files)}): "))
                if 1 <= selection <= len(excel_files):
                    return str(excel_files[selection - 1])
                else:
                    print("Invalid selection")
                    return None
            except ValueError:
                print("Invalid input")
                return None
        else:
            print("Default directory does not exist")
            return None
            
    elif choice == "3":
        # Custom path
        file_path = input("Enter path to Excel file: ").strip()
        if os.path.exists(file_path):
            return file_path
        else:
            print("File does not exist")
            return None
    
    else:
        print("Invalid choice")
        return None

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="DICOM Series NLP Labeling")
    parser.add_argument("--input", help="Input Excel file path")
    parser.add_argument("--output", help="Output Excel file path")
    parser.add_argument("--test", action="store_true", help="Test mode with sample data")
    
    args = parser.parse_args()
    
    # Initialize labeler
    labeler = DICOMSeriesLabeler()
    
    # Get input file
    if args.input:
        input_file = args.input
    else:
        input_file = select_input_file()
    
    if not input_file:
        print("No input file selected. Exiting.")
        return
    
    # Load data
    try:
        logger.info(f"Loading data from {input_file}")
        df = pd.read_excel(input_file)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        return
    
    # Test mode - process only first 100 rows
    if args.test:
        logger.info("Test mode: processing first 100 rows")
        df = df.head(100)
    
    # Process data
    try:
        labeled_df = labeler.process_dataframe(df)
    except Exception as e:
        logger.error(f"Error during NLP processing: {e}")
        return
    
    # Generate output filename
    if args.output:
        output_file = args.output
    else:
        input_path = Path(input_file)
        output_dir = Path("data/consolidated_summaries")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / f"{input_path.stem}_NLPlabel.xlsx")
    
    # Save results
    try:
        logger.info(f"Saving labeled data to {output_file}")
        labeled_df.to_excel(output_file, index=False)
        logger.info("Successfully saved labeled data!")
        
        # Print summary statistics
        print("\n=== LABELING SUMMARY ===")
        label_columns = [col for col in labeled_df.columns if not col.endswith('_confidence')]
        new_label_columns = [col for col in label_columns if col not in df.columns]
        
        for col in new_label_columns:
            print(f"\n{col}:")
            value_counts = labeled_df[col].value_counts()
            for value, count in value_counts.head(10).items():
                print(f"  {value}: {count}")
                
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return

if __name__ == "__main__":
    main()