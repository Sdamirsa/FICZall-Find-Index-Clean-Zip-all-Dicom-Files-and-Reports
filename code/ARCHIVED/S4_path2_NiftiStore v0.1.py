#!/usr/bin/env python3
"""
S4_path2_NiftiStore.py
======================

Expert-Level DICOM to NIfTI Conversion Pipeline
- Proper JSONL string parsing for DICOM geometry
- Accurate pixel data extraction from DICOM files  
- Medical imaging expertise in spatial registration
- Comprehensive error handling and quality control
- Excel export with processing results

Usage:
    python S4_path2_NiftiStore.py [--excel-file PATH] [--output-dir PATH] [--analysis-only]
"""

import os
import sys
import json
import hashlib
import logging
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
import pydicom
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict
import re
import random


class DICOMGeometryParser:
    """Expert-level parser for DICOM geometry stored as strings in JSONL."""
    
    @staticmethod
    def parse_image_orientation_patient(iop_str: str) -> Optional[np.ndarray]:
        """
        Parse ImageOrientationPatient from JSONL string format.
        
        Expected format: "[1, 7.044874666e-016, 0, 0, 0, -1]"
        Returns: numpy array of 6 direction cosines [Xx, Xy, Xz, Yx, Yy, Yz]
        
        Args:
            iop_str: String representation of orientation
            
        Returns:
            6-element numpy array or None if parsing fails
        """
        if not iop_str or iop_str == "Unknown":
            return None
            
        try:
            # Clean and parse string representation of list
            clean_str = iop_str.strip('[]').replace(' ', '')
            values = [float(x.strip()) for x in clean_str.split(',') if x.strip()]
            
            if len(values) >= 6:
                return np.array(values[:6], dtype=np.float64)
            else:
                return None
                
        except (ValueError, TypeError, AttributeError):
            return None
    
    @staticmethod
    def parse_image_position_patient(ipp_str: str) -> Optional[np.ndarray]:
        """
        Parse ImagePositionPatient from JSONL string format.
        
        Expected format: "[-511, -196, -766.5]"
        Returns: numpy array of 3 coordinates [x, y, z] in mm
        
        Args:
            ipp_str: String representation of position
            
        Returns:
            3-element numpy array or None if parsing fails
        """
        if not ipp_str or ipp_str == "Unknown":
            return None
            
        try:
            # Clean and parse string representation of list
            clean_str = ipp_str.strip('[]').replace(' ', '')
            values = [float(x.strip()) for x in clean_str.split(',') if x.strip()]
            
            if len(values) >= 3:
                return np.array(values[:3], dtype=np.float64)
            else:
                return None
                
        except (ValueError, TypeError, AttributeError):
            return None
    
    @staticmethod
    def parse_pixel_spacing(ps_str: str) -> Optional[Tuple[float, float]]:
        """
        Parse PixelSpacing from JSONL string format.
        
        Expected format: "2\\2" or "0.5\\0.5"
        Returns: tuple of (row_spacing, col_spacing) in mm
        
        Args:
            ps_str: String representation of pixel spacing
            
        Returns:
            Tuple of (row_spacing, col_spacing) or None if parsing fails
        """
        if not ps_str or ps_str == "Unknown":
            return None
            
        try:
            # Handle different separator formats
            if '\\\\' in ps_str:
                values = ps_str.split('\\\\')
            elif '\\' in ps_str:
                values = ps_str.split('\\')
            elif '/' in ps_str:
                values = ps_str.split('/')
            elif ',' in ps_str:
                values = ps_str.split(',')
            else:
                # Single value - assume square pixels
                return (float(ps_str), float(ps_str))
            
            if len(values) >= 2:
                row_spacing = float(values[0].strip())
                col_spacing = float(values[1].strip())
                return (row_spacing, col_spacing)
            elif len(values) == 1:
                spacing = float(values[0].strip())
                return (spacing, spacing)
            else:
                return None
                
        except (ValueError, TypeError, AttributeError):
            return None
    
    @staticmethod
    def parse_float_field(field_str: Union[str, float, int]) -> Optional[float]:
        """
        Parse float fields like slice_thickness, slice_location.
        
        Args:
            field_str: String, float, or int representation
            
        Returns:
            Float value or None if parsing fails
        """
        if field_str is None or field_str == "Unknown" or field_str == "":
            return None
            
        try:
            if isinstance(field_str, (int, float)):
                return float(field_str)
            else:
                return float(str(field_str).strip())
        except (ValueError, TypeError):
            return None


class NiftiStoreProcessor:
    """Expert-level DICOM to NIfTI processor with medical imaging expertise."""
    
    def __init__(self, output_dir: str = "data/nifti2share"):
        self.output_dir = Path(output_dir)
        self.setup_directories()
        self.setup_logging()
        
        # Processing statistics
        self.stats = {
            'total_studies': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'total_series': 0,
            'total_slices': 0,
            'analysis_exported': False,
            'start_time': datetime.now()
        }
        
        # Error tracking
        self.errors = []
        self.processing_results = []
        
        # Secure mapping for pseudonymization
        self.secure_mapping = {}
        self.patient_shifts = {}
        
        # Geometry parser
        self.geometry_parser = DICOMGeometryParser()
    
    def setup_directories(self):
        """Create necessary output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.output_dir / 'processing_log.txt')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def find_excel_files(self, search_dir: str = "data/consolidated_summaries") -> List[Path]:
        """Find available consolidated Excel files."""
        search_path = Path(search_dir)
        if not search_path.exists():
            self.logger.error(f"Search directory not found: {search_path}")
            return []
        
        pattern = "consolidated_patient_summary_*.xlsx"
        excel_files = list(search_path.glob(pattern))
        excel_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return excel_files
    
    def select_excel_file(self, custom_path: Optional[str] = None) -> Optional[Path]:
        """Interactive selection of Excel file."""
        if custom_path:
            excel_path = Path(custom_path)
            if excel_path.exists():
                return excel_path
            else:
                self.logger.error(f"Custom Excel file not found: {excel_path}")
                return None
        
        excel_files = self.find_excel_files()
        
        if not excel_files:
            self.logger.error("No consolidated Excel files found")
            return None
        
        if len(excel_files) == 1:
            self.logger.info(f"Using Excel file: {excel_files[0]}")
            return excel_files[0]
        
        # Interactive selection
        print("\\nAvailable Excel files:")
        for i, file_path in enumerate(excel_files):
            timestamp = datetime.fromtimestamp(file_path.stat().st_mtime)
            print(f"{i+1}. {file_path.name} (modified: {timestamp.strftime('%Y-%m-%d %H:%M')})")
        
        while True:
            try:
                choice = input(f"\\nSelect file (1-{len(excel_files)}): ").strip()
                index = int(choice) - 1
                if 0 <= index < len(excel_files):
                    return excel_files[index]
                else:
                    print(f"Please enter a number between 1 and {len(excel_files)}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\\nCancelled by user")
                return None
    
    def load_selected_studies(self, excel_path: Path) -> pd.DataFrame:
        """Load studies marked for processing."""
        self.logger.info(f"Loading Excel file: {excel_path}")
        
        try:
            df = pd.read_excel(excel_path)
            self.logger.info(f"Total records in Excel: {len(df)}")
            
            if 'final_selection' not in df.columns:
                self.logger.error("'final_selection' column not found")
                return pd.DataFrame()
            
            selected_df = df[df['final_selection'] == 1].copy()
            self.logger.info(f"Selected studies for processing: {len(selected_df)}")
            
            self.stats['total_studies'] = len(selected_df)
            return selected_df
            
        except Exception as e:
            self.logger.error(f"Error loading Excel file: {e}")
            return pd.DataFrame()
    
    def get_patient_shifts(self, patient_id: str) -> dict:
        """Get consistent date shift and age jitter for a patient."""
        if patient_id not in self.patient_shifts:
            random.seed(patient_id)
            date_shift_days = random.randint(-730, 730)
            age_jitter_years = random.randint(-2, 2)
            
            self.patient_shifts[patient_id] = {
                'date_shift_days': date_shift_days,
                'age_jitter_years': age_jitter_years
            }
            random.seed()
        
        return self.patient_shifts[patient_id]
    
    def apply_date_shift(self, date_str: str, patient_id: str) -> str:
        """Apply consistent date shift for a patient."""
        try:
            if date_str == "Unknown" or not date_str:
                return date_str
            
            original_date = datetime.strptime(date_str, "%Y%m%d")
            shifts = self.get_patient_shifts(patient_id)
            shifted_date = original_date + timedelta(days=shifts['date_shift_days'])
            
            return shifted_date.strftime("%Y%m%d")
        except (ValueError, TypeError):
            return date_str
    
    def apply_age_jitter(self, age_str: str, patient_id: str) -> str:
        """Apply consistent age jitter for a patient."""
        try:
            if age_str == "Unknown" or not age_str:
                return age_str
            
            age_match = age_str.rstrip('Y')
            original_age = int(age_match)
            shifts = self.get_patient_shifts(patient_id)
            jittered_age = max(1, original_age + shifts['age_jitter_years'])
            
            return f"{jittered_age:03d}Y"
        except (ValueError, TypeError):
            return age_str
    
    def generate_study_hash(self, patient_id: str, study_date: str, series_number: str, institution: str) -> str:
        """Generate unique hash ID for a study series."""
        hash_input = f"{patient_id}_{study_date}_{series_number}_{institution}"
        hash_object = hashlib.sha256(hash_input.encode('utf-8'))
        return hash_object.hexdigest()[:16]
    
    def parse_jsonl_file(self, jsonl_path: Path) -> List[Dict]:
        """Parse JSONL file and extract DICOM metadata."""
        try:
            dicom_data = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        dicom_data.append(data)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON at line {line_num} in {jsonl_path}: {e}")
            
            self.logger.info(f"Parsed {len(dicom_data)} DICOM entries from {jsonl_path}")
            return dicom_data
            
        except Exception as e:
            self.logger.error(f"Error parsing JSONL file {jsonl_path}: {e}")
            return []
    
    def parse_dicom_geometry(self, dicom_info: Dict) -> Dict[str, Any]:
        """
        Parse DICOM geometry from JSONL string representations.
        
        Expert medical imaging knowledge applied for proper spatial registration.
        
        Args:
            dicom_info: Raw DICOM metadata dictionary from JSONL
            
        Returns:
            Dictionary with parsed geometry and validation status
        """
        geometry = {
            'original_data': dicom_info,
            'parsed_geometry': {},
            'validation': {
                'is_valid': True,
                'errors': [],
                'warnings': []
            }
        }
        
        # Parse Image Orientation Patient (Direction Cosines)
        iop_raw = dicom_info.get('image_orientation_patient')
        iop_parsed = self.geometry_parser.parse_image_orientation_patient(iop_raw)
        
        if iop_parsed is not None:
            # Validate direction cosines (should be unit vectors)
            row_cosines = iop_parsed[:3]
            col_cosines = iop_parsed[3:6]
            
            row_norm = np.linalg.norm(row_cosines)
            col_norm = np.linalg.norm(col_cosines)
            
            if abs(row_norm - 1.0) > 0.1 or abs(col_norm - 1.0) > 0.1:
                geometry['validation']['warnings'].append("Direction cosines are not unit vectors")
            
            # Check orthogonality
            dot_product = np.dot(row_cosines, col_cosines)
            if abs(dot_product) > 0.1:
                geometry['validation']['warnings'].append("Row and column vectors are not orthogonal")
            
            geometry['parsed_geometry']['image_orientation_patient'] = iop_parsed
            geometry['parsed_geometry']['row_direction'] = row_cosines
            geometry['parsed_geometry']['col_direction'] = col_cosines
            geometry['parsed_geometry']['slice_direction'] = np.cross(row_cosines, col_cosines)
        else:
            geometry['validation']['errors'].append("Failed to parse ImageOrientationPatient")
            geometry['validation']['is_valid'] = False
        
        # Parse Image Position Patient (Origin coordinates)
        ipp_raw = dicom_info.get('image_position_patient')
        ipp_parsed = self.geometry_parser.parse_image_position_patient(ipp_raw)
        
        if ipp_parsed is not None:
            geometry['parsed_geometry']['image_position_patient'] = ipp_parsed
            geometry['parsed_geometry']['origin'] = ipp_parsed
        else:
            geometry['validation']['errors'].append("Failed to parse ImagePositionPatient")
            geometry['validation']['is_valid'] = False
        
        # Parse Pixel Spacing (In-plane resolution)
        ps_raw = dicom_info.get('pixel_spacing')
        ps_parsed = self.geometry_parser.parse_pixel_spacing(ps_raw)
        
        if ps_parsed is not None:
            row_spacing, col_spacing = ps_parsed
            geometry['parsed_geometry']['pixel_spacing'] = ps_parsed
            geometry['parsed_geometry']['row_spacing'] = row_spacing
            geometry['parsed_geometry']['col_spacing'] = col_spacing
            
            # Validate reasonable spacing values (0.1mm to 10mm typical for CT)
            if not (0.1 <= row_spacing <= 10.0) or not (0.1 <= col_spacing <= 10.0):
                geometry['validation']['warnings'].append(f"Unusual pixel spacing: {row_spacing}x{col_spacing} mm")
        else:
            geometry['validation']['warnings'].append("Failed to parse PixelSpacing")
            # Use default values
            geometry['parsed_geometry']['pixel_spacing'] = (1.0, 1.0)
            geometry['parsed_geometry']['row_spacing'] = 1.0
            geometry['parsed_geometry']['col_spacing'] = 1.0
        
        # Parse Slice Thickness with smart conversion logic
        st_raw = dicom_info.get('slice_thickness')
        st_parsed = self.geometry_parser.parse_float_field(st_raw)
        
        if st_parsed is not None:
            # Apply smart conversion: if > 100, divide by 1000 (e.g., 506 -> 0.506)
            if st_parsed > 100:
                original_value = st_parsed
                st_parsed = st_parsed / 1000.0
                geometry['validation']['warnings'].append(
                    f"Converted slice thickness: {original_value} -> {st_parsed:.3f}mm (divided by 1000)"
                )
            
            geometry['parsed_geometry']['slice_thickness'] = st_parsed
            geometry['parsed_geometry']['has_slice_thickness'] = True
            
            # Validate reasonable thickness (0.1mm to 50mm range for medical imaging)
            if not (0.1 <= st_parsed <= 50.0):
                geometry['validation']['warnings'].append(f"Unusual slice thickness: {st_parsed:.3f}mm")
        else:
            geometry['parsed_geometry']['has_slice_thickness'] = False
        
        # Parse Slice Location
        sl_raw = dicom_info.get('slice_location')
        sl_parsed = self.geometry_parser.parse_float_field(sl_raw)
        
        if sl_parsed is not None:
            geometry['parsed_geometry']['slice_location'] = sl_parsed
        
        return geometry
    
    def create_affine_matrix(self, parsed_geometries: List[Dict], spacing_analysis: Dict = None) -> Tuple[np.ndarray, bool]:
        """
        Create NIfTI affine matrix using medical imaging expertise.
        
        Implements proper DICOM to NIfTI coordinate transformation:
        - DICOM uses LPS (Left-Posterior-Superior) coordinate system
        - NIfTI uses RAS (Right-Anterior-Superior) coordinate system
        
        Args:
            parsed_geometries: List of parsed geometry dictionaries for series
            spacing_analysis: Optional spacing analysis from ImagePositionPatient
            
        Returns:
            Tuple of (4x4 affine matrix, has_warnings)
        """
        if not parsed_geometries:
            return np.eye(4), True
        
        has_warnings = False
        first_geometry = parsed_geometries[0]['parsed_geometry']
        
        try:
            # Extract spatial parameters
            row_direction = first_geometry['row_direction']
            col_direction = first_geometry['col_direction'] 
            slice_direction = first_geometry['slice_direction']
            origin = first_geometry['origin']
            row_spacing = first_geometry['row_spacing']
            col_spacing = first_geometry['col_spacing']
            
            # Determine slice spacing using expert medical imaging approach
            slice_spacing_method = "fallback"
            
            # Priority 1: Use ImagePositionPatient-calculated spacing
            if spacing_analysis and spacing_analysis.get('success'):
                slice_spacing = spacing_analysis['slice_spacing']
                slice_spacing_method = "image_position_patient"
                
                if not spacing_analysis.get('spacing_consistency', True):
                    has_warnings = True
                    self.logger.warning("Irregular slice spacing detected from ImagePositionPatient")
                    
                self.logger.info(f"Using ImagePositionPatient spacing: {slice_spacing:.3f}mm")
                
            # Priority 2: Calculate from slice locations if available
            elif len(parsed_geometries) > 1:
                slice_positions = []
                for geom in parsed_geometries:
                    if 'slice_location' in geom['parsed_geometry']:
                        slice_positions.append(geom['parsed_geometry']['slice_location'])
                
                if len(slice_positions) >= 2:
                    slice_positions.sort()
                    spacings = np.diff(slice_positions)
                    slice_spacing = np.mean(np.abs(spacings))
                    slice_spacing_method = "slice_location"
                    
                    # Check for irregular spacing
                    if np.std(spacings) > 0.1 * slice_spacing:
                        has_warnings = True
                        self.logger.warning(f"Irregular slice spacing from slice_location: std={np.std(spacings):.2f}")
                else:
                    # Priority 3: Fallback to slice thickness
                    slice_spacing = first_geometry.get('slice_thickness', 1.0)
                    has_warnings = True
                    slice_spacing_method = "slice_thickness"
            else:
                # Single slice: use slice thickness
                slice_spacing = first_geometry.get('slice_thickness', 1.0)
                slice_spacing_method = "slice_thickness"
            
            # Build affine matrix with proper DICOM LPS->NIfTI RAS transformation
            # DICOM coordinate system: LPS (Left-Posterior-Superior)
            # NIfTI coordinate system: RAS (Right-Anterior-Superior)
            
            # Convert DICOM LPS directions to NIfTI RAS
            # LPS->RAS: flip X and Y axes (multiply by -1)
            ras_row_direction = -row_direction      # L->R: flip X
            ras_col_direction = -col_direction      # P->A: flip Y  
            ras_slice_direction = slice_direction   # S->S: keep Z
            
            # Convert origin from LPS to RAS
            ras_origin = np.array([-origin[0], -origin[1], origin[2]])
            
            affine = np.zeros((4, 4))
            
            # Set rotation and scaling components in RAS coordinate system
            affine[0:3, 0] = ras_row_direction * row_spacing      # i-direction (rows)
            affine[0:3, 1] = ras_col_direction * col_spacing      # j-direction (columns)  
            affine[0:3, 2] = ras_slice_direction * slice_spacing  # k-direction (slices)
            
            # Set translation (origin) in RAS coordinates
            affine[0:3, 3] = ras_origin
            
            # Homogeneous coordinate
            affine[3, 3] = 1.0
            
            # Log transformation details
            self.logger.info(f"Affine matrix created using {slice_spacing_method}:")
            self.logger.info(f"  Spacing: [{row_spacing:.3f}, {col_spacing:.3f}, {slice_spacing:.3f}] mm")
            self.logger.info(f"  Origin (LPS): [{origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f}] mm")
            self.logger.info(f"  Origin (RAS): [{ras_origin[0]:.1f}, {ras_origin[1]:.1f}, {ras_origin[2]:.1f}] mm")
            self.logger.info("Applied DICOM LPS->NIfTI RAS coordinate transformation")
            
            return affine, has_warnings
            
        except Exception as e:
            self.logger.error(f"Error creating affine matrix: {e}")
            return np.eye(4), True
    
    def read_dicom_pixel_data(self, dicom_files: List[str]) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Read actual pixel data from DICOM files.
        
        Applies medical imaging expertise for proper CT data handling:
        - Reads pixel arrays from DICOM files
        - Applies rescale slope/intercept for Hounsfield Units
        - Handles different pixel data formats
        
        Args:
            dicom_files: List of DICOM file paths
            
        Returns:
            Tuple of (3D numpy array, metadata) or (None, error_info)
        """
        try:
            pixel_arrays = []
            slice_positions = []
            metadata = {}
            
            for file_path in dicom_files:
                if not os.path.exists(file_path):
                    self.logger.warning(f"DICOM file not found: {file_path}")
                    continue
                
                try:
                    # Read DICOM file
                    ds = pydicom.dcmread(file_path)
                    
                    # Extract pixel array with compression handling
                    if hasattr(ds, 'pixel_array'):
                        try:
                            pixel_array = ds.pixel_array.astype(np.float32)
                        except Exception as pixel_error:
                            # Special handling for compression errors
                            if "JPEG" in str(pixel_error) or "compress" in str(pixel_error).lower():
                                self.logger.warning(f"Compression error in {file_path}: {pixel_error}")
                                self.logger.info("Skipping compressed file - install: pip install pylibjpeg pylibjpeg-libjpeg gdcm")
                                continue
                            else:
                                raise pixel_error
                        
                        # Apply rescale slope and intercept for CT Hounsfield Units
                        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                            slope = float(ds.RescaleSlope)
                            intercept = float(ds.RescaleIntercept)
                            pixel_array = pixel_array * slope + intercept
                            
                            # Store rescale parameters
                            metadata['rescale_slope'] = slope
                            metadata['rescale_intercept'] = intercept
                        
                        pixel_arrays.append(pixel_array)
                        
                        # Extract slice position for sorting
                        if hasattr(ds, 'ImagePositionPatient'):
                            slice_positions.append(ds.ImagePositionPatient[2])
                        elif hasattr(ds, 'SliceLocation'):
                            slice_positions.append(float(ds.SliceLocation))
                        else:
                            slice_positions.append(len(slice_positions))  # Fallback ordering
                        
                        # Store additional metadata from first slice
                        if not metadata:
                            metadata.update({
                                'study_date': getattr(ds, 'StudyDate', ''),
                                'series_number': getattr(ds, 'SeriesNumber', ''),
                                'rows': getattr(ds, 'Rows', 0),
                                'columns': getattr(ds, 'Columns', 0),
                                'bits_allocated': getattr(ds, 'BitsAllocated', 16),
                                'pixel_representation': getattr(ds, 'PixelRepresentation', 0)
                            })
                    
                except Exception as e:
                    error_msg = str(e)
                    if "JPEG" in error_msg or "compress" in error_msg.lower():
                        self.logger.warning(f"DICOM compression error for {file_path}: {error_msg}")
                        self.logger.info("Consider installing: pip install pylibjpeg pylibjpeg-libjpeg gdcm")
                    else:
                        self.logger.warning(f"Error reading DICOM file {file_path}: {e}")
                    continue
            
            if not pixel_arrays:
                return None, {'error': 'No valid pixel data found'}
            
            # Sort arrays by slice position
            if len(slice_positions) == len(pixel_arrays):
                sorted_data = sorted(zip(slice_positions, pixel_arrays))
                pixel_arrays = [array for _, array in sorted_data]
            
            # Stack into 3D volume with proper DICOM->NIfTI orientation
            # DICOM: each slice is [rows, cols] 
            # NIfTI expects: [x, y, z] where z is slice axis
            
            # Stack slices along the last axis first
            volume_3d = np.stack(pixel_arrays, axis=-1)
            
            # Apply LPS->RAS coordinate transformation
            # For proper NIfTI orientation, we need to flip the first two axes
            # This handles the 90-degree rotation issue you mentioned
            volume_3d = np.flip(volume_3d, axis=0)  # Flip rows (L->R)
            volume_3d = np.flip(volume_3d, axis=1)  # Flip cols (P->A)
            
            metadata.update({
                'volume_shape': volume_3d.shape,
                'volume_dtype': str(volume_3d.dtype),
                'num_slices': len(pixel_arrays),
                'hounsfield_units': 'rescale_slope' in metadata
            })
            
            self.logger.info(f"Created 3D volume: {volume_3d.shape}, dtype: {volume_3d.dtype}")
            self.logger.info("Applied LPS->RAS flip corrections for proper NIfTI orientation")
            
            return volume_3d, metadata
            
        except Exception as e:
            self.logger.error(f"Error reading DICOM pixel data: {e}")
            return None, {'error': str(e)}
    
    def save_nifti_file(self, volume_3d: np.ndarray, affine: np.ndarray, output_path: Path) -> Tuple[bool, int]:
        """
        Save 3D volume as NIfTI file with proper medical imaging format.
        
        Args:
            volume_3d: 3D numpy array
            affine: 4x4 affine transformation matrix
            output_path: Output file path
            
        Returns:
            Tuple of (success, file_size_bytes)
        """
        try:
            # Create NIfTI image with proper header
            nifti_img = nib.Nifti1Image(volume_3d, affine)
            
            # Set proper NIfTI header information
            header = nifti_img.header
            header.set_data_dtype(volume_3d.dtype)
            header.set_xyzt_units('mm', 'sec')  # Spatial units in mm
            
            # Save with compression
            nib.save(nifti_img, str(output_path))
            
            # Get file size
            file_size = output_path.stat().st_size
            
            self.logger.info(f"Saved NIfTI file: {output_path} ({file_size / 1024 / 1024:.1f} MB)")
            
            return True, file_size
            
        except Exception as e:
            self.logger.error(f"Error saving NIfTI file {output_path}: {e}")
            return False, 0
    
    def process_single_series(self, series_data: List[Dict], study_row: pd.Series) -> Dict[str, Any]:
        """
        Process a single DICOM series with expert medical imaging handling.
        
        Args:
            series_data: List of DICOM metadata for the series
            study_row: Original study row from Excel
            
        Returns:
            Dictionary with processing results
        """
        result = {
            'success': False,
            'error_message': '',
            'file_size_bytes': 0,
            'hash_id': '',
            'nifti_path': '',
            'processing_time': 0,
            'geometry_warnings': []
        }
        
        start_time = datetime.now()
        
        try:
            if not series_data:
                result['error_message'] = "No series data provided"
                return result
            
            # Parse geometry from all slices
            parsed_geometries = []
            for slice_info in series_data:
                parsed_geom = self.parse_dicom_geometry(slice_info)
                parsed_geometries.append(parsed_geom)
                
                # Collect warnings
                if parsed_geom['validation']['warnings']:
                    result['geometry_warnings'].extend(parsed_geom['validation']['warnings'])
            
            # Check if any slices have critical geometry errors
            critical_errors = []
            for geom in parsed_geometries:
                if not geom['validation']['is_valid']:
                    critical_errors.extend(geom['validation']['errors'])
            
            if critical_errors:
                result['error_message'] = f"Critical geometry errors: {'; '.join(critical_errors)}"
                return result
            
            # Generate hash ID (without Er_ prefix initially)
            first_slice = series_data[0]
            base_hash_id = self.generate_study_hash(
                patient_id=first_slice.get('patient_id', ''),
                study_date=first_slice.get('study_date', ''),
                series_number=first_slice.get('series_number', ''),
                institution=first_slice.get('institution_name', '')
            )
            
            # Will apply Er_ prefix if ANY issues occur during processing
            has_issues = bool(result['geometry_warnings'])
            hash_id = base_hash_id  # Start without prefix
            
            result['hash_id'] = hash_id
            
            # Create output directory
            series_output_dir = self.output_dir / hash_id
            series_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate spacing from ImagePositionPatient for expert medical imaging
            spacing_analysis = self.calculate_slice_spacing_from_ipp(series_data)
            if spacing_analysis['warnings']:
                result['geometry_warnings'].extend(spacing_analysis['warnings'])
                has_issues = True
            
            # Create affine matrix with expert spacing calculation
            affine_matrix, affine_warnings = self.create_affine_matrix(parsed_geometries, spacing_analysis)
            if affine_warnings:
                result['geometry_warnings'].append("Affine matrix creation warnings")
                has_issues = True
            
            # Read actual pixel data from DICOM files
            dicom_files = [info['file_path'] for info in series_data]
            volume_3d, pixel_metadata = self.read_dicom_pixel_data(dicom_files)
            
            if volume_3d is None:
                # Apply Er_ prefix for pixel data failure
                hash_id = f"Er_{base_hash_id}"
                result['hash_id'] = hash_id
                result['error_message'] = f"Failed to read pixel data: {pixel_metadata.get('error', 'Unknown error')}"
                self.logger.warning(f"Pixel data reading failed for {hash_id}")
                return result
            
            # Apply Er_ prefix if any issues detected so far
            if has_issues:
                hash_id = f"Er_{base_hash_id}"
                result['hash_id'] = hash_id
                self.logger.warning(f"Processing issues detected for {hash_id}")
            
            # Update directory path with final hash_id
            series_output_dir = self.output_dir / hash_id
            series_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save NIfTI file
            nifti_path = series_output_dir / f"{hash_id}.nii.gz"
            save_success, file_size = self.save_nifti_file(volume_3d, affine_matrix, nifti_path)
            
            if not save_success:
                # Apply Er_ prefix for save failure (if not already applied)
                if not hash_id.startswith("Er_"):
                    hash_id = f"Er_{base_hash_id}"
                    result['hash_id'] = hash_id
                    # Recreate directory with Er_ prefix
                    series_output_dir = self.output_dir / hash_id
                    series_output_dir.mkdir(parents=True, exist_ok=True)
                    nifti_path = series_output_dir / f"{hash_id}.nii.gz"
                
                result['error_message'] = "Failed to save NIfTI file"
                self.logger.warning(f"NIfTI save failed for {hash_id}")
                return result
            
            result['nifti_path'] = str(nifti_path)
            result['file_size_bytes'] = file_size
            
            # Save metadata files
            self.save_metadata_files(series_output_dir, hash_id, parsed_geometries, 
                                   pixel_metadata, study_row, first_slice)
            
            # Update secure mapping
            self.update_secure_mapping(hash_id, first_slice, study_row)
            
            result['success'] = True
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Successfully processed series {hash_id}")
            return result
            
        except Exception as e:
            # Apply Er_ prefix for unexpected errors
            if 'hash_id' not in result or not result['hash_id']:
                # Generate basic hash if not already created
                first_slice = series_data[0] if series_data else {}
                base_hash_id = self.generate_study_hash(
                    patient_id=first_slice.get('patient_id', 'Unknown'),
                    study_date=first_slice.get('study_date', 'Unknown'),
                    series_number=first_slice.get('series_number', 'Unknown'),
                    institution=first_slice.get('institution_name', 'Unknown')
                )
                result['hash_id'] = f"Er_{base_hash_id}"
            elif not result['hash_id'].startswith("Er_"):
                # Add Er_ prefix if not already present
                result['hash_id'] = f"Er_{result['hash_id']}"
            
            result['error_message'] = str(e)
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Unexpected error processing series {result['hash_id']}: {e}")
            return result
    
    def save_metadata_files(self, output_dir: Path, hash_id: str, geometries: List[Dict], 
                          pixel_metadata: Dict, study_row: pd.Series, first_slice: Dict):
        """Save metadata files for the processed series."""
        try:
            # Safe metadata (shareable)
            safe_metadata = {
                'hash_id': hash_id,
                'nifti_shape': pixel_metadata.get('volume_shape'),
                'nifti_dtype': pixel_metadata.get('volume_dtype'),
                'num_slices': pixel_metadata.get('num_slices'),
                'hounsfield_units': pixel_metadata.get('hounsfield_units', False),
                'processing_timestamp': datetime.now().isoformat(),
                'geometry_summary': {
                    'has_orientation': len([g for g in geometries if 'image_orientation_patient' in g['parsed_geometry']]) > 0,
                    'has_position': len([g for g in geometries if 'image_position_patient' in g['parsed_geometry']]) > 0,
                    'pixel_spacing': geometries[0]['parsed_geometry'].get('pixel_spacing'),
                    'slice_thickness': geometries[0]['parsed_geometry'].get('slice_thickness'),
                    'validation_warnings': sum(len(g['validation']['warnings']) for g in geometries)
                },
                'shifted_data': {
                    'shifted_study_date': self.apply_date_shift(first_slice.get('study_date', ''), first_slice.get('patient_id', '')),
                    'shifted_patient_age': self.apply_age_jitter(first_slice.get('patient_age', ''), first_slice.get('patient_id', ''))
                }
            }
            
            safe_json_path = output_dir / f"{hash_id}.json"
            with open(safe_json_path, 'w') as f:
                json.dump(safe_metadata, f, indent=2)
            
            # Sensitive metadata (PHI)
            sensitive_metadata = {
                'original_patient_id': first_slice.get('patient_id', ''),
                'original_patient_name': first_slice.get('patient_name', ''),
                'original_study_date': first_slice.get('study_date', ''),
                'institution_name': first_slice.get('institution_name', ''),
                'source_folder': study_row.get('source_folder', ''),
                'source_file': study_row.get('source_file', ''),
                'dicom_files': [g['original_data']['file_path'] for g in geometries],
                'processing_timestamp': datetime.now().isoformat()
            }
            
            sensitive_txt_path = output_dir / f"SENSITIVE_{hash_id}.txt"
            with open(sensitive_txt_path, 'w') as f:
                f.write("SENSITIVE PATIENT DATA - HIPAA PROTECTED\\n")
                f.write("=" * 50 + "\\n\\n")
                for key, value in sensitive_metadata.items():
                    f.write(f"{key}: {value}\\n")
            
        except Exception as e:
            self.logger.error(f"Error saving metadata files: {e}")
    
    def update_secure_mapping(self, hash_id: str, first_slice: Dict, study_row: pd.Series):
        """Update secure mapping with shift information."""
        try:
            patient_id = first_slice.get('patient_id', '')
            shifts = self.get_patient_shifts(patient_id)
            
            self.secure_mapping[hash_id] = {
                'original_patient_id': patient_id,
                'original_patient_name': first_slice.get('patient_name', ''),
                'original_study_date': first_slice.get('study_date', ''),
                'shifted_study_date': self.apply_date_shift(first_slice.get('study_date', ''), patient_id),
                'date_shift_days': shifts['date_shift_days'],
                'age_jitter_years': shifts['age_jitter_years'],
                'series_number': first_slice.get('series_number', ''),
                'institution': first_slice.get('institution_name', ''),
                'source_folder': study_row.get('source_folder', ''),
                'created_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error updating secure mapping: {e}")
    
    def export_analysis_to_excel(self, studies_df: pd.DataFrame, excel_path: Path) -> bool:
        """
        Export DICOM analysis to Excel before processing.
        
        Args:
            studies_df: DataFrame with selected studies
            excel_path: Original Excel file path
            
        Returns:
            True if export successful
        """
        try:
            analysis_rows = []
            
            self.logger.info(f"Analyzing {len(studies_df)} studies...")
            
            for index, study_row in studies_df.iterrows():
                self.logger.info(f"Analyzing study {index + 1}/{len(studies_df)}")
                
                # Extract study information
                source_folder = study_row.get('source_folder', '')
                source_file = study_row.get('source_file', '')
                
                if not source_folder or not source_file:
                    continue
                
                # Construct JSONL file path
                base_dir = Path(__file__).parent.parent
                jsonl_path = base_dir / "data" / "processed" / source_folder / "S1_indexed_metadata" / source_file
                
                if not jsonl_path.exists():
                    continue
                
                # Parse JSONL and analyze
                dicom_data = self.parse_jsonl_file(jsonl_path)
                if not dicom_data:
                    continue
                
                # Group by series and analyze each
                series_groups = defaultdict(list)
                for dicom_info in dicom_data:
                    series_num = dicom_info.get('series_number', 'unknown')
                    series_groups[series_num].append(dicom_info)
                
                for series_number, series_data in series_groups.items():
                    # Analyze geometry for this series
                    first_slice = series_data[0]
                    parsed_geom = self.parse_dicom_geometry(first_slice)
                    
                    # Calculate spacing from ImagePositionPatient
                    spacing_analysis = self.calculate_slice_spacing_from_ipp(series_data)
                    
                    # Determine NIfTI readiness based on critical geometry requirements
                    has_orientation = 'image_orientation_patient' in parsed_geom['parsed_geometry']
                    has_position = 'image_position_patient' in parsed_geom['parsed_geometry']
                    has_pixel_spacing = 'pixel_spacing' in parsed_geom['parsed_geometry']
                    has_slice_thickness = parsed_geom['parsed_geometry'].get('has_slice_thickness', False)
                    has_enough_slices = len(series_data) > 10  # Series must have more than 10 slices
                    
                    # Final readiness assessment - ALL criteria must be met
                    is_ready_for_nifti = (has_orientation and has_position and 
                                         has_pixel_spacing and has_slice_thickness and 
                                         has_enough_slices)
                    
                    # Create analysis row
                    row_data = {
                        'original_index': index,
                        'unique_number': study_row.get('Unique_number', ''),
                        'patient_id': study_row.get('patient_id', ''),
                        'patient_name': study_row.get('patient_name', ''),
                        'study_date': study_row.get('study_date', ''),
                        'study_modality': study_row.get('study_modality', ''),
                        'series_number': series_number,
                        'final_selection': study_row.get('final_selection', 0),
                        'total_slices': len(series_data),
                        
                        # Additional DICOM metadata from JSONL
                        'study_description': first_slice.get('study_description', ''),
                        'study_body_part': first_slice.get('study_body_part', ''),
                        'study_series_description': first_slice.get('study_series_description', ''),
                        'study_protocol': first_slice.get('study_protocol', ''),
                        'image_type': first_slice.get('image_type', ''),
                        
                        # Raw JSONL data (as strings)
                        'raw_image_orientation_patient': first_slice.get('image_orientation_patient', ''),
                        'raw_image_position_patient': first_slice.get('image_position_patient', ''),
                        'raw_pixel_spacing': first_slice.get('pixel_spacing', ''),
                        'raw_slice_thickness': first_slice.get('slice_thickness', ''),
                        'raw_slice_location': first_slice.get('slice_location', ''),
                        
                        # Parsed analysis - critical geometry requirements
                        'has_orientation': has_orientation,
                        'has_position': has_position,
                        'has_pixel_spacing': has_pixel_spacing,
                        'has_slice_thickness': has_slice_thickness,
                        'has_enough_slices': has_enough_slices,
                        
                        # Final readiness assessment
                        'is_ready_for_nifti': is_ready_for_nifti,
                        
                        # Detailed parsed values
                        'parsed_slice_thickness': f"{parsed_geom['parsed_geometry'].get('slice_thickness', 0):.3f}" if has_slice_thickness else '',
                        'parsed_pixel_spacing': f"{parsed_geom['parsed_geometry'].get('row_spacing', 0):.3f}x{parsed_geom['parsed_geometry'].get('col_spacing', 0):.3f}" if has_pixel_spacing else '',
                        
                        # Geometry validation
                        'geometry_valid': parsed_geom['validation']['is_valid'],
                        'geometry_errors': '; '.join(parsed_geom['validation']['errors']),
                        'geometry_warnings': '; '.join(parsed_geom['validation']['warnings']),
                        
                        # Expert spacing analysis from ImagePositionPatient (informational)
                        'ipp_spacing_success': spacing_analysis.get('success', False),
                        'ipp_calculated_spacing': f"{spacing_analysis.get('slice_spacing', 0):.3f}" if spacing_analysis.get('success') else '',
                        'ipp_spacing_consistent': spacing_analysis.get('spacing_consistency', False),
                        'ipp_spacing_method': spacing_analysis.get('method_used', 'fallback'),
                        'ipp_spacing_warnings': '; '.join(spacing_analysis.get('warnings', [])),
                        
                        # Source information
                        'source_folder': source_folder,
                        'source_file': source_file
                    }
                    
                    analysis_rows.append(row_data)
            
            # Create analysis DataFrame
            analysis_df = pd.DataFrame(analysis_rows)
            
            # Export to Excel
            analysis_filename = f"{excel_path.stem}_Image3Ddata.xlsx"
            analysis_path = excel_path.parent / analysis_filename
            
            with pd.ExcelWriter(analysis_path, engine='openpyxl') as writer:
                analysis_df.to_excel(writer, sheet_name='DICOM_Analysis', index=False)
                
                # Summary sheet with comprehensive statistics
                summary_data = {
                    'Metric': [
                        'Total Studies',
                        'Total Series',
                        '--- NIfTI Readiness ---',
                        'Ready for NIfTI Conversion',
                        'Series with Valid Orientation',
                        'Series with Valid Position', 
                        'Series with Valid Pixel Spacing',
                        'Series with Valid Slice Thickness',
                        'Series with Enough Slices (>10)',
                        '--- Quality Metrics ---',
                        'IPP Spacing Calculation Success',
                        'IPP Spacing Consistent',
                        'Geometry Validation Errors',
                        'Geometry Validation Warnings',
                        '--- Selection Status ---',
                        'Final Selection = 1 & NIfTI Ready'
                    ],
                    'Count': [
                        len(studies_df),
                        len(analysis_df),
                        '',  # Separator
                        len(analysis_df[analysis_df['is_ready_for_nifti'] == True]),
                        len(analysis_df[analysis_df['has_orientation'] == True]),
                        len(analysis_df[analysis_df['has_position'] == True]),
                        len(analysis_df[analysis_df['has_pixel_spacing'] == True]),
                        len(analysis_df[analysis_df['has_slice_thickness'] == True]),
                        len(analysis_df[analysis_df['has_enough_slices'] == True]),
                        '',  # Separator
                        len(analysis_df[analysis_df['ipp_spacing_success'] == True]),
                        len(analysis_df[analysis_df['ipp_spacing_consistent'] == True]),
                        len(analysis_df[analysis_df['geometry_valid'] == False]),
                        len(analysis_df[analysis_df['geometry_warnings'] != '']),
                        '',  # Separator
                        len(analysis_df[(analysis_df['final_selection'] == 1) & (analysis_df['is_ready_for_nifti'] == True)])
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            self.logger.info(f"Analysis exported to: {analysis_path}")
            self.stats['analysis_exported'] = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting analysis: {e}")
            return False
    
    def calculate_slice_spacing_from_ipp(self, series_data: List[Dict]) -> Dict[str, Any]:
        """
        Calculate slice spacing using ImagePositionPatient coordinates.
        
        Expert medical imaging approach:
        1. Parse ImagePositionPatient and ImageOrientationPatient
        2. Determine slice direction vector (normal to slice plane)
        3. Sort slices by position along slice direction
        4. Calculate spacing between consecutive slices
        5. Validate consistency across series
        
        Args:
            series_data: List of DICOM metadata for a series
            
        Returns:
            Dictionary with spacing analysis and sorted slices
        """
        spacing_analysis = {
            'success': False,
            'slice_spacing': None,
            'spacing_consistency': False,
            'sorted_slices': series_data,
            'method_used': 'fallback',
            'warnings': [],
            'spacing_details': {}
        }
        
        if len(series_data) < 2:
            spacing_analysis['warnings'].append("Need at least 2 slices for spacing calculation")
            return spacing_analysis
        
        try:
            # Parse geometry from all slices
            slice_positions = []
            valid_slices = []
            
            for slice_info in series_data:
                # Parse ImagePositionPatient
                ipp_raw = slice_info.get('image_position_patient')
                ipp_parsed = self.geometry_parser.parse_image_position_patient(ipp_raw)
                
                # Parse ImageOrientationPatient  
                iop_raw = slice_info.get('image_orientation_patient')
                iop_parsed = self.geometry_parser.parse_image_orientation_patient(iop_raw)
                
                if ipp_parsed is not None and iop_parsed is not None:
                    # Calculate slice direction (normal to slice plane)
                    row_direction = iop_parsed[:3]
                    col_direction = iop_parsed[3:6]
                    slice_direction = np.cross(row_direction, col_direction)
                    slice_direction = slice_direction / np.linalg.norm(slice_direction)
                    
                    # Project position onto slice direction to get slice coordinate
                    slice_position = np.dot(ipp_parsed, slice_direction)
                    
                    slice_positions.append(slice_position)
                    valid_slices.append(slice_info)
                else:
                    # Fallback: try using just Z-coordinate if available
                    if ipp_parsed is not None:
                        slice_positions.append(ipp_parsed[2])  # Z-coordinate
                        valid_slices.append(slice_info)
                        spacing_analysis['warnings'].append("Using Z-coordinate fallback (no orientation info)")
            
            if len(slice_positions) < 2:
                spacing_analysis['warnings'].append("Insufficient valid spatial information")
                return spacing_analysis
            
            # Sort slices by position along slice direction
            sorted_data = sorted(zip(slice_positions, valid_slices))
            sorted_positions = [pos for pos, _ in sorted_data]
            sorted_slices = [slice_info for _, slice_info in sorted_data]
            
            # Calculate spacing between consecutive slices
            spacings = []
            for i in range(len(sorted_positions) - 1):
                spacing = abs(sorted_positions[i+1] - sorted_positions[i])
                spacings.append(spacing)
            
            if spacings:
                avg_spacing = np.mean(spacings)
                std_spacing = np.std(spacings)
                min_spacing = np.min(spacings)
                max_spacing = np.max(spacings)
                
                # Check consistency (within 5% tolerance)
                consistency_threshold = 0.05 * avg_spacing
                is_consistent = std_spacing <= consistency_threshold
                
                spacing_analysis.update({
                    'success': True,
                    'slice_spacing': avg_spacing,
                    'spacing_consistency': is_consistent,
                    'sorted_slices': sorted_slices,
                    'method_used': 'image_position_patient',
                    'spacing_details': {
                        'average': avg_spacing,
                        'std_deviation': std_spacing,
                        'min_spacing': min_spacing,
                        'max_spacing': max_spacing,
                        'all_spacings': spacings,
                        'num_intervals': len(spacings),
                        'consistency_threshold': consistency_threshold
                    }
                })
                
                if not is_consistent:
                    spacing_analysis['warnings'].append(
                        f"Inconsistent spacing detected: std={std_spacing:.3f}mm, avg={avg_spacing:.3f}mm"
                    )
                
                self.logger.info(f"Calculated slice spacing: {avg_spacing:.3f}mm (std: {std_spacing:.3f}mm)")
            
        except Exception as e:
            spacing_analysis['warnings'].append(f"Error in spacing calculation: {str(e)}")
            self.logger.warning(f"Slice spacing calculation failed: {e}")
        
        return spacing_analysis
    
    def group_dicom_by_series(self, dicom_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Group DICOM data by series number with expert spatial sorting."""
        series_groups = defaultdict(list)
        
        for dicom_info in dicom_data:
            series_num = dicom_info.get('series_number', 'unknown')
            series_groups[series_num].append(dicom_info)
        
        # Sort each series using ImagePositionPatient analysis
        for series_num in series_groups:
            spacing_analysis = self.calculate_slice_spacing_from_ipp(series_groups[series_num])
            
            if spacing_analysis['success']:
                # Use spatially sorted slices
                series_groups[series_num] = spacing_analysis['sorted_slices']
                self.logger.info(f"Series {series_num}: Used ImagePositionPatient sorting, "
                               f"spacing={spacing_analysis['slice_spacing']:.3f}mm")
            else:
                # Fallback to slice_location sorting
                series_groups[series_num].sort(
                    key=lambda x: self.geometry_parser.parse_float_field(x.get('slice_location', 0)) or 0
                )
                self.logger.warning(f"Series {series_num}: Fallback to slice_location sorting")
        
        return dict(series_groups)
    
    def save_processing_results(self, excel_path: Path):
        """Save final processing results and statistics."""
        try:
            # Create results Excel
            results_filename = f"{excel_path.stem}_NIfTI_Results.xlsx"
            results_path = excel_path.parent / results_filename
            
            results_df = pd.DataFrame(self.processing_results)
            
            with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Processing_Results', index=False)
            
            # Save secure mapping
            mapping_path = self.output_dir / "mapping_secure.json"
            with open(mapping_path, 'w') as f:
                json.dump(self.secure_mapping, f, indent=2)
            
            # Save patient shifts
            shifts_path = self.output_dir / "patient_shifts_secure.json"
            with open(shifts_path, 'w') as f:
                json.dump(self.patient_shifts, f, indent=2)
            
            self.logger.info("Processing results saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving processing results: {e}")
    
    def run_pipeline(self, excel_path: Optional[str] = None, analysis_only: bool = False):
        """Execute the complete pipeline."""
        try:
            self.logger.info("Starting Expert DICOM to NIfTI Pipeline")
            
            # Select Excel file
            selected_excel = self.select_excel_file(excel_path)
            if not selected_excel:
                return
            
            # Load selected studies
            studies_df = self.load_selected_studies(selected_excel)
            if studies_df.empty:
                return
            
            # Export analysis to Excel
            print("\\n📊 Exporting DICOM analysis to Excel...")
            if self.export_analysis_to_excel(studies_df, selected_excel):
                analysis_file = selected_excel.parent / f"{selected_excel.stem}_Image3Ddata.xlsx"
                print(f"✅ Analysis exported to: {analysis_file}")
                print(f"\\n🔍 Please review the analysis file to understand your data quality.")
                
                if analysis_only:
                    print("Analysis-only mode completed.")
                    return
                    
                input("\\nPress Enter to proceed with NIfTI conversion...")
            else:
                print("❌ Failed to export analysis. Proceeding with conversion...")
            
            # Process each study
            print(f"\\n🏗️ Processing {len(studies_df)} studies...")
            
            for index, study_row in studies_df.iterrows():
                self.logger.info(f"Processing study {index + 1}/{len(studies_df)}")
                
                try:
                    # Parse JSONL and group by series
                    source_folder = study_row.get('source_folder', '')
                    source_file = study_row.get('source_file', '')
                    
                    base_dir = Path(__file__).parent.parent
                    jsonl_path = base_dir / "data" / "processed" / source_folder / "S1_indexed_metadata" / source_file
                    
                    if not jsonl_path.exists():
                        continue
                    
                    dicom_data = self.parse_jsonl_file(jsonl_path)
                    if not dicom_data:
                        continue
                    
                    series_groups = self.group_dicom_by_series(dicom_data)
                    
                    # Process each series
                    for series_number, series_data in series_groups.items():
                        result = self.process_single_series(series_data, study_row)
                        
                        # Add to results tracking
                        result_row = {
                            'original_index': index,
                            'unique_number': study_row.get('Unique_number', ''),
                            'patient_id': study_row.get('patient_id', ''),
                            'series_number': series_number,
                            'success': result['success'],
                            'error_message': result['error_message'],
                            'file_size_mb': result['file_size_bytes'] / (1024 * 1024) if result['file_size_bytes'] > 0 else 0,
                            'hash_id': result['hash_id'],
                            'processing_time_sec': result['processing_time'],
                            'geometry_warnings': '; '.join(result['geometry_warnings'])
                        }
                        self.processing_results.append(result_row)
                        
                        if result['success']:
                            self.stats['successful_conversions'] += 1
                        else:
                            self.stats['failed_conversions'] += 1
                    
                    self.stats['total_series'] += len(series_groups)
                    
                except Exception as e:
                    self.logger.error(f"Error processing study {index}: {e}")
                    continue
            
            # Save final results
            self.save_processing_results(selected_excel)
            
            # Print summary
            duration = (datetime.now() - self.stats['start_time']).total_seconds()
            print(f"\\n{'='*60}")
            print("EXPERT NIFTI CONVERSION COMPLETED")
            print(f"{'='*60}")
            print(f"Duration: {duration:.1f} seconds")
            print(f"Studies processed: {self.stats['total_studies']}")
            print(f"Successful conversions: {self.stats['successful_conversions']}")
            print(f"Failed conversions: {self.stats['failed_conversions']}")
            print(f"Total series: {self.stats['total_series']}")
            print(f"Output directory: {self.output_dir}")
            print(f"{'='*60}")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description='Expert DICOM to NIfTI Conversion Pipeline')
    parser.add_argument('--excel-file', help='Path to consolidated Excel file')
    parser.add_argument('--output-dir', default='data/nifti2share', help='Output directory')
    parser.add_argument('--analysis-only', action='store_true', help='Export analysis only, no conversion')
    
    args = parser.parse_args()
    
    processor = NiftiStoreProcessor(output_dir=args.output_dir)
    processor.run_pipeline(excel_path=args.excel_file, analysis_only=args.analysis_only)


if __name__ == "__main__":
    main()