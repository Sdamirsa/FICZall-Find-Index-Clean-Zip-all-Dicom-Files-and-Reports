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
import shutil
import hashlib
import logging
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
import pydicom
import dicom2nifti
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict
import re
import random
from tqdm import tqdm
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import base64
import zipfile
import subprocess


def convert_to_json_serializable(obj):
    """
    Convert numpy types and other non-JSON serializable types to JSON serializable equivalents.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


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
    
    def __init__(self, output_dir: str = "data/nifti2share", force_reprocess: bool = False):
        self.output_dir = Path(output_dir)
        self.force_reprocess = force_reprocess
        self.setup_directories()
        self.setup_logging()
        self.setup_password_protection()
        
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
        
        # Progress tracking for resume functionality
        self.progress_file = None
        self.progress_data = {
            'processed_series': {},
            'total_series_count': 0,
            'current_batch': 0,
            'last_update': None,
            'excel_file_path': None
        }
        self.batch_size = 30  # Save progress every 30 series
        self.series_counter = 0
    
    def check_existing_image3d_file(self, excel_path: Path) -> Optional[Path]:
        """
        Check if processed Image3Ddata file exists for the selected Excel file.
        
        Args:
            excel_path: Path to the selected Excel file
            
        Returns:
            Path to existing Image3Ddata file or None if not found
        """
        image3d_file = excel_path.parent / f"{excel_path.stem}_Image3Ddata.xlsx"
        
        if image3d_file.exists():
            return image3d_file
        return None
    
    def ask_user_skip_first_step(self, image3d_file: Path) -> bool:
        """
        Ask user if they want to use existing Image3Ddata file and skip first step.
        
        Args:
            image3d_file: Path to existing Image3Ddata file
            
        Returns:
            True if user wants to skip first step, False otherwise
        """
        print(f"\n📊 Found existing processed file: {image3d_file.name}")
        print(f"📅 File last modified: {datetime.fromtimestamp(image3d_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📏 File size: {image3d_file.stat().st_size / (1024*1024):.2f} MB")
        
        while True:
            choice = input("\n🤔 Do you want to use this existing file and skip the analysis step? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                print("✅ Using existing Image3Ddata file. Skipping analysis step...")
                return True
            elif choice in ['n', 'no']:
                print("🔄 Will regenerate analysis file...")
                return False
            else:
                print("⚠️  Please enter 'y' for yes or 'n' for no.")
    
    def setup_progress_tracking(self, excel_path: Path):
        """
        Setup progress tracking file for resume functionality.
        
        Args:
            excel_path: Path to the Excel file being processed
        """
        self.progress_file = self.output_dir / f"{excel_path.stem}_nifti_progress.json"
        self.progress_data['excel_file_path'] = str(excel_path)
        
        if self.progress_file.exists() and not self.force_reprocess:
            try:
                with open(self.progress_file, 'r') as f:
                    saved_progress = json.load(f)
                    
                # Validate that progress file matches current excel file
                if saved_progress.get('excel_file_path') == str(excel_path):
                    self.progress_data = saved_progress
                    self.logger.info(f"Loaded existing progress: {len(self.progress_data['processed_series'])} series already processed")
                else:
                    self.logger.warning("Progress file doesn't match current Excel file. Starting fresh.")

            except Exception as e:
                self.logger.error(f"Error loading progress file: {e}. Starting fresh.")

    def ask_user_resume_or_restart(self) -> bool:
        """
        Ask user if they want to resume processing or start from beginning.
        
        Returns:
            True if user wants to resume, False to restart
        """
        processed_count = len(self.progress_data['processed_series'])
        last_update = self.progress_data.get('last_update')

        print(f"\n🔄 Found existing progress file:")
        print(f"   📊 Processed series: {processed_count}")
        if last_update:
            print(f"   📅 Last update: {last_update}")
        print(f"   📂 Progress file: {self.progress_file.name}")

        while True:
            choice = input("\n🤔 Do you want to resume from where you left off? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                print("✅ Resuming from previous progress...")
                return True
            elif choice in ['n', 'no']:
                print("🔄 Starting from beginning (progress will be overwritten)...")
                # Clear progress data
                self.progress_data = {
                    'processed_series': {},
                    'total_series_count': 0,
                    'current_batch': 0,
                    'last_update': None,
                    'excel_file_path': str(self.progress_data['excel_file_path'])
                }
                return False
            else:
                print("⚠️  Please enter 'y' for yes or 'n' for no.")

    def save_progress(self):
        """
        Save current progress to file.
        """
        if self.progress_file:
            try:
                self.progress_data['last_update'] = datetime.now().isoformat()
                self.progress_data['current_batch'] = self.series_counter // self.batch_size + 1
                
                with open(self.progress_file, 'w') as f:
                    json.dump(self.progress_data, f, indent=2)
                    
                self.logger.info(f"Progress saved: {len(self.progress_data['processed_series'])} series processed")
                
            except Exception as e:
                self.logger.error(f"Error saving progress: {e}")
    
    def is_series_already_processed(self, hash_id: str) -> bool:
        """
        Check if a series has already been processed based on hash_id.
        
        Args:
            hash_id: Unique identifier for the series
            
        Returns:
            True if already processed, False otherwise
        """
        return hash_id in self.progress_data['processed_series']
    
    def mark_series_processed(self, hash_id: str, success: bool, error_msg: str = None):
        """
        Mark a series as processed in progress tracking.
        
        Args:
            hash_id: Unique identifier for the series
            success: Whether processing was successful
            error_msg: Error message if processing failed
        """
        self.progress_data['processed_series'][hash_id] = {
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'error_message': error_msg if not success else None
        }
        
        self.series_counter += 1
        
        # Save progress every batch_size series
        if self.series_counter % self.batch_size == 0:
            self.save_progress()
            self.update_nifti_excel()
            print(f"\n💾 Progress saved after {self.series_counter} series (batch {self.series_counter // self.batch_size})")
    
    def create_nifti_excel_at_start(self, excel_path: Path, studies_df: pd.DataFrame):
        """
        Create the NIfTI processing Excel file at the beginning with all series to be processed.
        
        Args:
            excel_path: Original Excel file path
            studies_df: DataFrame with studies data
        """
        nifti_excel_path = self.output_dir / f"{excel_path.stem}_nifti_results.xlsx"

        # Create initial DataFrame with all series
        all_series_data = []
        
        for index, study_row in studies_df.iterrows():
            source_file = study_row.get('source_file', '')
            source_folder = study_row.get('source_folder', '')
            
            if source_file:
                # Try to load JSONL file to get series information
                jsonl_path = Path(source_folder) / source_file if source_folder else Path(source_file)
                
                try:
                    dicom_data = self.parse_jsonl_file(jsonl_path)
                    series_groups = self.group_dicom_by_series(dicom_data)
                    
                    for series_number, series_data in series_groups.items():
                        hash_id = self.generate_study_hash(
                            study_row.get('patient_id', ''),
                            study_row.get('study_date', ''),
                            str(series_number),
                            study_row.get('institution_name', '')
                        )
                        
                        series_info = {
                            'hash_id': hash_id,
                            'patient_id': study_row.get('patient_id', ''),
                            'study_date': study_row.get('study_date', ''),
                            'series_number': series_number,
                            'slice_count': len(series_data),
                            'institution_name': study_row.get('institution_name', ''),
                            'source_file': source_file,
                            'source_folder': source_folder,
                            'processing_status': 'pending',
                            'success': None,
                            'error_message': None,
                            'processing_timestamp': None,
                            'nifti_file_path': None
                        }
                        
                        all_series_data.append(series_info)
                        
                except Exception as e:
                    self.logger.error(f"Error reading JSONL file {jsonl_path}: {e}")
        
        # Create DataFrame and save to Excel
        if all_series_data:
            nifti_df = pd.DataFrame(all_series_data)
            
            try:
                with pd.ExcelWriter(nifti_excel_path, engine='openpyxl') as writer:
                    nifti_df.to_excel(writer, sheet_name='NIfTI_Processing', index=False)
                    
                    # Add summary sheet
                    summary_data = {
                        'Metric': ['Total Series', 'Pending', 'Successful', 'Failed', 'Processing Rate'],
                        'Value': [len(nifti_df), len(nifti_df), 0, 0, '0%']
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)

                self.logger.info(f"Initial NIfTI processing Excel created: {nifti_excel_path}")
                print(f"📊 Created NIfTI processing tracker: {nifti_excel_path.name}")
                print(f"   📈 Total series to process: {len(nifti_df)}")

                # Store reference for later updates
                self.nifti_excel_path = nifti_excel_path
                self.progress_data['total_series_count'] = len(nifti_df)

            except Exception as e:
                self.logger.error(f"Error creating NIfTI Excel file: {e}")

    def update_nifti_excel(self):
        """Update the NIfTI Excel file with current progress."""
        if not hasattr(self, 'nifti_excel_path') or not self.nifti_excel_path.exists():
            return
            
        try:
            # Read current Excel file
            nifti_df = pd.read_excel(self.nifti_excel_path, sheet_name='NIfTI_Processing')
            
            # Update status for processed series
            for hash_id, progress_info in self.progress_data['processed_series'].items():
                mask = nifti_df['hash_id'] == hash_id
                if mask.any():
                    nifti_df.loc[mask, 'processing_status'] = 'completed'
                    nifti_df.loc[mask, 'success'] = progress_info['success']
                    nifti_df.loc[mask, 'processing_timestamp'] = progress_info['timestamp']
                    
                    if not progress_info['success'] and progress_info.get('error_message'):
                        nifti_df.loc[mask, 'error_message'] = progress_info['error_message']
            
            # Save updated Excel file
            with pd.ExcelWriter(self.nifti_excel_path, engine='openpyxl') as writer:
                nifti_df.to_excel(writer, sheet_name='NIfTI_Processing', index=False)
                
                # Update summary
                total_series = len(nifti_df)
                completed_series = len(self.progress_data['processed_series'])
                successful_series = sum(1 for p in self.progress_data['processed_series'].values() if p['success'])
                failed_series = completed_series - successful_series
                pending_series = total_series - completed_series
                processing_rate = f"{(completed_series/total_series*100):.1f}%" if total_series > 0 else "0%"
                
                summary_data = {
                    'Metric': ['Total Series', 'Pending', 'Successful', 'Failed', 'Processing Rate'],
                    'Value': [total_series, pending_series, successful_series, failed_series, processing_rate]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

            self.logger.info(f"NIfTI Excel file updated: {completed_series}/{total_series} series processed")

        except Exception as e:
            self.logger.error(f"Error updating NIfTI Excel file: {e}")

    def setup_directories(self):
        """Create necessary output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
    
    def setup_logging(self):
        """Setup logging configuration with reduced verbosity for third-party libraries."""
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler for detailed logs
        log_file_path = self.output_dir / f'processing_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # Console handler for user feedback (less verbose)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Suppress verbose third-party library logging
        logging.getLogger('dicom2nifti').setLevel(logging.WARNING)
        logging.getLogger('pydicom').setLevel(logging.WARNING)
        logging.getLogger('nibabel').setLevel(logging.WARNING)
        
        # Our main logger
        self.logger = logging.getLogger(__name__)
        
        # Test password protection on startup
        self._test_password_protection()
    
    def setup_password_protection(self):
        """Setup password protection for sensitive data using .env password."""
        # Load .env file
        load_dotenv()
        
        # Get password from environment
        self.zip_password = os.getenv('Sensitive_data_zip_password')
        
        if not self.zip_password:
            self.logger.warning("No 'Sensitive_data_zip_password' found in .env file")
            self._generate_zip_password()
    
    def _generate_zip_password(self):
        """Generate a new ZIP password and save to .env."""
        # Generate a strong password
        import secrets
        import string
        alphabet = string.ascii_letters + string.digits + "!@#$%&*"
        self.zip_password = ''.join(secrets.choice(alphabet) for _ in range(16))
        
        # Write to .env file
        env_path = Path('.env')
        
        try:
            # Read existing .env content
            existing_content = ""
            if env_path.exists():
                with open(env_path, 'r') as f:
                    existing_content = f.read()
            
            # Check if password already exists in file
            if 'Sensitive_data_zip_password' not in existing_content:
                # Append new password
                with open(env_path, 'a') as f:
                    if existing_content and not existing_content.endswith('\n'):
                        f.write('\n')
                    f.write(f'Sensitive_data_zip_password={self.zip_password}\n')
                
                self.logger.info(f"Generated new ZIP password and saved to .env file")
            else:
                self.logger.info("ZIP password already exists in .env file")
                
        except Exception as e:
            self.logger.error(f"Failed to save ZIP password to .env: {e}")
    
    def _test_password_protection(self):
        """Test password protection capability on startup."""
        try:
            test_content = "Test sensitive data"
            test_file = self.output_dir / "test_protection.zip"
            
            success = self.create_password_protected_zip(test_file, test_content, "test123")
            
            if success and test_file.exists():
                self.logger.info("Password protection test: PASSED")
                # Clean up test file
                test_file.unlink()
                # Also clean up any fallback files
                for ext in ['.encrypted.txt', '.info.txt']:
                    fallback_file = test_file.with_suffix(ext)
                    if fallback_file.exists():
                        fallback_file.unlink()
            else:
                self.logger.warning("Password protection test: FAILED - check 7-Zip installation or pyminizip")
                
        except Exception as e:
            self.logger.warning(f"Password protection test failed: {e}")
    
    def create_password_protected_zip(self, file_path: Path, content: str, password: str = None) -> bool:
        """Create a password-protected ZIP file with the given content using multiple methods."""
        try:
            if password is None:
                password = self.zip_password
            
            # Create a temporary text file with the content
            temp_txt_path = file_path.with_suffix('.tmp.txt')
            with open(temp_txt_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            success = False
            
            # Method 1: Try 7-Zip (if available)
            try:
                # Look for 7-Zip in common locations
                possible_7zip_paths = [
                    "7z",  # If in PATH
                    "C:\\Program Files\\7-Zip\\7z.exe",
                    "C:\\Program Files (x86)\\7-Zip\\7z.exe"
                ]
                
                for zip_exe in possible_7zip_paths:
                    try:
                        # Test if 7-Zip is available
                        result = subprocess.run([zip_exe], capture_output=True, timeout=2)
                        
                        # Create password-protected ZIP with 7-Zip
                        cmd = [
                            zip_exe, "a", "-tzip",  # Add to ZIP format
                            f"-p{password}",        # Set password
                            "-mx=9",               # Maximum compression
                            str(file_path),        # Output ZIP file
                            str(temp_txt_path)     # Input file
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            success = True
                            self.logger.info(f"Created password-protected ZIP using 7-Zip")
                            break
                        else:
                            self.logger.warning(f"7-Zip failed: {result.stderr}")
                            
                    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                        continue
                        
            except Exception as e:
                self.logger.warning(f"7-Zip method failed: {e}")
            
            # Method 2: Fall back to pyminizip (if available)
            if not success:
                try:
                    import pyminizip
                    
                    # Create password-protected ZIP with pyminizip
                    pyminizip.compress(
                        str(temp_txt_path),    # Source file
                        None,                  # Path prefix in ZIP (None = use filename)
                        str(file_path),        # Output ZIP file
                        password,              # Password
                        9                      # Compression level (0-9)
                    )
                    
                    success = True
                    self.logger.info(f"Created password-protected ZIP using pyminizip")
                    
                except ImportError:
                    self.logger.warning("pyminizip not available. Install with: pip install pyminizip")
                except Exception as e:
                    self.logger.warning(f"pyminizip method failed: {e}")
            
            # Method 3: Fall back to encryption (if ZIP protection fails)
            if not success:
                self.logger.warning("ZIP password protection failed, falling back to Fernet encryption")
                
                # Use Fernet encryption as fallback
                from cryptography.fernet import Fernet
                import base64
                
                # Generate key from password
                key = base64.urlsafe_b64encode(password.encode().ljust(32)[:32])
                cipher_suite = Fernet(key)
                
                # Encrypt the content
                encrypted_content = cipher_suite.encrypt(content.encode())
                
                # Save as encrypted text file instead of ZIP
                encrypted_file_path = file_path.with_suffix('.encrypted.txt')
                with open(encrypted_file_path, 'wb') as f:
                    f.write(encrypted_content)
                
                # Create info file
                info_file_path = file_path.with_suffix('.info.txt')
                with open(info_file_path, 'w', encoding='utf-8') as f:
                    f.write("ENCRYPTED PATIENT DATA - FERNET ENCRYPTION\\n")
                    f.write("=" * 50 + "\\n")
                    f.write("This file contains Fernet-encrypted patient data.\\n")
                    f.write(f"Encrypted file: {encrypted_file_path.name}\\n")
                    f.write("Password is stored in .env file as 'Sensitive_data_zip_password'\\n")
                    f.write("\\n")
                    f.write("To decrypt:\\n")
                    f.write("1. Use the provided password with Fernet decryption\\n")
                    f.write("2. Or use the extract_sensitive_zip method in this tool\\n")
                    f.write("\\n")
                    f.write(f"Created: {datetime.now().isoformat()}\\n")
                
                success = True
                self.logger.info(f"Created Fernet-encrypted file as fallback")
            
            # Remove temporary file
            if temp_txt_path.exists():
                temp_txt_path.unlink()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to create password-protected file: {e}")
            if temp_txt_path.exists():
                temp_txt_path.unlink()
            return False
    
    def extract_sensitive_zip(self, zip_file_path: Path, password: str = None) -> Dict[str, Any]:
        """
        Extract a password-protected ZIP file and return the original metadata.
        
        Args:
            zip_file_path: Path to SENSITIVE_*.zip file
            password: Password for the ZIP file (uses self.zip_password if None)
            
        Returns:
            Dictionary with sensitive metadata
        """
        try:
            if password is None:
                password = self.zip_password
            
            with zipfile.ZipFile(zip_file_path, 'r') as zipf:
                zipf.setpassword(password.encode())
                
                # Get the first file in the ZIP (should only be one)
                file_names = zipf.namelist()
                if not file_names:
                    self.logger.error("ZIP file is empty")
                    return {}
                
                # Extract and read the content
                content = zipf.read(file_names[0]).decode('utf-8')
                
                # Parse JSON
                metadata = json.loads(content)
                
                return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting sensitive ZIP file: {e}")
            return {}
    
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
    
    def create_nifti_with_dicom2nifti(self, series_data: List[Dict], output_path: Path) -> Tuple[bool, int, Dict[str, Any]]:
        """
        Create NIfTI file using dicom2nifti library for robust conversion.
        
        This method uses the dicom2nifti library which handles:
        - Proper DICOM series organization
        - Automatic orientation handling
        - Slice ordering and spacing
        - Medical imaging standards compliance
        
        Args:
            series_data: List of DICOM metadata for the series
            output_path: Output path for NIfTI file
            
        Returns:
            Tuple of (success, file_size_bytes, conversion_info)
        """
        conversion_info = {
            'method': 'dicom2nifti',
            'dicom_files_count': len(series_data),
            'conversion_warnings': [],
            'conversion_errors': []
        }
        
        try:
            # Extract DICOM file paths
            dicom_files = [info['file_path'] for info in series_data]
            
            # Verify all DICOM files exist
            existing_files = []
            missing_files = []
            
            for file_path in dicom_files:
                if os.path.exists(file_path):
                    existing_files.append(file_path)
                else:
                    missing_files.append(file_path)
            
            if missing_files:
                conversion_info['conversion_warnings'].append(f"Missing {len(missing_files)} DICOM files")
                self.logger.warning(f"Missing {len(missing_files)} DICOM files from series")
            
            if not existing_files:
                conversion_info['conversion_errors'].append("No valid DICOM files found")
                return False, 0, conversion_info
            
            # Create temporary directory for DICOM files (dicom2nifti expects directory input)
            temp_dicom_dir = output_path.parent / f"temp_dicom_{output_path.stem}"
            temp_dicom_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Copy DICOM files to temporary directory with progress bar
                # (dicom2nifti works best with files in a single directory)
                with tqdm(existing_files, desc=f"Preparing DICOM files", leave=False, disable=len(existing_files) < 50) as pbar:
                    for i, dicom_file in enumerate(pbar):
                        if os.path.exists(dicom_file):
                            temp_file_name = f"slice_{i:04d}.dcm"
                            temp_file_path = temp_dicom_dir / temp_file_name
                            shutil.copy2(dicom_file, temp_file_path)
                
                # Create output directory for dicom2nifti
                output_dir = output_path.parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Use dicom2nifti to convert the series
                self.logger.info(f"Converting {len(existing_files)} DICOM files using dicom2nifti")
                
                # Temporarily suppress dicom2nifti logging during conversion
                dicom2nifti_logger = logging.getLogger('dicom2nifti')
                original_level = dicom2nifti_logger.level
                dicom2nifti_logger.setLevel(logging.ERROR)
                
                try:
                    # Convert using dicom2nifti with suppressed logging
                    with tqdm(total=1, desc="Converting to NIfTI", leave=False) as pbar:
                        dicom2nifti.convert_directory(
                            str(temp_dicom_dir), 
                            str(output_dir),
                            compression=True,  # Create .nii.gz files
                            reorient=True      # Apply proper orientation
                        )
                        pbar.update(1)
                finally:
                    # Restore original logging level
                    dicom2nifti_logger.setLevel(original_level)
                
                # Find the created NIfTI file (dicom2nifti creates its own filename)
                created_nifti_files = list(output_dir.glob("*.nii.gz"))
                if not created_nifti_files:
                    created_nifti_files = list(output_dir.glob("*.nii"))
                
                if created_nifti_files:
                    created_file = created_nifti_files[0]
                    
                    # Handle file renaming with retry logic for Windows file locking issues
                    final_file_path = output_path
                    if created_file != output_path:
                        import time
                        max_retries = 5
                        retry_delay = 0.5
                        
                        for attempt in range(max_retries):
                            try:
                                # Ensure no processes are holding the file
                                time.sleep(retry_delay)
                                created_file.rename(output_path)
                                break
                            except OSError as e:
                                if attempt < max_retries - 1:
                                    self.logger.warning(f"File rename attempt {attempt + 1} failed, retrying: {e}")
                                    retry_delay *= 1.5  # Exponential backoff
                                else:
                                    # If renaming fails, use the file as-is
                                    self.logger.warning(f"Could not rename {created_file.name} to {output_path.name}, using original name")
                                    final_file_path = created_file
                    else:
                        final_file_path = created_file
                    
                    # Get file size
                    file_size = final_file_path.stat().st_size
                    
                    # Load the created NIfTI to verify and get info
                    nifti_img = nib.load(str(final_file_path))
                    conversion_info.update({
                        'output_shape': convert_to_json_serializable(nifti_img.shape),
                        'output_dtype': str(nifti_img.get_fdata().dtype),
                        'voxel_spacing': convert_to_json_serializable(nifti_img.header.get_zooms()[:3]),
                        'orientation': 'dicom2nifti_standard',
                        'final_filename': final_file_path.name
                    })
                    
                    self.logger.info(f"Successfully created NIfTI file: {final_file_path.name}")
                    self.logger.info(f"  Shape: {nifti_img.shape}")
                    self.logger.info(f"  Voxel spacing: {nifti_img.header.get_zooms()[:3]}")
                    self.logger.info(f"  File size: {file_size / 1024 / 1024:.1f} MB")
                    
                    return True, file_size, conversion_info
                    
                else:
                    conversion_info['conversion_errors'].append("dicom2nifti did not create any NIfTI files")
                    self.logger.error("dicom2nifti conversion failed - no output files created")
                    return False, 0, conversion_info
                    
            finally:
                # Clean up temporary directory
                if temp_dicom_dir.exists():
                    shutil.rmtree(temp_dicom_dir)
                    
        except Exception as e:
            error_msg = f"dicom2nifti conversion error: {str(e)}"
            conversion_info['conversion_errors'].append(error_msg)
            self.logger.error(error_msg)
            return False, 0, conversion_info

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
    
    def check_series_already_processed(self, hash_id: str) -> bool:
        """
        Check if a series has already been successfully processed.
        
        Args:
            hash_id: Hash ID of the series to check
            
        Returns:
            True if series is already processed (NIfTI + metadata files exist)
        """
        try:
            # Check for both regular and Er_ prefixed versions
            for prefix in ["", "Er_"]:
                check_hash_id = f"{prefix}{hash_id}" if prefix else hash_id
                series_dir = self.output_dir / check_hash_id
                
                if series_dir.exists():
                    # Check for required files
                    nifti_file = None
                    json_file = series_dir / f"{check_hash_id}.json"
                    patient_data_zip = series_dir / f"PATIENT_DATA_{check_hash_id}.zip"
                    
                    # Look for NIfTI file (might have different names due to dicom2nifti)
                    nifti_files = list(series_dir.glob("*.nii.gz")) + list(series_dir.glob("*.nii"))
                    if nifti_files:
                        nifti_file = nifti_files[0]
                    
                    # Check if all required files exist
                    if nifti_file and nifti_file.exists() and json_file.exists() and patient_data_zip.exists():
                        self.logger.info(f"Series {check_hash_id} already processed - skipping")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking if series {hash_id} is processed: {e}")
            return False

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
            'geometry_warnings': [],
            'geometry_errors': []
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
            
            # Generate hash ID first (before checking critical errors)
            first_slice = series_data[0]
            base_hash_id = self.generate_study_hash(
                patient_id=first_slice.get('patient_id', ''),
                study_date=first_slice.get('study_date', ''),
                series_number=first_slice.get('series_number', ''),
                institution=first_slice.get('institution_name', '')
            )
            
            if critical_errors:
                result['error_message'] = f"Critical geometry errors: {'; '.join(critical_errors)}"
                result['hash_id'] = f"Er_{base_hash_id}"
                # Mark as failed with geometry errors
                self.mark_series_processed(result['hash_id'], False, result['error_message'])
                return result
            
            # Check if this series has already been processed using our new progress tracking
            if not self.force_reprocess and self.is_series_already_processed(base_hash_id):
                progress_info = self.progress_data['processed_series'][base_hash_id]
                result['success'] = progress_info['success']
                result['hash_id'] = base_hash_id
                result['processing_time'] = (datetime.now() - start_time).total_seconds()
                result['file_size_bytes'] = 0  # Unknown for existing files
                result['nifti_path'] = "already_processed"
                result['error_message'] = f"Skipped - already processed ({'success' if progress_info['success'] else 'failed'})"
                
                # Update stats for existing files
                if progress_info['success']:
                    self.stats['successful_conversions'] += 1
                else:
                    self.stats['failed_conversions'] += 1
                
                return result
            
            # Also check the older method for backward compatibility
            elif not self.force_reprocess and self.check_series_already_processed(base_hash_id):
                result['success'] = True
                result['hash_id'] = base_hash_id  # Will be updated if Er_ version exists
                result['processing_time'] = (datetime.now() - start_time).total_seconds()
                result['file_size_bytes'] = 0  # Unknown for existing files
                result['nifti_path'] = "already_processed"
                result['error_message'] = "Skipped - already processed (legacy check)"
                
                # Update stats for existing files
                self.stats['successful_conversions'] += 1
                
                return result
            
            # Track actual errors (not warnings) for Er_ prefix determination
            has_errors = bool(result['geometry_errors'])  # Only track errors, not warnings
            hash_id = base_hash_id  # Start without prefix
            
            result['hash_id'] = hash_id
            
            # Create output directory
            series_output_dir = self.output_dir / hash_id
            series_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate spacing from ImagePositionPatient for expert medical imaging
            spacing_analysis = self.calculate_slice_spacing_from_ipp(series_data)
            if spacing_analysis['warnings']:
                result['geometry_warnings'].extend(spacing_analysis['warnings'])
                # This is a warning, not an error - don't set has_errors = True
            
            # Er_ prefix is only applied for actual ERRORS, not warnings
            # (will be checked again after conversion attempts)
            
            # Create NIfTI file using dicom2nifti library
            nifti_path = series_output_dir / f"{hash_id}.nii.gz"
            save_success, file_size, conversion_info = self.create_nifti_with_dicom2nifti(series_data, nifti_path)
            
            # Handle conversion warnings and errors
            if conversion_info.get('conversion_warnings'):
                result['geometry_warnings'].extend(conversion_info['conversion_warnings'])
                # Warnings don't trigger Er_ prefix - only actual errors do
            
            if conversion_info.get('conversion_errors'):
                result['geometry_errors'].extend(conversion_info['conversion_errors'])
                has_errors = True
            
            # Apply Er_ prefix only for actual errors or conversion failures
            if not save_success or has_errors:
                if not hash_id.startswith("Er_"):
                    # Move files to new Er_ directory
                    old_hash_id = hash_id
                    hash_id = f"Er_{base_hash_id}"
                    result['hash_id'] = hash_id
                    
                    # Create new directory with Er_ prefix
                    new_series_output_dir = self.output_dir / hash_id
                    new_series_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Move any existing files to the new directory
                    if series_output_dir.exists():
                        for file_path in series_output_dir.iterdir():
                            if file_path.is_file():
                                new_file_path = new_series_output_dir / file_path.name.replace(old_hash_id, hash_id)
                                try:
                                    file_path.rename(new_file_path)
                                except OSError:
                                    pass  # File might be in use, continue
                        # Remove old directory if empty
                        try:
                            series_output_dir.rmdir()
                        except OSError:
                            pass  # Directory not empty or in use, leave it
                    
                    series_output_dir = new_series_output_dir
                    nifti_path = series_output_dir / f"{hash_id}.nii.gz"
                
                if not save_success:
                    error_messages = conversion_info.get('conversion_errors', ['Unknown conversion error'])
                    result['error_message'] = f"dicom2nifti conversion failed: {'; '.join(error_messages)}"
                    # Mark as failed in progress tracking
                    self.mark_series_processed(hash_id, False, result['error_message'])
                    self.logger.error(f"dicom2nifti conversion failed for {hash_id}")
                    return result
                else:
                    self.logger.warning(f"Series {hash_id} has errors but conversion succeeded")
            
            result['nifti_path'] = str(nifti_path)
            result['file_size_bytes'] = file_size
            
            # Save metadata files (using conversion_info instead of pixel_metadata)
            self.save_metadata_files(series_output_dir, hash_id, parsed_geometries, 
                                   conversion_info, study_row, first_slice)
            
            # Update secure mapping
            self.update_secure_mapping(hash_id, first_slice, study_row)
            
            result['success'] = True
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            # Mark series as successfully processed in progress tracking
            self.mark_series_processed(hash_id, True)
            
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
            
            # Mark as failed in progress tracking
            self.mark_series_processed(result['hash_id'], False, result['error_message'])
            
            self.logger.error(f"Unexpected error processing series {result['hash_id']}: {e}")
            return result
    
    def save_metadata_files(self, output_dir: Path, hash_id: str, geometries: List[Dict], 
                          conversion_info: Dict, study_row: pd.Series, first_slice: Dict):
        """Save metadata files: minimal sharing JSON + comprehensive patient data in protected ZIP."""
        try:
            # 1. MINIMAL SHARING JSON (for public sharing - spatial/orientation info only)
            sharing_metadata = {
                'hash_id': hash_id,
                'processing_timestamp': datetime.now().isoformat(),
                
                # Essential NIfTI conversion info
                'nifti_conversion': {
                    'shape': conversion_info.get('output_shape'),
                    'dtype': conversion_info.get('output_dtype'),
                    'method': conversion_info.get('method', 'dicom2nifti'),
                    'voxel_spacing': conversion_info.get('voxel_spacing'),
                    'orientation': conversion_info.get('orientation', 'dicom2nifti_standard')
                },
                
                # Essential spatial parameters only
                'spatial_parameters': {
                    'slice_thickness': first_slice.get('slice_thickness', ''),
                    'pixel_spacing': first_slice.get('pixel_spacing', ''),
                    'image_orientation_patient': first_slice.get('image_orientation_patient', ''),
                    'image_position_patient': first_slice.get('image_position_patient', ''),
                    'total_slices': len(geometries),
                    'rows': first_slice.get('rows', ''),
                    'columns': first_slice.get('columns', ''),
                    'modality': first_slice.get('study_modality', '')
                },
                
                # Technical imaging parameters
                'imaging_parameters': {
                    'bits_allocated': first_slice.get('bits_allocated', ''),
                    'bits_stored': first_slice.get('bits_stored', ''),
                    'pixel_representation': first_slice.get('pixel_representation', ''),
                    'rescale_slope': first_slice.get('rescale_slope', ''),
                    'rescale_intercept': first_slice.get('rescale_intercept', ''),
                    'image_type': first_slice.get('image_type', '')
                },
                
                # Geometry validation results
                'geometry_validation': {
                    'has_orientation': len([g for g in geometries if 'image_orientation_patient' in g['parsed_geometry']]) > 0,
                    'has_position': len([g for g in geometries if 'image_position_patient' in g['parsed_geometry']]) > 0,
                    'has_pixel_spacing': len([g for g in geometries if 'pixel_spacing' in g['parsed_geometry']]) > 0,
                    'has_slice_thickness': len([g for g in geometries if g['parsed_geometry'].get('has_slice_thickness', False)]) > 0,
                    'validation_warnings_count': sum(len(g['validation']['warnings']) for g in geometries),
                    'validation_errors_count': sum(len(g['validation']['errors']) for g in geometries)
                }
            }
            
            # Save minimal sharing JSON file
            sharing_json_path = output_dir / f"{hash_id}.json"
            sharing_metadata_serializable = convert_to_json_serializable(sharing_metadata)
            with open(sharing_json_path, 'w', encoding='utf-8') as f:
                json.dump(sharing_metadata_serializable, f, indent=2, ensure_ascii=False)
            
            # 2. COMPREHENSIVE PATIENT DATA JSON (includes ALL information)
            comprehensive_metadata = {
                # Basic identification
                'hash_id': hash_id,
                'processing_timestamp': datetime.now().isoformat(),
                
                # Patient Information (Original)
                'patient_information': {
                    'patient_id': first_slice.get('patient_id', ''),
                    'patient_name': first_slice.get('patient_name', ''),
                    'patient_age': first_slice.get('patient_age', ''),
                    'shifted_study_date': self.apply_date_shift(first_slice.get('study_date', ''), first_slice.get('patient_id', '')),
                    'shifted_study_time': first_slice.get('study_time', ''),
                    'shifted_patient_age': self.apply_age_jitter(first_slice.get('patient_age', ''), first_slice.get('patient_id', '')),
                    'original_study_date': first_slice.get('study_date', ''),
                    'original_study_time': first_slice.get('study_time', '')
                },
                
                # Institution and Study Information
                'study_information': {
                    'study_description': first_slice.get('study_description', ''),
                    'series_description': first_slice.get('study_series_description', first_slice.get('series_description', '')),
                    'study_modality': first_slice.get('study_modality', ''),
                    'study_body_part': first_slice.get('study_body_part', ''),
                    'study_protocol': first_slice.get('study_protocol', ''),
                    'study_comments': first_slice.get('study_comments', ''),
                    'institution_name': first_slice.get('institution_name', ''),
                    'station_name': first_slice.get('station_name', ''),
                    'series_number': first_slice.get('series_number', '')
                },
                
                # Complete NIfTI conversion info
                'nifti_conversion': {
                    'shape': conversion_info.get('output_shape'),
                    'dtype': conversion_info.get('output_dtype'),
                    'method': conversion_info.get('method', 'dicom2nifti'),
                    'voxel_spacing': conversion_info.get('voxel_spacing'),
                    'orientation': conversion_info.get('orientation', 'dicom2nifti_standard'),
                    'warnings': conversion_info.get('conversion_warnings', []),
                    'errors': conversion_info.get('conversion_errors', [])
                },
                
                # Complete technical DICOM parameters
                'technical_parameters': {
                    'slice_thickness': first_slice.get('slice_thickness', ''),
                    'pixel_spacing': first_slice.get('pixel_spacing', ''),
                    'image_orientation_patient': first_slice.get('image_orientation_patient', ''),
                    'image_position_patient': first_slice.get('image_position_patient', ''),
                    'slice_location': first_slice.get('slice_location', ''),
                    'rows': first_slice.get('rows', ''),
                    'columns': first_slice.get('columns', ''),
                    'bits_allocated': first_slice.get('bits_allocated', ''),
                    'bits_stored': first_slice.get('bits_stored', ''),
                    'pixel_representation': first_slice.get('pixel_representation', ''),
                    'rescale_slope': first_slice.get('rescale_slope', ''),
                    'rescale_intercept': first_slice.get('rescale_intercept', ''),
                    'window_center': first_slice.get('window_center', ''),
                    'window_width': first_slice.get('window_width', ''),
                    'image_type': first_slice.get('image_type', '')
                },
                
                # Detailed geometry analysis
                'geometry_analysis': {
                    'has_orientation': len([g for g in geometries if 'image_orientation_patient' in g['parsed_geometry']]) > 0,
                    'has_position': len([g for g in geometries if 'image_position_patient' in g['parsed_geometry']]) > 0,
                    'has_pixel_spacing': len([g for g in geometries if 'pixel_spacing' in g['parsed_geometry']]) > 0,
                    'has_slice_thickness': len([g for g in geometries if g['parsed_geometry'].get('has_slice_thickness', False)]) > 0,
                    'parsed_pixel_spacing': geometries[0]['parsed_geometry'].get('pixel_spacing'),
                    'parsed_slice_thickness': geometries[0]['parsed_geometry'].get('slice_thickness'),
                    'validation_warnings': sum(len(g['validation']['warnings']) for g in geometries),
                    'validation_errors': sum(len(g['validation']['errors']) for g in geometries),
                    'geometry_warnings': [w for g in geometries for w in g['validation']['warnings']],
                    'geometry_errors': [e for g in geometries for e in g['validation']['errors']]
                },
                
                # Source file information
                'source_information': {
                    'source_folder': study_row.get('source_folder', ''),
                    'source_file': study_row.get('source_file', ''),
                    'dicom_folder_path': study_row.get('dicom_folder_path', ''),
                    'original_unique_number': study_row.get('Unique_number', ''),
                    'dicom_file_paths': [g.get('original_data', {}).get('file_path', '') for g in geometries if g.get('original_data', {}).get('file_path')],
                    'total_dicom_files': len(geometries),
                    'total_files_in_series': len(geometries)
                },
                
                # All additional DICOM fields
                'additional_dicom_fields': {
                    k: v for k, v in first_slice.items() 
                    if not k.startswith('original_') 
                    and v not in ['', None, 'Unknown']
                },
                
                # Metadata
                'data_classification': {
                    'protection_method': 'ZIP_Password',
                    'data_type': 'COMPREHENSIVE_PATIENT_DATA',
                    'contains_phi': True,
                    'created_timestamp': datetime.now().isoformat()
                }
            }
            
            # Convert comprehensive patient data to JSON and create password-protected ZIP
            comprehensive_metadata_serializable = convert_to_json_serializable(comprehensive_metadata)
            comprehensive_json_str = json.dumps(comprehensive_metadata_serializable, indent=2, ensure_ascii=False)
            
            # Save comprehensive patient data in password-protected ZIP file
            patient_data_zip_path = output_dir / f"PATIENT_DATA_{hash_id}.zip"
            success = self.create_password_protected_zip(patient_data_zip_path, comprehensive_json_str)
            
            if success:
                # Create an info file with instructions
                info_path = output_dir / f"PATIENT_DATA_{hash_id}_INFO.txt"
                with open(info_path, 'w', encoding='utf-8') as f:
                    f.write("PASSWORD-PROTECTED COMPREHENSIVE PATIENT DATA\\n")
                    f.write("=" * 50 + "\\n")
                    f.write("This file contains comprehensive patient and study information.\\n")
                    f.write(f"Protected file: PATIENT_DATA_{hash_id}.zip (or .encrypted.txt if ZIP failed)\\n")
                    f.write("Password is stored in .env file as 'Sensitive_data_zip_password'\\n")
                    f.write("\\n")
                    f.write("Contents: Complete patient data, study information,\\n")
                    f.write("technical parameters, and source file details\\n")
                    f.write("\\n")
                    f.write("Protection Methods (tried in order):\\n")
                    f.write("1. 7-Zip password-protected ZIP (if 7-Zip available)\\n")
                    f.write("2. pyminizip password-protected ZIP (if pyminizip installed)\\n")
                    f.write("3. Fernet encryption (fallback)\\n")
                    f.write("\\n")
                    f.write("To open on Windows/Mac:\\n")
                    f.write("- ZIP files: Double-click -> Enter password when prompted\\n")
                    f.write("- Encrypted files: Use this tool's extraction method\\n")
                    f.write("\\n")
                    f.write(f"Hash ID: {hash_id}\\n")
                    f.write(f"Created: {datetime.now().isoformat()}\\n")
                    f.write(f"Security: Multi-layer password protection\\n")
            
            self.logger.info(f"Saved metadata files for {hash_id}: minimal sharing JSON and comprehensive patient data ZIP")
            
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
            # Create index-based sorting to avoid comparing dictionaries
            indexed_data = list(enumerate(zip(slice_positions, valid_slices)))
            sorted_indexed_data = sorted(indexed_data, key=lambda x: x[1][0])  # Sort by position only
            sorted_positions = [pos for _, (pos, _) in sorted_indexed_data]
            sorted_slices = [slice_info for _, (_, slice_info) in sorted_indexed_data]
            
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
        """Save final processing results and statistics with iterative updates."""
        try:
            # Create results Excel with comprehensive tracking
            results_filename = f"{excel_path.stem}_NIfTI_Results.xlsx"
            results_path = excel_path.parent / results_filename
            
            results_df = pd.DataFrame(self.processing_results)
            
            # Add processing summary columns
            if not results_df.empty:
                results_df['output_directory'] = str(self.output_dir)
                results_df['processing_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                results_df['pipeline_version'] = 'S4_v2.1_smart_filtering'
            
            with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Processing_Results', index=False)
                
                # Add summary statistics sheet
                if not results_df.empty:
                    summary_stats = {
                        'Metric': [
                            'Total Processed Series',
                            'Successful Conversions', 
                            'Failed Conversions',
                            'Already Processed (Skipped)',
                            'Success Rate (%)',
                            'Processing Mode',
                            'Output Directory',
                            'Processing Date'
                        ],
                        'Value': [
                            len(results_df),
                            len(results_df[results_df['success'] == True]),
                            len(results_df[results_df['success'] == False]),
                            len(results_df[results_df['error_message'] == 'Skipped - already processed']),
                            f"{(len(results_df[results_df['success'] == True]) / len(results_df) * 100):.1f}%" if len(results_df) > 0 else "0%",
                            results_df['processing_mode'].iloc[0] if 'processing_mode' in results_df.columns else 'unknown',
                            str(self.output_dir),
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        ]
                    }
                    summary_df = pd.DataFrame(summary_stats)
                    summary_df.to_excel(writer, sheet_name='Processing_Summary', index=False)
            
            # Save secure mapping
            mapping_path = self.output_dir / "mapping_secure.json"
            with open(mapping_path, 'w') as f:
                json.dump(convert_to_json_serializable(self.secure_mapping), f, indent=2)
            
            # Save patient shifts
            shifts_path = self.output_dir / "patient_shifts_secure.json"
            with open(shifts_path, 'w') as f:
                json.dump(convert_to_json_serializable(self.patient_shifts), f, indent=2)
            
            self.logger.info("Processing results saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving processing results: {e}")
    
    def run_pipeline(self, excel_path: Optional[str] = None, analysis_only: bool = False):
        """Execute the complete pipeline with resume functionality."""
        try:
            self.logger.info("Starting Expert DICOM to NIfTI Pipeline")
            
            # Select Excel file
            selected_excel = self.select_excel_file(excel_path)
            if not selected_excel:
                return
            
            # Setup progress tracking
            self.setup_progress_tracking(selected_excel)
            
            # Check if we should resume from previous progress
            if self.progress_file.exists() and len(self.progress_data['processed_series']) > 0:
                if not self.ask_user_resume_or_restart():
                    # User chose to restart - progress_data already cleared
                    pass
            
            # Load selected studies
            studies_df = self.load_selected_studies(selected_excel)
            if studies_df.empty:
                return
            
            # Export analysis to Excel (or check if already exists)
            analysis_file = selected_excel.parent / f"{selected_excel.stem}_Image3Ddata.xlsx"
            
            # Check for existing Image3Ddata file and ask user if they want to skip first step
            existing_image3d = self.check_existing_image3d_file(selected_excel)
            skip_analysis = False
            
            if existing_image3d and not self.force_reprocess:
                skip_analysis = self.ask_user_skip_first_step(existing_image3d)
                if skip_analysis:
                    analysis_file = existing_image3d
            
            if skip_analysis or analysis_file.exists():
                if not skip_analysis:
                    print(f"\\n📊 Found existing analysis file: {analysis_file.name}")
                print("🔍 Loading NIfTI readiness data for smart filtering...")
                
                # Load analysis to filter for NIfTI-ready series
                try:
                    analysis_df = pd.read_excel(analysis_file, sheet_name='DICOM_Analysis')
                    ready_series = analysis_df[
                        (analysis_df['final_selection'] == 1) & 
                        (analysis_df['is_ready_for_nifti'] == True)
                    ]
                    
                    print(f"📈 Analysis Summary:")
                    print(f"   Total series in analysis: {len(analysis_df)}")
                    print(f"   Selected series (final_selection=1): {len(analysis_df[analysis_df['final_selection'] == 1])}")
                    print(f"   ✅ NIfTI-ready series: {len(ready_series)}")
                    print(f"   ❌ Not ready series: {len(analysis_df[analysis_df['final_selection'] == 1]) - len(ready_series)}")
                    
                    if len(ready_series) == 0:
                        print("\\n⚠️  No NIfTI-ready series found!")
                        print("Consider running analysis-only mode first to identify issues.")
                        return
                        
                except Exception as e:
                    self.logger.error(f"Error loading analysis file: {e}")
                    print("❌ Could not load analysis file. Proceeding without filtering...")
                    ready_series = None
            else:
                print("\\n📊 No existing analysis found. Generating analysis...")
                if self.export_analysis_to_excel(studies_df, selected_excel):
                    print(f"✅ Analysis exported to: {analysis_file}")
                    print(f"\\n🔍 Please review the analysis file to understand your data quality.")
                    
                    if analysis_only:
                        print("Analysis-only mode completed.")
                        return
                        
                    # Load the newly created analysis
                    try:
                        analysis_df = pd.read_excel(analysis_file, sheet_name='DICOM_Analysis')
                        ready_series = analysis_df[
                            (analysis_df['final_selection'] == 1) & 
                            (analysis_df['is_ready_for_nifti'] == True)
                        ]
                        print(f"\\n📈 Found {len(ready_series)} NIfTI-ready series for conversion.")
                    except:
                        ready_series = None
                        
                    input("\\nPress Enter to proceed with NIfTI conversion...")
                else:
                    print("❌ Failed to export analysis. Proceeding with conversion...")
                    ready_series = None
            
            # Create NIfTI tracking Excel file at the beginning (if not resuming)
            if len(self.progress_data['processed_series']) == 0:
                print("\\n📊 Creating NIfTI processing tracker...")
                self.create_nifti_excel_at_start(selected_excel, studies_df)
            else:
                print(f"\\n🔄 Resuming processing ({len(self.progress_data['processed_series'])} series already processed)")
                # Set up reference to existing NIfTI excel
                self.nifti_excel_path = self.output_dir / f"{selected_excel.stem}_nifti_results.xlsx"
            
            # Process each study with smart filtering
            if ready_series is not None and len(ready_series) > 0:
                print(f"\\n🏗️ Processing {len(ready_series)} NIfTI-ready series (filtered for efficiency)...")
                processing_mode = "filtered"
            else:
                print(f"\\n🏗️ Processing {len(studies_df)} studies (no filtering available)...")
                processing_mode = "unfiltered"
            
            # Determine what to process based on filtering mode
            if processing_mode == "filtered":
                # Group ready series by study for efficient processing
                ready_series_by_study = {}
                for _, series_row in ready_series.iterrows():
                    study_key = (series_row['source_folder'], series_row['source_file'])
                    if study_key not in ready_series_by_study:
                        ready_series_by_study[study_key] = []
                    ready_series_by_study[study_key].append(series_row['series_number'])
                
                studies_to_process = []
                for index, study_row in studies_df.iterrows():
                    study_key = (study_row.get('source_folder', ''), study_row.get('source_file', ''))
                    if study_key in ready_series_by_study:
                        studies_to_process.append((index, study_row, ready_series_by_study[study_key]))
                
                # Create progress bar for filtered processing
                study_progress = tqdm(studies_to_process,
                                    desc="NIfTI-ready studies", 
                                    unit="study",
                                    position=0)
                
                for index, study_row, ready_series_numbers in study_progress:
                    study_progress.set_description(f"Study {index + 1} ({len(ready_series_numbers)} ready series)")
                    self.logger.info(f"Processing study {index + 1} - NIfTI-ready series: {ready_series_numbers}")
                    
                    try:
                        # Parse JSONL and group by series
                        source_folder = study_row.get('source_folder', '')
                        source_file = study_row.get('source_file', '')
                        
                        base_dir = Path(__file__).parent.parent
                        jsonl_path = base_dir / "data" / "processed" / source_folder / "S1_indexed_metadata" / source_file
                        
                        if not jsonl_path.exists():
                            study_progress.write(f"⚠️  JSONL file not found: {jsonl_path}")
                            continue
                        
                        dicom_data = self.parse_jsonl_file(jsonl_path)
                        if not dicom_data:
                            study_progress.write(f"⚠️  No DICOM data found in: {source_file}")
                            continue
                        
                        series_groups = self.group_dicom_by_series(dicom_data)
                        
                        # Process only NIfTI-ready series with nested progress bar
                        # Convert both ready_series_numbers and series_groups keys to strings for comparison
                        ready_series_str = [str(num) for num in ready_series_numbers]
                        ready_series_items = [(series_num, series_data) for series_num, series_data in series_groups.items() 
                                            if str(series_num) in ready_series_str]
                        
                        # Debug logging
                        self.logger.info(f"Study {index + 1}: Found {len(series_groups)} total series, {len(ready_series_items)} ready for processing")
                        if len(ready_series_items) == 0:
                            self.logger.warning(f"No matching series found. Ready series: {ready_series_str}, Available series: {list(series_groups.keys())}")
                        
                        series_progress = tqdm(ready_series_items,
                                             desc=f"  Ready series",
                                             unit="series", 
                                             position=1,
                                             leave=False)
                        
                        for series_number, series_data in series_progress:
                            series_progress.set_description(f"  ✅ Series {series_number} ({len(series_data)} slices)")
                            result = self.process_single_series(series_data, study_row)
                            
                            # Update progress with result
                            if result['success']:
                                series_progress.write(f"✅ Series {series_number}: {result['hash_id']}")
                                self.stats['successful_conversions'] += 1
                            else:
                                series_progress.write(f"❌ Series {series_number}: {result['error_message'][:50]}...")
                                self.stats['failed_conversions'] += 1
                            
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
                                'geometry_warnings': '; '.join(result['geometry_warnings']),
                                'processing_mode': 'filtered_ready'
                            }
                            self.processing_results.append(result_row)
                        
                        self.stats['total_series'] += len(ready_series_items)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing study {index}: {e}")
                        continue
            
            else:
                # Original unfiltered processing for backward compatibility
                study_progress = tqdm(studies_df.iterrows(), 
                                    total=len(studies_df),
                                    desc="Studies",
                                    unit="study",
                                    position=0)
                
                for index, study_row in study_progress:
                    study_progress.set_description(f"Study {index + 1}/{len(studies_df)}")
                    self.logger.info(f"Processing study {index + 1}/{len(studies_df)}")
                    
                    try:
                        # Parse JSONL and group by series
                        source_folder = study_row.get('source_folder', '')
                        source_file = study_row.get('source_file', '')
                        
                        base_dir = Path(__file__).parent.parent
                        jsonl_path = base_dir / "data" / "processed" / source_folder / "S1_indexed_metadata" / source_file
                        
                        if not jsonl_path.exists():
                            study_progress.write(f"⚠️  JSONL file not found: {jsonl_path}")
                            continue
                        
                        dicom_data = self.parse_jsonl_file(jsonl_path)
                        if not dicom_data:
                            study_progress.write(f"⚠️  No DICOM data found in: {source_file}")
                            continue
                        
                        series_groups = self.group_dicom_by_series(dicom_data)
                        
                        # Process each series with nested progress bar
                        series_progress = tqdm(series_groups.items(),
                                             desc=f"  Series",
                                             unit="series",
                                             position=1,
                                             leave=False)
                        
                        for series_number, series_data in series_progress:
                            series_progress.set_description(f"  Series {series_number} ({len(series_data)} slices)")
                            result = self.process_single_series(series_data, study_row)
                            
                            # Update progress with result
                            if result['success']:
                                series_progress.write(f"✅ Series {series_number}: {result['hash_id']}")
                                self.stats['successful_conversions'] += 1
                            else:
                                series_progress.write(f"❌ Series {series_number}: {result['error_message'][:50]}...")
                                self.stats['failed_conversions'] += 1
                            
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
                                'geometry_warnings': '; '.join(result['geometry_warnings']),
                                'processing_mode': 'unfiltered'
                            }
                            self.processing_results.append(result_row)
                        
                        self.stats['total_series'] += len(series_groups)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing study {index}: {e}")
                        continue
            
            # Save final results
            self.save_processing_results(selected_excel)
            
            # Save final progress and update NIfTI Excel
            self.save_progress()
            self.update_nifti_excel()
            print(f"\\n💾 Final progress saved and NIfTI Excel updated")
            
            # Print summary
            duration = (datetime.now() - self.stats['start_time']).total_seconds()
            print(f"\\n{'='*60}")
            print("EXPERT NIFTI CONVERSION COMPLETED")
            print(f"{'='*60}")
            print(f"Duration: {duration:.1f} seconds")
            print(f"Processing mode: {processing_mode}")
            if processing_mode == "filtered":
                print(f"Series processed: {self.stats['total_series']} (NIfTI-ready only)")
            else:
                print(f"Studies processed: {self.stats['total_studies']}")
                print(f"Total series: {self.stats['total_series']}")
            print(f"✅ Successful conversions: {self.stats['successful_conversions']}")
            print(f"❌ Failed conversions: {self.stats['failed_conversions']}")
            if self.stats['total_series'] > 0:
                success_rate = (self.stats['successful_conversions'] / self.stats['total_series']) * 100
                print(f"📊 Success rate: {success_rate:.1f}%")
            print(f"📁 Output directory: {self.output_dir}")
            if hasattr(self, 'nifti_excel_path'):
                print(f"📊 NIfTI processing tracker: {self.nifti_excel_path.name}")
            if self.progress_file:
                print(f"🔄 Progress file: {self.progress_file.name}")
            print(f"{'='*60}")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description='Expert DICOM to NIfTI Conversion Pipeline with Smart Resume')
    parser.add_argument('--excel-file', help='Path to consolidated Excel file')
    parser.add_argument('--output-dir', default='data/nifti2share', help='Output directory')
    parser.add_argument('--analysis-only', action='store_true', help='Export analysis only, no conversion')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Force reprocessing of already completed series')
    parser.add_argument('--smart-filter', action='store_true', default=True,
                       help='Use smart filtering for NIfTI-ready series only (default: True)')
    parser.add_argument('--batch-size', type=int, default=30,
                       help='Number of series to process before saving progress (default: 30)')
    
    args = parser.parse_args()
    
    processor = NiftiStoreProcessor(output_dir=args.output_dir, force_reprocess=args.force_reprocess)
    processor.batch_size = args.batch_size  # Set custom batch size
    processor.run_pipeline(excel_path=args.excel_file, analysis_only=args.analysis_only)


if __name__ == "__main__":
    main()