#!/usr/bin/env python3
"""
Enhanced Consolidate Patient Summaries Script
==============================================

This script finds processed folders, collects S2_patient_summary.json files,
and creates consolidated Excel and JSON files with all patient data.

Features:
- GUI mode with interactive folder selection
- CLI mode with command-line arguments
- Support for single folder or multiple folders
- Automatic discovery of S2 summaries in processed folders

Usage:
    # GUI Mode (interactive)
    python consolidate_patient_summaries.py --gui
    
    # CLI Mode - process entire processed directory
    python consolidate_patient_summaries.py --processed-dir ../data/processed
    
    # CLI Mode - process specific folders
    python consolidate_patient_summaries.py --folders folder1 folder2 folder3
    
    # CLI Mode - custom output directory
    python consolidate_patient_summaries.py --output-dir custom_output
"""

import os
import json
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from threading import Thread
import queue
from difflib import SequenceMatcher


class ConsolidationEngine:
    """Core engine for patient summary consolidation."""
    
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
    
    def log(self, message):
        """Log message to console and progress callback."""
        print(message)
        if self.progress_callback:
            self.progress_callback(message)
    
    def calculate_name_similarity(self, name1, name2):
        """
        Calculate similarity between two patient names using word-level matching.
        
        Args:
            name1 (str): First patient name
            name2 (str): Second patient name
            
        Returns:
            float: Similarity ratio (0.0 to 1.0)
        """
        if not name1 or not name2 or name1 == "Unknown" or name2 == "Unknown":
            return 0.0
        
        # Clean and split names into words
        words1 = set(name1.upper().replace("^", " ").split())
        words2 = set(name2.upper().replace("^", " ").split())
        
        # Remove empty strings
        words1 = {w for w in words1 if w.strip()}
        words2 = {w for w in words2 if w.strip()}
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def is_duplicate_patient(self, patient1, patient2):
        """
        Check if two patient records are duplicates based on the specified criteria.
        
        Criteria:
        - study_date, study_modality, and ct_objects_count must exactly match
        - AND one of:
          A: patient_id is valid (>3 chars) and exactly matches
          B: patient_name words are 80%+ similar AND patient_age exactly matches (not "Unknown")
        
        Args:
            patient1 (dict): First patient record
            patient2 (dict): Second patient record
            
        Returns:
            bool: True if patients are duplicates
        """
        # Core criteria - must all match exactly
        core_match = (
            patient1.get('study_date') == patient2.get('study_date') and
            patient1.get('study_modality') == patient2.get('study_modality') and
            patient1.get('ct_objects_count') == patient2.get('ct_objects_count')
        )
        
        if not core_match:
            return False
        
        # Criteria A: Patient ID match
        pid1 = patient1.get('patient_id', '')
        pid2 = patient2.get('patient_id', '')
        
        if (pid1 and pid2 and 
            pid1 != "Unknown" and pid2 != "Unknown" and
            len(str(pid1)) > 3 and len(str(pid2)) > 3 and
            str(pid1) == str(pid2)):
            return True
        
        # Criteria B: Name similarity + age match
        name1 = patient1.get('patient_name', '')
        name2 = patient2.get('patient_name', '')
        age1 = patient1.get('patient_age', '')
        age2 = patient2.get('patient_age', '')
        
        if (name1 and name2 and age1 and age2 and
            name1 != "Unknown" and name2 != "Unknown" and
            age1 != "Unknown" and age2 != "Unknown" and
            age1 == age2):
            
            name_similarity = self.calculate_name_similarity(name1, name2)
            if name_similarity >= 0.8:
                return True
        
        return False
    
    def detect_duplicates(self, consolidated_data):
        """
        Detect duplicates in consolidated patient data and add readable duplicate columns.
        
        Args:
            consolidated_data (list): List of patient records
            
        Returns:
            list: Updated patient records with is_duplicate and is_duplicate_first_occurrence_un columns
        """
        self.log("Detecting duplicate patients...")
        
        # Initialize duplicate columns
        for i, patient in enumerate(consolidated_data):
            patient['is_duplicate'] = "unique"  # Default to unique
            patient['is_duplicate_first_occurrence_un'] = ""  # Empty for non-duplicates
        
        duplicate_count = 0
        first_occurrence_count = 0
        
        # Compare each patient with all previous patients
        for i in range(len(consolidated_data)):
            if consolidated_data[i]['is_duplicate'] != "unique":
                continue  # Skip already processed records
                
            duplicates_found = []
            
            for j in range(i + 1, len(consolidated_data)):
                if consolidated_data[j]['is_duplicate'] != "unique":
                    continue  # Skip already processed records
                
                if self.is_duplicate_patient(consolidated_data[i], consolidated_data[j]):
                    duplicates_found.append(j)
                    
                    # Mark the duplicate record
                    original_unique_num = consolidated_data[i]['Unique_number']
                    duplicate_unique_num = consolidated_data[j]['Unique_number']
                    consolidated_data[j]['is_duplicate'] = "duplicate"
                    consolidated_data[j]['is_duplicate_first_occurrence_un'] = str(original_unique_num)
                    duplicate_count += 1
                    
                    self.log(f"Duplicate found: {duplicate_unique_num} is duplicate of {original_unique_num}")
                    self.log(f"  Original: {consolidated_data[i].get('patient_name', 'Unknown')} "
                            f"({consolidated_data[i].get('patient_id', 'Unknown')}) - "
                            f"{consolidated_data[i].get('study_date', 'Unknown')}")
                    self.log(f"  Duplicate: {consolidated_data[j].get('patient_name', 'Unknown')} "
                            f"({consolidated_data[j].get('patient_id', 'Unknown')}) - "
                            f"{consolidated_data[j].get('study_date', 'Unknown')}")
            
            # If duplicates were found, mark the first occurrence
            if duplicates_found:
                consolidated_data[i]['is_duplicate'] = "first_occurrence"
                first_occurrence_count += 1
        
        self.log(f"Duplicate detection completed. Found {duplicate_count} duplicates with {first_occurrence_count} first occurrences.")
        return consolidated_data
    
    def is_same_patient(self, patient1, patient2):
        """
        Check if two records belong to the same patient (but different studies/dates).
        Uses similar logic to duplicate detection but allows different study dates.
        
        Args:
            patient1 (dict): First patient record
            patient2 (dict): Second patient record
            
        Returns:
            tuple: (is_same_patient: bool, matching_method: str)
        """
        # Skip if they are the exact same record (same study date)
        if patient1.get('study_date') == patient2.get('study_date'):
            return False, ""
        
        # Criteria A: Patient ID match
        pid1 = patient1.get('patient_id', '')
        pid2 = patient2.get('patient_id', '')
        
        if (pid1 and pid2 and 
            pid1 != "Unknown" and pid2 != "Unknown" and
            len(str(pid1)) > 3 and len(str(pid2)) > 3 and
            str(pid1) == str(pid2)):
            return True, "ID_match"
        
        # Criteria B: Name similarity + age match
        name1 = patient1.get('patient_name', '')
        name2 = patient2.get('patient_name', '')
        age1 = patient1.get('patient_age', '')
        age2 = patient2.get('patient_age', '')
        
        if (name1 and name2 and age1 and age2 and
            name1 != "Unknown" and name2 != "Unknown" and
            age1 != "Unknown" and age2 != "Unknown" and
            age1 == age2):
            
            name_similarity = self.calculate_name_similarity(name1, name2)
            if name_similarity >= 0.8:
                similarity_percent = int(name_similarity * 100)
                return True, f"Name_age_match({similarity_percent}%)"
        
        return False, ""
    
    def find_patient_other_images(self, consolidated_data):
        """
        Find other studies for each patient and add improved readable columns.
        Only processes records that are not duplicates (unique or first_occurrence).
        
        Args:
            consolidated_data (list): List of patient records with duplicate columns
            
        Returns:
            list: Updated patient records with has_other_studies and other_studies_details columns
        """
        self.log("Finding other studies for each patient...")
        
        # Initialize new columns
        for patient in consolidated_data:
            patient['Patient_other_image'] = ""  # Keep for backward compatibility
            patient['has_other_studies'] = 0
            patient['other_studies_details'] = ""
        
        match_count = 0
        
        # Only process records that are not duplicates (unique or first_occurrence)
        unique_records = [patient for patient in consolidated_data 
                         if patient.get('is_duplicate', 'unique') in ['unique', 'first_occurrence']]
        
        self.log(f"Processing {len(unique_records)} unique records for patient matching...")
        
        # Compare each unique record with all other unique records
        for i, patient1 in enumerate(unique_records):
            other_studies = []
            other_studies_details = []
            
            for j, patient2 in enumerate(unique_records):
                if i != j:
                    is_same, matching_method = self.is_same_patient(patient1, patient2)
                    if is_same:
                        other_studies.append(str(patient2['Unique_number']))
                        other_studies_details.append(f"{patient2['Unique_number']}({matching_method})")
                        match_count += 1
            
            # Update columns
            if other_studies:
                patient1['Patient_other_image'] = ",".join(other_studies)  # Backward compatibility
                patient1['has_other_studies'] = len(other_studies)
                patient1['other_studies_details'] = "||".join(other_studies_details)  # || is safe separator
                
                self.log(f"Patient {patient1.get('patient_name', 'Unknown')} "
                        f"({patient1.get('patient_id', 'Unknown')}) has {len(other_studies)} other studies: "
                        f"{patient1['other_studies_details']}")
        
        self.log(f"Patient matching completed. Found {match_count} patient-study relationships.")
        return consolidated_data
    
    def find_s2_summary_files_in_folders(self, folders):
        """
        Find S2_patient_summary.json files in specific folders.
        
        Args:
            folders (list): List of folder paths to check
            
        Returns:
            list: List of tuples (folder_name, json_file_path, excel_file_path)
        """
        summary_files = []
        
        self.log(f"Checking {len(folders)} specific folders for S2 summaries...")
        
        for folder_path in folders:
            folder = Path(folder_path)
            if not folder.exists():
                self.log(f"Warning: Folder not found: {folder}")
                continue
                
            if not folder.is_dir():
                self.log(f"Warning: Not a directory: {folder}")
                continue
            
            s2_folder = folder / "S2_concatenated_summaries"
            if s2_folder.exists():
                json_file = s2_folder / "S2_patient_summary.json"
                excel_file = s2_folder / "S2_patient_summary.xlsx"
                
                if json_file.exists():
                    summary_files.append((folder.name, json_file, excel_file))
                    self.log(f"Found: {folder.name}")
                else:
                    self.log(f"Warning: No S2_patient_summary.json in {s2_folder}")
            else:
                self.log(f"Warning: No S2_concatenated_summaries folder in {folder}")
        
        return summary_files
    
    def find_s2_summary_files_in_processed_dir(self, base_processed_dir):
        """
        Find all S2_patient_summary.json files in processed folders.
        
        Args:
            base_processed_dir (Path): Base directory containing processed folders
            
        Returns:
            list: List of tuples (folder_name, json_file_path, excel_file_path)
        """
        summary_files = []
        
        if not base_processed_dir.exists():
            self.log(f"Warning: Processed directory not found: {base_processed_dir}")
            return summary_files
        
        self.log(f"Searching for S2 summaries in: {base_processed_dir}")
        
        # Look for processed folders
        for folder in base_processed_dir.iterdir():
            if folder.is_dir():
                s2_folder = folder / "S2_concatenated_summaries"
                if s2_folder.exists():
                    json_file = s2_folder / "S2_patient_summary.json"
                    excel_file = s2_folder / "S2_patient_summary.xlsx"
                    
                    if json_file.exists():
                        summary_files.append((folder.name, json_file, excel_file))
                        self.log(f"Found: {folder.name}")
                    else:
                        self.log(f"Warning: No S2_patient_summary.json in {s2_folder}")
        
        return summary_files
    
    def load_patient_data(self, summary_files):
        """
        Load patient data from all summary files.
        
        Args:
            summary_files (list): List of (folder_name, json_file_path, excel_file_path)
            
        Returns:
            tuple: (consolidated_data, folder_stats)
        """
        consolidated_data = []
        folder_stats = {}
        
        for folder_name, json_file, excel_file in summary_files:
            try:
                self.log(f"Loading data from: {folder_name}")
                
                with open(json_file, 'r', encoding='utf-8') as f:
                    folder_data = json.load(f)
                
                # Add source folder information to each patient record
                for patient in folder_data:
                    patient_copy = patient.copy()
                    patient_copy['source_folder'] = folder_name
                    consolidated_data.append(patient_copy)
                
                folder_stats[folder_name] = {
                    'patient_count': len(folder_data),
                    'json_file': str(json_file),
                    'excel_file': str(excel_file) if excel_file.exists() else None
                }
                
                self.log(f"  - Loaded {len(folder_data)} patients")
                
            except Exception as e:
                self.log(f"Error loading {json_file}: {e}")
                folder_stats[folder_name] = {
                    'patient_count': 0,
                    'error': str(e),
                    'json_file': str(json_file)
                }
        
        # Add unique numbers after loading all data
        if consolidated_data:
            # Add Unique_number column starting from 10001
            for i, patient in enumerate(consolidated_data):
                patient['Unique_number'] = 10001 + i
            
            consolidated_data = self.detect_duplicates(consolidated_data)
            # Find other images for each patient (only for non-duplicates)
            consolidated_data = self.find_patient_other_images(consolidated_data)
        
        return consolidated_data, folder_stats
    
    def create_consolidated_files(self, consolidated_data, folder_stats, output_dir):
        """
        Create consolidated JSON and Excel files.
        
        Args:
            consolidated_data (list): All patient data
            folder_stats (dict): Statistics per folder
            output_dir (Path): Output directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output files
        consolidated_json = output_dir / f"consolidated_patient_summary_{timestamp}.json"
        consolidated_excel = output_dir / f"consolidated_patient_summary_{timestamp}.xlsx"
        stats_json = output_dir / f"consolidation_stats_{timestamp}.json"
        
        # Save consolidated JSON
        self.log(f"Creating consolidated JSON: {consolidated_json}")
        with open(consolidated_json, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
        
        # Create consolidated Excel with multiple sheets
        self.log(f"Creating consolidated Excel: {consolidated_excel}")
        
        try:
            with pd.ExcelWriter(consolidated_excel, engine='openpyxl') as writer:
                # Main patients sheet
                if consolidated_data:
                    df_patients = pd.DataFrame(consolidated_data)
                    df_patients.to_excel(writer, sheet_name='All_Patients', index=False)
                    
                    # Summary statistics sheet
                    summary_stats = {
                        'total_patients': len(consolidated_data),
                        'total_folders': len(folder_stats),
                        'consolidation_date': datetime.now().isoformat(),
                        'folders_processed': list(folder_stats.keys())
                    }
                    
                    # Create folder statistics sheet
                    folder_stats_data = []
                    for folder_name, stats in folder_stats.items():
                        folder_stats_data.append({
                            'folder_name': folder_name,
                            'patient_count': stats.get('patient_count', 0),
                            'has_error': 'error' in stats,
                            'error_message': stats.get('error', ''),
                            'json_file': stats.get('json_file', ''),
                            'excel_file': stats.get('excel_file', '')
                        })
                    
                    df_stats = pd.DataFrame(folder_stats_data)
                    df_stats.to_excel(writer, sheet_name='Folder_Statistics', index=False)
                    
                    # Summary sheet
                    df_summary = pd.DataFrame([summary_stats])
                    df_summary.to_excel(writer, sheet_name='Summary', index=False)
                    
                    self.log(f"Excel file created with {len(consolidated_data)} patient records")
                else:
                    # Create empty file with message
                    df_empty = pd.DataFrame([{'message': 'No patient data found'}])
                    df_empty.to_excel(writer, sheet_name='No_Data', index=False)
                    self.log("Warning: No patient data found, created empty Excel file")
                    
        except ImportError:
            self.log("Warning: pandas/openpyxl not available, skipping Excel creation")
            self.log("Run: pip install pandas openpyxl")
        
        # Save consolidation statistics
        stats_data = {
            'consolidation_timestamp': datetime.now().isoformat(),
            'total_patients': len(consolidated_data),
            'total_folders_processed': len(folder_stats),
            'output_files': {
                'json': str(consolidated_json),
                'excel': str(consolidated_excel)
            },
            'folder_statistics': folder_stats
        }
        
        with open(stats_json, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        self.log(f"\nConsolidation completed!")
        self.log(f"Output files created:")
        self.log(f"  - JSON: {consolidated_json}")
        self.log(f"  - Excel: {consolidated_excel}")
        self.log(f"  - Stats: {stats_json}")
        self.log(f"\nTotal patients consolidated: {len(consolidated_data)}")
        self.log(f"Total folders processed: {len(folder_stats)}")
        
        return {
            'json': str(consolidated_json),
            'excel': str(consolidated_excel),
            'stats': str(stats_json),
            'total_patients': len(consolidated_data),
            'total_folders': len(folder_stats)
        }


class ConsolidationGUI:
    """GUI interface for patient summary consolidation."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Patient Summary Consolidation Tool")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Variables
        self.mode_var = tk.StringVar(value="processed_dir")
        self.processed_dir_var = tk.StringVar(value="../data/processed")
        self.output_dir_var = tk.StringVar(value="../data/consolidated_summaries")
        self.selected_folders = []
        
        # Progress queue for thread communication
        self.progress_queue = queue.Queue()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the GUI interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Patient Summary Consolidation Tool", 
                               font=('TkDefaultFont', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Processing Mode", padding="10")
        mode_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        mode_frame.columnconfigure(1, weight=1)
        
        ttk.Radiobutton(mode_frame, text="Process entire processed directory", 
                       variable=self.mode_var, value="processed_dir",
                       command=self.on_mode_change).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        ttk.Radiobutton(mode_frame, text="Select specific folders", 
                       variable=self.mode_var, value="specific_folders",
                       command=self.on_mode_change).grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # Processed directory section
        self.processed_frame = ttk.Frame(mode_frame)
        self.processed_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        self.processed_frame.columnconfigure(1, weight=1)
        
        ttk.Label(self.processed_frame, text="Processed Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.processed_entry = ttk.Entry(self.processed_frame, textvariable=self.processed_dir_var, width=50)
        self.processed_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(self.processed_frame, text="Browse", command=self.browse_processed_dir).grid(row=0, column=2)
        
        # Specific folders section
        self.folders_frame = ttk.Frame(mode_frame)
        self.folders_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        self.folders_frame.columnconfigure(0, weight=1)
        
        ttk.Label(self.folders_frame, text="Selected Folders:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Folders listbox with scrollbar
        listbox_frame = ttk.Frame(self.folders_frame)
        listbox_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        listbox_frame.columnconfigure(0, weight=1)
        
        self.folders_listbox = tk.Listbox(listbox_frame, height=6, selectmode=tk.EXTENDED)
        self.folders_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        folders_scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.folders_listbox.yview)
        folders_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.folders_listbox.configure(yscrollcommand=folders_scrollbar.set)
        
        # Folder buttons
        folder_buttons_frame = ttk.Frame(self.folders_frame)
        folder_buttons_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W)
        
        ttk.Button(folder_buttons_frame, text="Add Folder", command=self.add_folder).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(folder_buttons_frame, text="Remove Selected", command=self.remove_selected_folders).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(folder_buttons_frame, text="Clear All", command=self.clear_all_folders).grid(row=0, column=2)
        
        # Output directory
        output_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding="10")
        output_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        output_frame.columnconfigure(1, weight=1)
        
        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(output_frame, textvariable=self.output_dir_var, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(output_frame, text="Browse", command=self.browse_output_dir).grid(row=0, column=2)
        
        # Progress and log section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(1, weight=1)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.log_text = scrolledtext.ScrolledText(progress_frame, height=12, state=tk.DISABLED)
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=(10, 0))
        
        self.start_button = ttk.Button(button_frame, text="Start Consolidation", command=self.start_consolidation)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(button_frame, text="Clear Log", command=self.clear_log).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(button_frame, text="Exit", command=self.root.quit).grid(row=0, column=2)
        
        # Initial mode setup
        self.on_mode_change()
        
    def on_mode_change(self):
        """Handle mode selection changes."""
        if self.mode_var.get() == "processed_dir":
            # Show processed directory controls
            for widget in self.processed_frame.winfo_children():
                try:
                    widget.configure(state='normal')
                except:
                    pass
            
            # Hide specific folders controls
            for widget in self.folders_frame.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for subwidget in widget.winfo_children():
                        try:
                            subwidget.configure(state='disabled')
                        except:
                            pass
                else:
                    try:
                        widget.configure(state='disabled')
                    except:
                        pass
        else:
            # Hide processed directory controls
            for widget in self.processed_frame.winfo_children():
                try:
                    widget.configure(state='disabled')
                except:
                    pass
            
            # Show specific folders controls
            for widget in self.folders_frame.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for subwidget in widget.winfo_children():
                        try:
                            subwidget.configure(state='normal')
                        except:
                            pass
                else:
                    try:
                        widget.configure(state='normal')
                    except:
                        pass
    
    def browse_processed_dir(self):
        """Browse for processed directory."""
        directory = filedialog.askdirectory(title="Select Processed Directory", 
                                          initialdir=self.processed_dir_var.get())
        if directory:
            self.processed_dir_var.set(directory)
    
    def browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory",
                                          initialdir=self.output_dir_var.get())
        if directory:
            self.output_dir_var.set(directory)
    
    def add_folder(self):
        """Add a folder to the selection."""
        directory = filedialog.askdirectory(title="Select Folder with S2_concatenated_summaries")
        if directory and directory not in self.selected_folders:
            self.selected_folders.append(directory)
            self.update_folders_listbox()
    
    def remove_selected_folders(self):
        """Remove selected folders from the list."""
        selected_indices = self.folders_listbox.curselection()
        for index in reversed(selected_indices):
            del self.selected_folders[index]
        self.update_folders_listbox()
    
    def clear_all_folders(self):
        """Clear all selected folders."""
        self.selected_folders.clear()
        self.update_folders_listbox()
    
    def update_folders_listbox(self):
        """Update the folders listbox."""
        self.folders_listbox.delete(0, tk.END)
        for folder in self.selected_folders:
            self.folders_listbox.insert(tk.END, folder)
    
    def log_message(self, message):
        """Add a message to the log."""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.configure(state='disabled')
        self.log_text.see(tk.END)
    
    def clear_log(self):
        """Clear the log."""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
    
    def progress_callback(self, message):
        """Callback for progress updates from consolidation engine."""
        self.progress_queue.put(message)
    
    def check_progress_queue(self):
        """Check for progress updates from the consolidation thread."""
        try:
            while True:
                message = self.progress_queue.get_nowait()
                self.log_message(message)
        except queue.Empty:
            pass
        finally:
            # Schedule next check
            self.root.after(100, self.check_progress_queue)
    
    def start_consolidation(self):
        """Start the consolidation process."""
        # Validate inputs
        if self.mode_var.get() == "processed_dir":
            if not self.processed_dir_var.get().strip():
                messagebox.showerror("Error", "Please specify a processed directory.")
                return
            if not Path(self.processed_dir_var.get()).exists():
                messagebox.showerror("Error", "Processed directory does not exist.")
                return
        else:
            if not self.selected_folders:
                messagebox.showerror("Error", "Please select at least one folder.")
                return
        
        if not self.output_dir_var.get().strip():
            messagebox.showerror("Error", "Please specify an output directory.")
            return
        
        # Disable start button and start progress bar
        self.start_button.configure(state='disabled')
        self.progress_bar.start()
        self.clear_log()
        
        # Start consolidation in a separate thread
        def consolidation_thread():
            try:
                engine = ConsolidationEngine(progress_callback=self.progress_callback)
                
                script_dir = Path(__file__).parent.parent  # Go up one level from my_code
                output_dir = script_dir / self.output_dir_var.get()
                
                if self.mode_var.get() == "processed_dir":
                    processed_dir = script_dir / self.processed_dir_var.get()
                    summary_files = engine.find_s2_summary_files_in_processed_dir(processed_dir)
                else:
                    summary_files = engine.find_s2_summary_files_in_folders(self.selected_folders)
                
                if not summary_files:
                    self.progress_queue.put("Error: No S2 patient summary files found!")
                    return
                
                consolidated_data, folder_stats = engine.load_patient_data(summary_files)
                
                if not consolidated_data:
                    self.progress_queue.put("Error: No patient data could be loaded!")
                    return
                
                results = engine.create_consolidated_files(consolidated_data, folder_stats, output_dir)
                self.progress_queue.put(f"\n=== Consolidation Summary ===")
                self.progress_queue.put(f"Total patients: {results['total_patients']}")
                self.progress_queue.put(f"Total folders: {results['total_folders']}")
                self.progress_queue.put(f"Files created:")
                self.progress_queue.put(f"  - JSON: {Path(results['json']).name}")
                self.progress_queue.put(f"  - Excel: {Path(results['excel']).name}")
                self.progress_queue.put(f"  - Stats: {Path(results['stats']).name}")
                
            except Exception as e:
                self.progress_queue.put(f"Error during consolidation: {e}")
            finally:
                # Re-enable UI
                self.root.after(0, lambda: (
                    self.progress_bar.stop(),
                    self.start_button.configure(state='normal')
                ))
        
        # Start the consolidation thread
        Thread(target=consolidation_thread, daemon=True).start()
        
        # Start checking for progress updates
        self.check_progress_queue()
    
    def run(self):
        """Run the GUI."""
        self.root.mainloop()


def run_cli(args):
    """Run the CLI version."""
    engine = ConsolidationEngine()
    
    # Set up paths
    script_dir = Path(__file__).parent.parent  # Go up one level from my_code
    output_dir = script_dir / args.output_dir
    
    print("=== Patient Summary Consolidation (CLI Mode) ===")
    print(f"Output directory: {output_dir}")
    print()
    
    # Determine input mode
    if args.folders:
        # Process specific folders
        print(f"Processing {len(args.folders)} specific folders")
        summary_files = engine.find_s2_summary_files_in_folders(args.folders)
    else:
        # Process entire processed directory
        processed_dir = script_dir / args.processed_dir
        print(f"Processed directory: {processed_dir}")
        summary_files = engine.find_s2_summary_files_in_processed_dir(processed_dir)
    
    if not summary_files:
        print("Error: No S2 patient summary files found!")
        print("Make sure the directories contain S2_concatenated_summaries/S2_patient_summary.json files")
        sys.exit(1)
    
    print(f"\nFound {len(summary_files)} folders with patient summaries")
    
    # Load all patient data
    consolidated_data, folder_stats = engine.load_patient_data(summary_files)
    
    if not consolidated_data:
        print("Error: No patient data could be loaded!")
        sys.exit(1)
    
    # Create consolidated files
    results = engine.create_consolidated_files(consolidated_data, folder_stats, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Consolidate patient summaries from processed folders')
    parser.add_argument('--gui', action='store_true', 
                        help='Run in GUI mode with interactive interface')
    parser.add_argument('--output-dir', 
                        default='data/consolidated_summaries',
                        help='Output directory for consolidated files (default: data/consolidated_summaries)')
    parser.add_argument('--processed-dir',
                        default='data/processed',
                        help='Base processed directory (default: data/processed)')
    parser.add_argument('--folders', nargs='*',
                        help='Specific folders to process (alternative to --processed-dir)')
    
    args = parser.parse_args()
    
    if args.gui:
        # Run GUI mode
        try:
            app = ConsolidationGUI()
            app.run()
        except ImportError as e:
            print(f"Error: GUI mode requires tkinter: {e}")
            print("Run in CLI mode instead (remove --gui flag)")
            sys.exit(1)
    else:
        # Run CLI mode
        run_cli(args)


if __name__ == "__main__":
    main()