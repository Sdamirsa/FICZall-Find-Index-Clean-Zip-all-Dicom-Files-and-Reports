#!/usr/bin/env python3
"""
Multi-Location DICOM Processing Pipeline Orchestrator
====================================================

This script extends the functionality of run.py to process multiple data locations
sequentially using the same configuration. It asks for all settings upfront,
then applies them to each location one by one.

Features:
- Process multiple DICOM data locations in sequence
- Single configuration setup for all locations
- Progress tracking across all locations
- Resume capability if interrupted
- Comprehensive error handling and reporting
- Same interface as run.py but with multi-location support

Usage:
    python runMulti.py                    # Interactive mode
    python runMulti.py --config           # Show configuration
    python runMulti.py --setup            # Setup only (venv + dependencies)
    python runMulti.py --non-interactive  # Use existing configuration
    python runMulti.py --resume           # Resume interrupted batch processing

This is the final script in the FICZall pipeline suite.
"""

import os
import sys
import subprocess
import json
import argparse
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Ensure we're using Python 3.7+
if sys.version_info < (3, 7):
    print("Error: Python 3.7 or higher is required.")
    print(f"Current version: {sys.version}")
    sys.exit(1)

# Add current directory to path to import run_config and run.py functions
sys.path.insert(0, str(Path(__file__).parent))

# Import functions from run.py
try:
    from run import (
        Colors, print_header, print_step, print_success, print_error, print_warning,
        get_python_executable, run_command, check_python_version, 
        setup_virtual_environment, install_dependencies, get_user_input,
        validate_directory, BASE_DIR, CODE_DIR, REQUIREMENTS_FILE, DEFAULT_VENV_DIR
    )
except ImportError as e:
    print(f"Error: Cannot import required functions from run.py: {e}")
    print("Make sure run.py is in the same directory as runMulti.py")
    sys.exit(1)

# Try to import run_config, but don't fail if it doesn't exist
try:
    import run_config
    VENV_DIR = Path(run_config.VENV_PATH) if run_config.VENV_PATH else DEFAULT_VENV_DIR
except ImportError:
    # run_config.py doesn't exist yet, we'll create it
    VENV_DIR = DEFAULT_VENV_DIR
    run_config = None

# =============================================================================
# BATCH PROCESSING CONSTANTS
# =============================================================================

BATCH_CONFIG_FILE = BASE_DIR / "runMulti_config.json"
BATCH_PROGRESS_FILE = BASE_DIR / "runMulti_progress.json"

# =============================================================================
# BATCH CONFIGURATION FUNCTIONS
# =============================================================================

def get_multiple_locations() -> List[str]:
    """Get multiple data locations from user input."""
    print_header("MULTI-LOCATION DATA INPUT")
    print(f"\n{Colors.BLUE}Enter multiple data locations to process{Colors.ENDC}")
    print(f"{Colors.GREEN}TIP: You can process different hospitals, studies, or time periods{Colors.ENDC}")
    print(f"   Examples:")
    print(f"   - C:/Hospital_A/CT_2024")
    print(f"   - C:/Hospital_B/MRI_Studies") 
    print(f"   - /data/Research_Project_1")
    print(f"   - /data/Research_Project_2")
    print(f"\n{Colors.WARNING}WARNING: Each location will be processed using the SAME configuration{Colors.ENDC}")
    print(f"{Colors.WARNING}   Make sure all locations have similar data structure{Colors.ENDC}")
    
    locations = []
    location_num = 1
    
    while True:
        print(f"\nData Location #{location_num}:")
        location = get_user_input(
            f"Enter path to DICOM data location #{location_num} (or press Enter to finish)",
            required=(location_num == 1)  # First location is required
        )
        
        if not location:
            if location_num == 1:
                print_warning("At least one location is required")
                continue
            else:
                break  # User finished entering locations
        
        # Validate directory
        if not validate_directory(location):
            print_error(f"Invalid or inaccessible directory: {location}")
            continue
        
        # Check for duplicates
        if location in locations:
            print_warning(f"Location already added: {location}")
            continue
        
        locations.append(location)
        print_success(f"Added location #{location_num}: {location}")
        location_num += 1
        
        # Ask if user wants to add more
        if location_num > 2:  # After adding 2+ locations
            more = get_user_input(
                "Add another location? (y/n)", 
                "n"
            )
            if more.lower() not in ['y', 'yes']:
                break
    
    if not locations:
        print_error("No valid locations provided")
        return None
    
    print_success(f"Total locations to process: {len(locations)}")
    print("\nLocations summary:")
    for i, loc in enumerate(locations, 1):
        print(f"  {i}. {loc}")
    
    return locations

def configure_batch_pipeline() -> Dict[str, Any]:
    """Configure pipeline for batch processing with multiple locations."""
    print_header("BATCH PIPELINE CONFIGURATION")
    print(f"\n{Colors.BLUE}Configure pipeline settings for ALL locations{Colors.ENDC}")
    print(f"{Colors.GREEN}These settings will be applied to every data location{Colors.ENDC}")
    
    # Get multiple locations first
    locations = get_multiple_locations()
    if not locations:
        return None
    
    # Configure pipeline stages (same as run.py but adapted for batch)
    print(f"\n{Colors.BOLD}Select pipeline stages to run for ALL locations:{Colors.ENDC}")
    print(f"{Colors.WARNING}Note: These stages will run for EVERY location{Colors.ENDC}")
    print(f"   Required stages: S1, S2, S3")
    print(f"   Optional stages: S0 (extraction), S4 (ZIP archives), S5 (AI)")
    
    config = {'locations': locations}
    
    stages = {
        'S0_ZIP': ('ZIP File Extraction (OPTIONAL - extract ZIP files)', 'RUN_S0_ZIP_EXTRACT', False),
        'S0_ISO': ('ISO Extraction (OPTIONAL - only if you have ISO files)', 'RUN_S0_ISO_EXTRACT', False),
        'S1': ('DICOM Indexing (REQUIRED)', 'RUN_S1_INDEXING', True),
        'S2': ('Concatenate Index (REQUIRED)', 'RUN_S2_CONCAT', True),
        'S3': ('Process for Storage (REQUIRED)', 'RUN_S3_PROCESS', True),
        'S4': ('Create ZIP Archives (OPTIONAL - for long-term storage)', 'RUN_S4_ZIP', False),
        'S5': ('LLM Document Extraction (OPTIONAL - AI-powered extraction)', 'RUN_S5_LLM_EXTRACT', False)
    }
    
    for stage, (desc, env_var, default) in stages.items():
        current = os.getenv(env_var, 'true' if default else 'false')
        response = get_user_input(
            f"Run {stage} - {desc}? (y/n)",
            'y' if current == 'true' else 'n'
        )
        config[env_var] = 'true' if response.lower() in ['y', 'yes'] else 'false'
    
    # Configure S0_ZIP if enabled
    if config.get('RUN_S0_ZIP_EXTRACT') == 'true':
        print(f"\n{Colors.BOLD}S0 - ZIP Extraction Configuration:{Colors.ENDC}")
        print(f"{Colors.WARNING}WARNING: ZIP extraction will look for ZIP files in each data location{Colors.ENDC}")
        
        workers = get_user_input(
            "Number of worker threads for parallel extraction",
            run_config.S0_ZIP_WORKERS if run_config and hasattr(run_config, 'S0_ZIP_WORKERS') else '4'
        )
        config['S0_ZIP_WORKERS'] = workers or '4'
        
        integrity = get_user_input(
            "Enable integrity checking (yes/no)",
            run_config.S0_ZIP_INTEGRITY_CHECK if run_config and hasattr(run_config, 'S0_ZIP_INTEGRITY_CHECK') else 'yes'
        )
        config['S0_ZIP_INTEGRITY_CHECK'] = integrity or 'yes'
        
        overwrite = get_user_input(
            "Overwrite existing extractions (yes/no)",
            run_config.S0_ZIP_OVERWRITE if run_config and hasattr(run_config, 'S0_ZIP_OVERWRITE') else 'no'
        )
        config['S0_ZIP_OVERWRITE'] = overwrite or 'no'
    
    # Configure S0_ISO if enabled
    if config.get('RUN_S0_ISO_EXTRACT') == 'true':
        print(f"\n{Colors.BOLD}S0 - ISO Extraction Configuration:{Colors.ENDC}")
        print(f"{Colors.WARNING}WARNING: ISO extraction will look for ISO files in each data location{Colors.ENDC}")
        
        poweriso_default = run_config.POWERISO_PATH if run_config else r"C:\Users\LEGION\Downloads\PowerISO.9.0.Portable\PowerISO.9.0.Portable\App\PowerISO\piso.exe"
        poweriso_path = get_user_input(
            "Path to PowerISO executable",
            poweriso_default
        )
        if poweriso_path and not Path(poweriso_path).exists():
            print_warning(f"PowerISO not found at: {poweriso_path}")
            print_warning("Please ensure PowerISO is installed before running S0")
        config['POWERISO_PATH'] = poweriso_path
    
    # Configure S3 processing parameters
    if config.get('RUN_S3_PROCESS') == 'true':
        print(f"\n{Colors.BOLD}S3 - Processing Configuration:{Colors.ENDC}")
        
        min_files = get_user_input(
            "Minimum number of files per study",
            run_config.S3_MIN_FILES if run_config else '10'
        )
        config['S3_MIN_FILES'] = min_files or '10'
    
    # Configure S4 ZIP settings
    if config.get('RUN_S4_ZIP') == 'true':
        print(f"\n{Colors.BOLD}S4 - ZIP Configuration:{Colors.ENDC}")
        
        compression = get_user_input(
            "ZIP compression level (0-9, 9=max)",
            run_config.ZIP_COMPRESSION_LEVEL if run_config else '6'
        )
        config['ZIP_COMPRESSION_LEVEL'] = compression or '6'
        
        workers = get_user_input(
            "Number of parallel workers for ZIP creation",
            run_config.S4_WORKERS if run_config and hasattr(run_config, 'S4_WORKERS') else '2'
        )
        config['S4_WORKERS'] = workers or '2'
    
    # Configure S5 AI settings
    if config.get('RUN_S5_LLM_EXTRACT') == 'true':
        print(f"\n{Colors.BOLD}S5 - LLM Document Extraction Configuration:{Colors.ENDC}")
        print(f"{Colors.WARNING}IMPORTANT: AI extraction will process documents from ALL locations{Colors.ENDC}")
        print(f"{Colors.WARNING}   Check S5_llmExtract_config.py for API settings and privacy{Colors.ENDC}")
        
        # Check if S5 dependencies are available
        try:
            import importlib.util
            pydantic_spec = importlib.util.find_spec("pydantic")
            openai_spec = importlib.util.find_spec("openai")
            ollama_spec = importlib.util.find_spec("ollama")
            if pydantic_spec is None or openai_spec is None:
                print(f"{Colors.WARNING}WARNING: S5 dependencies not fully installed. Run setup first.{Colors.ENDC}")
        except Exception:
            pass
        
        batch_size = get_user_input(
            "Batch size for progress saving",
            run_config.S5_BATCH_SIZE if run_config and hasattr(run_config, 'S5_BATCH_SIZE') else '10'
        )
        config['S5_BATCH_SIZE'] = batch_size or '10'
        
        chunk_size = get_user_input(
            "Chunk size for large datasets",
            run_config.S5_CHUNK_SIZE if run_config and hasattr(run_config, 'S5_CHUNK_SIZE') else '1000'
        )
        config['S5_CHUNK_SIZE'] = chunk_size or '1000'
    
    # Global settings
    print(f"\n{Colors.BOLD}Global Processing Settings:{Colors.ENDC}")
    
    max_workers = get_user_input(
        "Maximum worker threads (affects all stages)",
        run_config.MAX_WORKERS if run_config else '4'
    )
    config['MAX_WORKERS'] = max_workers or '4'
    
    # Project naming strategy
    print(f"\n{Colors.BOLD}Project Naming Strategy:{Colors.ENDC}")
    naming_strategy = get_user_input(
        "Project naming strategy: 'auto' (use folder names) or 'custom' (you specify)",
        "auto"
    )
    config['NAMING_STRATEGY'] = naming_strategy or "auto"
    
    if config['NAMING_STRATEGY'] == 'custom':
        base_name = get_user_input(
            "Base project name (will be numbered: name_001, name_002, etc.)",
            "batch_project"
        )
        config['BASE_PROJECT_NAME'] = base_name or "batch_project"
    
    # Add batch processing metadata
    config['batch_processing'] = True
    config['created_at'] = datetime.now().isoformat()
    config['total_locations'] = len(locations)
    
    print(f"\n{Colors.GREEN}Batch configuration complete!{Colors.ENDC}")
    print(f"   Total locations: {len(locations)}")
    print(f"   Stages per location: {sum(1 for k, v in config.items() if k.startswith('RUN_') and v == 'true')}")
    
    return config

def save_batch_config(config: Dict[str, Any]) -> bool:
    """Save batch configuration to file."""
    try:
        with open(BATCH_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print_success(f"Batch configuration saved to: {BATCH_CONFIG_FILE}")
        return True
    except Exception as e:
        print_error(f"Failed to save batch configuration: {e}")
        return False

def load_batch_config() -> Optional[Dict[str, Any]]:
    """Load batch configuration from file."""
    try:
        if BATCH_CONFIG_FILE.exists():
            with open(BATCH_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            print_success(f"Loaded batch configuration from: {BATCH_CONFIG_FILE}")
            return config
        else:
            return None
    except Exception as e:
        print_error(f"Failed to load batch configuration: {e}")
        return None

# =============================================================================
# BATCH PROGRESS TRACKING
# =============================================================================

def save_batch_progress(progress: Dict[str, Any]) -> bool:
    """Save batch processing progress."""
    try:
        with open(BATCH_PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
        return True
    except Exception as e:
        print_error(f"Failed to save progress: {e}")
        return False

def load_batch_progress() -> Optional[Dict[str, Any]]:
    """Load batch processing progress."""
    try:
        if BATCH_PROGRESS_FILE.exists():
            with open(BATCH_PROGRESS_FILE, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print_error(f"Failed to load progress: {e}")
        return None

# =============================================================================
# BATCH PIPELINE EXECUTION
# =============================================================================

def generate_project_name(location: str, config: Dict[str, Any], location_index: int) -> str:
    """Generate project name for a location using last two parent folders."""
    if config.get('NAMING_STRATEGY') == 'custom':
        base_name = config.get('BASE_PROJECT_NAME', 'batch_project')
        return f"{base_name}_{location_index:03d}"
    else:
        # Auto-generate from last two parent folders
        location_path = Path(location)
        path_parts = location_path.parts
        
        if len(path_parts) >= 2:
            # Take last two parts and join with _._
            project_name = f"{path_parts[-2]}_._{path_parts[-1]}"
        else:
            # Fallback to just the last part
            project_name = path_parts[-1] if path_parts else f"location_{location_index:03d}"
        
        # Clean the project name (remove special characters that might cause issues, preserve _._)
        # First replace the _._ temporarily to preserve it
        project_name = project_name.replace("_._", "|||DELIMITER|||")
        # Clean other characters
        clean_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in project_name)
        # Restore the _._ delimiter
        clean_name = clean_name.replace("|||DELIMITER|||", "_._")
        
        if not clean_name:
            clean_name = f"location_{location_index:03d}"
        
        return clean_name

def run_single_location(location: str, config: Dict[str, Any], venv_path: Path, 
                       location_index: int, total_locations: int) -> bool:
    """Run pipeline for a single location."""
    print_header(f"PROCESSING LOCATION {location_index}/{total_locations}")
    print(f"Location: {location}")
    
    # Generate project name
    project_name = generate_project_name(location, config, location_index)
    print(f"Project name: {project_name}")
    
    # Prepare configuration for this location
    location_config = config.copy()
    location_config['S1_ROOT_DIR'] = location
    location_config['desired_name_of_project'] = project_name
    
    # Set up output directories relative to this project
    data_dir = BASE_DIR / "data" / "processed" / project_name
    location_config['S1_OUTPUT_DIR'] = str(data_dir / "S1_indexed_metadata")
    location_config['S2_INPUT_DIR'] = str(data_dir / "S1_indexed_metadata")
    location_config['S2_OUTPUT_JSON'] = str(data_dir / "S2_concatenated_summaries" / "S2_patient_summary.json")
    location_config['S2_OUTPUT_EXCEL'] = str(data_dir / "S2_concatenated_summaries" / "S2_patient_summary.xlsx")
    location_config['S3_INPUT_DIR'] = str(data_dir / "S1_indexed_metadata")
    location_config['S3_OUTPUT_DIR'] = str(data_dir / "S3_filtered_studies")
    location_config['S4_INPUT_DIR'] = str(data_dir / "S3_filtered_studies")
    location_config['S4_OUTPUT_DIR'] = str(data_dir / "S4_zip_archives")
    location_config['S5_INPUT_FILE'] = str(data_dir / "S1_indexed_metadata" / "S1_indexingFiles_allDocuments.json")
    location_config['S5_OUTPUT_DIR'] = str(data_dir / "S5_llm_extractions")
    
    # For S0 stages, use the location as root directory
    if config.get('RUN_S0_ZIP_EXTRACT') == 'true':
        location_config['S0_ZIP_ROOT_DIR'] = location
    if config.get('RUN_S0_ISO_EXTRACT') == 'true':
        location_config['ISO_SEARCH_DIR'] = location
        location_config['ISO_OUTPUT_DIR'] = 'flat_iso_extract'  # Will be created in location
    
    # Import run_pipeline function from run.py
    try:
        from run import run_pipeline
        success = run_pipeline(location_config, venv_path)
        return success
    except Exception as e:
        print_error(f"Error running pipeline for location {location}: {e}")
        return False

def run_batch_pipeline(config: Dict[str, Any], venv_path: Path, resume: bool = False) -> bool:
    """
    Run pipeline for all locations in batch - SEQUENTIAL PROCESSING.
    
    Each location is processed completely (all stages) before moving to the next.
    This ensures:
    1. No resource conflicts between locations
    2. Clear progress tracking
    3. Ability to resume from any point
    4. Individual project folders for each location
    """
    print_header("BATCH PIPELINE EXECUTION")
    
    locations = config['locations']
    total_locations = len(locations)
    
    # Load progress if resuming
    start_index = 0
    if resume:
        progress = load_batch_progress()
        if progress:
            start_index = progress.get('completed_locations', 0)
            print_success(f"Resuming from location {start_index + 1}/{total_locations}")
    
    # Initialize progress tracking
    progress = {
        'total_locations': total_locations,
        'completed_locations': start_index,
        'failed_locations': [],
        'start_time': datetime.now().isoformat(),
        'current_location': None
    }
    
    print(f"Processing {total_locations} locations SEQUENTIALLY with shared configuration")
    print(f"Each location will be completed fully before moving to the next")
    print(f"Starting from location: {start_index + 1}")
    
    # Process each location SEQUENTIALLY (one after another, not in parallel)
    overall_success = True
    for i in range(start_index, total_locations):
        location = locations[i]
        location_index = i + 1
        
        print(f"\n{'='*60}")
        print(f"LOCATION {location_index}/{total_locations}: {Path(location).name}")
        print(f"{'='*60}")
        
        # Update progress
        progress['current_location'] = location
        progress['current_location_index'] = location_index
        save_batch_progress(progress)
        
        try:
            # Run pipeline for this location (COMPLETE ALL STAGES before moving to next location)
            start_time = time.time()
            success = run_single_location(location, config, venv_path, location_index, total_locations)
            elapsed = time.time() - start_time
            
            if success:
                print_success(f"Location {location_index} completed in {elapsed/60:.1f} minutes")
                progress['completed_locations'] = location_index
            else:
                print_error(f"Location {location_index} failed")
                progress['failed_locations'].append({
                    'location': location,
                    'index': location_index,
                    'error': "Pipeline execution failed"
                })
                overall_success = False
                
                # Ask if user wants to continue
                response = input(f"\nContinue with remaining {total_locations - location_index} locations? (y/n): ")
                if response.lower() != 'y':
                    break
                    
        except KeyboardInterrupt:
            print_warning(f"\nBatch processing interrupted at location {location_index}")
            progress['interrupted_at'] = location_index
            save_batch_progress(progress)
            return False
            
        except Exception as e:
            print_error(f"Unexpected error processing location {location_index}: {e}")
            progress['failed_locations'].append({
                'location': location,
                'index': location_index,
                'error': str(e)
            })
            overall_success = False
        
        # Save progress after each location
        save_batch_progress(progress)
    
    # Final progress update
    progress['end_time'] = datetime.now().isoformat()
    progress['completed'] = True
    save_batch_progress(progress)
    
    return overall_success

# =============================================================================
# REPORTING FUNCTIONS
# =============================================================================

def print_batch_summary(config: Dict[str, Any], progress: Dict[str, Any] = None):
    """Print summary of batch configuration and progress."""
    print_header("BATCH PROCESSING SUMMARY")
    
    if progress is None:
        progress = load_batch_progress()
    
    # Configuration summary
    print(f"\n{Colors.BOLD}Configuration:{Colors.ENDC}")
    print(f"   Total locations: {config.get('total_locations', len(config.get('locations', [])))}")
    print(f"   Naming strategy: {config.get('NAMING_STRATEGY', 'auto')}")
    
    # Stages summary
    enabled_stages = [k.replace('RUN_', '').replace('_', ' ') for k, v in config.items() 
                     if k.startswith('RUN_') and v == 'true']
    print(f"   Enabled stages: {', '.join(enabled_stages)}")
    
    # Progress summary
    if progress:
        print(f"\n{Colors.BOLD}Progress:{Colors.ENDC}")
        completed = progress.get('completed_locations', 0)
        total = progress.get('total_locations', 0)
        failed = len(progress.get('failed_locations', []))
        
        print(f"   Completed: {completed}/{total} locations")
        if failed > 0:
            print(f"   Failed: {failed} locations")
            
        if progress.get('start_time'):
            start_time = datetime.fromisoformat(progress['start_time'])
            if progress.get('end_time'):
                end_time = datetime.fromisoformat(progress['end_time'])
                duration = end_time - start_time
                print(f"   Duration: {duration}")
            else:
                current_time = datetime.now()
                duration = current_time - start_time
                print(f"   Running time: {duration}")
    
    # Locations summary
    if config.get('locations'):
        print(f"\n{Colors.BOLD}Data Locations:{Colors.ENDC}")
        for i, location in enumerate(config['locations'], 1):
            status = "[OK]" if progress and i <= progress.get('completed_locations', 0) else "[PENDING]"
            if progress and any(f['index'] == i for f in progress.get('failed_locations', [])):
                status = "[FAILED]"
            print(f"   {status} {i}. {location}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description="Multi-Location DICOM Processing Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python runMulti.py                    # Interactive multi-location setup
    python runMulti.py --config           # Show current batch configuration
    python runMulti.py --setup            # Setup environment only
    python runMulti.py --resume           # Resume interrupted batch processing
    python runMulti.py --non-interactive  # Use existing configuration
        """
    )
    
    parser.add_argument('--config', action='store_true',
                       help='Show current batch configuration and exit')
    parser.add_argument('--setup', action='store_true',
                       help='Setup virtual environment and install dependencies only')
    parser.add_argument('--skip-setup', action='store_true',
                       help='Skip setup and run batch processing directly')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Run with existing batch configuration without prompts')
    parser.add_argument('--resume', action='store_true',
                       help='Resume interrupted batch processing')
    
    args = parser.parse_args()
    
    # Show configuration and exit
    if args.config:
        config = load_batch_config()
        if config:
            print_batch_summary(config)
        else:
            print_warning("No batch configuration found. Run without --config to create one.")
        return 0
    
    print_header("MULTI-LOCATION DICOM PROCESSING PIPELINE")
    print(f"Repository: {BASE_DIR}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{Colors.BLUE}Welcome to the BATCH processing tool!{Colors.ENDC}")
    print(f"{Colors.GREEN}This tool processes multiple data locations with the same settings{Colors.ENDC}")
    print(f"{Colors.GREEN}   Perfect for processing multiple hospitals, studies, or projects{Colors.ENDC}")
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Setup phase
    if not args.skip_setup:
        print_header("SETUP PHASE")
        
        # Setup virtual environment
        if not setup_virtual_environment(VENV_DIR):
            return 1
        
        # Install dependencies
        if not install_dependencies(VENV_DIR):
            print_warning("Some dependencies failed to install, continuing anyway...")
        
        if args.setup:
            print_success("Setup completed successfully")
            print(f"Next step: Run 'python runMulti.py' to configure batch processing")
            return 0
    
    # Ask user if they want to manually provide info or use configuration file
    if not args.resume and not args.non_interactive:
        print(f"\n{Colors.BLUE}How would you like to provide processing information?{Colors.ENDC}")
        input_method = get_user_input(
            "Enter 'm' for manual input in terminal or 'f' to use a configuration file",
            "m"
        ).lower()
        
        if input_method == 'f':
            # Ask for configuration file location
            config_file_path = get_user_input(
                "Enter the path to the configuration file (or press Enter for default runMulti_config.json)",
                str(BATCH_CONFIG_FILE)
            )
            
            # Load configuration from the specified file
            try:
                config_file = Path(config_file_path)
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    print_success(f"Successfully loaded configuration from {config_file}")
                else:
                    print_error(f"Configuration file not found: {config_file}")
                    print("Please run the script again and choose manual input or provide a valid configuration file.")
                    return 1
            except Exception as e:
                print_error(f"Failed to load configuration file: {e}")
                print("Please run the script again and choose manual input or provide a valid configuration file.")
                return 1
        else:
            # Interactive configuration
            config = configure_batch_pipeline()
            if config is None:
                print_error("Batch configuration cancelled")
                return 1
            
            # Save configuration
            if not save_batch_config(config):
                return 1
    else:
        # Load existing configuration for resume or non-interactive mode
        config = load_batch_config()
        if not config:
            print_error("No batch configuration found. Run without --resume or --non-interactive first.")
            return 1
    
    if not config:
        print_error("No configuration available")
        return 1
    
    # Show summary before starting
    print_batch_summary(config)
    
    if not args.non_interactive and not args.resume:
        print(f"\n{Colors.WARNING}WARNING: About to process {len(config['locations'])} locations{Colors.ENDC}")
        response = input("Proceed with batch processing? (y/n): ")
        if response.lower() != 'y':
            print("Batch processing cancelled")
            return 0
    
    # Run batch pipeline
    print_header("STARTING BATCH PROCESSING")
    start_time = time.time()
    
    success = run_batch_pipeline(config, VENV_DIR, resume=args.resume)
    
    elapsed = time.time() - start_time
    
    # Final summary
    if success:
        print_header("BATCH PROCESSING COMPLETED")
        print_success(f"All locations processed successfully in {elapsed/3600:.1f} hours")
        print(f"\n{Colors.GREEN}Batch processing complete!{Colors.ENDC}")
        print(f"   Check the 'data/processed/' directory for results from each location")
    else:
        print_header("BATCH PROCESSING FINISHED WITH ERRORS")
        print_error("Some locations failed to process")
        print(f"   Total time: {elapsed/3600:.1f} hours")
        print(f"   Use 'python runMulti.py --resume' to continue from where it stopped")
    
    # Show final summary
    progress = load_batch_progress()
    if progress:
        print_batch_summary(config, progress)
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n" + Colors.WARNING + "Batch processing interrupted by user" + Colors.ENDC)
        print("Use 'python runMulti.py --resume' to continue from where it stopped")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)