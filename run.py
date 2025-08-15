#!/usr/bin/env python3
"""
Main orchestrator for DICOM Processing Pipeline
================================================

This script is designed to work with basic Python installation (no dependencies).
It handles the complete pipeline execution including:
1. Virtual environment setup
2. Dependency installation
3. Configuration management
4. Sequential execution of all pipeline stages
5. Progress tracking and reporting

This script DOES NOT require any pip packages to be installed.
It will create a virtual environment and install all dependencies automatically.

Usage:
    python run.py                    # Interactive mode
    python run.py --config           # Show configuration
    python run.py --setup            # Setup only (venv + dependencies)
    python run.py --skip-setup       # Skip setup and run pipeline
    python run.py --stage S1         # Run specific stage only
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

# Ensure we're using Python 3.6+
if sys.version_info < (3, 6):
    print("Error: Python 3.6 or higher is required.")
    print(f"Current version: {sys.version}")
    sys.exit(1)

# Add current directory to path to import run_config
sys.path.insert(0, str(Path(__file__).parent))

# We import run_config later, after ensuring it exists

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent.resolve()
CODE_DIR = BASE_DIR / "code"
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"

# Default virtual environment location
DEFAULT_VENV_DIR = BASE_DIR / "venv"

# Try to import run_config, but don't fail if it doesn't exist
try:
    import run_config
    VENV_DIR = Path(run_config.VENV_PATH) if run_config.VENV_PATH else DEFAULT_VENV_DIR
except ImportError:
    # run_config.py doesn't exist yet, we'll create it
    VENV_DIR = DEFAULT_VENV_DIR
    run_config = None

# Colors for terminal output (Windows compatible)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

def print_step(text: str):
    """Print a step message."""
    print(f"{Colors.BLUE}► {text}{Colors.ENDC}")

def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def get_python_executable(venv_path: Path = None) -> str:
    """Get the Python executable path."""
    if venv_path and venv_path.exists():
        if os.name == 'nt':  # Windows
            python_exe = venv_path / "Scripts" / "python.exe"
        else:  # Unix/Linux/Mac
            python_exe = venv_path / "bin" / "python"
        
        if python_exe.exists():
            return str(python_exe)
    
    return sys.executable

def run_command(cmd: List[str], cwd: str = None, env: Dict = None, 
                timeout: int = None, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env or os.environ.copy(),
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
        return result
    except subprocess.TimeoutExpired:
        print_error(f"Command timed out: {' '.join(cmd)}")
        raise
    except Exception as e:
        print_error(f"Error running command: {e}")
        raise

# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print_error(f"Python 3.7+ required. Current version: {version.major}.{version.minor}")
        return False
    print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def setup_virtual_environment(venv_path: Path) -> bool:
    """Create virtual environment if it doesn't exist."""
    print_step(f"Setting up virtual environment at: {venv_path}")
    
    if venv_path.exists():
        print_success("Virtual environment already exists")
        return True
    
    try:
        # Create parent directories if needed
        venv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create virtual environment
        result = run_command([sys.executable, "-m", "venv", str(venv_path)])
        
        if result.returncode == 0:
            print_success("Virtual environment created successfully")
            return True
        else:
            print_error(f"Failed to create virtual environment: {result.stderr}")
            return False
            
    except Exception as e:
        print_error(f"Error creating virtual environment: {e}")
        return False

def install_dependencies(venv_path: Path) -> bool:
    """Install required Python packages."""
    print_step("Installing dependencies from requirements.txt")
    
    python_exe = get_python_executable(venv_path)
    pip_exe = str(venv_path / "Scripts" / "pip.exe") if os.name == 'nt' else str(venv_path / "bin" / "pip")
    
    if not Path(pip_exe).exists():
        pip_exe = python_exe.replace("python", "pip")
    
    try:
        # Upgrade pip first
        print_step("Upgrading pip...")
        run_command([python_exe, "-m", "pip", "install", "--upgrade", "pip"], capture_output=False)
        
        # Install requirements
        if REQUIREMENTS_FILE.exists():
            print_step("Installing packages...")
            result = run_command(
                [python_exe, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
                capture_output=False
            )
            
            if result.returncode == 0:
                print_success("All dependencies installed successfully")
                return True
            else:
                print_warning("Some dependencies may have failend to install")
                return True  # Continue anyway, some packages are optional
        else:
            print_warning("requirements.txt not found")
            return True
            
    except Exception as e:
        print_error(f"Error installing dependencies: {e}")
        return False

# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def get_user_input(prompt: str, default: str = None, required: bool = False) -> str:
    """Get input from user with optional default value."""
    if default:
        # Enhanced prompt with emoji and clear instructions
        print(f"💡 {prompt}")
        print(f"   Default: {Colors.GREEN}{default}{Colors.ENDC}")
        user_prompt = f"   Press Enter for default or enter new value: "
    else:
        user_prompt = f"{prompt}: "
    
    while True:
        value = input(user_prompt).strip()
        
        if not value and default:
            print(f"   ✓ Using default: {Colors.GREEN}{default}{Colors.ENDC}")
            return default
        
        if value:
            # Remove quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            print(f"   ✓ Using: {Colors.BLUE}{value}{Colors.ENDC}")
            return value
        
        if not required:
            return ""
        
        print_warning("This field is required. Please enter a value.")

def validate_directory(path: str, create: bool = False) -> bool:
    """Validate that a directory exists or can be created."""
    path_obj = Path(path)
    
    if path_obj.exists():
        if path_obj.is_dir():
            return True
        else:
            print_error(f"Path exists but is not a directory: {path}")
            return False
    
    if create:
        try:
            path_obj.mkdir(parents=True, exist_ok=True)
            print_success(f"Created directory: {path}")
            return True
        except Exception as e:
            print_error(f"Failed to create directory: {e}")
            return False
    
    return False

def configure_pipeline() -> Dict[str, str]:
    """Interactive configuration of pipeline parameters."""
    print_header("PIPELINE CONFIGURATION")
    
    # Add helpful tip at the beginning
    print(f"\n{Colors.BLUE}💡 TIP: Most settings have good defaults. Just press Enter to accept them!{Colors.ENDC}")
    print(f"{Colors.GREEN}   This makes setup quick and easy for most users.{Colors.ENDC}")
    
    config = {}
    
    # Ask which stages to run
    print(f"\n{Colors.BOLD}🔧 Select pipeline stages to run:{Colors.ENDC}")
    print(f"{Colors.WARNING}ℹ️  Note: S0 (ISO Extraction) and S4 (ZIP Archives) are optional{Colors.ENDC}")
    print(f"   📋 Required stages: S1, S2, S3")
    print(f"   ⚙️  Optional stages: S0 (if you have ISO files), S4 (if you want ZIP archives)")
    print()
    
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
        print(f"\n{Colors.BOLD}📂 S0 - ZIP Extraction Configuration:{Colors.ENDC}")
        
        # Root directory containing ZIP files
        root_dir = get_user_input(
            "Root directory containing ZIP files",
            run_config.S0_ZIP_ROOT_DIR if run_config and hasattr(run_config, 'S0_ZIP_ROOT_DIR') else os.getcwd()
        )
        config['S0_ZIP_ROOT_DIR'] = root_dir or os.getcwd()
        
        # Number of worker threads
        workers = get_user_input(
            "Number of worker threads for parallel extraction",
            run_config.S0_ZIP_WORKERS if run_config and hasattr(run_config, 'S0_ZIP_WORKERS') else '4'
        )
        config['S0_ZIP_WORKERS'] = workers or '4'
        
        # Integrity checking
        integrity = get_user_input(
            "Enable integrity checking (yes/no)",
            run_config.S0_ZIP_INTEGRITY_CHECK if run_config and hasattr(run_config, 'S0_ZIP_INTEGRITY_CHECK') else 'yes'
        )
        config['S0_ZIP_INTEGRITY_CHECK'] = integrity or 'yes'
        
        # Overwrite existing extractions
        overwrite = get_user_input(
            "Overwrite existing extractions (yes/no)",
            run_config.S0_ZIP_OVERWRITE if run_config and hasattr(run_config, 'S0_ZIP_OVERWRITE') else 'no'
        )
        config['S0_ZIP_OVERWRITE'] = overwrite or 'no'
    
    # Configure S0_ISO if enabled
    if config.get('RUN_S0_ISO_EXTRACT') == 'true':
        print(f"\n{Colors.BOLD}💿 S0 - ISO Extraction Configuration:{Colors.ENDC}")
        
        # PowerISO path
        poweriso_default = run_config.POWERISO_PATH if run_config else r"C:\Users\LEGION\Downloads\PowerISO.9.0.Portable\PowerISO.9.0.Portable\App\PowerISO\piso.exe"
        poweriso_path = get_user_input(
            "Path to PowerISO executable",
            poweriso_default
        )
        if poweriso_path and not Path(poweriso_path).exists():
            print_warning(f"PowerISO not found at: {poweriso_path}")
            print_warning("Please ensure PowerISO is installed before running S0")
        config['POWERISO_PATH'] = poweriso_path
        
        # ISO search directory
        iso_dir = get_user_input(
            "Directory containing ISO files",
            run_config.ISO_SEARCH_DIR if run_config else '',
            required=True
        )
        if not validate_directory(iso_dir):
            print_error(f"Invalid directory: {iso_dir}")
            return None
        config['ISO_SEARCH_DIR'] = iso_dir
    
    # Configure S1 if enabled
    if config.get('RUN_S1_INDEXING') == 'true':
        print(f"\n{Colors.BOLD}📂 S1 - DICOM Indexing Configuration:{Colors.ENDC}")
        
        # Get root directory for DICOM files
        if config.get('RUN_S0_ISO_EXTRACT') == 'true':
            # Use ISO output as default
            iso_output_dir = run_config.ISO_OUTPUT_DIR if run_config else 'flat_iso_extract'
            default_root = str(Path(config.get('ISO_SEARCH_DIR', '')) / iso_output_dir)
        else:
            default_root = run_config.S1_ROOT_DIR if run_config else ''
        
        root_dir = get_user_input(
            "Root directory containing DICOM files",
            default_root,
            required=True
        )
        if not validate_directory(root_dir):
            print_error(f"Invalid directory: {root_dir}")
            return None
        config['S1_ROOT_DIR'] = root_dir
        
        # Output directory
        s1_output_default = run_config.S1_OUTPUT_DIR if run_config else str(BASE_DIR / "data" / "S1_output")
        output_dir = get_user_input(
            "Output directory for S1",
            s1_output_default
        )
        config['S1_OUTPUT_DIR'] = output_dir or str(BASE_DIR / "data" / "S1_output")
    
    # Configure S3 if enabled
    if config.get('RUN_S3_PROCESS') == 'true':
        print(f"\n{Colors.BOLD}⚙️  S3 - Processing Configuration:{Colors.ENDC}")
        
        min_files = get_user_input(
            "Minimum number of files per study",
            run_config.S3_MIN_FILES if run_config else '10'
        )
        config['S3_MIN_FILES'] = min_files or '10'
    
    # Configure S4 if enabled
    if config.get('RUN_S4_ZIP') == 'true':
        print(f"\n{Colors.BOLD}📦 S4 - ZIP Configuration:{Colors.ENDC}")
        
        compression = get_user_input(
            "ZIP compression level (0-9, 9=max)",
            run_config.ZIP_COMPRESSION_LEVEL if run_config else '6'
        )
        config['ZIP_COMPRESSION_LEVEL'] = compression or '6'
    
    # Configure S5 if enabled
    if config.get('RUN_S5_LLM_EXTRACT') == 'true':
        print(f"\n{Colors.BOLD}🤖 S5 - LLM Document Extraction Configuration:{Colors.ENDC}")
        print(f"{Colors.WARNING}⚠️  IMPORTANT: This stage sends document data to an AI model.{Colors.ENDC}")
        print(f"{Colors.WARNING}   Check S5_llmExtract_config.py for API settings.{Colors.ENDC}")
        print(f"{Colors.WARNING}   Ensure HIPAA compliance if using external APIs!{Colors.ENDC}")
        
        # Check if S5 dependencies are available
        try:
            import importlib.util
            pydantic_spec = importlib.util.find_spec("pydantic")
            openai_spec = importlib.util.find_spec("openai")
            ollama_spec = importlib.util.find_spec("ollama")
            if pydantic_spec is None or openai_spec is None:
                print(f"{Colors.WARNING}⚠️  S5 dependencies not fully installed. Configuration will use defaults.{Colors.ENDC}")
                print(f"{Colors.WARNING}   Run 'python run.py --setup' first for full S5 functionality.{Colors.ENDC}")
            if ollama_spec is None:
                print(f"{Colors.WARNING}⚠️  Ollama client not installed. Only OpenAI client will be available for S5.{Colors.ENDC}")
                print(f"{Colors.WARNING}   For local LLM support, run 'python run.py --setup' to install ollama library.{Colors.ENDC}")
        except Exception:
            pass
        
        # Try to import and check LLM config (only if dependencies are installed)
        try:
            # First check if required packages are available
            import importlib.util
            pydantic_spec = importlib.util.find_spec("pydantic")
            if pydantic_spec is None:
                print(f"{Colors.WARNING}⚠️  S5 dependencies not installed yet. S5 configuration will be available after setup.{Colors.ENDC}")
            else:
                from S5_llmExtract_config import LLMConfig
                print(f"\nCurrent LLM Base URL: {LLMConfig.BASE_URL}")
                if LLMConfig.is_local():
                    print(f"{Colors.GREEN}✓ Using LOCAL LLM - Data stays on this machine{Colors.ENDC}")
                else:
                    print(f"{Colors.WARNING}⚠️  Using EXTERNAL API - Data will leave this machine!{Colors.ENDC}")
        except ImportError as e:
            if "pydantic" in str(e) or "openai" in str(e) or "ollama" in str(e):
                print(f"{Colors.WARNING}⚠️  S5 dependencies not fully installed yet. S5 will be available after running setup.{Colors.ENDC}")
            else:
                print(f"{Colors.WARNING}⚠️  S5_llmExtract_config.py not found. It will be needed to run S5.{Colors.ENDC}")
                print(f"{Colors.WARNING}   Error details: {str(e)}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.WARNING}⚠️  Error loading S5 configuration: {str(e)}{Colors.ENDC}")
            print(f"{Colors.WARNING}   S5 may not work properly. Check S5_llmExtract_config.py{Colors.ENDC}")
        
        # Input file path (default from S1 output)
        s1_output = config.get('S1_OUTPUT_DIR', 'data/processed/project/S1_indexed_metadata')
        default_input = f"{s1_output}/S1_indexingFiles_allDocuments.json"
        
        input_file = get_user_input(
            "Input file path (S1_indexingFiles_allDocuments.json)",
            run_config.S5_INPUT_FILE if run_config and hasattr(run_config, 'S5_INPUT_FILE') else default_input
        )
        config['S5_INPUT_FILE'] = input_file or default_input
        
        # Output directory
        parent_dir = Path(config['S5_INPUT_FILE']).parent.parent
        default_output = str(parent_dir / "S5_llm_extractions")
        
        output_dir = get_user_input(
            "Output directory",
            run_config.S5_OUTPUT_DIR if run_config and hasattr(run_config, 'S5_OUTPUT_DIR') else default_output
        )
        config['S5_OUTPUT_DIR'] = output_dir or default_output
        
        # Batch size
        batch_size = get_user_input(
            "Batch size for progress saving",
            run_config.S5_BATCH_SIZE if run_config and hasattr(run_config, 'S5_BATCH_SIZE') else '10'
        )
        config['S5_BATCH_SIZE'] = batch_size or '10'
        
        # Chunk size for large datasets
        chunk_size = get_user_input(
            "Chunk size for large datasets",
            run_config.S5_CHUNK_SIZE if run_config and hasattr(run_config, 'S5_CHUNK_SIZE') else '1000'
        )
        config['S5_CHUNK_SIZE'] = chunk_size or '1000'
    
    # Configuration complete message
    print(f"\n{Colors.GREEN}✅ Configuration complete! Ready to run pipeline.{Colors.ENDC}")
    
    return config

# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def run_stage(stage: str, config: Dict[str, str], venv_path: Path) -> bool:
    """Run a single pipeline stage."""
    python_exe = get_python_executable(venv_path)
    env = os.environ.copy()
    env.update(config)
    
    # Special handling for S5 - Privacy confirmation
    if stage == 'S5':
        # Test if S5 can import using the venv python before proceeding
        test_cmd = [python_exe, '-c', 'from S5_llmExtract_config import LLMConfig; print("SUCCESS")']
        test_result = run_command(test_cmd, capture_output=True)
        
        if test_result.returncode != 0:
            print_error("S5 dependencies (pydantic, openai, ollama) not fully installed in virtual environment.")
            print_error("Please run setup first: python run.py --setup")
            print_error(f"Test error: {test_result.stderr}")
            return False
        
        # Get S5 configuration info using venv python (not local import)
        config_cmd = [python_exe, '-c', 'from S5_llmExtract_config import LLMConfig; print("BASE_URL:", LLMConfig.BASE_URL); print("IS_LOCAL:", LLMConfig.is_local())']
        config_result = run_command(config_cmd, capture_output=True)
        
        if config_result.returncode != 0:
            print_error("Failed to get S5 configuration from virtual environment.")
            print_error(f"Config error: {config_result.stderr}")
            return False
        
        # Parse the output
        output_lines = config_result.stdout.strip().split('\n')
        base_url = next((line.split('BASE_URL: ')[1] for line in output_lines if line.startswith('BASE_URL: ')), "Unknown")
        is_local = next((line.split('IS_LOCAL: ')[1] == 'True' for line in output_lines if line.startswith('IS_LOCAL: ')), False)
        
        # Generate warning message locally (no Unicode issues)
        if is_local:
            warning = f"{Colors.GREEN}✓ Using LOCAL LLM - Data stays on this machine{Colors.ENDC}"
        else:
            warning = f"{Colors.WARNING}⚠️ WARNING: Using EXTERNAL API at {base_url}\n   Patient data will be sent to external servers!\n   Ensure HIPAA compliance and data agreements are in place.{Colors.ENDC}"
        
        print(f"\n{Colors.WARNING}{'='*55}{Colors.ENDC}")
        print(f"{Colors.WARNING}     S5 LLM EXTRACTION - PRIVACY CONFIRMATION{Colors.ENDC}")
        print(f"{Colors.WARNING}{'='*55}{Colors.ENDC}")
        print(f"\nBase URL: {Colors.BOLD}{base_url}{Colors.ENDC}")
        print(warning)
        print(f"\n{Colors.BOLD}Do you want to proceed with S5 LLM extraction? [y/N]:{Colors.ENDC} ", end="")
        
        response = input().strip().lower()
        if response != 'y':
            print("S5 stage skipped by user.")
            return True  # Return True to continue with other stages
    
    stage_scripts = {
        'S0_ZIP': 'S0_zipExtract.py',
        'S0_ISO': 'S0_isoExtract.py',
        'S1': 'S1_indexingFiles_E2.py',
        'S2': 'S2_concatIndex_E2.py',
        'S3': 'S3_processForStore.py',
        'S4': 'S4_zipStore.py',
        'S5': 'S5_llmExtract.py'
    }
    
    script_name = stage_scripts.get(stage)
    if not script_name:
        print_error(f"Unknown stage: {stage}")
        return False
    
    script_path = CODE_DIR / script_name
    if not script_path.exists():
        print_error(f"Script not found: {script_path}")
        return False
    
    print_step(f"Running {stage} - {script_name}")
    
    # Build command based on stage
    cmd = [python_exe, str(script_path)]
    
    if stage == 'S0_ZIP':
        if 'S0_ZIP_ROOT_DIR' in config:
            cmd.extend(['--root-dir', config['S0_ZIP_ROOT_DIR']])
        if 'S0_ZIP_WORKERS' in config:
            cmd.extend(['--workers', config['S0_ZIP_WORKERS']])
        if config.get('S0_ZIP_OVERWRITE', 'no').lower() in ['yes', 'true', '1']:
            cmd.append('--overwrite')
        if config.get('S0_ZIP_INTEGRITY_CHECK', 'yes').lower() in ['no', 'false', '0']:
            cmd.append('--no-integrity-check')
    
    elif stage == 'S0_ISO':
        if 'ISO_SEARCH_DIR' in config:
            cmd.extend(['--search-dir', config['ISO_SEARCH_DIR']])
        if 'ISO_OUTPUT_DIR' in config:
            cmd.extend(['--output-dir', config['ISO_OUTPUT_DIR']])
    
    elif stage == 'S1':
        if 'S1_ROOT_DIR' in config:
            cmd.extend(['--root', config['S1_ROOT_DIR']])
        if 'S1_OUTPUT_DIR' in config:
            cmd.extend(['--output_dir', config['S1_OUTPUT_DIR']])
    
    elif stage == 'S2':
        input_dir = config.get('S2_INPUT_DIR', config.get('S1_OUTPUT_DIR', ''))
        output_json = config.get('S2_OUTPUT_JSON', '')
        output_excel = config.get('S2_OUTPUT_EXCEL', '')
        
        if input_dir:
            cmd.extend(['--folder_path', input_dir])
        if output_json:
            cmd.extend(['--output_json', output_json])
        if output_excel:
            cmd.extend(['--output_excel', output_excel])
        cmd.extend(['--overwrite'])  # Auto-overwrite in pipeline mode
    
    elif stage == 'S3':
        input_dir = config.get('S3_INPUT_DIR', config.get('S1_OUTPUT_DIR', ''))
        output_dir = config.get('S3_OUTPUT_DIR', str(BASE_DIR / "data" / "S3_processForStore"))
        if input_dir:
            cmd.extend(['--input_dir', input_dir])
        cmd.extend(['--output_dir', output_dir])
        cmd.extend(['--min_files', config.get('S3_MIN_FILES', '10')])
    
    elif stage == 'S4':
        input_dir = config.get('S4_INPUT_DIR', config.get('S3_OUTPUT_DIR', ''))
        output_dir = config.get('S4_OUTPUT_DIR', str(BASE_DIR / "data" / "S4_zipStore"))
        if input_dir:
            cmd.extend(['--input-dir', input_dir])
        cmd.extend(['--output-dir', output_dir])
        cmd.extend(['--workers', config.get('S4_WORKERS', '2')])
    
    elif stage == 'S5':
        input_file = config.get('S5_INPUT_FILE', '')
        output_dir = config.get('S5_OUTPUT_DIR', '')
        
        # Auto-generate paths if not provided
        if not input_file:
            s1_output = config.get('S1_OUTPUT_DIR', str(BASE_DIR / "data" / "processed" / "project" / "S1_indexed_metadata"))
            input_file = str(Path(s1_output) / "S1_indexingFiles_allDocuments.json")
        
        if not output_dir:
            parent_dir = Path(input_file).parent.parent if input_file else BASE_DIR / "data" / "processed" / "project"
            output_dir = str(parent_dir / "S5_llm_extractions")
        
        cmd.extend(['--input', input_file])
        cmd.extend(['--output', output_dir])
        
        # Add optional parameters
        batch_size = config.get('S5_BATCH_SIZE', '10')
        chunk_size = config.get('S5_CHUNK_SIZE', '1000')
        overwrite = config.get('S5_OVERWRITE', 'false')
        
        cmd.extend(['--batch-size', batch_size])
        cmd.extend(['--chunk-size', chunk_size])
        
        if overwrite.lower() in ['true', 'yes', '1']:
            cmd.append('--overwrite')
    
    try:
        # Run the stage
        start_time = time.time()
        result = run_command(cmd, cwd=str(CODE_DIR), env=env, capture_output=False)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print_success(f"{stage} completed successfully in {elapsed:.1f} seconds")
            return True
        else:
            print_error(f"{stage} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print_error(f"Error running {stage}: {e}")
        return False

def run_pipeline(config: Dict[str, str], venv_path: Path, stages: List[str] = None) -> bool:
    """Run the complete pipeline or specific stages."""
    print_header("RUNNING PIPELINE")
    
    # Determine which stages to run
    if stages:
        stages_to_run = stages
    else:
        stages_to_run = []
        stage_map = {
            'S0_ZIP': 'RUN_S0_ZIP_EXTRACT',
            'S0_ISO': 'RUN_S0_ISO_EXTRACT',
            'S1': 'RUN_S1_INDEXING',
            'S2': 'RUN_S2_CONCAT',
            'S3': 'RUN_S3_PROCESS',
            'S4': 'RUN_S4_ZIP',
            'S5': 'RUN_S5_LLM_EXTRACT'
        }
        
        for stage, env_var in stage_map.items():
            if config.get(env_var, 'false') == 'true':
                stages_to_run.append(stage)
    
    if not stages_to_run:
        print_warning("No stages selected to run")
        return True
    
    print(f"Stages to run: {', '.join(stages_to_run)}")
    
    # Create data directory if needed
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Run each stage
    success = True
    for i, stage in enumerate(stages_to_run, 1):
        print(f"\n[{i}/{len(stages_to_run)}] Stage {stage}")
        print("-" * 40)
        
        if not run_stage(stage, config, venv_path):
            print_error(f"Stage {stage} failed")
            response = input("Continue with next stage? (y/n): ")
            if response.lower() != 'y':
                success = False
                break
    
    return success

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DICOM Processing Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', action='store_true',
                       help='Show current configuration and exit')
    parser.add_argument('--setup', action='store_true',
                       help='Setup virtual environment and install dependencies only')
    parser.add_argument('--skip-setup', action='store_true',
                       help='Skip setup and run pipeline directly')
    parser.add_argument('--stage', choices=['S0', 'S1', 'S2', 'S3', 'S4'],
                       help='Run specific stage only')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Run with current configuration without prompts')
    
    args = parser.parse_args()
    
    # Show configuration and exit
    if args.config:
        if run_config:
            run_config.print_config()
        else:
            print_warning("run_config.py not found. Run setup first.")
        return 0
    
    print_header("DICOM PROCESSING PIPELINE")
    print(f"📁 Repository: {BASE_DIR}")
    print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{Colors.BLUE}🚀 Welcome! This tool will guide you through DICOM processing.{Colors.ENDC}")
    print(f"{Colors.GREEN}💡 Quick tip: Most settings have smart defaults - just press Enter!{Colors.ENDC}")
    
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
            return 0
    
    # Load configuration
    if run_config:
        run_config.set_environment_variables()
        config = run_config.get_config_dict()
    else:
        print_warning("run_config.py not found. Using default configuration.")
        config = {
            'BASE_DIR': str(BASE_DIR),
            'CODE_DIR': str(CODE_DIR),
            'VENV_PATH': str(VENV_DIR),
            'RUN_S1_INDEXING': 'true',
            'RUN_S2_CONCAT': 'true',
            'RUN_S3_PROCESS': 'true',
            'RUN_S4_ZIP': 'false',      # Changed to false (optional)
            'RUN_S0_ZIP_EXTRACT': 'false',  # ZIP extraction (optional)
            'RUN_S0_ISO_EXTRACT': 'false',  # ISO extraction (optional)
            'RUN_S5_LLM_EXTRACT': 'false'   # AI extraction (optional)
        }
    
    # Interactive configuration if not non-interactive
    if not args.non_interactive and not args.stage:
        user_config = configure_pipeline()
        if user_config is None:
            print_error("Configuration cancelled")
            return 1
        config.update(user_config)
    
    # Run pipeline
    if args.stage:
        # Run single stage
        stages = [args.stage]
    else:
        stages = None
    
    success = run_pipeline(config, VENV_DIR, stages)
    
    if success:
        print_header("PIPELINE COMPLETED")
        print_success("All selected stages completed successfully")
        return 0
    else:
        print_header("PIPELINE FAILED")
        print_error("Pipeline execution failed")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n" + Colors.WARNING + "Pipeline interrupted by user" + Colors.ENDC)
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)