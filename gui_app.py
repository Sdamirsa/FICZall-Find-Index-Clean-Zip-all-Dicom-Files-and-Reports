#!/usr/bin/env python3
"""
DICOM Processing Pipeline GUI
============================

A simple, user-friendly GUI application for running DICOM processing pipelines.
Supports both single-location (run.py) and multi-location (runMulti.py) processing.

Features:
- Simple tabbed interface
- Memory system for recent inputs
- Real-time progress monitoring
- Built-in log viewer
- Virtual environment integration

Author: Claude Code Assistant
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
import queue
import pickle
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def get_system_info():
    """Get system information for optimal worker count recommendation."""
    try:
        import multiprocessing
        import shutil
        
        # Get CPU count
        cpu_count = multiprocessing.cpu_count()
        
        # Get available memory (cross-platform approach)
        try:
            # Try psutil first if available
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            memory_available_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            # Fallback for systems without psutil
            if os.name == 'nt':  # Windows
                try:
                    # Windows memory detection
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    c_ulong = ctypes.c_ulong
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ('dwLength', c_ulong),
                            ('dwMemoryLoad', c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong),
                            ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong),
                            ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong),
                            ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                        ]
                    
                    memInfo = MEMORYSTATUSEX()
                    memInfo.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    result = kernel32.GlobalMemoryStatusEx(ctypes.byref(memInfo))
                    if result:
                        memory_gb = memInfo.ullTotalPhys / (1024**3)
                        memory_available_gb = memInfo.ullAvailPhys / (1024**3)
                    else:
                        # Fallback if the call fails - couldn't retrieve memory
                        memory_gb = 0.0
                        memory_available_gb = 0.0
                except:
                    memory_gb = 8.0  # Default assumption
                    memory_available_gb = 4.0
            else:
                # Unix/Linux/Mac fallback
                try:
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = f.read()
                    total_mem = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1]) * 1024
                    available_mem = int([line for line in meminfo.split('\n') if 'MemAvailable' in line][0].split()[1]) * 1024
                    memory_gb = total_mem / (1024**3)
                    memory_available_gb = available_mem / (1024**3)
                except:
                    # Mac doesn't have /proc/meminfo, use vm_stat command
                    try:
                        import subprocess
                        result = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            lines = result.stdout.strip().split('\n')
                            page_size = 4096  # Default Mac page size
                            for line in lines:
                                if 'page size of' in line:
                                    page_size = int(line.split()[-2])
                                    break
                            
                            free_pages = 0
                            for line in lines:
                                if 'Pages free:' in line:
                                    free_pages = int(line.split()[-1].rstrip('.'))
                                    break
                            
                            memory_gb = 16.0  # Mac default assumption  
                            memory_available_gb = (free_pages * page_size) / (1024**3) if free_pages > 0 else 8.0
                        else:
                            memory_gb = 16.0  # Mac default assumption
                            memory_available_gb = 8.0
                    except:
                        memory_gb = 8.0  # Default assumption
                        memory_available_gb = 4.0
        
        return {
            'cpu_count': cpu_count,
            'total_memory_gb': round(memory_gb, 1),
            'available_memory_gb': round(memory_available_gb, 1),
            'platform': sys.platform
        }
    except Exception as e:
        # Fallback if all detection fails
        return {
            'cpu_count': 4,
            'total_memory_gb': 8.0,
            'available_memory_gb': 4.0,
            'platform': sys.platform,
            'error': str(e)
        }

def get_recommended_workers(system_info=None):
    """Get recommended number of workers based on system capabilities."""
    if system_info is None:
        system_info = get_system_info()
    
    cpu_count = system_info['cpu_count']
    memory_gb = system_info['available_memory_gb']
    
    # Conservative worker calculation
    # Rule: Don't use all CPUs, leave some for OS and other processes
    # Rule: Each worker needs ~1-2GB RAM for DICOM processing
    
    # CPU-based limit (use 75% of CPUs, minimum 1, maximum 8)
    cpu_workers = max(1, min(8, int(cpu_count * 0.75)))
    
    # Memory-based limit (assume 1.5GB per worker)
    memory_workers = max(1, int(memory_gb / 1.5))
    
    # Take the smaller of the two limits
    recommended = min(cpu_workers, memory_workers)
    
    # Ensure reasonable bounds
    recommended = max(2, min(8, recommended))
    
    return recommended

def get_worker_hint(system_info=None):
    """Get user-friendly hint about worker count."""
    if system_info is None:
        system_info = get_system_info()
    
    recommended = get_recommended_workers(system_info)
    cpu_count = system_info['cpu_count']
    memory_gb = system_info['total_memory_gb']
    
    hint = f"Recommended: {recommended} "
    
    if memory_gb > 0:
        hint += f"(System: {cpu_count} CPUs, {memory_gb}GB RAM)"
        if memory_gb < 8:
            hint += " - Low memory detected"
        elif memory_gb >= 16:
            hint += " - High memory available"
    else:
        hint += f"(System: {cpu_count} CPUs, couldn't retrieve memory)"
    
    return hint

# Define constants and functions locally to avoid importing run.py on startup
DEFAULT_VENV_DIR = Path("venv")

def get_python_executable(venv_path=None):
    """Get Python executable, preferring virtual environment if available."""
    if venv_path and venv_path.exists():
        if os.name == 'nt':  # Windows
            python_exe = venv_path / "Scripts" / "python.exe"
        else:  # Unix/Linux/Mac
            python_exe = venv_path / "bin" / "python"
        
        if python_exe.exists():
            return str(python_exe)
    return sys.executable

class MemoryManager:
    """Manages user input memory using pickle."""
    
    def __init__(self, memory_file: str = "gui_memory.pkl"):
        self.memory_file = Path(memory_file)
        self.memory: Dict[str, List[str]] = {}
        self.max_items = 3
        self.load_memory()
    
    def load_memory(self):
        """Load memory from pickle file."""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'rb') as f:
                    self.memory = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load memory: {e}")
            self.memory = {}
    
    def save_memory(self):
        """Save memory to pickle file."""
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.memory, f)
        except Exception as e:
            print(f"Warning: Could not save memory: {e}")
    
    def add_entry(self, key: str, value: str):
        """Add a new entry to memory."""
        if not value or not value.strip():
            return
        
        if key not in self.memory:
            self.memory[key] = []
        
        # Remove if already exists
        if value in self.memory[key]:
            self.memory[key].remove(value)
        
        # Add to beginning
        self.memory[key].insert(0, value)
        
        # Keep only max_items
        if len(self.memory[key]) > self.max_items:
            self.memory[key] = self.memory[key][:self.max_items]
        
        self.save_memory()
    
    def get_entries(self, key: str) -> List[str]:
        """Get recent entries for a key."""
        return self.memory.get(key, [])

class ProcessRunner:
    """Handles running subprocess with real-time output."""
    
    def __init__(self, output_callback):
        self.output_callback = output_callback
        self.process = None
        self.is_running = False
    
    def run_command(self, cmd: List[str], cwd: str = None, env: Dict[str, str] = None):
        """Run command with real-time output."""
        self.is_running = True
        output_queue = queue.Queue()
        
        def enqueue_output(out, q):
            """Read output from subprocess and put in queue."""
            try:
                for line in iter(out.readline, ''):
                    if not self.is_running:
                        break
                    line_content = line.rstrip()
                    # Print to terminal (console) as well as sending to GUI
                    print(line_content)
                    q.put(('stdout', line_content))
                q.put(('done', None))
            except Exception as e:
                error_msg = str(e)
                print(f"[DEBUG] Output read error: {error_msg}")
                q.put(('error', error_msg))
            finally:
                out.close()
        
        try:
            # Set environment to ensure unbuffered output
            process_env = os.environ.copy()
            process_env['PYTHONUNBUFFERED'] = '1'
            
            # Add any additional environment variables
            if env:
                process_env.update(env)
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd,
                bufsize=1,
                universal_newlines=True,
                env=process_env
            )
            
            # Start thread to read output
            output_thread = threading.Thread(target=enqueue_output, 
                                           args=(self.process.stdout, output_queue))
            output_thread.daemon = True
            output_thread.start()
            
            # Process output in real-time
            done = False
            while not done and self.is_running:
                try:
                    # Check for output with timeout
                    msg_type, content = output_queue.get(timeout=0.5)
                    if msg_type == 'stdout':
                        self.output_callback(content)
                    elif msg_type == 'error':
                        self.output_callback(f"[DEBUG] Output read error: {content}")
                    elif msg_type == 'done':
                        done = True
                except queue.Empty:
                    # Check if process is still running
                    if self.process.poll() is not None:
                        break
                    continue
            
            # Wait for process to complete
            self.process.wait()
            
            # Process any remaining queued output
            while not output_queue.empty():
                try:
                    msg_type, content = output_queue.get_nowait()
                    if msg_type == 'stdout':
                        self.output_callback(content)
                except queue.Empty:
                    break
            
            completion_msg = f"\n[COMPLETED] Process finished with exit code: {self.process.returncode}"
            print(completion_msg)  # Also print to console
            self.output_callback(completion_msg)
            
        except Exception as e:
            error_msg = f"\n[ERROR] Failed to run command: {e}"
            print(error_msg)  # Also print to console
            self.output_callback(error_msg)
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the running process."""
        self.is_running = False
        if self.process:
            self.process.terminate()

class SingleLocationTab:
    """Tab for single location processing (run.py)."""
    
    def __init__(self, parent, memory_manager: MemoryManager, process_callback):
        self.memory_manager = memory_manager
        self.process_callback = process_callback
        
        # Main frame
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title = ttk.Label(main_frame, text="Single Location DICOM Processing", 
                         font=('Arial', 14, 'bold'))
        title.pack(pady=(0, 5))
        
        # Info label
        info_label = ttk.Label(main_frame, 
                              text="Process DICOM files from one location. S1-S3 are required for basic processing.",
                              font=('Arial', 9), foreground='blue')
        info_label.pack(pady=(0, 10))
        
        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Configuration", padding=10)
        input_frame.pack(fill='x', pady=(0, 10))
        
        # DICOM data location
        ttk.Label(input_frame, text="DICOM Data Location:").grid(row=0, column=0, sticky='w', pady=5)
        self.data_location_var = tk.StringVar()
        self.data_location_combo = ttk.Combobox(input_frame, textvariable=self.data_location_var, 
                                              width=50, values=self.memory_manager.get_entries('single_data_location'))
        self.data_location_combo.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        ttk.Button(input_frame, text="Browse", 
                  command=self.browse_data_location).grid(row=0, column=2, padx=5)
        
        # Project name
        ttk.Label(input_frame, text="Project Name (optional):").grid(row=1, column=0, sticky='w', pady=5)
        self.project_name_var = tk.StringVar()
        self.project_name_combo = ttk.Combobox(input_frame, textvariable=self.project_name_var,
                                             width=50, values=self.memory_manager.get_entries('single_project_name'))
        self.project_name_combo.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        
        # Stages frame
        stages_frame = ttk.LabelFrame(input_frame, text="Pipeline Stages")
        stages_frame.grid(row=2, column=0, columnspan=3, sticky='ew', pady=10)
        
        self.stage_vars = {}
        stages = [
            ('S0_ZIP', 'S0a - ZIP File Extraction (Optional)', False),
            ('S0_ISO', 'S0b - ISO File Extraction (Optional)', False),
            ('S1', 'S1 - DICOM Indexing (Required)', True),
            ('S2', 'S2 - Concatenate Index (Required)', True),
            ('S3', 'S3 - Process for Storage (Required)', True),
            ('S4', 'S4 - Create ZIP Archives (Optional)', False),
            ('S5', 'S5 - AI Document Extraction (Optional)', False)
        ]
        
        # Arrange stages in single column (S0 to S5)
        for i, (key, desc, default) in enumerate(stages):
            var = tk.BooleanVar(value=default)
            self.stage_vars[key] = var
            ttk.Checkbutton(stages_frame, text=desc, variable=var).grid(row=i, column=0, sticky='w', padx=10, pady=2)
        
        # Advanced settings frame
        advanced_frame = ttk.LabelFrame(input_frame, text="Advanced Settings")
        advanced_frame.grid(row=3, column=0, columnspan=3, sticky='ew', pady=10)
        
        # Min files with hint
        ttk.Label(advanced_frame, text="Min files per study:").grid(row=0, column=0, sticky='w', padx=5)
        self.min_files_var = tk.StringVar(value="10")
        min_files_entry = ttk.Entry(advanced_frame, textvariable=self.min_files_var, width=10)
        min_files_entry.grid(row=0, column=1, padx=5)
        
        # Min files hint
        min_files_hint = ttk.Label(advanced_frame, text="(5-20 typical, lower = more studies included)", 
                                  font=('Arial', 8), foreground='gray')
        min_files_hint.grid(row=1, column=0, columnspan=2, sticky='w', padx=5)
        
        # Max workers with hint
        ttk.Label(advanced_frame, text="Max workers:").grid(row=0, column=2, sticky='w', padx=5)
        recommended_workers = get_recommended_workers()
        self.max_workers_var = tk.StringVar(value=str(recommended_workers))
        max_workers_entry = ttk.Entry(advanced_frame, textvariable=self.max_workers_var, width=10)
        max_workers_entry.grid(row=0, column=3, padx=5)
        
        # Max workers hint with system detection
        worker_hint_text = get_worker_hint()
        max_workers_hint = ttk.Label(advanced_frame, text=worker_hint_text, 
                                    font=('Arial', 8), foreground='gray')
        max_workers_hint.grid(row=1, column=2, columnspan=2, sticky='w', padx=5)
        
        # Configure grid weights
        input_frame.columnconfigure(1, weight=1)
        stages_frame.columnconfigure(0, weight=1)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=5)
        
        self.run_button = ttk.Button(button_frame, text="Start Processing", 
                                    command=self.start_processing, style='Accent.TButton')
        self.run_button.pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="Setup Only", 
                  command=self.setup_only).pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="Show Config", 
                  command=self.show_config).pack(side='left', padx=5)
    
    def browse_data_location(self):
        """Browse for DICOM data location."""
        folder = filedialog.askdirectory(title="Select DICOM Data Location")
        if folder:
            self.data_location_var.set(folder)
    
    def start_processing(self):
        """Start the processing pipeline."""
        data_location = self.data_location_var.get().strip()
        if not data_location:
            messagebox.showerror("Error", "Please specify a DICOM data location")
            return
        
        if not os.path.exists(data_location):
            messagebox.showerror("Error", f"Data location does not exist: {data_location}")
            return
        
        # Save to memory
        self.memory_manager.add_entry('single_data_location', data_location)
        project_name = self.project_name_var.get().strip()
        if project_name:
            self.memory_manager.add_entry('single_project_name', project_name)
        
        # Update comboboxes
        self.data_location_combo['values'] = self.memory_manager.get_entries('single_data_location')
        self.project_name_combo['values'] = self.memory_manager.get_entries('single_project_name')
        
        # Build command
        python_exe = get_python_executable(DEFAULT_VENV_DIR)
        cmd = [python_exe, "run.py", "--non-interactive"]
        
        # Set environment variables
        env = os.environ.copy()
        env['S1_ROOT_DIR'] = data_location
        if project_name:
            env['desired_name_of_project'] = project_name
        
        # Stage configuration
        for stage_key, var in self.stage_vars.items():
            stage_map = {
                'S0_ZIP': 'RUN_S0_ZIP_EXTRACT',
                'S0_ISO': 'RUN_S0_ISO_EXTRACT',
                'S1': 'RUN_S1_INDEXING',
                'S2': 'RUN_S2_CONCAT',
                'S3': 'RUN_S3_PROCESS',
                'S4': 'RUN_S4_ZIP',
                'S5': 'RUN_S5_LLM_EXTRACT'
            }
            env_var = stage_map.get(stage_key)
            if env_var:
                env[env_var] = 'true' if var.get() else 'false'
        
        # Advanced settings
        env['S3_MIN_FILES'] = self.min_files_var.get()
        env['MAX_WORKERS'] = self.max_workers_var.get()
        
        # CRITICAL: Set the project name from GUI (this overrides run_config.py auto-naming)
        project_name = self.project_name_var.get().strip()
        if project_name:
            env['desired_name_of_project'] = project_name
            print(f"[GUI] Using project name from GUI: {project_name}")
        
        # If S5 is enabled, apply LLM configuration from the LLM Config tab
        if self.stage_vars.get('S5', tk.BooleanVar()).get():
            try:
                # Get the app instance through the process_callback to access LLM config tab
                app_instance = getattr(self.process_callback, '__self__', None)
                llm_tab = getattr(app_instance, 'llm_config_tab', None) if app_instance else None
                if llm_tab:
                    env['S5_LLM_CLIENT_TYPE'] = llm_tab.client_type_var.get()
                    env['OPENAI_BASE_URL'] = llm_tab.base_url_var.get()
                    env['OLLAMA_HOST'] = llm_tab.ollama_host_var.get()
                    env['S5_LLM_MODEL'] = llm_tab.model_name_var.get()
                    env['S5_LLM_TEMPERATURE'] = llm_tab.temperature_var.get()
                    env['S5_LLM_MAX_TOKENS'] = llm_tab.max_tokens_var.get()
                    env['S5_LLM_BATCH_SIZE'] = llm_tab.batch_size_var.get()
                    env['S5_LLM_CHUNK_SIZE'] = llm_tab.chunk_size_var.get()
                    env['S5_LLM_MAX_RETRIES'] = llm_tab.max_retries_var.get()
                    env['S5_LLM_PARALLEL_WORKERS'] = llm_tab.parallel_workers_var.get()
                    
                    # Apply API key if not a placeholder
                    api_key = llm_tab.api_key_var.get().strip()
                    if api_key and not api_key.startswith('*'):
                        env['OPENAI_API_KEY'] = api_key
            except Exception as e:
                print(f"Warning: Could not apply LLM configuration: {e}")
        
        # Setup status tracking for enabled stages only
        enabled_stages = [key for key, var in self.stage_vars.items() if var.get()]
        app_instance = getattr(self.process_callback, '__self__', None)
        if app_instance and hasattr(app_instance, 'setup_status_for_enabled_stages'):
            app_instance.setup_status_for_enabled_stages(enabled_stages)
        
        # Start processing
        self.process_callback(cmd, env)
    
    def setup_only(self):
        """Run setup only."""
        python_exe = get_python_executable(DEFAULT_VENV_DIR)
        cmd = [python_exe, "run.py", "--setup"]
        self.process_callback(cmd)
    
    def show_config(self):
        """Show current configuration."""
        python_exe = get_python_executable(DEFAULT_VENV_DIR)
        cmd = [python_exe, "run.py", "--config"]
        self.process_callback(cmd)

class MultiLocationTab:
    """Tab for multi-location processing (currently not implemented)."""
    
    def __init__(self, parent, memory_manager: MemoryManager, process_callback):
        self.memory_manager = memory_manager
        self.process_callback = process_callback
        
        # Main frame
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Center frame for the message
        center_frame = ttk.Frame(main_frame)
        center_frame.pack(expand=True)
        
        # Large warning message
        warning_label = ttk.Label(center_frame, 
                                 text="MULTI-LOCATION BATCH PROCESSING", 
                                 font=('Arial', 20, 'bold'),
                                 foreground='red')
        warning_label.pack(pady=20)
        
        # Not implemented message
        message_label = ttk.Label(center_frame, 
                                 text="NOT IMPLEMENTED YET", 
                                 font=('Arial', 16, 'bold'),
                                 foreground='red')
        message_label.pack(pady=10)
        
        # Explanation
        explanation_label = ttk.Label(center_frame, 
                                     text="This feature is currently under development.\n\n"
                                          "For batch processing of multiple locations, please use:\n"
                                          "• Single Location tab for individual processing\n"
                                          "• Command line: python runMulti.py (advanced users)\n\n"
                                          "This GUI implementation will be available in a future update.",
                                     font=('Arial', 11),
                                     foreground='black',
                                     justify='center')
        explanation_label.pack(pady=20)
        
        # Alternative suggestion frame
        alt_frame = ttk.LabelFrame(center_frame, text="Alternative Solutions", padding=20)
        alt_frame.pack(pady=20, padx=40, fill='x')
        
        alt_text = ttk.Label(alt_frame, 
                           text="1. Use the 'Single Location' tab to process one location at a time\n"
                                "2. Advanced users can use: python runMulti.py from command line\n"
                                "3. Contact support for assistance with batch processing",
                           font=('Arial', 10),
                           justify='left')
        alt_text.pack()
        
        # Add a dummy batch_run_button for compatibility with main GUI
        # This prevents AttributeError when main GUI tries to disable it
        self.batch_run_button = ttk.Button(alt_frame, text="Not Available", state='disabled')
        # Don't pack it - it's just for compatibility

class LLMConfigTab:
    """Tab for LLM configuration (S5 stage)."""
    
    def __init__(self, parent, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self._initializing = True  # Flag to prevent callbacks during init
        
        # Main frame
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title = ttk.Label(main_frame, text="LLM Configuration for S5 AI Document Extraction", 
                         font=('Arial', 14, 'bold'))
        title.pack(pady=(0, 5))
        
        # Warning label
        warning_label = ttk.Label(main_frame, 
                                 text="Configure AI settings for medical document extraction. Changes modify S5_llmExtract_config.py",
                                 font=('Arial', 9), foreground='orange')
        warning_label.pack(pady=(0, 10))
        
        # Create scrollable frame with fixed height to prevent overlap
        canvas = tk.Canvas(main_frame, height=500)  # Fixed height to prevent overlap
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Enable mouse wheel scrolling on canvas and all child widgets
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_mousewheel(widget):
            """Recursively bind mouse wheel to widget and all its children"""
            widget.bind("<MouseWheel>", _on_mousewheel)
            for child in widget.winfo_children():
                _bind_mousewheel(child)
        
        canvas.bind("<MouseWheel>", _on_mousewheel)
        # Bind to all child widgets after they are created (done later)
        
        # Client Configuration
        client_frame = ttk.LabelFrame(scrollable_frame, text="Client Configuration", padding=10)
        client_frame.pack(fill='x', pady=(0, 10))
        
        # Client Type
        ttk.Label(client_frame, text="Client Type:").grid(row=0, column=0, sticky='w', pady=5)
        self.client_type_var = tk.StringVar(value="ollama")
        client_combo = ttk.Combobox(client_frame, textvariable=self.client_type_var, 
                                   values=["auto", "openai", "ollama"], state="readonly", width=15)
        client_combo.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        # Client type hint
        client_hint = ttk.Label(client_frame, 
                               text="auto: detects based on URL | openai: force OpenAI API | ollama: force local Ollama",
                               font=('Arial', 8), foreground='gray')
        client_hint.grid(row=0, column=2, padx=10, pady=5, sticky='w')
        
        # API Configuration
        api_frame = ttk.LabelFrame(scrollable_frame, text="API Configuration", padding=10)
        api_frame.pack(fill='x', pady=(0, 10))
        
        # Base URL
        ttk.Label(api_frame, text="Base URL:").grid(row=0, column=0, sticky='w', pady=5)
        self.base_url_var = tk.StringVar(value="http://localhost:11434/v1")
        ttk.Entry(api_frame, textvariable=self.base_url_var, width=50).grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        # Base URL hint
        base_url_hint = ttk.Label(api_frame, 
                                 text="OpenAI: https://api.openai.com/v1 | Ollama: http://localhost:11434/v1",
                                 font=('Arial', 8), foreground='gray')
        base_url_hint.grid(row=0, column=2, padx=10, pady=5, sticky='w')
        
        # API Key
        ttk.Label(api_frame, text="API Key (for OpenAI):").grid(row=1, column=0, sticky='w', pady=5)
        self.api_key_var = tk.StringVar()
        api_key_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=50, show="*")
        api_key_entry.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        
        # API Key load from .env button
        load_env_button = ttk.Button(api_frame, text="Load from .env", 
                                   command=self.load_api_key_from_env, width=12)
        load_env_button.grid(row=1, column=2, padx=5, pady=5, sticky='w')
        
        # API Key security note
        api_security_note = ttk.Label(api_frame, 
                                     text="API key will be saved to .env file (not config files or memory pickle)",
                                     font=('Arial', 8), foreground='orange')
        api_security_note.grid(row=2, column=0, columnspan=3, sticky='w', padx=5, pady=2)
        
        # Ollama Host
        ttk.Label(api_frame, text="Ollama Host:").grid(row=3, column=0, sticky='w', pady=5)
        self.ollama_host_var = tk.StringVar(value="http://localhost:11434")
        ttk.Entry(api_frame, textvariable=self.ollama_host_var, width=50).grid(row=3, column=1, padx=5, pady=5, sticky='ew')
        
        # Ollama Host hint
        ollama_hint = ttk.Label(api_frame, 
                               text="Default: http://localhost:11434 (for local Ollama server)",
                               font=('Arial', 8), foreground='gray')
        ollama_hint.grid(row=3, column=2, padx=10, pady=5, sticky='w')
        
        # Model Configuration
        model_frame = ttk.LabelFrame(scrollable_frame, text="Model Configuration", padding=10)
        model_frame.pack(fill='x', pady=(0, 10))
        
        # Model Name
        ttk.Label(model_frame, text="Model Name:").grid(row=0, column=0, sticky='w', pady=5)
        self.model_name_var = tk.StringVar(value="qwen3:4b-instruct-2507-q8_0")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_name_var, width=40,
                                  values=["qwen3:4b-instruct-2507-q8_0", "gpt-4o-mini", "gpt-4o", "llama3.1:8b", "llama3.2:3b", "llama3-groq-tool-use:8b-q8_0"])
        model_combo.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        # Temperature and Max Tokens
        ttk.Label(model_frame, text="Temperature:").grid(row=1, column=0, sticky='w', pady=5)
        self.temperature_var = tk.StringVar(value="0.3")
        ttk.Entry(model_frame, textvariable=self.temperature_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Label(model_frame, text="Max Tokens:").grid(row=1, column=2, sticky='w', pady=5, padx=20)
        self.max_tokens_var = tk.StringVar(value="4096")
        ttk.Entry(model_frame, textvariable=self.max_tokens_var, width=10).grid(row=1, column=3, padx=5, pady=5, sticky='w')
        
        # Processing Configuration
        processing_frame = ttk.LabelFrame(scrollable_frame, text="Processing Configuration", padding=10)
        processing_frame.pack(fill='x', pady=(0, 10))
        
        # Batch Size and Chunk Size
        ttk.Label(processing_frame, text="Batch Size:").grid(row=0, column=0, sticky='w', pady=5)
        self.batch_size_var = tk.StringVar(value="10")
        ttk.Entry(processing_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky='w')
        batch_hint = ttk.Label(processing_frame, text="documents per batch (10-50 recommended)",
                              font=('Arial', 8), foreground='gray')
        batch_hint.grid(row=0, column=2, padx=10, pady=5, sticky='w')
        
        ttk.Label(processing_frame, text="Chunk Size:").grid(row=1, column=0, sticky='w', pady=5)
        self.chunk_size_var = tk.StringVar(value="1000")
        ttk.Entry(processing_frame, textvariable=self.chunk_size_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky='w')
        chunk_hint = ttk.Label(processing_frame, text="characters per document chunk (500-2000 recommended)",
                              font=('Arial', 8), foreground='gray')
        chunk_hint.grid(row=1, column=2, padx=10, pady=5, sticky='w')
        
        # Max Retries and Parallel Workers
        ttk.Label(processing_frame, text="Max Retries:").grid(row=2, column=0, sticky='w', pady=5)
        self.max_retries_var = tk.StringVar(value="3")
        ttk.Entry(processing_frame, textvariable=self.max_retries_var, width=10).grid(row=2, column=1, padx=5, pady=5, sticky='w')
        retry_hint = ttk.Label(processing_frame, text="retries on API failures (2-5 recommended)",
                              font=('Arial', 8), foreground='gray')
        retry_hint.grid(row=2, column=2, padx=10, pady=5, sticky='w')
        
        ttk.Label(processing_frame, text="Parallel Workers:").grid(row=3, column=0, sticky='w', pady=5)
        self.parallel_workers_var = tk.StringVar(value="1")
        self.parallel_workers_entry = ttk.Entry(processing_frame, textvariable=self.parallel_workers_var, width=10)
        self.parallel_workers_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        self.parallel_hint = ttk.Label(processing_frame, text="LOCKED to 1 (no async support currently)",
                                      font=('Arial', 8), foreground='red')
        self.parallel_hint.grid(row=3, column=2, padx=10, pady=5, sticky='w')
        
        # Lock parallel workers to 1 and disable entry since no async support
        self.parallel_workers_var.set("1")
        self.parallel_workers_entry.config(state='disabled')
        
        # Privacy Status
        privacy_frame = ttk.LabelFrame(scrollable_frame, text="Privacy Status", padding=10)
        privacy_frame.pack(fill='x', pady=(0, 10))
        
        self.privacy_label = ttk.Label(privacy_frame, text="Status will update when you change settings", 
                                      font=('Arial', 10), foreground='blue', wraplength=500)
        self.privacy_label.pack(pady=5)
        
        # Pydantic Models Info
        pydantic_frame = ttk.LabelFrame(scrollable_frame, text="Data Extraction Models (Read-Only)", padding=10)
        pydantic_frame.pack(fill='x', pady=(0, 10))
        
        pydantic_info = ttk.Label(pydantic_frame, 
                                 text="To modify what medical data is extracted (patient info, findings, diagnoses, etc.),\nedit the Pydantic models directly in S5_llmExtract_config.py\n\nAvailable models: SimpleMedicalExtraction, MedicalReportExtraction",
                                 font=('Arial', 9), foreground='gray')
        pydantic_info.pack(pady=5)
        
        # Buttons - organized in two rows
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill='x', pady=10)
        
        # First row - Configuration buttons
        config_buttons = ttk.Frame(button_frame)
        config_buttons.pack(fill='x', pady=(0, 5))
        
        ttk.Button(config_buttons, text="Load Current Config", 
                  command=lambda: self.load_current_config(show_message=True)).pack(side='left', padx=5)
        
        ttk.Button(config_buttons, text="Save Configuration", 
                  command=self.save_config, style='Accent.TButton').pack(side='left', padx=5)
        
        ttk.Button(config_buttons, text="Reset to Defaults", 
                  command=self.reset_to_defaults).pack(side='left', padx=5)
        
        # Second row - Testing buttons
        test_buttons = ttk.Frame(button_frame)
        test_buttons.pack(fill='x')
        
        ttk.Button(test_buttons, text="Test Connection", 
                  command=self.test_connection).pack(side='left', padx=5)
        
        ttk.Button(test_buttons, text="Run Full LLM Test", 
                  command=self.run_full_llm_test, style='Accent.TButton').pack(side='left', padx=5)
        
        # Test status label
        self.test_status_label = ttk.Label(test_buttons, text="", 
                                          font=('Arial', 9), foreground='blue')
        self.test_status_label.pack(side='right', padx=5)
        
        # Configure grid weights
        api_frame.columnconfigure(1, weight=1)
        model_frame.columnconfigure(1, weight=1)
        
        # Pack scrollable components
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to all widgets for proper scrolling
        _bind_mousewheel(scrollable_frame)
        
        # Load current configuration (without showing success message)
        self.load_current_config(show_message=False)
        
        # Bind events to update privacy status and auto-configure (after loading config)
        self.client_type_var.trace('w', self.on_client_type_change)
        self.base_url_var.trace('w', self.update_privacy_status)
        self.ollama_host_var.trace('w', self.update_privacy_status)
        self.api_key_var.trace('w', self.update_privacy_status)
        
        # Mark initialization as complete
        self._initializing = False
        
        # Initial privacy status update
        self.update_privacy_status()
    
    def load_current_config(self, show_message=True):
        """Load current LLM configuration from the config file."""
        try:
            # Import the config to get current values
            import sys
            if 'S5_llmExtract_config' in sys.modules:
                # Reload if already imported
                import importlib
                importlib.reload(sys.modules['S5_llmExtract_config'])
            
            from S5_llmExtract_config import LLMConfig
            
            # Load values into GUI
            self.client_type_var.set(str(LLMConfig.CLIENT_TYPE.value))
            self.base_url_var.set(LLMConfig.BASE_URL)
            self.ollama_host_var.set(LLMConfig.OLLAMA_HOST)
            self.model_name_var.set(LLMConfig.MODEL_NAME)
            self.temperature_var.set(str(LLMConfig.TEMPERATURE))
            self.max_tokens_var.set(str(LLMConfig.MAX_TOKENS))
            self.batch_size_var.set(str(LLMConfig.BATCH_SIZE))
            self.chunk_size_var.set(str(LLMConfig.CHUNK_SIZE))
            self.max_retries_var.set(str(LLMConfig.MAX_RETRIES))
            self.parallel_workers_var.set(str(LLMConfig.PARALLEL_WORKERS))
            
            # Load API key from environment (but don't show it for security)
            api_key = os.getenv(LLMConfig.API_KEY_ENV_VAR, "")
            if api_key:
                self.api_key_var.set("*" * 20)  # Show placeholder
            
            self.update_privacy_status()
            
            if show_message:
                messagebox.showinfo("Success", "Current configuration loaded successfully")
            
        except Exception as e:
            if show_message:
                messagebox.showerror("Error", f"Failed to load current configuration: {e}")
            else:
                # Just print to console for silent load
                print(f"Warning: Could not load LLM configuration: {e}")
    
    def save_config(self):
        """Save LLM configuration by modifying the config file safely."""
        try:
            config_file = Path("S5_llmExtract_config.py")
            if not config_file.exists():
                messagebox.showerror("Error", "S5_llmExtract_config.py not found")
                return
            
            # Read the current file
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create a backup
            backup_file = config_file.with_suffix('.py.backup')
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Update the configuration values in the file content
            # Force parallel workers to 1 (no async support currently)
            parallel_workers = "1" if self.parallel_workers_var.get() != "1" else self.parallel_workers_var.get()
            
            updates = {
                'CLIENT_TYPE': f'LLMClientType(os.getenv("S5_LLM_CLIENT_TYPE", "{self.client_type_var.get()}"))',
                'BASE_URL': f'os.getenv("OPENAI_BASE_URL", "{self.base_url_var.get()}")',
                'OLLAMA_HOST': f'os.getenv("OLLAMA_HOST", "{self.ollama_host_var.get()}")',
                'MODEL_NAME': f'os.getenv("S5_LLM_MODEL", "{self.model_name_var.get()}")',
                'TEMPERATURE': f'float(os.getenv("S5_LLM_TEMPERATURE", "{self.temperature_var.get()}"))',
                'MAX_TOKENS': f'int(os.getenv("S5_LLM_MAX_TOKENS", "{self.max_tokens_var.get()}"))',
                'BATCH_SIZE': f'int(os.getenv("S5_LLM_BATCH_SIZE", "{self.batch_size_var.get()}"))',
                'CHUNK_SIZE': f'int(os.getenv("S5_LLM_CHUNK_SIZE", "{self.chunk_size_var.get()}"))',
                'MAX_RETRIES': f'int(os.getenv("S5_LLM_MAX_RETRIES", "{self.max_retries_var.get()}"))',
                'PARALLEL_WORKERS': f'int(os.getenv("S5_LLM_PARALLEL_WORKERS", "{parallel_workers}"))',
            }
            
            # Apply updates using regex replacement to preserve file structure
            import re
            for key, new_value in updates.items():
                pattern = rf'(\s+{key}:\s*[^=]*=\s*)[^#\n]+([#\n].*)?'
                replacement = rf'\g<1>{new_value}\g<2>'
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
            # Write the updated content
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Handle environment variables and .env file
            self._update_env_file()
            
            messagebox.showinfo("Success", 
                               f"Configuration saved successfully!\nBackup created: {backup_file.name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def _update_env_file(self):
        """Update .env file with current settings, ensuring OpenAI and Ollama settings coexist."""
        try:
            env_file = Path(".env")
            env_vars = {}
            
            # Read existing .env file if it exists
            if env_file.exists():
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip()
            
            client_type = self.client_type_var.get().strip()
            
            # Always update the client type and model
            env_vars['S5_LLM_CLIENT_TYPE'] = client_type
            env_vars['S5_LLM_MODEL'] = self.model_name_var.get().strip()
            env_vars['S5_LLM_TEMPERATURE'] = self.temperature_var.get().strip()
            env_vars['S5_LLM_MAX_TOKENS'] = self.max_tokens_var.get().strip()
            env_vars['S5_LLM_BATCH_SIZE'] = self.batch_size_var.get().strip()
            env_vars['S5_LLM_CHUNK_SIZE'] = self.chunk_size_var.get().strip()
            env_vars['S5_LLM_MAX_RETRIES'] = self.max_retries_var.get().strip()
            # Force parallel workers to 1 (no async support currently)
            env_vars['S5_LLM_PARALLEL_WORKERS'] = "1"
            
            # Handle API settings based on client type
            if client_type == "ollama":
                # For Ollama: only update Ollama settings, preserve OpenAI settings if they exist
                env_vars['OLLAMA_HOST'] = self.ollama_host_var.get().strip()
                # Don't touch OPENAI_BASE_URL or OPENAI_API_KEY if they exist
                
            elif client_type == "openai":
                # For OpenAI: update OpenAI settings, preserve Ollama settings if they exist  
                env_vars['OPENAI_BASE_URL'] = self.base_url_var.get().strip()
                api_key = self.api_key_var.get().strip()
                if api_key and not api_key.startswith('*'):
                    env_vars['OPENAI_API_KEY'] = api_key
                # Don't touch OLLAMA_HOST if it exists
                
            else:  # auto or other
                # For auto: update both but preserve existing values
                base_url = self.base_url_var.get().strip()
                ollama_host = self.ollama_host_var.get().strip()
                
                if "localhost" in base_url or "127.0.0.1" in base_url:
                    # Looks like Ollama URL
                    env_vars['OLLAMA_HOST'] = ollama_host
                else:
                    # Looks like external API
                    env_vars['OPENAI_BASE_URL'] = base_url
                    api_key = self.api_key_var.get().strip()
                    if api_key and not api_key.startswith('*'):
                        env_vars['OPENAI_API_KEY'] = api_key
            
            # Write updated .env file
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write("# LLM Configuration Environment Variables\n")
                f.write("# Generated by DICOM Processing Pipeline GUI\n\n")
                
                # Group related settings
                f.write("# Client Configuration\n")
                if 'S5_LLM_CLIENT_TYPE' in env_vars:
                    f.write(f"S5_LLM_CLIENT_TYPE={env_vars['S5_LLM_CLIENT_TYPE']}\n")
                
                f.write("\n# OpenAI Configuration\n")
                if 'OPENAI_BASE_URL' in env_vars:
                    f.write(f"OPENAI_BASE_URL={env_vars['OPENAI_BASE_URL']}\n")
                if 'OPENAI_API_KEY' in env_vars:
                    f.write(f"OPENAI_API_KEY={env_vars['OPENAI_API_KEY']}\n")
                
                f.write("\n# Ollama Configuration\n")
                if 'OLLAMA_HOST' in env_vars:
                    f.write(f"OLLAMA_HOST={env_vars['OLLAMA_HOST']}\n")
                
                f.write("\n# Model Configuration\n")
                for key in ['S5_LLM_MODEL', 'S5_LLM_TEMPERATURE', 'S5_LLM_MAX_TOKENS']:
                    if key in env_vars:
                        f.write(f"{key}={env_vars[key]}\n")
                
                f.write("\n# Processing Configuration\n")
                for key in ['S5_LLM_BATCH_SIZE', 'S5_LLM_CHUNK_SIZE', 'S5_LLM_MAX_RETRIES', 'S5_LLM_PARALLEL_WORKERS']:
                    if key in env_vars:
                        f.write(f"{key}={env_vars[key]}\n")
                
                # Write any other existing variables that we didn't handle
                f.write("\n# Other Variables\n")
                handled_keys = {
                    'S5_LLM_CLIENT_TYPE', 'OPENAI_BASE_URL', 'OPENAI_API_KEY', 'OLLAMA_HOST',
                    'S5_LLM_MODEL', 'S5_LLM_TEMPERATURE', 'S5_LLM_MAX_TOKENS', 
                    'S5_LLM_BATCH_SIZE', 'S5_LLM_CHUNK_SIZE', 'S5_LLM_MAX_RETRIES', 
                    'S5_LLM_PARALLEL_WORKERS'
                }
                for key, value in env_vars.items():
                    if key not in handled_keys:
                        f.write(f"{key}={value}\n")
            
            # Also set in current environment for immediate effect
            for key, value in env_vars.items():
                os.environ[key] = value
                
        except Exception as e:
            print(f"Warning: Could not update .env file: {e}")
    
    def load_api_key_from_env(self):
        """Load API key from .env file if it exists."""
        try:
            env_file = Path(".env")
            if not env_file.exists():
                messagebox.showinfo("Load API Key", "No .env file found. Create one first by saving configuration.")
                return
            
            # Read .env file and look for API key
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('OPENAI_API_KEY='):
                        api_key = line.split('=', 1)[1].strip('"\'')
                        if api_key:
                            self.api_key_var.set(api_key)
                            messagebox.showinfo("Load API Key", "API key loaded from .env file successfully!")
                            return
            
            messagebox.showinfo("Load API Key", "No OPENAI_API_KEY found in .env file.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load API key from .env: {e}")
    
    def on_client_type_change(self, *args):
        """Handle client type change - auto-update base_url and model."""
        try:
            # Skip during initialization or if required variables don't exist
            if getattr(self, '_initializing', True):
                return
            if not hasattr(self, 'client_type_var') or not hasattr(self, 'base_url_var') or not hasattr(self, 'model_name_var'):
                return
                
            client_type = self.client_type_var.get()
            
            if client_type == "openai":
                # Set OpenAI defaults
                self.base_url_var.set("https://api.openai.com/v1")
                self.model_name_var.set("gpt-4o-mini")
                
            elif client_type == "ollama":
                # Set Ollama defaults
                self.base_url_var.set("http://localhost:11434/v1")
                self.model_name_var.set("qwen3:4b-instruct-2507-q8_0")
                
            # Update privacy status regardless of client type
            if hasattr(self, 'update_privacy_status'):
                self.update_privacy_status()
            
        except Exception as e:
            # Ignore errors during initialization
            print(f"Debug: on_client_type_change error: {e}")
    
    def reset_to_defaults(self):
        """Reset all values to defaults."""
        self.client_type_var.set("ollama")
        self.base_url_var.set("http://localhost:11434/v1")
        self.ollama_host_var.set("http://localhost:11434")
        self.model_name_var.set("qwen3:4b-instruct-2507-q8_0")
        self.temperature_var.set("0.3")
        self.max_tokens_var.set("4096")
        self.batch_size_var.set("10")
        self.chunk_size_var.set("1000")
        self.max_retries_var.set("3")
        # Keep parallel workers locked to 1 (no async support)
        self.parallel_workers_var.set("1")
        self.api_key_var.set("")
        self.update_privacy_status()
    
    def test_connection(self):
        """Test the LLM connection."""
        try:
            base_url = self.base_url_var.get().strip()
            client_type = self.client_type_var.get().strip()
            
            if client_type == "ollama" or "localhost" in base_url or "127.0.0.1" in base_url:
                # Test Ollama connection
                import urllib.request
                import urllib.error
                
                ollama_host = self.ollama_host_var.get().strip()
                test_url = f"{ollama_host}/api/version"
                
                try:
                    with urllib.request.urlopen(test_url, timeout=5) as response:
                        if response.getcode() == 200:
                            messagebox.showinfo("Connection Test", 
                                               f"Ollama server is running at {ollama_host}")
                        else:
                            messagebox.showwarning("Connection Test", 
                                                  f"Ollama responded with status {response.getcode()}")
                except urllib.error.URLError:
                    messagebox.showerror("Connection Test", 
                                        f"Cannot connect to Ollama at {ollama_host}\n\nMake sure Ollama is installed and running:\n1. Download from https://ollama.ai/\n2. Run: ollama serve")
            else:
                # Test OpenAI API connection
                api_key = self.api_key_var.get().strip()
                if not api_key or api_key.startswith('*'):
                    messagebox.showwarning("Connection Test", 
                                          "Please enter a valid API key to test external API connection")
                    return
                
                messagebox.showinfo("Connection Test", 
                                   f"API configuration looks valid for {base_url}\nUse 'Start Processing' with S5 enabled to test actual API calls")
            
        except Exception as e:
            messagebox.showerror("Connection Test", f"Test failed: {e}")
    
    def update_privacy_status(self, *args):
        """Update privacy status based on current settings."""
        try:
            # Skip during initialization or if required variables don't exist
            if getattr(self, '_initializing', True):
                return
            if not all(hasattr(self, attr) for attr in ['base_url_var', 'client_type_var', 'ollama_host_var', 'api_key_var', 'privacy_label']):
                return
                
            base_url = self.base_url_var.get().strip()
            client_type = self.client_type_var.get().strip()
            ollama_host = self.ollama_host_var.get().strip()
            api_key = self.api_key_var.get().strip()
            
            # Determine if this is a local configuration
            is_local = False
            
            # Check client type first
            if client_type == "ollama":
                is_local = True
            # Check URLs for localhost/127.0.0.1
            elif "localhost" in base_url or "127.0.0.1" in base_url:
                is_local = True
            elif "localhost" in ollama_host or "127.0.0.1" in ollama_host:
                # If Ollama host contains localhost, likely using Ollama
                if client_type == "auto" and not ("api.openai.com" in base_url or "openai" in base_url.lower()):
                    is_local = True
            
            if is_local:
                status_text = "LOCAL LLM CONFIGURATION - Patient data stays on this machine (HIPAA safe)"
                self.privacy_label.config(text=status_text, foreground='green')
            else:
                # Check if using external API with API key
                if api_key and not api_key.startswith("*"):  # Real API key, not placeholder
                    status_text = f"WARNING: EXTERNAL API WITH API KEY\nSending medical data to third party: {base_url}\nEnsure HIPAA compliance and data use agreements are in place!"
                    self.privacy_label.config(text=status_text, foreground='red')
                elif "api.openai.com" in base_url or "openai" in base_url.lower():
                    status_text = f"WARNING: THIRD-PARTY API DETECTED\nData will be sent to: {base_url}\nAPI key required. Medical data will be processed externally!"
                    self.privacy_label.config(text=status_text, foreground='orange')
                else:
                    status_text = f"EXTERNAL API CONFIGURATION\nData will be sent to: {base_url}\nVerify privacy and compliance requirements!"
                    self.privacy_label.config(text=status_text, foreground='orange')
                
        except Exception as e:
            # More robust error handling
            if hasattr(self, 'privacy_label'):
                try:
                    self.privacy_label.config(text="Error determining privacy status", foreground='orange')
                except:
                    pass
            print(f"Debug: update_privacy_status error: {e}")
    
    def run_full_llm_test(self):
        """Run comprehensive LLM test using the S5_llmExtract_test.py script."""
        try:
            self.test_status_label.config(text="Running LLM test...", foreground='blue')
            
            # First, apply current GUI settings to environment
            import os
            os.environ['S5_LLM_CLIENT_TYPE'] = self.client_type_var.get()
            os.environ['OPENAI_BASE_URL'] = self.base_url_var.get()
            os.environ['OLLAMA_HOST'] = self.ollama_host_var.get()
            os.environ['S5_LLM_MODEL'] = self.model_name_var.get()
            os.environ['S5_LLM_TEMPERATURE'] = self.temperature_var.get()
            os.environ['S5_LLM_MAX_TOKENS'] = self.max_tokens_var.get()
            
            # Apply API key if provided
            api_key = self.api_key_var.get().strip()
            if api_key and not api_key.startswith('*'):
                os.environ['OPENAI_API_KEY'] = api_key
            
            # Get python executable
            python_exe = get_python_executable(DEFAULT_VENV_DIR)
            
            # Build test command
            test_script = Path(__file__).parent / "code" / "S5_llmExtract_test.py"
            cmd = [python_exe, str(test_script), "--client-type", self.client_type_var.get()]
            
            # Run test in a separate thread to not block GUI
            import threading
            import subprocess
            
            def run_test():
                try:
                    result = subprocess.run(cmd, 
                                          capture_output=True, 
                                          text=True, 
                                          timeout=120,  # 2 minute timeout
                                          cwd=str(Path(__file__).parent))
                    
                    # Update UI on main thread (get root from widget hierarchy)
                    root_widget = self.test_status_label.winfo_toplevel()
                    root_widget.after(0, lambda: self.handle_test_result(result))
                except subprocess.TimeoutExpired:
                    root_widget = self.test_status_label.winfo_toplevel()
                    root_widget.after(0, lambda: self.test_status_label.config(
                        text="Test timeout (>2 min)", foreground='red'))
                except Exception as e:
                    root_widget = self.test_status_label.winfo_toplevel()
                    root_widget.after(0, lambda: self.test_status_label.config(
                        text=f"Test error: {str(e)[:50]}...", foreground='red'))
            
            test_thread = threading.Thread(target=run_test, daemon=True)
            test_thread.start()
            
        except Exception as e:
            self.test_status_label.config(text=f"Failed to start test: {str(e)[:30]}...", foreground='red')
    
    def handle_test_result(self, result):
        """Handle the test result and update UI."""
        try:
            if result.returncode == 0:
                # Test passed
                self.test_status_label.config(text="✓ LLM test passed successfully!", foreground='green')
                
                # Show detailed results in a popup
                success_msg = "LLM Configuration Test Results:\n\n"
                success_msg += "✓ Connection successful\n"
                success_msg += "✓ Model responding correctly\n" 
                success_msg += "✓ Medical data extraction working\n\n"
                success_msg += "Detailed output:\n" + result.stdout[-500:]  # Last 500 chars
                
                messagebox.showinfo("LLM Test Results", success_msg)
            else:
                # Test failed
                self.test_status_label.config(text="✗ LLM test failed", foreground='red')
                
                # Show error details
                error_msg = "LLM Test Failed:\n\n"
                error_msg += f"Exit Code: {result.returncode}\n\n"
                error_msg += "Error Output:\n" + result.stderr[-800:]  # Last 800 chars
                error_msg += "\n\nStdout:\n" + result.stdout[-400:]  # Last 400 chars
                
                messagebox.showerror("LLM Test Failed", error_msg)
        except Exception as e:
            self.test_status_label.config(text="Error processing test results", foreground='red')
            messagebox.showerror("Error", f"Failed to process test results: {e}")

class DICOMProcessingApp:
    """Main GUI application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DICOM Processing Pipeline - GUI")
        self.root.geometry("900x900")  # Increased width and height for better visibility
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Memory manager
        self.memory_manager = MemoryManager()
        
        # Process runner
        self.process_runner = None
        self.output_queue = queue.Queue()
        
        # Create GUI
        self.create_gui()
        
        # Initialize status tracking
        self.init_status_tracking()
        
        # Configure text tags for colored output
        self.configure_output_tags()
        
        # Start checking for output
        self.check_output()
    
    def create_gui(self):
        """Create the main GUI."""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill='both', expand=True, pady=(0, 10))
        
        # Single location tab
        single_tab = ttk.Frame(notebook)
        notebook.add(single_tab, text="Single Location")
        self.single_location_tab = SingleLocationTab(single_tab, self.memory_manager, self.run_process)
        
        # Multi location tab
        multi_tab = ttk.Frame(notebook)
        notebook.add(multi_tab, text="Multi Location (Batch)")
        self.multi_location_tab = MultiLocationTab(multi_tab, self.memory_manager, self.run_process)
        
        # LLM Configuration tab
        llm_tab = ttk.Frame(notebook)
        notebook.add(llm_tab, text="LLM Config (S5)")
        self.llm_config_tab = LLMConfigTab(llm_tab, self.memory_manager)
        
        # Output frame
        output_frame = ttk.LabelFrame(main_container, text="Processing Status & Logs")
        output_frame.pack(fill='both', expand=True)
        
        # Create horizontal paned window for status and logs
        paned_window = ttk.PanedWindow(output_frame, orient='horizontal')
        paned_window.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left side: Status tracking
        status_frame = ttk.LabelFrame(paned_window, text="Pipeline Status", padding=5)
        paned_window.add(status_frame, weight=1)
        
        # Status list
        self.status_tree = ttk.Treeview(status_frame, columns=('Status',), show='tree headings', height=8)
        self.status_tree.heading('#0', text='Pipeline Stage')
        self.status_tree.heading('Status', text='Status')
        self.status_tree.column('#0', width=200, minwidth=150)
        self.status_tree.column('Status', width=100, minwidth=80)
        
        # Add scrollbar to status tree
        status_scrollbar = ttk.Scrollbar(status_frame, orient='vertical', command=self.status_tree.yview)
        self.status_tree.configure(yscrollcommand=status_scrollbar.set)
        
        self.status_tree.pack(side='left', fill='both', expand=True)
        status_scrollbar.pack(side='right', fill='y')
        
        # Right side: Log output
        log_frame = ttk.LabelFrame(paned_window, text="Processing Logs", padding=5)
        paned_window.add(log_frame, weight=2)
        
        # Text area for output with improved formatting
        self.output_text = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD, 
                                                     font=('Consolas', 10), 
                                                     bg='#f8f9fa', fg='#212529')
        self.output_text.pack(fill='both', expand=True)
        
        # Control buttons
        control_frame = ttk.Frame(output_frame)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Process", 
                                     command=self.stop_process, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Clear Output", 
                  command=self.clear_output).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Save Log", 
                  command=self.save_log).pack(side='right', padx=5)
    
    def run_process(self, cmd: List[str], env: Dict[str, str] = None):
        """Run a process with real-time output."""
        if self.process_runner and self.process_runner.is_running:
            messagebox.showwarning("Warning", "A process is already running")
            return
        
        # Clear output
        self.output_text.delete(1.0, tk.END)
        
        # Add command to output (both GUI and console)
        cmd_msg = f"[COMMAND] {' '.join(cmd)}"
        time_msg = f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        separator = "-" * 60
        
        print(cmd_msg)      # Print to console
        print(time_msg)     # Print to console  
        print(separator)    # Print to console
        
        self.add_output(cmd_msg)
        self.add_output(time_msg)
        self.add_output(separator)
        
        # Update UI
        self.stop_button.config(state='normal')
        self.single_location_tab.run_button.config(state='disabled')
        self.multi_location_tab.batch_run_button.config(state='disabled')
        
        # Start process in thread
        self.process_runner = ProcessRunner(self.add_output_threadsafe)
        thread = threading.Thread(target=self._run_process_thread, 
                                 args=(cmd, env), daemon=True)
        thread.start()
    
    def _run_process_thread(self, cmd: List[str], env: Dict[str, str]):
        """Run process in separate thread."""
        try:
            self.process_runner.run_command(cmd, cwd=str(Path(__file__).parent), env=env)
        except Exception as e:
            self.add_output_threadsafe(f"[ERROR] {e}")
        finally:
            # Re-enable buttons
            self.root.after(0, self._process_finished)
    
    def _process_finished(self):
        """Called when process finishes."""
        self.stop_button.config(state='disabled')
        self.single_location_tab.run_button.config(state='normal')
        self.multi_location_tab.batch_run_button.config(state='normal')
    
    def add_output_threadsafe(self, text: str):
        """Add output from thread safely."""
        self.output_queue.put(text)
    
    def init_status_tracking(self):
        """Initialize the status tracking system."""
        self.stage_status = {}
        self.status_items = {}
        
        # Define all possible stages
        self.all_stages = [
            ('setup', 'Environment Setup'),
            ('s0_zip', 'S0a - ZIP Extraction'),
            ('s0_iso', 'S0b - ISO Extraction'),
            ('s1', 'S1 - DICOM Indexing'),
            ('s2', 'S2 - Create Summaries'),
            ('s3', 'S3 - Filter Studies'),
            ('s4', 'S4 - Create Archives'),
            ('s5', 'S5 - AI Extraction')
        ]
        
        # Clear the tree first
        for item in self.status_tree.get_children():
            self.status_tree.delete(item)
            
        # Only add Environment Setup (always runs)
        item = self.status_tree.insert('', 'end', text='Environment Setup', values=('Waiting',))
        self.status_items['setup'] = item
        self.stage_status['setup'] = 'waiting'
    
    def setup_status_for_enabled_stages(self, enabled_stages):
        """Add status tracking for enabled stages only."""
        # Clear existing stages (except setup)
        for item_id, item in list(self.status_items.items()):
            if item_id != 'setup':
                self.status_tree.delete(item)
                del self.status_items[item_id]
                del self.stage_status[item_id]
        
        # Stage mapping
        stage_names = {
            'S0_ZIP': ('s0_zip', 'S0a - ZIP Extraction'),
            'S0_ISO': ('s0_iso', 'S0b - ISO Extraction'), 
            'S1': ('s1', 'S1 - DICOM Indexing'),
            'S2': ('s2', 'S2 - Create Summaries'),
            'S3': ('s3', 'S3 - Filter Studies'),
            'S4': ('s4', 'S4 - Create Archives'),
            'S5': ('s5', 'S5 - AI Extraction')
        }
        
        # Add only enabled stages
        for stage_key in enabled_stages:
            if stage_key in stage_names:
                stage_id, stage_name = stage_names[stage_key]
                item = self.status_tree.insert('', 'end', text=stage_name, values=('Waiting',))
                self.status_items[stage_id] = item
                self.stage_status[stage_id] = 'waiting'
    
    def configure_output_tags(self):
        """Configure text tags for colored output."""
        # Configure different text styles
        self.output_text.tag_configure('command', foreground='#0066cc', font=('Consolas', 10, 'bold'))
        self.output_text.tag_configure('time', foreground='#666666', font=('Consolas', 9))
        self.output_text.tag_configure('success', foreground='#28a745', font=('Consolas', 10, 'bold'))
        self.output_text.tag_configure('error', foreground='#dc3545', font=('Consolas', 10, 'bold'))
        self.output_text.tag_configure('warning', foreground='#ffc107', font=('Consolas', 10, 'bold'))
        self.output_text.tag_configure('step', foreground='#17a2b8', font=('Consolas', 10))
        self.output_text.tag_configure('progress', foreground='#6f42c1', font=('Consolas', 10))
        self.output_text.tag_configure('separator', foreground='#6c757d', font=('Consolas', 10))
    
    def update_stage_status(self, stage_id: str, status: str):
        """Update the status of a pipeline stage."""
        if stage_id in self.status_items:
            item = self.status_items[stage_id]
            
            # Status mapping with colors
            status_display = {
                'waiting': ('Waiting', '#6c757d'),
                'running': ('Running...', '#007bff'),
                'success': ('Success', '#28a745'),
                'error': ('Failed', '#dc3545'),
                'skipped': ('Skipped', '#ffc107')
            }
            
            if status in status_display:
                display_text, color = status_display[status]
                self.status_tree.set(item, 'Status', display_text)
                
                # Update the item appearance (this is a limitation of ttk.Treeview)
                # We'll use tags to at least show different formatting
                self.status_tree.item(item, tags=(status,))
                
        self.stage_status[stage_id] = status
    
    def detect_stage_from_output(self, text: str):
        """Detect which stage is running from output text and update status."""
        text_lower = text.lower()
        
        # Stage detection patterns
        if '[step]' in text_lower and 'virtual environment' in text_lower:
            self.update_stage_status('setup', 'running')
        elif 'setup completed' in text_lower or 'environment already exists' in text_lower:
            self.update_stage_status('setup', 'success')
        elif 's0_zipextract' in text_lower or 'zip extraction' in text_lower:
            self.update_stage_status('s0_zip', 'running')
        elif 's0_isoextract' in text_lower or 'iso extraction' in text_lower:
            self.update_stage_status('s0_iso', 'running')
        elif 's1_indexingfiles' in text_lower or 'dicom indexing' in text_lower:
            self.update_stage_status('s1', 'running')
        elif 's2_concatindex' in text_lower or 'concatenat' in text_lower:
            self.update_stage_status('s2', 'running')
        elif 's3_processforstore' in text_lower or 'process for storage' in text_lower:
            self.update_stage_status('s3', 'running')
        elif 's4_zipstore' in text_lower or 'zip archiv' in text_lower:
            self.update_stage_status('s4', 'running')
        elif 's5_llmextract' in text_lower or 'ai extract' in text_lower:
            self.update_stage_status('s5', 'running')
        
        # Success patterns - specific stage completion detection
        if 'completed successfully' in text_lower:
            if 's1 completed' in text_lower:
                self.update_stage_status('s1', 'success')
            elif 's2 completed' in text_lower:
                self.update_stage_status('s2', 'success')
            elif 's3 completed' in text_lower:
                self.update_stage_status('s3', 'success')
            elif 's4 completed' in text_lower:
                self.update_stage_status('s4', 'success')
            elif 's5 completed' in text_lower:
                self.update_stage_status('s5', 'success')
        elif 'all dependencies installed successfully' in text_lower:
            self.update_stage_status('setup', 'success')
        elif '[completed]' in text_lower:
            # Find which stage just completed
            for stage_id, status in self.stage_status.items():
                if status == 'running':
                    self.update_stage_status(stage_id, 'success')
                    break
        
        # Error patterns
        if '[error]' in text_lower and 'failed' in text_lower:
            # Find which stage is failing
            for stage_id, status in self.stage_status.items():
                if status == 'running':
                    self.update_stage_status(stage_id, 'error')
                    break

    def add_output(self, text: str):
        """Add text to output area with intelligent formatting."""
        # Detect stage changes and update status
        self.detect_stage_from_output(text)
        
        # Determine the appropriate tag based on content
        tag = None
        if text.startswith('[COMMAND]'):
            tag = 'command'
        elif text.startswith('[TIME]'):
            tag = 'time'
        elif '[OK]' in text or '[COMPLETED]' in text or 'Success' in text:
            tag = 'success'
        elif '[ERROR]' in text or 'Error:' in text or 'Traceback' in text:
            tag = 'error'
        elif '[WARNING]' in text or 'Warning:' in text:
            tag = 'warning'
        elif '[STEP]' in text:
            tag = 'step'
        elif text.startswith('---') or text.startswith('==='):
            tag = 'separator'
        elif '%' in text or 'progress' in text.lower():
            tag = 'progress'
        
        # Insert text with appropriate formatting
        start_pos = self.output_text.index(tk.END + "-1c")
        self.output_text.insert(tk.END, text + "\n")
        
        if tag:
            end_pos = self.output_text.index(tk.END + "-1c")
            self.output_text.tag_add(tag, start_pos, end_pos)
        
        self.output_text.see(tk.END)
        self.root.update_idletasks()
    
    def check_output(self):
        """Check for output from queue."""
        try:
            while True:
                text = self.output_queue.get_nowait()
                self.add_output(text)
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_output)
    
    def stop_process(self):
        """Stop the current process."""
        if self.process_runner:
            self.process_runner.stop()
            stop_msg = "[STOPPED] Process stopped by user"
            print(stop_msg)  # Also print to console
            self.add_output(stop_msg)
    
    def clear_output(self):
        """Clear the output area."""
        self.output_text.delete(1.0, tk.END)
    
    def save_log(self):
        """Save the current log to file."""
        content = self.output_text.get(1.0, tk.END)
        if not content.strip():
            messagebox.showinfo("Info", "No log content to save")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
            initialname=f"dicom_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Log saved to: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save log: {e}")
    
    def run(self):
        """Start the GUI application."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            if self.process_runner:
                self.process_runner.stop()

def main():
    """Main entry point."""
    app = DICOMProcessingApp()
    app.run()

if __name__ == "__main__":
    main()