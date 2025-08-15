import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MultiRunner:
    def __init__(self, venv_path: str = None):
        """
        Initialize the MultiRunner
        
        Args:
            venv_path: Path to virtual environment. If None, uses system Python
        """
        self.venv_path = venv_path
        self.python_executable = self._get_python_executable()
        
    def _get_python_executable(self) -> str:
        """Get the Python executable path"""
        if self.venv_path:
            if os.name == 'nt':  # Windows
                return os.path.join(self.venv_path, "Scripts", "python.exe")
            else:  # Unix/Linux/Mac
                return os.path.join(self.venv_path, "bin", "python")
        return sys.executable
    
    def run_s1_instance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single S1.py instance with given configuration
        
        Args:
            config: Dictionary containing S1 configuration parameters
            
        Returns:
            Dictionary with execution results
        """
        instance_id = config.get('instance_id', 'unknown')
        start_time = time.time()
        
        try:
            # Prepare command arguments
            cmd = [
                self.python_executable,
                "S1.py",
                "--root", config['root_directory'],
                "--output_dir", config['output_directory']
            ]
            
            if config.get('overwrite', False):
                cmd.append("--overwrite")
            
            logger.info(f"Starting S1 instance {instance_id}: {' '.join(cmd)}")
            
            # Run the subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=config.get('working_directory', '.'),
                timeout=config.get('timeout', 3600)  # 1 hour default timeout
            )
            
            execution_time = time.time() - start_time
            
            return {
                'instance_id': instance_id,
                'script': 'S1',
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': execution_time,
                'config': config
            }
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            logger.error(f"S1 instance {instance_id} timed out after {execution_time:.2f} seconds")
            return {
                'instance_id': instance_id,
                'script': 'S1',
                'success': False,
                'return_code': -1,
                'stdout': '',
                'stderr': 'Process timed out',
                'execution_time': execution_time,
                'config': config
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error running S1 instance {instance_id}: {e}")
            return {
                'instance_id': instance_id,
                'script': 'S1',
                'success': False,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'execution_time': execution_time,
                'config': config
            }
    
    def run_s2_instance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single S2.py instance with given configuration
        
        Args:
            config: Dictionary containing S2 configuration parameters
            
        Returns:
            Dictionary with execution results
        """
        instance_id = config.get('instance_id', 'unknown')
        start_time = time.time()
        
        try:
            # Prepare command arguments
            cmd = [
                self.python_executable,
                "S2.py",
                "--folder_path", config['folder_path']
            ]
            
            if 'output_json' in config:
                cmd.extend(["--output_json", config['output_json']])
            
            if 'output_excel' in config:
                cmd.extend(["--output_excel", config['output_excel']])
            
            if config.get('overwrite', False):
                cmd.append("--overwrite")
            
            logger.info(f"Starting S2 instance {instance_id}: {' '.join(cmd)}")
            
            # Set environment variables if specified
            env = os.environ.copy()
            if 'skip_files' in config:
                env['SKIP_FILES'] = ','.join(config['skip_files'])
            
            # Run the subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=config.get('working_directory', '.'),
                timeout=config.get('timeout', 1800)  # 30 minutes default timeout
            )
            
            execution_time = time.time() - start_time
            
            return {
                'instance_id': instance_id,
                'script': 'S2',
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': execution_time,
                'config': config
            }
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            logger.error(f"S2 instance {instance_id} timed out after {execution_time:.2f} seconds")
            return {
                'instance_id': instance_id,
                'script': 'S2',
                'success': False,
                'return_code': -1,
                'stdout': '',
                'stderr': 'Process timed out',
                'execution_time': execution_time,
                'config': config
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error running S2 instance {instance_id}: {e}")
            return {
                'instance_id': instance_id,
                'script': 'S2',
                'success': False,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'execution_time': execution_time,
                'config': config
            }
    
    def run_multiple_instances(self, 
                             s1_configs: List[Dict[str, Any]] = None,
                             s2_configs: List[Dict[str, Any]] = None,
                             max_workers: int = None,
                             sequential: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run multiple instances of S1 and/or S2 scripts
        
        Args:
            s1_configs: List of S1 configurations
            s2_configs: List of S2 configurations
            max_workers: Maximum number of parallel workers
            sequential: If True, run sequentially instead of parallel
            
        Returns:
            Dictionary with results for each script type
        """
        if max_workers is None:
            max_workers = min(4, os.cpu_count() or 1)
        
        s1_configs = s1_configs or []
        s2_configs = s2_configs or []
        
        results = {
            'S1': [],
            'S2': []
        }
        
        if sequential:
            # Run sequentially
            logger.info("Running instances sequentially...")
            
            for config in s1_configs:
                result = self.run_s1_instance(config)
                results['S1'].append(result)
                logger.info(f"S1 instance {config.get('instance_id')} completed: {'SUCCESS' if result['success'] else 'FAILED'}")
            
            for config in s2_configs:
                result = self.run_s2_instance(config)
                results['S2'].append(result)
                logger.info(f"S2 instance {config.get('instance_id')} completed: {'SUCCESS' if result['success'] else 'FAILED'}")
        
        else:
            # Run in parallel
            logger.info(f"Running instances in parallel with max_workers={max_workers}...")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all jobs
                future_to_config = {}
                
                for config in s1_configs:
                    future = executor.submit(self.run_s1_instance, config)
                    future_to_config[future] = ('S1', config)
                
                for config in s2_configs:
                    future = executor.submit(self.run_s2_instance, config)
                    future_to_config[future] = ('S2', config)
                
                # Collect results as they complete
                for future in as_completed(future_to_config):
                    script_type, config = future_to_config[future]
                    try:
                        result = future.result()
                        results[script_type].append(result)
                        logger.info(f"{script_type} instance {config.get('instance_id')} completed: {'SUCCESS' if result['success'] else 'FAILED'}")
                    except Exception as e:
                        logger.error(f"Exception in {script_type} instance {config.get('instance_id')}: {e}")
                        results[script_type].append({
                            'instance_id': config.get('instance_id', 'unknown'),
                            'script': script_type,
                            'success': False,
                            'return_code': -1,
                            'stdout': '',
                            'stderr': str(e),
                            'execution_time': 0,
                            'config': config
                        })
        
        return results
    
    def save_results(self, results: Dict[str, List[Dict[str, Any]]], output_file: str = None):
        """Save execution results to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"execution_results_{timestamp}.json"
        
        # Add summary statistics
        summary = {
            'execution_timestamp': datetime.now().isoformat(),
            'total_s1_instances': len(results['S1']),
            'successful_s1_instances': sum(1 for r in results['S1'] if r['success']),
            'total_s2_instances': len(results['S2']),
            'successful_s2_instances': sum(1 for r in results['S2'] if r['success']),
        }
        
        output_data = {
            'summary': summary,
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
        return output_file


def main():
    """Example usage and configuration"""
    
    # Initialize the runner with your virtual environment
    venv_path = r"C:\Users\LEGION\Documents\GIT\PanCanAID - BACKUP 2024-06-05\venv_blindCleaning"
    runner = MultiRunner(venv_path=venv_path)
    
    # Example configurations for S1 instances
    s1_configs = [
        {
            'instance_id': 'dataset_1',
            'root_directory': r'C:\Data\Dataset1',
            'output_directory': r'C:\Output\Dataset1',
            'overwrite': False,
            'timeout': 3600,  # 1 hour
            'working_directory': '.'
        },
        {
            'instance_id': 'dataset_2',
            'root_directory': r'C:\Data\Dataset2',
            'output_directory': r'C:\Output\Dataset2',
            'overwrite': True,
            'timeout': 3600,
            'working_directory': '.'
        }
    ]
    
    # Example configurations for S2 instances
    s2_configs = [
        {
            'instance_id': 'analysis_1',
            'folder_path': r'C:\Output\Dataset1',
            'output_json': r'C:\Results\dataset1_patients.json',
            'output_excel': r'C:\Results\dataset1_patients.xlsx',
            'overwrite': True,
            'skip_files': ['temp.jsonl', 'backup.jsonl'],
            'timeout': 1800,  # 30 minutes
            'working_directory': '.'
        },
        {
            'instance_id': 'analysis_2',
            'folder_path': r'C:\Output\Dataset2',
            'output_json': r'C:\Results\dataset2_patients.json',
            'output_excel': r'C:\Results\dataset2_patients.xlsx',
            'overwrite': True,
            'skip_files': ['temp.jsonl', 'test.jsonl'],
            'timeout': 1800,
            'working_directory': '.'
        }
    ]
    
    # Run all instances
    logger.info("Starting multi-instance execution...")
    
    results = runner.run_multiple_instances(
        s1_configs=s1_configs,
        s2_configs=s2_configs,
        max_workers=2,  # Adjust based on your system capabilities
        sequential=False  # Set to True for sequential execution
    )
    
    # Save results
    results_file = runner.save_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    
    total_s1 = len(results['S1'])
    successful_s1 = sum(1 for r in results['S1'] if r['success'])
    
    total_s2 = len(results['S2'])
    successful_s2 = sum(1 for r in results['S2'] if r['success'])
    
    print(f"S1 Script Results: {successful_s1}/{total_s1} successful")
    print(f"S2 Script Results: {successful_s2}/{total_s2} successful")
    print(f"Results saved to: {results_file}")
    
    # Print detailed results
    for script_type, script_results in results.items():
        if script_results:
            print(f"\n{script_type} Detailed Results:")
            print("-" * 40)
            for result in script_results:
                status = "SUCCESS" if result['success'] else "FAILED"
                print(f"  {result['instance_id']}: {status} ({result['execution_time']:.2f}s)")
                if not result['success'] and result['stderr']:
                    print(f"    Error: {result['stderr'][:100]}...")


if __name__ == "__main__":
    main()

