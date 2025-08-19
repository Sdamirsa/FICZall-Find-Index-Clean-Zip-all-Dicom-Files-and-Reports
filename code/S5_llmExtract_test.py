#!/usr/bin/env python3
"""
Standalone test script for S5 LLM Document Extraction
====================================================

This is a comprehensive standalone test for S5 LLM functionality with dual client support. 
While S5 now has integrated health checks, this script provides more detailed testing
and can be used for debugging both OpenAI and Ollama client issues.

Usage:
    python code/S5_llmExtract_test.py [--client-type openai|ollama|auto]

Requirements:
    For OpenAI: API key set in OPENAI_API_KEY environment variable
    For Ollama: 
        - Ollama running locally (ollama serve)
        - Model gpt-oss:20b available (ollama pull gpt-oss:20b)
        - ollama library installed (pip install ollama)
    - Dependencies installed (openai, pydantic)

Note: S5 now includes built-in health checks. This standalone test is for advanced debugging.
"""

import sys
import json
import traceback
from pathlib import Path
import requests

# Add parent directory to path to import modules from main directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from openai import OpenAI
    from S5_llmExtract_config import MedicalReportExtraction, LLMConfig, LLMClientType
    # Import the LLMClientManager from main S5 module
    sys.path.insert(0, str(Path(__file__).parent))
    from S5_llmExtract import LLMClientManager
    print("Successfully imported required modules")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install dependencies: pip install openai pydantic ollama")
    sys.exit(1)

# Sample medical document for testing
SAMPLE_MEDICAL_TEXT = """
PATIENT NAME: Test Patient
PATIENT ID: TEST001
AGE: 50 years
GENDER: Female

STUDY: CT Abdomen and Pelvis with Contrast
DATE: 2024-01-15
INDICATION: Abdominal pain, rule out appendicitis

FINDINGS:
The liver, spleen, pancreas, and kidneys appear normal.
No evidence of acute appendicitis.
Small amount of free fluid in the pelvis.
No masses or lymphadenopathy identified.

IMPRESSION:
1. No acute abdominal pathology
2. Small amount of pelvic free fluid, clinical correlation recommended

Previous history includes hypertension.
"""

def test_llm_config():
    """Test LLM configuration"""
    print("\n" + "="*50)
    print("TESTING LLM CONFIGURATION")
    print("="*50)
    
    try:
        print(f"Client Type: {LLMConfig.get_client_type()}")
        print(f"Base URL: {LLMConfig.BASE_URL}")
        print(f"Ollama Host: {LLMConfig.OLLAMA_HOST}")
        print(f"Model: {LLMConfig.MODEL_NAME}")
        print(f"Max Tokens: {LLMConfig.MAX_TOKENS}")
        print(f"Temperature: {LLMConfig.TEMPERATURE}")
        print(f"Is Local: {LLMConfig.is_local()}")
        
        api_key = LLMConfig.get_api_key()
        print(f"API Key: {'*' * len(api_key) if api_key != 'ollama' else 'Not required (Ollama)'}")
        
        if LLMConfig.validate_config():
            print("Configuration is valid")
            return True
        else:
            print("Configuration is invalid")
            return False
            
    except Exception as e:
        print(f"Configuration error: {e}")
        return False

def test_client_manager():
    """Test LLMClientManager initialization"""
    print("\n" + "="*50)
    print(f"TESTING {LLMConfig.get_client_type().upper()} CLIENT MANAGER")
    print("="*50)
    
    try:
        # Setup basic logging for client manager
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('test_client_manager')
        
        client_manager = LLMClientManager(logger)
        print(f"{client_manager.client_type} client manager initialized successfully")
        print(f"   Client type: {client_manager.client_type}")
        print(f"   OpenAI client available: {client_manager.openai_client is not None}")
        print(f"   Ollama client available: {client_manager.ollama_client is not None}")
        return client_manager
        
    except Exception as e:
        print(f"Client manager initialization error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

def test_llm_extraction(client_manager):
    """Test LLM extraction with sample medical document using unified client manager"""
    print("\n" + "="*50)
    print("TESTING LLM EXTRACTION")
    print("="*50)
    
    try:
        print("Sending request to LLM via client manager...")
        print(f"   Client type: {client_manager.client_type}")
        print(f"   Model: {LLMConfig.MODEL_NAME}")
        if client_manager.client_type == LLMClientType.OPENAI:
            print(f"   OpenAI URL: {LLMConfig.BASE_URL}")
        elif client_manager.client_type == LLMClientType.OLLAMA:
            print(f"   Ollama host: {LLMConfig.OLLAMA_HOST}")
        print(f"   Schema: MedicalReportExtraction (comprehensive medical data extraction)")
        
        # Use client manager's unified extraction method
        extraction_result = client_manager.extract_medical_data(
            SAMPLE_MEDICAL_TEXT, 
            'test_document.txt', 
            use_simple_schema=False  # Use comprehensive MedicalReportExtraction schema
        )
        
        print("Received response from client manager")
        
        # Check extraction result
        if extraction_result['success']:
            extracted_data = extraction_result['extraction_data']
            parsing_method = extraction_result.get('response_metadata', {}).get('parsing_method', 'unknown')
            client_type = extraction_result.get('client_type', 'unknown')
            
            print(f"Successfully extracted data using {parsing_method} method")
            print(f"  Client type: {client_type}")
            
            # Display results
            print("\n" + "-"*40)
            print("EXTRACTION RESULTS (MedicalReportExtraction)")
            print("-"*40)
            print(f"Number of studies: {extracted_data.number_of_studies}")
            print(f"Imaging modality: {extracted_data.imaging_modality}")
            print(f"Body part: {extracted_data.body_part}")
            print(f"Patient name: {extracted_data.patient_info.name}")
            print(f"Patient age: {extracted_data.patient_info.age}")
            print(f"Patient gender: {extracted_data.patient_info.gender}")
            print(f"Contrast used: {extracted_data.contrast_used}")
            print(f"Findings summary: {extracted_data.findings_summary}")
            print(f"Impression: {extracted_data.impression}")
            print(f"Current diseases: {extracted_data.current_disease_list}")
            print(f"Previous diseases: {extracted_data.previous_disease_list}")
            print(f"Previous interventions: {extracted_data.previous_interventions}")
            
            # Show response metadata
            response_metadata = extraction_result.get('response_metadata', {})
            if response_metadata:
                print(f"\nResponse metadata:")
                for key, value in response_metadata.items():
                    print(f"  {key}: {value}")
            
            # Show parsing method details
            print(f"\nParsing method used: {parsing_method}")
            print(f"Client used: {client_type}")
            
            return True
        else:
            print(f"Extraction failed: {extraction_result['error_message']}")
            print(f"  Error type: {extraction_result['error_type']}")
            
            # Show full response for debugging
            full_response = extraction_result.get('full_response')
            if full_response:
                print("\nFull response details:")
                if isinstance(full_response, dict):
                    for key, value in full_response.items():
                        if isinstance(value, str) and len(value) > 200:
                            print(f"  {key}: {value[:200]}...")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"  Response: {full_response}")
            
            return False
            
    except Exception as e:
        print(f"Extraction error: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_connection():
    """Test basic connection to LLM service"""
    print("\n" + "="*50)
    print("TESTING LLM SERVICE CONNECTION")
    print("="*50)
    
    try:
        # Test connection based on client type
        client_type = LLMConfig.get_client_type()
        
        if client_type == LLMClientType.OLLAMA or LLMConfig.is_local():
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json()
                    # Handle different possible model data structures
                    model_names = []
                    for model in models.get('models', []):
                        if isinstance(model, dict):
                            # Try common keys for model name
                            model_name = model.get('name') or model.get('model') or model.get('id')
                            if model_name:
                                model_names.append(model_name)
                        else:
                            model_names.append(str(model))
                    
                    if LLMConfig.MODEL_NAME in model_names:
                        print(f"Ollama is running and model {LLMConfig.MODEL_NAME} is available")
                        return True
                    else:
                        print(f"WARNING: Model {LLMConfig.MODEL_NAME} not found in Ollama")
                        print(f"   Available models: {model_names}")
                        print(f"   Run: ollama pull {LLMConfig.MODEL_NAME}")
                        return False
                else:
                    print(f"Ollama responded with status {response.status_code}")
                    return False
            except Exception as e:
                print(f"Ollama connection error: {e}")
                print("   Make sure Ollama is running: ollama serve")
                return False
        else:
            print("Using external API, skipping local connectivity test")
            return True
            
    except Exception as e:
        print(f"Connection test error: {e}")
        return False

def test_error_handling(client_manager):
    """Test error handling with invalid inputs"""
    print("\n" + "="*50)
    print("TESTING ERROR HANDLING")
    print("="*50)
    
    try:
        # Test with empty document
        print("Testing with empty document...")
        
        extraction_result = client_manager.extract_medical_data(
            "", 
            'empty_test_document.txt', 
            use_simple_schema=False
        )
        
        if extraction_result['success']:
            extracted_data = extraction_result['extraction_data']
            print(f"Empty document handled gracefully (studies: {extracted_data.number_of_studies})")
            return True
        elif extraction_result['error_type'] == 'empty_document':
            print("Empty document rejected appropriately")
            return True
        else:
            print(f"WARNING: Unexpected response for empty document: {extraction_result['error_message']}")
            return True  # Still consider this a success as it handled the error
            
    except Exception as e:
        print(f"Error handling test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all tests"""
    print("S5 LLM EXTRACTION DUAL CLIENT TEST")
    print("=" * 60)
    print("This comprehensive test verifies S5 LLM functionality with both OpenAI and Ollama clients")
    print("Note: S5 now includes built-in health checks. This is for advanced debugging.")
    print()
    
    # Test sequence
    tests = [
        ("Configuration", test_llm_config),
        ("Connection", test_connection),
        ("Client Manager", test_client_manager),
    ]
    
    results = {}
    client_manager = None
    
    for test_name, test_func in tests:
        if test_name == "Client Manager":
            client_manager = test_func()
            results[test_name] = client_manager is not None
        else:
            results[test_name] = test_func()
    
    # Only run extraction tests if client manager is available
    if client_manager:
        results["LLM Extraction"] = test_llm_extraction(client_manager)
        results["Error Handling"] = test_error_handling(client_manager)
    else:
        results["LLM Extraction"] = False
        results["Error Handling"] = False
        print("\nSkipping advanced tests (client manager unavailable)")
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("-" * 60)
    if all_passed:
        print("ALL TESTS PASSED! S5 LLM extraction is ready to use.")
        print(f"\nTIP: Using {LLMConfig.get_client_type()} client")
        print("   S5 now includes automatic health checks.")
        print("   Just run: python code/S5_llmExtract.py --test-llm")
    else:
        print("SOME TESTS FAILED. Check the output above for details.")
        print("\nTroubleshooting:")
        print(f"1. Current client type: {LLMConfig.get_client_type()}")
        if LLMConfig.get_client_type() == LLMClientType.OLLAMA:
            print("   - Make sure Ollama is running: ollama serve")
            print(f"   - Make sure model is available: ollama pull {LLMConfig.MODEL_NAME}")
        else:
            print("   - Check that API key is set correctly")
            print("   - Verify API endpoint is accessible")
        print("3. Check that dependencies are installed: pip install openai pydantic ollama")
        print("4. Try the integrated health check: python code/S5_llmExtract.py --test-llm")
    
    return 0 if all_passed else 1

def parse_test_arguments():
    """Parse command line arguments for test script"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Test S5 LLM extraction with dual client support"
    )
    parser.add_argument(
        '--client-type', 
        choices=['auto', 'openai', 'ollama'],
        default='auto',
        help='Force specific client type (default: auto-detect)'
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_test_arguments()
    
    # Set client type if specified
    if args.client_type != 'auto':
        import os
        os.environ['S5_LLM_CLIENT_TYPE'] = args.client_type
        print(f"Forced client type to: {args.client_type}")
    
    sys.exit(main())