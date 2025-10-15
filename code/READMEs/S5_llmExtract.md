# S5_llmExtract.py

## Purpose
Extracts structured medical information from documents using Large Language Models (OpenAI/Ollama).

## Code Logic
- Supports dual client architecture (OpenAI API and local Ollama)
- Uses structured outputs with Pydantic models for medical data
- Implements robust parsing with multiple fallback strategies
- Processes documents from S1_indexingFiles_allDocuments.json
- Includes comprehensive health checks and error handling

## Inputs
- S1_indexingFiles_allDocuments.json (document text from S1)
- LLM configuration (model, API keys, endpoints)
- Processing parameters (batch size, chunk size)

## Outputs
- "S5_extracted_medical_data.json": Structured medical extractions
- "S5_failed_extractions.json": Failed processing attempts
- "S5_llmExtract_progress.json": Resume tracking
- "S5_processing_stats.json": Comprehensive statistics

## User Changeable Settings
- LLM client type: OpenAI or Ollama (via S5_llmExtract_config.py)
- Model selection and API parameters
- `--batch-size`: Progress save frequency
- `--chunk-size`: Memory management for large datasets
- `--overwrite`: Force reprocessing
- Temperature, max tokens, and other LLM parameters in config file