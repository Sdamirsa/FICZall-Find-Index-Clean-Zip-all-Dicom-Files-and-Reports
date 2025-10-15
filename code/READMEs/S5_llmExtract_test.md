# S5_llmExtract_test.py

## Purpose
Standalone comprehensive test script for S5 LLM functionality with dual client support.

## Code Logic
- Tests LLM configuration validation
- Verifies connectivity to OpenAI API or local Ollama service
- Tests client manager initialization for both service types
- Performs actual medical text extraction with sample document
- Tests error handling with edge cases (empty documents)

## Inputs
- Configuration from S5_llmExtract_config.py
- Sample medical document (built-in test data)
- Client type preference (auto-detect, OpenAI, or Ollama)

## Outputs
- Comprehensive test results and status
- Extracted medical data from test document
- Troubleshooting guidance for failed tests
- Response metadata and parsing method details

## User Changeable Settings
- `--client-type`: Force specific client (auto/openai/ollama)
- Test document content can be modified in SAMPLE_MEDICAL_TEXT
- LLM configuration via S5_llmExtract_config.py
- All LLM parameters (model, temperature, etc.) inherited from main config

**Note**: This is primarily for debugging. S5 main script includes integrated health checks.