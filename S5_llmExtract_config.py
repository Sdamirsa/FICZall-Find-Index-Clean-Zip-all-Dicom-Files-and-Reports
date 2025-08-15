"""
LLM Configuration and Pydantic Models for S5 Document Processing

This module contains ONLY the LLM-specific settings and Pydantic models for 
medical data extraction. All other configuration (paths, etc.) should be in run_config.py

IMPORTANT: This file contains settings for AI processing. If using commercial APIs
(OpenAI, Claude, etc.), patient data will be sent to external servers. For HIPAA
compliance, use local models (Ollama) or ensure proper data agreements are in place.
"""

import os
from typing import Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# CLIENT CONFIGURATION ENUMS
# =============================================================================

class LLMClientType(str, Enum):
    """LLM client types supported"""
    AUTO = "auto"           # Automatic selection based on BASE_URL
    OPENAI = "openai"       # Force OpenAI client (for external APIs)
    OLLAMA = "ollama"       # Force native Ollama client (for local models)

# =============================================================================
# PYDANTIC    for EXTRACTION of DATA FROM REPORTS
# =============================================================================


class ImagingModality(str, Enum):
    CT = "CT"
    MRI = "MRI"
    XRAY = "X-ray"
    ULTRASOUND = "Ultrasound"
    PET = "PET"
    SPECT = "SPECT"
    MAMMOGRAPHY = "Mammography"
    FLUOROSCOPY = "Fluoroscopy"
    NUCLEAR_MEDICINE = "Nuclear Medicine"
    OTHER = "Other"
    UNKNOWN = "Unknown"


class BodyPart(str, Enum):
    HEAD = "Head"
    NECK = "Neck"
    CHEST = "Chest"
    ABDOMEN = "Abdomen"
    PELVIS = "Pelvis"
    SPINE = "Spine"
    EXTREMITIES = "Extremities"
    WHOLE_BODY = "Whole Body"
    ABDOMEN_PELVIS = "Abdomen and Pelvis"
    CHEST_ABDOMEN_PELVIS = "Chest, Abdomen and Pelvis"
    OTHER = "Other"
    UNKNOWN = "Unknown"


class PatientID(BaseModel):
    id_value: str = Field(description="The ID value")
    id_type: str = Field(description="Type of ID (e.g., 'Medical Record Number', 'National ID', 'Patient Number')")


class PatientInfo(BaseModel):
    name: Optional[str] = Field(None, description="Patient's full name")
    ids: list[PatientID] = Field(default_factory=list, description="List of patient identifiers")
    patient_number: Optional[str] = Field(None, description="Patient number from the report")
    age: Optional[str] = Field(None, description="Patient age (with units if specified)")
    gender: Optional[str] = Field(None, description="Patient gender/sex")


class SimpleMedicalExtraction(BaseModel):
    """
    Simplified Pydantic model for testing and open-source model compatibility.
    
    This is a reduced complexity version for better parsing with open-source models.
    """
    
    number_of_studies: int = Field(
        default=0,
        description="Number of imaging studies in this document (0 if no medical studies)"
    )
    
    modality: str = Field(
        default="Unknown",
        description="Imaging type: CT, MRI, X-ray, Ultrasound, PET, or Unknown"
    )
    
    body_part: str = Field(
        default="Unknown", 
        description="Body region: Head, Chest, Abdomen, Pelvis, Spine, or Unknown"
    )
    
    patient_name: Optional[str] = Field(
        None,
        description="Patient's name if mentioned"
    )
    
    patient_age: Optional[str] = Field(
        None, 
        description="Patient's age if mentioned"
    )
    
    contrast_used: Optional[bool] = Field(
        None,
        description="True if contrast was used, False if not, None if unclear"
    )
    
    findings: Optional[str] = Field(
        None,
        description="Key findings from the report"
    )


class MedicalReportExtraction(BaseModel):
    """
    Pydantic model for extracting structured data from medical imaging reports.
    
    This model is designed to extract key information from radiology reports,
    particularly CT scans and other imaging studies.
    """
    
    number_of_studies: int = Field(
        default=0,
        description="Number of distinct imaging studies reported in this document. Return 0 if this is just plain text without useful medical information about imaging exams."
    )
    
    imaging_indication: Optional[str] = Field(
        None,
        description="The clinical indication or reason for the imaging study"
    )
    
    imaging_modality: ImagingModality = Field(
        default=ImagingModality.UNKNOWN,
        description="Primary imaging modality used (CT, MRI, X-ray, etc.)"
    )
    
    body_part: BodyPart = Field(
        default=BodyPart.UNKNOWN,
        description="Primary body part or region examined"
    )
    
    current_disease_list: list[str] = Field(
        default_factory=list,
        description="List of current diseases, conditions, or abnormal findings mentioned in the report"
    )
    
    previous_disease_list: list[str] = Field(
        default_factory=list,
        description="List of previous diseases, conditions, or medical history mentioned"
    )
    
    previous_interventions: list[str] = Field(
        default_factory=list,
        description="List of previous interventions, surgeries, chemotherapy, or treatments mentioned"
    )
    
    patient_info: PatientInfo = Field(
        default_factory=PatientInfo,
        description="Patient demographic and identifier information"
    )
    
    contrast_used: Optional[bool] = Field(
        None,
        description="Whether contrast material was used in the imaging study"
    )
    
    findings_summary: Optional[str] = Field(
        None,
        description="Brief summary of key imaging findings"
    )
    
    impression: Optional[str] = Field(
        None,
        description="Radiologist's impression or conclusion"
    )

# =============================================================================
# LLM  CONFIG    for EXTRACTION of DATA FROM REPORTS
# =============================================================================

# LLM API Configuration
class LLMConfig:
    """Configuration for LLM API settings with dual client support
    
    Supports both OpenAI-compatible APIs and native Ollama client:
    - OpenAI: https://api.openai.com/v1 - External API (requires API key)
    - Ollama: http://localhost:11434 - Local model server (no API key needed)
    - Custom: Your organization's API endpoint
    
    For medical data, ensure HIPAA compliance when using external APIs.
    """
    
    # Client Selection Configuration
    CLIENT_TYPE: LLMClientType = LLMClientType(os.getenv("S5_LLM_CLIENT_TYPE", "auto"))
    
    # API Configuration (OpenAI-compatible)
    BASE_URL: str = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")  
    API_KEY_ENV_VAR: str = "OPENAI_API_KEY"
    
    # Ollama-specific Configuration
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    
    # Model Configuration
    MODEL_NAME: str = os.getenv("S5_LLM_MODEL", "llama3-groq-tool-use:8b-q8_0")  # gpt-oss:20b
    MAX_TOKENS: int = int(os.getenv("S5_LLM_MAX_TOKENS", "4096"))
    TEMPERATURE: float = float(os.getenv("S5_LLM_TEMPERATURE", "0.3"))
    
    # Processing Configuration (for LLM-specific operations)
    MAX_RETRIES: int = int(os.getenv("S5_LLM_MAX_RETRIES", "3"))
    RETRY_DELAY: float = float(os.getenv("S5_LLM_RETRY_DELAY", "1.0"))
    BATCH_SIZE: int = int(os.getenv("S5_LLM_BATCH_SIZE", "10"))
    CHUNK_SIZE: int = int(os.getenv("S5_LLM_CHUNK_SIZE", "1000"))
    PARALLEL_WORKERS: int = int(os.getenv("S5_LLM_PARALLEL_WORKERS", "1"))
    
    # Ollama-specific options
    OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
    OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", str(MAX_TOKENS)))
    OLLAMA_TOP_K: int = int(os.getenv("OLLAMA_TOP_K", "40"))
    OLLAMA_TOP_P: float = float(os.getenv("OLLAMA_TOP_P", "0.9"))
    OLLAMA_THINKING: bool = os.getenv("OLLAMA_THINKING", "false").lower() == "true"
    
    # System Prompt for the LLM
    SYSTEM_PROMPT: str = """You are a medical AI assistant specialized in extracting structured information from radiology reports and medical documents. 

Your task is to analyze medical reports and extract key information in a structured format.

Key guidelines:
- If the document contains no useful medical imaging information (just plain text, contact info, etc.), set number_of_studies to 0
- Extract patient information carefully, including all available IDs and their types
- Identify imaging modality (CT, MRI, etc.) and body part examined
- List current diseases/findings and any mentioned previous conditions
- Note any previous interventions (surgery, chemotherapy, etc.)
- Be conservative with interpretations - only extract information that is clearly stated
- For Persian/Farsi text, translate medical terms to English for consistency
- Maintain accuracy and do not hallucinate information not present in the text"""

    @classmethod
    def get_client_type(cls) -> LLMClientType:
        """Determine which client type to use"""
        if cls.CLIENT_TYPE == LLMClientType.AUTO:
            # Auto-detect based on configuration
            if cls.is_local():
                return LLMClientType.OLLAMA
            else:
                return LLMClientType.OPENAI
        return cls.CLIENT_TYPE
    
    @classmethod
    def get_api_key(cls) -> str:
        """Get API key from environment variable (for OpenAI client)"""
        api_key = os.getenv(cls.API_KEY_ENV_VAR)
        if not api_key:
            # For Ollama, API key is not required
            if cls.get_client_type() == LLMClientType.OLLAMA:
                return "ollama"  # Dummy key for compatibility
            raise ValueError(f"API key not found. Please set the {cls.API_KEY_ENV_VAR} environment variable.")
        return api_key

    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is available"""
        try:
            client_type = cls.get_client_type()
            if client_type == LLMClientType.OPENAI:
                cls.get_api_key()  # Will raise if missing for OpenAI
            # For Ollama, no API key validation needed
            return True
        except ValueError:
            return False
    
    @classmethod
    def is_local(cls) -> bool:
        """Check if using local LLM (Ollama or similar)"""
        return ("localhost" in cls.BASE_URL or "127.0.0.1" in cls.BASE_URL or 
                "localhost" in cls.OLLAMA_HOST or "127.0.0.1" in cls.OLLAMA_HOST)
    
    @classmethod
    def get_privacy_warning(cls) -> str:
        """Get privacy warning based on configuration"""
        client_type = cls.get_client_type()
        if client_type == LLMClientType.OLLAMA or cls.is_local():
            return "✓ Using LOCAL LLM - Patient data stays on this machine"
        else:
            return f"⚠️ WARNING: Using EXTERNAL API at {cls.BASE_URL}\n   Patient data will be sent to external servers!\n   Ensure HIPAA compliance and data agreements are in place."
    
    @classmethod
    def get_ollama_options(cls) -> Dict[str, Any]:
        """Get Ollama-specific options for chat requests"""
        return {
            "temperature": cls.TEMPERATURE,
            "num_predict": cls.OLLAMA_NUM_PREDICT,
            "top_k": cls.OLLAMA_TOP_K,
            "top_p": cls.OLLAMA_TOP_P,
        }


# Validate configuration on import
def _validate_llm_config():
    """Validate that all required attributes are present in LLMConfig"""
    required_attrs = [
        'CLIENT_TYPE', 'BASE_URL', 'API_KEY_ENV_VAR', 'OLLAMA_HOST', 'MODEL_NAME', 
        'MAX_TOKENS', 'TEMPERATURE', 'MAX_RETRIES', 'RETRY_DELAY', 'BATCH_SIZE', 
        'CHUNK_SIZE', 'SYSTEM_PROMPT', 'OLLAMA_TIMEOUT', 'PARALLEL_WORKERS', 'OLLAMA_THINKING'
    ]
    
    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(LLMConfig, attr):
            missing_attrs.append(attr)
    
    if missing_attrs:
        raise AttributeError(f"LLMConfig missing required attributes: {missing_attrs}")

# Run validation
_validate_llm_config()

# Export the main classes and configuration
__all__ = [
    "MedicalReportExtraction",
    "SimpleMedicalExtraction",
    "PatientInfo", 
    "PatientID",
    "ImagingModality",
    "BodyPart",
    "LLMConfig",
    "LLMClientType"
]