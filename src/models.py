from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"
    UNKNOWN = "Unknown"


class PersonalInfo(BaseModel):
    """Patient personal information"""
    first_name: str
    last_name: str
    date_of_birth: Optional[str] = None
    gender: Optional[Gender] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None


class InsuranceDetails(BaseModel):
    """Insurance policy details"""
    member_id: str
    policy_number: Optional[str] = None
    group_number: Optional[str] = None
    plan_type: Optional[str] = None  # HMO, PPO, EPO, HDHP, etc.
    plan_name: Optional[str] = None
    effective_date: Optional[str] = None
    termination_date: Optional[str] = None
    carrier_name: Optional[str] = None


class CoverageInfo(BaseModel):
    """Coverage details and benefits"""
    deductible: Optional[float] = None
    out_of_pocket_max: Optional[float] = None
    copay_visit: Optional[float] = None
    copay_specialist: Optional[float] = None
    copay_emergency: Optional[float] = None
    coinsurance: Optional[float] = None
    covered_services: List[str] = Field(default_factory=list)
    excluded_services: List[str] = Field(default_factory=list)


class ProviderInfo(BaseModel):
    """Primary care provider information"""
    pcp_name: Optional[str] = None
    pcp_specialty: Optional[str] = None
    pcp_phone: Optional[str] = None
    pcp_office_address: Optional[str] = None
    network_status: Optional[str] = None  # In-network, Out-of-network
    referred_specialists: List[str] = Field(default_factory=list)


class MedicalHistory(BaseModel):
    """Medical history and conditions"""
    pre_existing_conditions: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    chronic_conditions: List[str] = Field(default_factory=list)


class PatientInfo(BaseModel):
    """Complete patient information extracted from document"""
    personal_info: PersonalInfo
    insurance_details: InsuranceDetails
    coverage_info: CoverageInfo
    provider_info: ProviderInfo
    medical_history: MedicalHistory
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extraction_notes: Optional[str] = None


class ExtractionResult(BaseModel):
    """Final extraction result with metadata"""
    patient_info: PatientInfo
    document_source: str
    extraction_timestamp: datetime
    pages_processed: int
    success: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    raw_extracted_text: Optional[str] = None


class DocumentMetadata(BaseModel):
    """Document metadata"""
    file_path: str
    file_type: str  # pdf, image, text
    file_size: int
    pages: int
    extraction_method: str
    extracted_at: datetime
    processing_time_seconds: float
