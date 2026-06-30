import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from .document_processor import DocumentProcessor
from .llm_client import OpenAIClient, AnthropicClient, LLMClient
from .models import (
    PatientInfo,
    PersonalInfo,
    InsuranceDetails,
    CoverageInfo,
    ProviderInfo,
    MedicalHistory,
    ExtractionResult,
)

logger = logging.getLogger(__name__)


class InsuranceParsingAgent:
    """
    Main agentic system for parsing insurance documents and extracting patient information.

    Uses a multi-step workflow:
    1. Document Analysis - Understand document structure and content
    2. Information Extraction - Extract relevant patient data
    3. Validation - Verify extracted data quality
    4. Refinement - Improve extraction based on validation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the parsing agent

        Args:
            api_key: API key for LLM provider
            provider: LLM provider ('openai' or 'anthropic')
            model: Specific model to use
            config_path: Path to custom configuration
        """
        self.provider = provider.lower()
        self.llm_client = self._initialize_llm(api_key, model)
        self.document_processor = DocumentProcessor()
        self.config = self._load_config(config_path)
        self.extraction_schema = self.config.get("extraction_schema", {})
        self.validation_rules = self.config.get("validation_rules", {})

        logger.info(f"Initialized InsuranceParsingAgent with {self.provider} provider")

    def process_document(
        self,
        document_path: str,
        refinement_iterations: int = 2,
        verbose: bool = False,
    ) -> ExtractionResult:
        """
        Process an insurance document and extract patient information

        Args:
            document_path: Path to insurance document
            refinement_iterations: Number of refinement iterations
            verbose: Print detailed processing logs

        Returns:
            ExtractionResult with extracted patient information
        """
        if verbose:
            logger.setLevel(logging.DEBUG)

        logger.info(f"Processing document: {document_path}")
        errors = []
        warnings = []

        try:
            # Step 1: Process document
            logger.debug("Step 1: Processing document...")
            document_text, metadata = self.document_processor.process_document(
                document_path
            )
            logger.debug(f"Extracted {len(document_text)} characters from document")

            # Step 2: Analyze document
            logger.debug("Step 2: Analyzing document structure...")
            analysis = self._analyze_document(document_text)
            logger.debug(f"Analysis: {analysis}")

            # Step 3: Extract information
            logger.debug("Step 3: Extracting patient information...")
            extracted_data = self._extract_information(document_text)
            logger.debug(f"Extracted {len(extracted_data)} top-level fields")

            # Step 4: Validate extraction
            logger.debug("Step 4: Validating extracted data...")
            is_valid, validation_issues = self._validate_extraction(extracted_data)

            if not is_valid:
                warnings.extend(validation_issues)
                logger.warning(f"Validation issues found: {validation_issues}")

                # Step 5: Refine extraction
                for i in range(refinement_iterations):
                    logger.debug(f"Refinement iteration {i+1}...")
                    extracted_data = self._refine_extraction(
                        extracted_data, document_text, validation_issues
                    )

                    is_valid, validation_issues = self._validate_extraction(
                        extracted_data
                    )
                    if is_valid:
                        logger.info(f"Extraction validated after {i+1} refinements")
                        break

                if not is_valid:
                    warnings.extend(validation_issues)
                    logger.warning(f"Could not fully validate extraction")

            # Step 6: Build result
            logger.debug("Step 6: Building result object...")
            patient_info = self._build_patient_info(extracted_data)
            confidence = self._calculate_confidence(extracted_data, is_valid)

            result = ExtractionResult(
                patient_info=PatientInfo(
                    **extracted_data,
                    extraction_confidence=confidence,
                ),
                document_source=str(document_path),
                extraction_timestamp=datetime.now(),
                pages_processed=metadata.pages,
                success=is_valid,
                errors=errors,
                warnings=warnings,
                raw_extracted_text=document_text[:500],  # Store sample
            )

            logger.info(f"Successfully processed document with {confidence:.1%} confidence")
            return result

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            errors.append(str(e))

            # Return partial result on error
            return ExtractionResult(
                patient_info=self._get_empty_patient_info(),
                document_source=str(document_path),
                extraction_timestamp=datetime.now(),
                pages_processed=0,
                success=False,
                errors=errors,
                warnings=warnings,
            )

    def _analyze_document(self, document_text: str) -> Dict[str, Any]:
        """Analyze document structure and content"""
        prompt = f"""
Analyze this insurance document and provide:
1. Document type (explanation of benefits, policy document, insurance card, etc.)
2. Key sections identified
3. Presence of patient information
4. Presence of coverage information
5. Document quality (complete/incomplete)

Document:
{document_text[:2000]}

Respond as JSON with keys: type, sections, has_patient_info, has_coverage_info, quality
"""

        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        analysis = json.loads(response.choices[0].message.content)
        return analysis

    def _extract_information(self, document_text: str) -> Dict[str, Any]:
        """Extract patient information from document"""
        logger.debug("Calling LLM for information extraction...")

        extracted = self.llm_client.extract_information(
            document_text,
            self.extraction_schema,
            system_prompt=self._get_extraction_system_prompt(),
        )

        return extracted

    def _validate_extraction(self, extracted_data: Dict[str, Any]) -> tuple[bool, list]:
        """Validate extracted data"""
        logger.debug("Validating extracted data...")

        is_valid, issues = self.llm_client.validate_extraction(
            extracted_data,
            self.validation_rules,
        )

        return is_valid, issues

    def _refine_extraction(
        self,
        extracted_data: Dict[str, Any],
        document_text: str,
        issues: list,
    ) -> Dict[str, Any]:
        """Refine extraction based on validation issues"""
        logger.debug(f"Refining extraction with {len(issues)} issues...")

        refined = self.llm_client.refine_extraction(
            extracted_data,
            document_text,
            issues,
        )

        return refined

    def _build_patient_info(self, extracted_data: Dict[str, Any]) -> PatientInfo:
        """Build PatientInfo object from extracted data"""
        try:
            personal = PersonalInfo(
                first_name=extracted_data.get("personal_info", {}).get("first_name", ""),
                last_name=extracted_data.get("personal_info", {}).get("last_name", ""),
                **{
                    k: v
                    for k, v in extracted_data.get("personal_info", {}).items()
                    if k not in ["first_name", "last_name"]
                },
            )

            insurance = InsuranceDetails(
                member_id=extracted_data.get("insurance_details", {}).get(
                    "member_id", ""
                ),
                **{
                    k: v
                    for k, v in extracted_data.get("insurance_details", {}).items()
                    if k != "member_id"
                },
            )

            coverage = CoverageInfo(
                **extracted_data.get("coverage_info", {})
            )

            provider = ProviderInfo(
                **extracted_data.get("provider_info", {})
            )

            medical = MedicalHistory(
                **extracted_data.get("medical_history", {})
            )

            return PatientInfo(
                personal_info=personal,
                insurance_details=insurance,
                coverage_info=coverage,
                provider_info=provider,
                medical_history=medical,
            )

        except Exception as e:
            logger.warning(f"Error building patient info: {str(e)}")
            return self._get_empty_patient_info()

    def _get_empty_patient_info(self) -> PatientInfo:
        """Create empty PatientInfo object"""
        return PatientInfo(
            personal_info=PersonalInfo(first_name="", last_name=""),
            insurance_details=InsuranceDetails(member_id=""),
            coverage_info=CoverageInfo(),
            provider_info=ProviderInfo(),
            medical_history=MedicalHistory(),
        )

    def _calculate_confidence(self, extracted_data: Dict[str, Any], is_valid: bool) -> float:
        """Calculate extraction confidence score"""
        # Base score on validation
        base_score = 0.9 if is_valid else 0.6

        # Adjust based on data completeness
        total_fields = 0
        filled_fields = 0

        for section in extracted_data.values():
            if isinstance(section, dict):
                for field, value in section.items():
                    total_fields += 1
                    if value is not None and value != "" and value != []:
                        filled_fields += 1

        if total_fields > 0:
            completeness = filled_fields / total_fields
            return base_score * (0.7 + 0.3 * completeness)

        return base_score

    def _get_extraction_system_prompt(self) -> str:
        return """You are an expert insurance document analyst specializing in extracting patient information.
Your task is to carefully read insurance documents and extract all relevant patient-related data.

Guidelines:
- Extract only information explicitly present in the document
- Return null for missing fields
- Be precise with dates (use YYYY-MM-DD format when possible)
- Normalize phone numbers and zip codes
- List multiple items (allergies, medications, etc.) as arrays
- Maintain data consistency across fields

Return ONLY valid JSON matching the required schema."""

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return json.load(f)

        # Return default configuration
        return {
            "extraction_schema": self._get_default_extraction_schema(),
            "validation_rules": self._get_default_validation_rules(),
        }

    def _get_default_extraction_schema(self) -> Dict[str, Any]:
        """Default extraction schema"""
        return {
            "personal_info": {
                "first_name": "string",
                "last_name": "string",
                "date_of_birth": "string (YYYY-MM-DD)",
                "gender": "string (Male/Female/Other)",
                "email": "string",
                "phone": "string",
                "address": "string",
                "city": "string",
                "state": "string",
                "zip_code": "string",
            },
            "insurance_details": {
                "member_id": "string (required)",
                "policy_number": "string",
                "group_number": "string",
                "plan_type": "string",
                "plan_name": "string",
                "effective_date": "string (YYYY-MM-DD)",
                "termination_date": "string (YYYY-MM-DD)",
                "carrier_name": "string",
            },
            "coverage_info": {
                "deductible": "number",
                "out_of_pocket_max": "number",
                "copay_visit": "number",
                "copay_specialist": "number",
                "copay_emergency": "number",
                "coinsurance": "number",
                "covered_services": ["string"],
                "excluded_services": ["string"],
            },
            "provider_info": {
                "pcp_name": "string",
                "pcp_specialty": "string",
                "pcp_phone": "string",
                "pcp_office_address": "string",
                "network_status": "string",
                "referred_specialists": ["string"],
            },
            "medical_history": {
                "pre_existing_conditions": ["string"],
                "current_medications": ["string"],
                "allergies": ["string"],
                "chronic_conditions": ["string"],
            },
        }

    def _get_default_validation_rules(self) -> Dict[str, Any]:
        """Default validation rules"""
        return {
            "required_fields": [
                "insurance_details.member_id",
                "personal_info.first_name",
                "personal_info.last_name",
            ],
            "field_constraints": {
                "personal_info.date_of_birth": "valid ISO date format",
                "insurance_details.effective_date": "valid ISO date format",
                "coverage_info.deductible": "positive number",
                "coverage_info.out_of_pocket_max": "positive number",
            },
            "logical_rules": [
                "member_id should not be null",
                "at least one of: email or phone should be present",
                "plan_type should be one of: HMO, PPO, EPO, HDHP, POS",
            ],
        }

    def _initialize_llm(self, api_key: Optional[str], model: Optional[str]) -> LLMClient:
        """Initialize LLM client"""
        if self.provider == "openai":
            return OpenAIClient(
                api_key=api_key,
                model=model or "gpt-4-turbo",
            )
        elif self.provider == "anthropic":
            return AnthropicClient(
                api_key=api_key,
                model=model or "claude-3-opus-20240229",
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def extract_batch(
        self,
        document_paths: list,
        verbose: bool = False,
    ) -> list:
        """
        Process multiple documents

        Args:
            document_paths: List of document paths
            verbose: Print detailed logs

        Returns:
            List of ExtractionResult objects
        """
        results = []
        for i, path in enumerate(document_paths, 1):
            logger.info(f"Processing document {i}/{len(document_paths)}: {path}")
            result = self.process_document(path, verbose=verbose)
            results.append(result)

        return results
