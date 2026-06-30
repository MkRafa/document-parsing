import os
import json
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import base64


class LLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    def extract_information(
        self,
        document_text: str,
        extraction_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract structured information from document text"""
        pass

    @abstractmethod
    def validate_extraction(
        self,
        extracted_data: Dict[str, Any],
        validation_rules: Dict[str, Any],
    ) -> tuple[bool, List[str]]:
        """Validate extracted data against rules"""
        pass

    @abstractmethod
    def refine_extraction(
        self,
        extracted_data: Dict[str, Any],
        original_text: str,
        issues: List[str],
    ) -> Dict[str, Any]:
        """Refine extraction based on issues found"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        try:
            import openai
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model

    def extract_information(
        self,
        document_text: str,
        extraction_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract structured information using OpenAI"""

        if not system_prompt:
            system_prompt = self._get_default_system_prompt()

        user_prompt = self._build_extraction_prompt(document_text, extraction_schema)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        response_text = response.choices[0].message.content
        extracted_data = json.loads(response_text)
        return extracted_data

    def validate_extraction(
        self,
        extracted_data: Dict[str, Any],
        validation_rules: Dict[str, Any],
    ) -> tuple[bool, List[str]]:
        """Validate extracted data"""

        prompt = f"""
Validate the following extracted insurance data against these rules:

Data:
{json.dumps(extracted_data, indent=2)}

Validation Rules:
{json.dumps(validation_rules, indent=2)}

Respond with JSON in this format:
{{
    "is_valid": true/false,
    "issues": ["list of issues found"]
}}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a data validation expert. Validate insurance documents strictly.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        return result.get("is_valid", False), result.get("issues", [])

    def refine_extraction(
        self,
        extracted_data: Dict[str, Any],
        original_text: str,
        issues: List[str],
    ) -> Dict[str, Any]:
        """Refine extraction based on validation issues"""

        prompt = f"""
The following extraction has issues. Please refine it:

Original Document Text (relevant excerpt):
{original_text[:2000]}

Current Extraction:
{json.dumps(extracted_data, indent=2)}

Issues Found:
{json.dumps(issues, indent=2)}

Please fix these issues and return the corrected extraction as JSON with the same structure.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at extracting insurance information. Fix the issues in the extraction.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        refined_data = json.loads(response.choices[0].message.content)
        return refined_data

    def _get_default_system_prompt(self) -> str:
        return """You are an expert at extracting insurance information from documents.
Extract patient-related information carefully and accurately.
Return only valid JSON with no additional text.
If information is not found, use null.
Be precise with dates, numbers, and formatting."""

    def _build_extraction_prompt(
        self, document_text: str, schema: Dict[str, Any]
    ) -> str:
        return f"""
Extract insurance information from this document:

{document_text}

Extract data to match this schema:
{json.dumps(schema, indent=2)}

Return the extracted data as JSON matching the schema exactly.
"""


class AnthropicClient(LLMClient):
    """Anthropic Claude API client"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic is required. Install with: pip install anthropic")

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

    def extract_information(
        self,
        document_text: str,
        extraction_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract structured information using Claude"""

        if not system_prompt:
            system_prompt = self._get_default_system_prompt()

        user_prompt = self._build_extraction_prompt(document_text, extraction_schema)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        response_text = message.content[0].text
        extracted_data = json.loads(response_text)
        return extracted_data

    def validate_extraction(
        self,
        extracted_data: Dict[str, Any],
        validation_rules: Dict[str, Any],
    ) -> tuple[bool, List[str]]:
        """Validate extracted data"""

        prompt = f"""
Validate the following extracted insurance data against these rules:

Data:
{json.dumps(extracted_data, indent=2)}

Validation Rules:
{json.dumps(validation_rules, indent=2)}

Respond with JSON in this format:
{{
    "is_valid": true/false,
    "issues": ["list of issues found"]
}}
"""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system="You are a data validation expert. Validate insurance documents strictly. Return only JSON.",
            messages=[{"role": "user", "content": prompt}],
        )

        result = json.loads(message.content[0].text)
        return result.get("is_valid", False), result.get("issues", [])

    def refine_extraction(
        self,
        extracted_data: Dict[str, Any],
        original_text: str,
        issues: List[str],
    ) -> Dict[str, Any]:
        """Refine extraction based on validation issues"""

        prompt = f"""
The following extraction has issues. Please refine it:

Original Document Text (relevant excerpt):
{original_text[:2000]}

Current Extraction:
{json.dumps(extracted_data, indent=2)}

Issues Found:
{json.dumps(issues, indent=2)}

Please fix these issues and return the corrected extraction as JSON with the same structure.
"""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system="You are an expert at extracting insurance information. Fix the issues in the extraction. Return only JSON.",
            messages=[{"role": "user", "content": prompt}],
        )

        refined_data = json.loads(message.content[0].text)
        return refined_data

    def _get_default_system_prompt(self) -> str:
        return """You are an expert at extracting insurance information from documents.
Extract patient-related information carefully and accurately.
Return only valid JSON with no additional text.
If information is not found, use null.
Be precise with dates, numbers, and formatting."""

    def _build_extraction_prompt(
        self, document_text: str, schema: Dict[str, Any]
    ) -> str:
        return f"""
Extract insurance information from this document:

{document_text}

Extract data to match this schema:
{json.dumps(schema, indent=2)}

Return the extracted data as JSON matching the schema exactly.
"""
