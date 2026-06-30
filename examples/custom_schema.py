#!/usr/bin/env python3
"""
Custom extraction schema example

This example shows how to:
1. Create a custom extraction schema
2. Use custom validation rules
3. Process documents with custom requirements
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agent import InsuranceParsingAgent


def main():
    """Main custom schema function"""

    # Create custom configuration
    custom_config = {
        "extraction_schema": {
            "personal_info": {
                "first_name": "string",
                "last_name": "string",
                "date_of_birth": "string (YYYY-MM-DD)",
                "member_id": "string",
            },
            "insurance_details": {
                "policy_number": "string",
                "plan_type": "string",
                "effective_date": "string (YYYY-MM-DD)",
            },
            "coverage_info": {
                "deductible": "number",
                "out_of_pocket_max": "number",
            },
        },
        "validation_rules": {
            "required_fields": [
                "personal_info.first_name",
                "personal_info.last_name",
                "insurance_details.policy_number",
            ],
            "field_constraints": {
                "coverage_info.deductible": "must be positive",
                "personal_info.date_of_birth": "must be valid date",
            },
        },
    }

    # Save custom config
    config_file = "custom_config.json"
    with open(config_file, "w") as f:
        json.dump(custom_config, f, indent=2)

    # Initialize agent with custom config
    agent = InsuranceParsingAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        provider="openai",
        config_path=config_file,
    )

    # Process document
    document_path = "sample_insurance_document.pdf"

    if not Path(document_path).exists():
        print(f"Document not found: {document_path}")
        print("Please provide a valid insurance document path")
        return

    print(f"Processing with custom schema: {document_path}")
    result = agent.process_document(document_path, verbose=True)

    print("\n" + "=" * 60)
    print("CUSTOM EXTRACTION RESULTS")
    print("=" * 60)

    if result.success:
        print("✓ Extraction successful")
    else:
        print("✗ Extraction had issues")

    print(f"Confidence: {result.patient_info.extraction_confidence:.1%}")

    # Display extracted data
    patient_info = result.patient_info
    print("\nExtracted Data:")
    print(json.dumps(patient_info.model_dump(), indent=2))


if __name__ == "__main__":
    main()
