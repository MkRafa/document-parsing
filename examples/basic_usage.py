#!/usr/bin/env python3
"""
Basic usage example of the Insurance Parsing Agent

This example shows how to:
1. Initialize the agent
2. Process a single document
3. Handle the results
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agent import InsuranceParsingAgent


def main():
    """Main example function"""

    # Initialize the agent with OpenAI
    agent = InsuranceParsingAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        provider="openai",
        model="gpt-4-turbo",
    )

    # Example document path - replace with your own
    document_path = "sample_insurance_document.pdf"

    # Check if document exists
    if not Path(document_path).exists():
        print(f"Document not found: {document_path}")
        print("Please provide a valid insurance document path")
        return

    # Process the document
    print(f"Processing document: {document_path}")
    result = agent.process_document(
        document_path,
        refinement_iterations=2,
        verbose=True,
    )

    # Display results
    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)

    print(f"\nSuccess: {result.success}")
    print(f"Confidence: {result.patient_info.extraction_confidence:.1%}")
    print(f"Pages Processed: {result.pages_processed}")

    # Patient Information
    print("\n--- PATIENT INFORMATION ---")
    personal = result.patient_info.personal_info
    print(f"Name: {personal.first_name} {personal.last_name}")
    print(f"DOB: {personal.date_of_birth}")
    print(f"Email: {personal.email}")
    print(f"Phone: {personal.phone}")

    # Insurance Details
    print("\n--- INSURANCE DETAILS ---")
    insurance = result.patient_info.insurance_details
    print(f"Member ID: {insurance.member_id}")
    print(f"Policy Number: {insurance.policy_number}")
    print(f"Plan Type: {insurance.plan_type}")
    print(f"Carrier: {insurance.carrier_name}")
    print(f"Effective Date: {insurance.effective_date}")

    # Coverage Information
    print("\n--- COVERAGE INFORMATION ---")
    coverage = result.patient_info.coverage_info
    print(f"Deductible: ${coverage.deductible}")
    print(f"Out-of-Pocket Max: ${coverage.out_of_pocket_max}")
    print(f"Office Visit Copay: ${coverage.copay_visit}")
    print(f"Specialist Copay: ${coverage.copay_specialist}")

    # Provider Information
    print("\n--- PROVIDER INFORMATION ---")
    provider = result.patient_info.provider_info
    print(f"PCP Name: {provider.pcp_name}")
    print(f"PCP Specialty: {provider.pcp_specialty}")
    print(f"Network Status: {provider.network_status}")

    # Medical History
    print("\n--- MEDICAL HISTORY ---")
    medical = result.patient_info.medical_history
    if medical.pre_existing_conditions:
        print(f"Pre-existing Conditions: {', '.join(medical.pre_existing_conditions)}")
    if medical.allergies:
        print(f"Allergies: {', '.join(medical.allergies)}")
    if medical.current_medications:
        print(f"Medications: {', '.join(medical.current_medications)}")

    # Export results
    print("\n" + "=" * 60)
    export_results(result)


def export_results(result):
    """Export results to JSON file"""
    output_file = "extraction_results.json"

    output_data = {
        "success": result.success,
        "extraction_timestamp": result.extraction_timestamp.isoformat(),
        "pages_processed": result.pages_processed,
        "confidence": result.patient_info.extraction_confidence,
        "patient_info": result.patient_info.model_dump(),
        "errors": result.errors,
        "warnings": result.warnings,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results exported to: {output_file}")


if __name__ == "__main__":
    main()
