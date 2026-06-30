#!/usr/bin/env python3
"""
Batch processing example - process multiple insurance documents

This example shows how to:
1. Process multiple documents in batch
2. Aggregate results
3. Generate a summary report
"""

import sys
import os
import json
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agent import InsuranceParsingAgent
from src.models import ExtractionResult


def main():
    """Main batch processing function"""

    # Initialize the agent
    agent = InsuranceParsingAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        provider="openai",
        model="gpt-4-turbo",
    )

    # Get list of documents to process
    document_dir = Path("insurance_documents")
    
    if not document_dir.exists():
        print(f"Directory not found: {document_dir}")
        print("Please create a directory with insurance documents")
        return

    document_paths = list(document_dir.glob("*.pdf")) + list(document_dir.glob("*.txt"))

    if not document_paths:
        print(f"No documents found in {document_dir}")
        return

    print(f"Found {len(document_paths)} documents to process")

    # Process all documents
    results = agent.extract_batch(document_paths, verbose=False)

    # Generate report
    generate_report(results, document_paths)


def generate_report(results: List[ExtractionResult], document_paths: List[Path]):
    """Generate a summary report of all processed documents"""

    print("\n" + "=" * 70)
    print("BATCH PROCESSING REPORT")
    print("=" * 70)

    # Summary statistics
    total = len(results)
    successful = sum(1 for r in results if r.success)
    success_rate = (successful / total * 100) if total > 0 else 0

    print(f"\nTotal Documents: {total}")
    print(f"Successfully Processed: {successful}")
    print(f"Success Rate: {success_rate:.1f}%")

    # Confidence statistics
    confidences = [r.patient_info.extraction_confidence for r in results if r.success]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        print(f"Average Confidence: {avg_confidence:.1%}")

    # Detailed results
    print("\n" + "-" * 70)
    print("DETAILED RESULTS")
    print("-" * 70)

    for i, result in enumerate(results, 1):
        doc_name = document_paths[i - 1].name if i <= len(document_paths) else f"Document {i}"

        status = "✓ SUCCESS" if result.success else "✗ FAILED"
        confidence = result.patient_info.extraction_confidence

        print(f"\n{i}. {doc_name} [{status}]")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Pages: {result.pages_processed}")

        if result.patient_info.personal_info.first_name:
            name = f"{result.patient_info.personal_info.first_name} {result.patient_info.personal_info.last_name}"
            print(f"   Patient: {name}")

        member_id = result.patient_info.insurance_details.member_id
        if member_id:
            print(f"   Member ID: {member_id}")

        if result.errors:
            print(f"   Errors: {', '.join(result.errors)}")

        if result.warnings:
            print(f"   Warnings: {len(result.warnings)} issue(s)")

    # Export detailed results
    export_batch_results(results)


def export_batch_results(results: List[ExtractionResult]):
    """Export batch results to JSON file"""
    output_file = "batch_extraction_results.json"

    output_data = {
        "total_processed": len(results),
        "successful": sum(1 for r in results if r.success),
        "results": [
            {
                "success": r.success,
                "extraction_timestamp": r.extraction_timestamp.isoformat(),
                "pages_processed": r.pages_processed,
                "confidence": r.patient_info.extraction_confidence,
                "patient_info": {
                    "name": f"{r.patient_info.personal_info.first_name} {r.patient_info.personal_info.last_name}",
                    "member_id": r.patient_info.insurance_details.member_id,
                    "plan_type": r.patient_info.insurance_details.plan_type,
                    "carrier": r.patient_info.insurance_details.carrier_name,
                },
                "errors": r.errors,
                "warning_count": len(r.warnings),
            }
            for r in results
        ],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults exported to: {output_file}")


if __name__ == "__main__":
    main()
