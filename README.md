# Insurance Document Parsing Agent System

An intelligent agentic system for parsing insurance documents and extracting patient-related information using LLM capabilities.

## Features

- 📄 Multi-format document support (PDF, images, text)
- 🤖 LLM-powered intelligent extraction
- 👤 Patient information extraction (demographics, coverage details, etc.)
- 🔄 Agentic workflow with validation and refinement
- 💾 Structured output in JSON format
- ✅ Built-in validation and error handling
- 🔀 Chainable processing pipeline

## Architecture

```
Document Input
    ↓
Document Processor (PDF/Image extraction)
    ↓
Agentic System
    ├── Document Analyzer Agent
    ├── Information Extractor Agent
    ├── Validator Agent
    └── Refinement Agent
    ↓
Structured Output (Patient Data)
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from insurance_parsing_agent import InsuranceParsingAgent

# Initialize agent
agent = InsuranceParsingAgent(
    api_key="your-openai-api-key",
    model="gpt-4"
)

# Process document
result = agent.process_document(
    document_path="insurance_document.pdf",
    extraction_schema="patient_info"
)

print(result)
```

## Supported Patient Information

- Personal Information (Name, DOB, Gender, Contact)
- Insurance Details (Policy Number, Member ID, Group)
- Coverage Information (Plan Type, Deductible, Co-pay)
- Provider Information (Primary Care, Network Status)
- Medical History (Pre-existing conditions, Medications)

## Configuration

See `config/extraction_schema.json` for customizable extraction templates.
