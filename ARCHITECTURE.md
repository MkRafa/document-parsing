# Insurance Document Parsing Agent - Architecture

## System Overview

The Insurance Parsing Agent is an intelligent system designed to extract patient-related information from insurance documents using LLM capabilities. It follows an agentic workflow pattern with validation and refinement loops.

## Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│              (Examples: CLI, API, Batch Processing)          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│         InsuranceParsingAgent (Orchestrator)                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • Document Processing                               │   │
│  │  • LLM Coordination                                  │   │
│  │  • Workflow Management                               │   │
│  │  • Result Assembly                                   │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────┬─────────────────────┬──────────────────────────┘
             │                     │
    ┌────────▼─────────┐   ┌──────▼─────────────┐
    │ DocumentProcessor │   │   LLM Client      │
    │                  │   │                    │
    │ • PDF Extract    │   │ • OpenAI (GPT-4)   │
    │ • Image Extract  │   │ • Anthropic Claude │
    │ • Text Extract   │   │ • Custom LLMs      │
    │ • OCR Support    │   │ • Multi-provider   │
    └────────┬─────────┘   └────────┬───────────┘
             │                     │
    ┌────────▼──────────────────────▼───────────────────────┐
    │           Processing Pipeline                         │
    │                                                       │
    │  1. Document Analysis                                │
    │     ├─ Detect document type                          │
    │     ├─ Identify sections                             │
    │     └─ Assess quality                                │
    │                                                       │
    │  2. Information Extraction                           │
    │     ├─ Personal Information                          │
    │     ├─ Insurance Details                             │
    │     ├─ Coverage Information                          │
    │     ├─ Provider Information                          │
    │     └─ Medical History                               │
    │                                                       │
    │  3. Validation                                       │
    │     ├─ Required field checks                         │
    │     ├─ Format validation                             │
    │     ├─ Logical consistency                           │
    │     └─ Business rule validation                      │
    │                                                       │
    │  4. Refinement (Iterative)                           │
    │     ├─ Issue identification                          │
    │     ├─ Context-aware correction                      │
    │     ├─ Re-validation                                 │
    │     └─ Confidence scoring                            │
    └──────────────────────────────────────────────────────┘
             │
    ┌────────▼──────────────────────────────────────────────┐
    │            Structured Output (Models)                 │
    │                                                       │
    │  ├─ PersonalInfo                                      │
    │  ├─ InsuranceDetails                                  │
    │  ├─ CoverageInfo                                      │
    │  ├─ ProviderInfo                                      │
    │  ├─ MedicalHistory                                    │
    │  ├─ PatientInfo (aggregated)                          │
    │  └─ ExtractionResult (with metadata)                  │
    └──────────────────────────────────────────────────────┘
```

## Module Structure

```
src/
├── __init__.py              # Package initialization
├── agent.py                 # Main InsuranceParsingAgent class
├── models.py                # Pydantic data models
├── document_processor.py     # Document handling and extraction
├── llm_client.py            # LLM provider abstraction
└── utils.py                 # Helper utilities (optional)

config/
├── extraction_schema.json   # Extraction schema templates
└── validation_rules.json    # Validation rule configurations

examples/
├── basic_usage.py           # Single document processing
├── batch_processing.py      # Multiple document processing
└── custom_schema.py         # Custom extraction configuration

tests/
├── test_models.py           # Model validation tests
├── test_document_processor.py
└── test_agent.py            # Integration tests
```

## Data Models

### PatientInfo Hierarchy

```
PatientInfo
├── PersonalInfo
│   ├── first_name (required)
│   ├── last_name (required)
│   ├── date_of_birth
│   ├── gender
│   ├── email
│   ├── phone
│   ├── address
│   ├── city
│   ├── state
│   └── zip_code
├── InsuranceDetails
│   ├── member_id (required)
│   ├── policy_number
│   ├── group_number
│   ├── plan_type (HMO/PPO/EPO/HDHP/POS)
│   ├── plan_name
│   ├── effective_date
│   ├── termination_date
│   └── carrier_name
├── CoverageInfo
│   ├── deductible
│   ├── out_of_pocket_max
│   ├── copay_visit
│   ├── copay_specialist
│   ├── copay_emergency
│   ├── coinsurance
│   ├── covered_services[]
│   └── excluded_services[]
├── ProviderInfo
│   ├── pcp_name
│   ├── pcp_specialty
│   ├── pcp_phone
│   ├── pcp_office_address
│   ├── network_status
│   └── referred_specialists[]
├── MedicalHistory
│   ├── pre_existing_conditions[]
│   ├── current_medications[]
│   ├── allergies[]
│   └── chronic_conditions[]
└── Metadata
    ├── extraction_confidence
    ├── extraction_notes
    └── extraction_timestamp
```

## Processing Workflow

### Single Document Processing

1. **Input Validation**
   - Verify document path exists
   - Check file format is supported

2. **Document Processing**
   - Extract text/images from document
   - Generate document metadata
   - Prepare content for LLM

3. **Document Analysis**
   - Identify document type
   - Locate key sections
   - Assess completeness

4. **Information Extraction**
   - Call LLM with extraction schema
   - Parse structured JSON response
   - Map to PatientInfo models

5. **Validation Loop**
   - Check required fields
   - Validate data formats
   - Check logical consistency
   - Calculate confidence score

6. **Refinement (if needed)**
   - Identify issues
   - Send corrected extraction back to LLM
   - Re-validate
   - Iterate (configurable iterations)

7. **Result Assembly**
   - Combine all extracted information
   - Add metadata
   - Return ExtractionResult

### Batch Processing

- Sequential or parallel document processing
- Aggregated statistics and reporting
- Error handling and logging
- Export to JSON format

## LLM Integration

### Provider Abstraction

```python
LLMClient (Abstract)
├── OpenAIClient
│   └── Uses: gpt-4, gpt-4-turbo, gpt-3.5-turbo
└── AnthropicClient
    └── Uses: claude-3-opus, claude-3-sonnet, claude-3-haiku
```

### LLM Prompting Strategy

1. **System Prompt**: Establishes role and constraints
2. **Extraction Prompt**: Includes document text + schema
3. **Validation Prompt**: Checks extracted data against rules
4. **Refinement Prompt**: Fixes identified issues

### Response Format

- JSON-only responses with structured schemas
- Type validation using Pydantic
- Error handling for malformed responses

## Configuration System

### Default Configuration

- Extraction schema defined in `config/extraction_schema.json`
- Validation rules defined separately
- Customizable per use case

### Custom Configuration

- Load from JSON file
- Override in code
- Schema validation on load

## Error Handling

### Document Processing Errors

- FileNotFoundError: File not found
- ValueError: Unsupported file type
- RuntimeError: OCR/extraction failed

### LLM Processing Errors

- API connectivity issues
- Rate limiting
- Invalid response format
- Retry with backoff

### Validation Errors

- Missing required fields
- Invalid data formats
- Logical inconsistencies
- Refinement iteration limit

## Performance Considerations

- Document size limits
- Batch processing optimization
- Caching strategies
- Async/parallel processing options

## Security

- API key management via environment variables
- No sensitive data logging by default
- Configurable data retention
- Document access control (user responsibility)

## Extensibility

- Custom LLM providers via LLMClient interface
- Custom extraction schemas
- Custom validation rules
- Plugin architecture for document processors
- Custom output formats
