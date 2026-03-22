# Multi-Agent Form Schema ETL Pipeline

A sophisticated, layout-aware, schema-guided approach to structured information extraction from documents using multi-modal transformers, vision-based Large Language Models, and intelligent orchestration. This system demonstrates a production-ready pipeline for extracting structured data from complex documents like contracts, NDAs, and forms.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [System Components](#system-components)
5. [Installation & Setup](#installation--setup)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [Project Structure](#project-structure)
9. [Data Flow](#data-flow)
10. [API Documentation](#api-documentation)
11. [Examples](#examples)
12. [Supported Models](#supported-models)
13. [Development](#development)
14. [License](#license)

---

## Overview

This project implements an end-to-end pipeline for extracting structured information from unstructured documents. Unlike naive LLM-based extraction that processes documents in their entirety, this system employs a **multi-stage, research-backed approach**:

1. **Document Ingestion**: Extract text and metadata from PDFs with OCR fallback
2. **Layout Analysis**: Perform token-level structural classification using LayoutLMv3
3. **Clause Graph Construction**: Build a hierarchical, deterministic representation of document structure
4. **Schema-Guided Extraction**: Use LLMs to extract specific fields within constrained contexts
5. **Validation & Recovery**: Apply rule-based validation and recover from extraction failures
6. **Orchestration**: Coordinate all stages via a stateful LangGraph pipeline

**Research Benefits**:
- ✅ Reduces LLM hallucination through context grounding
- ✅ Improves field extraction accuracy via structural awareness
- ✅ Provides reproducibility through deterministic processing stages
- ✅ Enables cost-efficient extraction via micro-decoding (field-level) instead of full-document processing

---

## Key Features

### Multi-Modal Processing
- **Text Extraction**: Native PDF text with fallback to Tesseract OCR
- **Layout Awareness**: Spatial features via LayoutLMv3 token classification
- **Vision Capabilities**: Optional Gemini and Llama 3.2 vision models for image-based extraction

### Schema Management
- **Flexible Field Definitions**: Support for multiple data types (date, string, number, boolean)
- **Type-Specific Extraction**: Regex patterns, keyword matching, and LLM-based extraction per field
- **Validation Rules**: Built-in validators for dates, required fields, and custom patterns
- **Easy Extensibility**: Simple JSON-based schema system for new document types

### Intelligent Extraction
- **Clause Graph**: Deterministic document structure representation preserving hierarchy
- **Schema-Guided LLM**: Micro-decoding approach extracts fields within relevant contexts only
- **Multiple LLM Support**: Works with Ollama (local), Hugging Face, and Google Gemini
- **Recovery Mechanisms**: Automatic fallback strategies for failed extractions

### Orchestration
- **LangGraph Pipeline**: Stateful, reliable workflow management with clear node separation
- **Error Handling**: Comprehensive logging, error tracking, and recovery at each stage
- **Metadata Tracking**: Pipeline metadata including timestamps, processing details, and performance metrics

### Evaluation
- **Extraction Metrics**: Field-level accuracy, completeness, and confidence scoring
- **Comparison Framework**: Evaluate multiple extraction methods on the same documents
- **Test Datasets**: Support for benchmark evaluation against gold-standard annotations

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PDF Input                                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 1. PDF Ingestion     │
                    │ - Text extraction    │
                    │ - OCR (fallback)     │
                    │ - Bounding boxes     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 2. Layout Analysis   │
                    │ - Token classification
                    │ - LayoutLMv3         │
                    │ - Structural labels  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 3. Clause Graph      │
                    │ - Hierarchical build │
                    │ - Section grouping   │
                    │ - Context grounding  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 4. Schema Loading    │
                    │ - Field definitions  │
                    │ - Type constraints   │
                    │ - Validation rules   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 5. Field Extraction  │
                    │ - LLM-based          │
                    │ - Schema-guided      │
                    │ - Context-aware      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 6. Validation        │
                    │ - Field validation   │
                    │ - Error recovery     │
                    │ - Completeness check │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 7. JSON Output       │
                    │ - Formatted results  │
                    │ - Metadata included  │
                    │ - Performance stats  │
                    └──────────────────────┘
```

---

## System Components

### 1. **Ingestion Module** (`ingestion/`)
Handles PDF processing and text extraction.

**Key Classes**:
- `DocumentBlock`: Represents a text block with metadata (page, bbox, confidence, source)
- `PDFIngester`: Main ingestion class that extracts text and applies OCR fallback

**Features**:
- Native PDF text extraction via PyMuPDF
- OCR fallback when text confidence is low (configurable threshold)
- Bounding box extraction for spatial awareness
- Metadata collection (page count, document properties)

**Example Use**:
```python
from ingestion.ingestion import ingest_pdf

blocks, metadata = ingest_pdf("data/raw/contract.pdf")
print(f"Extracted {len(blocks)} blocks from {metadata['total_pages']} pages")
```

### 2. **Layout Analysis Module** (`layout_analysis/`)
Performs structural classification of document content.

**Key Classes**:
- `LayoutAnalyzer`: Analyzes document layout and builds clause graphs

**Labels**:
- `heading`: Section titles, numbered sections, all-caps text
- `paragraph`: Body text paragraphs
- `list_item`: Bulleted or numbered list items
- `table`: Tabular content
- `caption`: Captions or figure titles
- `other`: Unclassified content

**Features**:
- Two-mode operation: Fine-tuned LayoutLMv3 (if available) or heuristic fallback
- Regex and pattern-based classification for reliability
- Clause graph construction from labeled blocks
- Token-to-block mapping for precise localization

**Example Use**:
```python
from layout_analysis.layout_structure import LayoutAnalyzer

analyzer = LayoutAnalyzer()
result = analyzer.analyze(blocks, page_image)
clause_graph = result["clause_graph"]
predictions = result["predictions"]
```

### 3. **Schema Module** (`schema/`)
Manages field definitions and extraction schemas.

**Key Classes**:
- `SchemaManager`: Loads and manages schemas from JSON files
- Schema utility function: `load_schema(form_name)`

**Schema Structure** (JSON):
```json
{
  "form_name": "NDA_Form",
  "version": "1.0",
  "description": "Non-Disclosure Agreement extraction schema",
  "fields": {
    "field_name": {
      "type": "date|string|number|boolean",
      "description": "Field description",
      "section": "Document section hint",
      "required": true|false,
      "examples": ["example1", "example2"],
      "keywords": ["keyword1", "keyword2"],
      "patterns": ["regex_pattern"]
    }
  }
}
```

**Default Schemas**:
- `NDA_Form`: Pre-configured schema for NDAs with fields like:
  - `effective_date`: Agreement effective date
  - `termination_notice`: Notice period for termination
  - `governing_law`: Jurisdiction and governing law

### 4. **Extraction Module** (`extraction/`)
Performs field-level extraction using LLMs.

**Key Classes**:
- `LLMExtractor`: Handles LLM-based field extraction with structured output
- `FormFiller`: Populates form instances using clause graphs and schemas
- `GeminiDirectExtractor`: Direct extraction using Google Gemini vision
- `LlamaDirectExtractor`: Direct extraction using Llama 3.2 vision (Ollama)

**Extraction Strategies**:
1. **Regex-Based**: Fast, reliable for structured fields (dates, phone numbers)
2. **Keyword-Based**: Context matching using field keywords
3. **LLM-Based**: Flexible extraction for complex fields (descriptions, open-ended text)

### 5. **Validation Module** (`utils/`)
Validates extracted fields and handles recovery.

### 6. **Orchestration Module** (`orchestration/`)
Coordinates the entire pipeline using LangGraph.

### 7. **Evaluation Module** (`evaluation/`)
Metrics and benchmarking for extraction quality.

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- pip or conda
- For OCR: Tesseract-OCR system package
- For Ollama support: Ollama installed and running (optional)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd multiagent-form-schema-etl
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n form-etl python=3.9
conda activate form-etl
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Spacy Model (Optional)
```bash
python -m spacy download en_core_web_sm
```

### Step 5: Configure Models
Edit `.env` file (create if not exists):
```env
# Gemini API (optional, for vision extraction)
GEMINI_API_KEY=your-api-key-here
```

### Step 6: Prepare Data
```bash
mkdir -p data/raw data/outputs data/schemas data/test
# Place PDF files in data/raw/
```

### Step 7: Verify Installation
```bash
python -c "from ingestion.ingestion import ingest_pdf; print('Installation successful')"
```

---

## Configuration

Configuration is managed through `config/config.py` using dataclasses.

### Model Configuration (`ModelConfig`)
- `gemini_model`: Gemini model name (default: "gemini-2.0-flash")
- `layout_model`: Path to LayoutLMv3 checkpoint or "fallback"
- `llm_model`: LLM model spec: "ollama/model", "hf-model", etc. (default: "ollama/llama3.2")
- `llm_temperature`: Sampling temperature (default: 0.1)
- `llm_max_tokens`: Max output tokens (default: 256)
- `device`: "auto", "cpu", "cuda" (default: "auto")

### Programmatic Configuration
```python
from config.config import get_config, update_config

cfg = get_config()
update_config(enable_validation=False, verbose=True)
```

---

## Usage

### Quick Start
```bash
# Process a single PDF
python main.py --input data/raw/sample_contract.pdf

# Use Gemini vision (requires API key)
python main.py --input data/raw/contract.pdf --use-gemini

# Use Llama 3.2 vision (requires Ollama)
python main.py --input data/raw/contract.pdf --use-llama --form-name NDA_Form
```

### Programmatic Usage
```python
from pathlib import Path
import json
from ingestion.ingestion import ingest_pdf
from orchestration.orchestrator import get_orchestrator

pdf_path = Path("data/raw/contract.pdf")
blocks, metadata = ingest_pdf(str(pdf_path))

orchestrator = get_orchestrator()
output = orchestrator.process(blocks, pdf_path, "NDA_Form")

output_path = Path("data/outputs") / f"{pdf_path.stem}_extracted.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)
```

---

## Project Structure
```
multiagent-form-schema-etl/
├── main.py                          # Entry point
├── requirements.txt                 # Dependencies
├── config/                          # Configuration
├── data/                            # Data directory
│   ├── raw/                         # Input PDFs
│   ├── outputs/                     # Extraction results
│   ├── schemas/                     # Schema definitions
│   └── intermediate/                # Pipeline intermediate results
├── ingestion/                       # PDF text/OCR extraction
├── layout_analysis/                 # LayoutLMv3-based classification
├── schema/                          # Schema management
├── extraction/                      # LLM-based field extraction
├── orchestration/                   # Pipeline orchestration
├── utils/                           # Utilities
├── evaluation/                      # Metrics and benchmarking
├── finetune/                        # LayoutLMv3 fine-tuning
└── models/                          # Model checkpoints
```

---

## Supported Models

### Layout Models
| Model | Location | Status |
|-------|----------|--------|
| **LayoutLMv3 Fine-tuned** | `models/layoutlmv3-nda/` | ✅ Recommended |
| **Fallback (Heuristic)** | Built-in | ✅ Always available |

### LLM Models
| Provider | Model | Config | Status |
|----------|-------|--------|--------|
| **Ollama (Local)** | `ollama/llama3.2` | Local, private | ✅ Recommended |
| **Google Gemini** | `gemini-2.0-flash` | Cloud, reliable | ✅ Works well |

---

## Development

### Setup Development Environment
```bash
pip install -r requirements.txt
pip install pytest pytest-cov black flake8
```

### Running Tests
```bash
pytest
pytest --cov=. --cov-report=html
```

### Code Formatting
```bash
black . --line-length 100
flake8 . --max-line-length 100
```

### Fine-tuning LayoutLMv3
```bash
python finetune/train.py --generate --n_samples 300 --epochs 5
```

---

## Performance

### Typical Processing Times
| Stage | Time |
|-------|------|
| Ingestion | 0.5-2s |
| Layout Analysis | 2-5s |
| Extraction | 3-10s |
| Validation | 0.5-1s |
| **Total** | **7-20s** |

### Resource Requirements
- **Minimum**: CPU-only, 4GB RAM
- **Recommended**: GPU (CUDA), 8GB RAM
- **Optimal**: GPU with 16GB+ VRAM

### Throughput
- **Local (CPU)**: 3-5 documents/minute
- **Local (GPU)**: 10-20 documents/minute
- **Batch with Gemini**: 20-30 documents/minute

---

## Troubleshooting

### OCR Not Working
```
Solution: Install Tesseract-OCR system package
  - Ubuntu: sudo apt-get install tesseract-ocr
  - macOS: brew install tesseract
```

### Ollama Connection Issues
```
Solution:
  1. Ensure Ollama is running: ollama serve
  2. Check model is available: ollama list
  3. Verify config: llm_model = "ollama/llama3.2"
```

### Extraction Returns None
```
Solution:
  1. Enable verbose logging for context
  2. Verify schema keywords/patterns
  3. Check LLM max_tokens
  4. Enable recovery: enable_recovery = True
```

---

## License

MIT License - Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction.

---

## Support & Contributing

### Issues & Feature Requests
Please open GitHub issues for bugs or feature requests.

### Contributing Guidelines
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

**Questions?** Check the examples directory or open an issue on GitHub.