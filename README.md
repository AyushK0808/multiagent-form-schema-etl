# Multi-Agent Form Schema ETL Pipeline

A layout-aware, schema-guided document extraction system using multi-modal transformers, LoRA-adapted vision-language models, and a reflexive multi-agent orchestration fabric. Designed for production-grade extraction from complex documents such as contracts, NDAs, invoices, and forms.

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
9. [Fine-tuning](#fine-tuning)
10. [Supported Models](#supported-models)
11. [Development](#development)
12. [License](#license)

---

## Overview

This system implements an end-to-end pipeline for extracting structured information from unstructured documents. Rather than naive full-document LLM processing, it uses a **multi-stage, research-backed approach**:

1. **Document Ingestion** — Extract text and bounding boxes from PDFs with OCR fallback via Tesseract
2. **Layout Analysis** — Token-level structural classification using LayoutLMv3, with LoRA adapter routing per document group
3. **Parallel Extraction** — Donut (OCR-free, pixel-level) and LayoutLM/FormFiller pipelines run concurrently
4. **Reflexive Policy Fusion** — Confidence-weighted field-level fusion with consistency validation and jargon normalisation
5. **Groq Repair Pass** — LLM-assisted field repair and normalisation before schema resolution
6. **Schema Resolution** — Groq agent matches, maps, or synthesises schemas via a semantic embedding registry
7. **Validation & Recovery** — Rule-based validation with default-value and retry recovery strategies
8. **Database Population** — Schema-driven SQLite storage with dynamic table creation

**Research benefits**:
- Reduces LLM hallucination by grounding extraction in layout-classified clause contexts
- Improves accuracy through parallel extraction and policy-layer fusion
- Enables cost-efficient micro-decoding (field-level) instead of full-document prompting
- Supports reproducibility through deterministic preprocessing and curriculum-ordered fine-tuning

---

## Key Features

### Multi-Modal Parallel Extraction
- **Donut** (`naver-clova-ix/donut-base-finetuned-docvqa`) — OCR-free extraction directly from pixel data via DocVQA-style question answering
- **LayoutLMv3** — Spatially-anchored token classification with bounding-box-aware preprocessing
- Both extractors run concurrently via `ThreadPoolExecutor`; results are fused by a `ReflexivePolicyLayer`

### LoRA Adapter Architecture
- Three adapter groups trained on distinct dataset curricula (layout-primitive, structural, reasoning-heavy)
- Adapters are hot-swapped at inference via `AdapterRouter` using PEFT's `set_adapter` / `load_adapter` APIs
- Full model checkpoints (~500 MB) replaced by per-group LoRA adapters (~8–15 MB each)
- Group routing is resolved once per document from the recognised schema name via `group_for_schema()`

### Reflexive Policy Fusion
- Per-field source selection based on confidence scores, field coverage, and spatial preference
- Spatially-anchored fields (dates, amounts, signatures) prefer LayoutLM; semantic fields prefer Donut
- Cross-field consistency rules: date ordering, party name uniqueness, required field presence
- Jargon normalisation strips legalese patterns (`as defined herein`, `(the "Company")`) post-fusion

### Schema Management
- **Schema registry** — Sentence-transformer embeddings (MiniLM-L6-v2) over field descriptions enable semantic similarity search
- **Schema agent** — Groq-hosted LLM normalises, maps, and synthesises schemas; falls back to rule-based construction
- **SQLite backend** — Unified storage for both schema definitions and extracted records, with `UPSERT` semantics and dynamic `ALTER TABLE` for new fields
- Schemas are registered automatically when no close match exists in the registry

### Three-Pass Extraction
1. **Text LLM pass** — Full document text + schema fed to Flan-T5 or Ollama for initial JSON extraction
2. **Vision LLM fallback** — Null fields after pass 1 are sent to a Groq vision model (Llama 4 Scout) for image-based recovery
3. **Sentinel pass** — Remaining null fields are marked `"NaN"` for downstream handling

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      PDF Input                          │
└──────────────────────────┬──────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   1. PDF Ingestion      │
              │   PyMuPDF + Tesseract   │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   2. Layout Analysis    │
              │   LayoutLMv3 + LoRA     │
              │   adapter routing       │
              └────────────┬────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
 ┌──────────▼──────────┐    ┌─────────────▼────────────┐
 │  3a. Donut          │    │  3b. LayoutLM + FormFiller│
 │  OCR-free, DocVQA   │    │  Token classification     │
 │  pixel → fields     │    │  + 3-pass LLM extraction  │
 └──────────┬──────────┘    └─────────────┬────────────┘
            │                             │
            └──────────────┬──────────────┘
                           │  ExtractionBundle
              ┌────────────▼────────────┐
              │  4. Reflexive Policy    │
              │  Fusion                 │
              │  Confidence weighting   │
              │  + consistency checks   │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  5. Groq Repair Pass    │
              │  Field normalisation    │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  6. Schema Resolution   │
              │  Registry match or      │
              │  synthesis via LLM      │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  7. Validation +        │
              │  Recovery               │
              └────────────┬────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
 ┌──────────▼──────────┐    ┌─────────────▼────────────┐
 │  8a. SQLite DB       │    │  8b. JSON Output          │
 │  Schema-driven       │    │  Fields + metadata        │
 │  dynamic tables      │    │  + pipeline stats         │
 └─────────────────────┘    └──────────────────────────┘
```

---

## System Components

### 1. Ingestion (`ingestion/`)

Handles PDF processing and text extraction.

**Key classes**: `DocumentBlock`, `PDFIngester`

- Native PDF text via PyMuPDF; OCR fallback (Tesseract) when text volume is below `ocr_threshold`
- Bounding box extraction for spatial layout features
- Outputs a list of `DocumentBlock` dicts with `page`, `text`, `bbox`, `confidence`, `source`

```python
from ingestion.ingestion import ingest_pdf
blocks, metadata = ingest_pdf("data/raw/contract.pdf")
```

### 2. Layout Analysis (`layout_analysis/`)

Structural classification of document tokens with backend selection:

1. **`AdapterLayoutAnalyzer`** — LoRA-adapted LayoutLMv3 (preferred when adapters exist)
2. **`FineTunedLayoutAnalyzer`** — Legacy monolithic fine-tuned checkpoint
3. **`HeuristicLayoutAnalyzer`** — Regex-based fallback, always available

**Labels**: `heading`, `paragraph`, `list_item`, `table`, `caption`, `other`

The `AdapterRouter` singleton manages base model loading, adapter hot-swapping, and LRU caching (max 3 groups in memory simultaneously).

```python
from layout_analysis.layout_structure import LayoutAnalyzer
analyzer = LayoutAnalyzer()
result = analyzer.analyze(blocks, page_image, group_name="group_3")
clause_graph = result["clause_graph"]
```

### 3. Parallel Extraction (`extraction/`)

**`ParallelExtractor`** launches Donut and LayoutLM/FormFiller concurrently and returns an `ExtractionBundle`.

- `DonutExtractor` — one VQA round-trip per schema field; confidence derived from mean max token probability
- `FormFiller` — 3-pass extraction (text LLM → vision LLM fallback → sentinel); forwards `adapter_group` to LayoutAnalyzer
- `ExtractionBundle` exposes per-extractor field coverage and mean confidence

**`ReflexivePolicyLayer`** fuses the bundle:

| Field type | Preferred source |
|---|---|
| Dates, amounts, spatial fields | LayoutLM |
| Party names, governing law, clauses | Donut |
| Tie (Δconfidence < 5%) | Higher-coverage extractor |

### 4. Schema Agent (`schema/schema_agent.py`)

Groq-powered agent with three operations:

- **Normalise** — clean values, fix OCR artefacts, standardise dates to ISO 8601
- **Map** — semantically align extracted fields to a candidate schema's field names
- **Synthesise** — generate a new schema definition from observed fields when no registry match is found

All operations use JSON-only prompts with one automatic retry on parse failure.

### 5. Schema Registry (`schema/schema_registry.py`)

- Encodes field descriptions as MiniLM-L6-v2 embeddings; computes cosine similarity against stored centroids
- Persists `.npy` embedding files and full schema JSON per `schema_id`
- Mirrors registrations to SQLite via `DatabaseManager.upsert_schema`

### 6. Database Manager (`database/db_manager.py`)

Schema-driven SQLite manager:

- `ensure_table` — creates or `ALTER TABLE`s to add new fields dynamically based on schema definition
- `insert_record` — inserts extracted fields with system columns (`_record_id`, `_schema_version`, `_ingested_at`, `_confidence_avg`)
- `upsert_schema` — stores schema definitions in `_schemas` table with versioning

### 7. Orchestrator (`orchestration/orchestrator.py`)

LangGraph `StateGraph` with 8 sequential nodes. The `ContractState` TypedDict carries all pipeline state including the resolved `adapter_group`.

Adapter group is resolved once before the graph runs via `_resolve_adapter_group`, which looks up `group_for_schema(form_name)` from `finetune/adapter_groups.py`.

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Tesseract-OCR system package (for OCR fallback)
- Groq API key (for schema agent and vision fallback)
- Ollama (optional, for local LLM)

### Steps

```bash
git clone <repository-url>
cd multiagent-form-schema-etl

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Optional: spaCy model
python -m spacy download en_core_web_sm
```

### Environment

Create a `.env` file:

```env
HF_TOKEN=hf_...          # Required for gated datasets (DocVQA, Kleister-NDA, etc.)
GROQ_API_KEY=gsk_...     # Required for schema agent and vision fallback
```

### Directory bootstrap

```bash
mkdir -p data/raw data/outputs data/schemas data/test data/schema_registry
```

---

## Configuration

Configuration lives in `config/config.py` as a hierarchy of frozen dataclasses, accessed via `get_config()`.

### Key settings

| Section | Key | Default | Description |
|---|---|---|---|
| `model` | `layout_model` | `models/layoutlmv3-nda/checkpoint_best` | Legacy layout checkpoint path |
| `model` | `adapter_root` | `models/adapters` | LoRA adapter root directory |
| `model` | `donut_model` | `naver-clova-ix/donut-base-finetuned-docvqa` | Donut checkpoint |
| `model` | `llm_model` | `google/flan-t5-base` | Text LLM for pass 1 extraction |
| `groq` | `vision_model` | `meta-llama/llama-4-scout-17b-16e-instruct` | Vision model for pass 2 |
| `groq` | `synthesis_model` | `llama-3.3-70b-versatile` | Model for schema synthesis |
| `processing` | `schema_sim_threshold` | `0.70` | Cosine similarity floor for registry match |
| `processing` | `confidence_threshold` | `0.70` | Confidence floor for policy fusion |

Feature flags: `enable_parallel_extraction`, `enable_schema_agent`, `enable_schema_recognition`, `enable_db_population`, `enable_lora_adapters`.

```python
from config.config import update_config
update_config(enable_schema_agent=False, verbose=True)
```

---

## Usage

### CLI

```bash
# Process a PDF with a named schema
python main.py --pdf data/raw/contract.pdf --form NDA_Form

# Use a stored schema ID
python main.py --pdf contract.pdf --schema-id <uuid>

# Offline mode (no Groq)
python main.py --pdf contract.pdf --form NDA_Form --no-schema-agent

# LayoutLM only (skip Donut)
python main.py --pdf contract.pdf --form NDA_Form --no-donut

# List all registered schemas
python main.py --list-schemas

# Query extracted records
python main.py --query-db NDA_Form
```

### Programmatic

```python
from pathlib import Path
from ingestion.ingestion import ingest_pdf
from orchestration.orchestrator import get_orchestrator
import fitz
from PIL import Image

pdf_path = Path("data/raw/contract.pdf")
blocks, metadata = ingest_pdf(str(pdf_path))

doc = fitz.open(str(pdf_path))
pix = doc.load_page(0).get_pixmap()
page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
doc.close()

state = {
    "blocks": blocks,
    "page_image": page_image,
    "pdf_metadata": metadata,
    "schema_recognition": {"form_name": "NDA_Form"},
    "schema": {},
    "clause_graph": {},
    "output": {},
    "errors": [],
    "warnings": [],
}

final_state = get_orchestrator().process(state)
print(final_state["output"]["fields"])
```

---

## Project Structure

```
multiagent-form-schema-etl/
├── main.py                        # CLI entry point
├── requirements.txt
├── config/
│   └── config.py                  # Dataclass config hierarchy
├── ingestion/
│   └── ingestion.py               # PyMuPDF + Tesseract OCR
├── layout_analysis/
│   ├── layout_structure.py        # LayoutAnalyzer (adapter / legacy / heuristic)
│   └── adapter_router.py          # LoRA adapter hot-swap manager
├── extraction/
│   ├── parallel_extractor.py      # ThreadPoolExecutor orchestrator + ExtractionBundle
│   ├── donut_extractor.py         # OCR-free DocVQA extraction
│   ├── form_filler.py             # 3-pass text+vision extraction
│   ├── llama_extractor.py         # Groq vision extractor (GroqVisionExtractor)
│   └── policy_layer.py            # ReflexivePolicyLayer fusion + normalisation
├── schema/
│   ├── schema.py                  # SchemaManager (SQLite-backed)
│   ├── schema_agent.py            # Groq normalise / map / synthesise agent
│   ├── schema_registry.py         # MiniLM embedding registry
│   └── schema_recognizer.py       # Auto schema recognition from page image
├── orchestration/
│   └── orchestrator.py            # LangGraph 8-node pipeline
├── database/
│   └── db_manager.py              # Schema-driven SQLite manager
├── finetune/
│   ├── train.py                   # Full fine-tune entry point
│   ├── train_lora.py              # LoRA adapter training entry point
│   ├── adapter_groups.py          # 3-group curriculum definition + SCHEMA_TO_GROUP
│   ├── data_loader.py             # Multi-dataset loader with normalised/augmented cache
│   ├── layoutlmv3_trainer.py      # LLRD + ForTokenClassification trainer
│   ├── lora_layoutlmv3_trainer.py # PEFT LoRA adapter trainer
│   ├── donut_trainer.py           # Donut seq2seq schema recognition trainer
│   ├── lora_donut_trainer.py      # PEFT LoRA Donut trainer
│   ├── normalizers.py             # Per-dataset normalisation to unified format
│   ├── metrics.py                 # CER, F1, label-containment assignment
│   ├── config.py                  # Label space, DatasetSpec, DATASET_SPECS
│   └── augmentation.py            # Albumentations image augmentation
├── evaluation/
│   └── evaluator.py               # ExtractionMetrics, baseline comparison, report
├── utils/
│   ├── form.py                    # FormInstance data structure
│   └── validation.py              # FieldValidator + ValidationRecoveryManager
└── data/
    ├── raw/                       # Input PDFs
    ├── outputs/                   # Extracted JSON results
    ├── schemas/                   # Schema JSON files
    ├── schema_registry/           # MiniLM embeddings + schema JSONs
    └── intermediate/              # Per-stage pipeline summaries
```

---

## Fine-tuning

### Dataset groups

Datasets are organised into three curriculum groups defined in `finetune/adapter_groups.py`:

| Group | Datasets | Focus |
|---|---|---|
| `group_1` | CORD, SROIE, SynthDog-EN | Layout-primitive / receipt-like |
| `group_2` | FUNSD, RVL-CDIP, DocLayNet, DocBank | Structural classification |
| `group_3` | DocVQA, Kleister-NDA, InfographicVQA | Reasoning-heavy / clause extraction |

> **Note**: Several datasets are gated on HuggingFace (DocVQA, Kleister-NDA, InfographicVQA). Accept their terms at `huggingface.co/datasets/<repo_id>` and set `HF_TOKEN` in `.env` before training.

### LoRA adapter training (recommended)

```bash
# Train all three groups (both LayoutLMv3 and Donut adapters)
python finetune/train_lora.py

# Train a single group
python finetune/train_lora.py --groups group_3

# LayoutLMv3 adapters only, quick smoke-test
python finetune/train_lora.py --model layoutlmv3 --max-train-samples 100 --no-augment
```

Adapters are saved to `models/adapters/group_{1,2,3}/{layoutlmv3,donut}/`. The pipeline automatically detects and routes to them at inference.

### Full fine-tune

```bash
# Both models, all datasets, curriculum ordering
python finetune/train.py --all-datasets --curriculum

# LayoutLMv3 only
python finetune/train.py --model layoutlmv3 --epochs 10 --batch-size 2
```

### Training features

- **LLRD** (layer-wise LR decay) for full LayoutLMv3 fine-tune — head gets `base_lr`, lower encoder layers decay by `llrd_factor` (default 0.9)
- **Cosine LR schedule** with 6% warmup ratio
- **bf16** on CUDA, gradient accumulation (steps=8)
- **Early stopping** patience=3 on macro-F1 (LayoutLMv3) or CER (Donut)
- **Normalised and augmented dataset caches** persist between runs to avoid redundant preprocessing
- **Albumentations augmentation**: rotation ±3°, brightness/contrast jitter, JPEG compression, Gaussian blur

---

## Supported Models

### Layout analysis

| Backend | Location | Notes |
|---|---|---|
| LoRA adapters | `models/adapters/group_*/layoutlmv3/` | Preferred; ~8–15 MB per group |
| Legacy checkpoint | `models/layoutlmv3-nda/checkpoint_best` | ~500 MB monolithic fine-tune |
| Heuristic | Built-in | Regex fallback, always available |

### LLM backends

| Role | Model | Provider |
|---|---|---|
| Pass 1 text extraction | `google/flan-t5-base` or `ollama/llama3.2` | HuggingFace / Ollama |
| Pass 2 vision fallback | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq |
| Schema normalise / map | `llama-3.1-8b-instant` | Groq |
| Schema synthesis | `llama-3.3-70b-versatile` | Groq |

---

## Performance

### Resource requirements

- **Minimum**: CPU only, 8 GB RAM
- **Recommended**: CUDA GPU, 16 GB RAM
- **LoRA adapters**: 3 × ~12 MB vs one ~500 MB monolithic checkpoint

---

## Troubleshooting

**Gated dataset 401 errors**
Set `HF_TOKEN` in `.env` and accept the dataset terms at `huggingface.co`.

**Groq API errors**
Set `GROQ_API_KEY` in `.env`. Use `--no-schema-agent` for offline operation.

**Tesseract not found**
```bash
# Ubuntu
sudo apt-get install tesseract-ocr
# macOS
brew install tesseract
```

**No schema found**
Register a schema before running the pipeline:
```python
from schema.schema import SchemaManager
SchemaManager().add_schema(your_schema_dict)
```

**Adapter not loading**
Train adapters first with `python finetune/train_lora.py`. Until then the pipeline silently falls back to the heuristic analyzer.

## License

MIT License — Copyright (c) 2024