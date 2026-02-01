# Form Schema ETL Pipeline

A layout-aware, schema-guided approach to structured information extraction from documents using multi-modal transformers and LLM-based micro-decoding.

## Research Contributions

1. **Layout-Aware Structural Modeling**: Token-level classification using LayoutLMv3 for spatial context
2. **Clause Graph Construction**: Hierarchical document representation preserving structural relationships
3. **Schema-Guided LLM Extraction**: Constrained micro-decoding for field-level extraction
4. **Orchestrated Pipeline**: Stateful workflow using LangGraph for reproducible processing

## Architecture

```
PDF Input → Ingestion → Layout Analysis → Clause Graph → Schema-Guided Extraction → Validated Output
```

### Components

- **Ingestion**: PDF text/OCR extraction with bounding boxes
- **Layout Analysis**: LayoutLMv3-based structural classification
- **Clause Graph**: Hierarchical document representation
- **Schema Management**: Flexible field definitions with type constraints
- **Extraction**: LLM-based field extraction with JSON constraints
- **Validation**: Rule-based field validation and error recovery
- **Orchestration**: LangGraph-based stateful pipeline

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

```bash
python main.py --input data/raw/sample_contract.pdf --output data/outputs/
```

## Evaluation

```bash
python -m evaluation.evaluator --test-set data/test/
```

## Project Structure

```
contract_extraction/
├── config/              # Configuration files
├── data/                # Data directories
├── extraction/          # LLM-based field extraction
├── ingestion/           # PDF processing
├── layout_analysis/     # Layout modeling
├── orchestration/       # Pipeline orchestration
├── schema/              # Schema definitions
├── utils/               # Utility functions
├── evaluation/          # Evaluation metrics
└── tests/               # Unit tests
```

## Key Features

- **Multi-modal Processing**: Combines text, layout, and spatial features
- **Deterministic Clause Graphs**: Reproducible document structure
- **Schema Flexibility**: Easy addition of new document types
- **Validation Framework**: Automated field validation with recovery
- **Baseline Comparison**: Evaluation against naive extraction

## Thesis Validation

This implementation validates:
- ✅ Structure-first processing improves extraction accuracy
- ✅ Clause graphs reduce hallucination via context grounding
- ✅ Schema-guided micro-decoding outperforms full-document LLM extraction
- ✅ Layout awareness improves field localization

## License

MIT