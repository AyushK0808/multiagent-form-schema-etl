# Multi Agent Form Schema ETL Pipeline

A production-grade modular system for extracting structured data from contract PDFs using a multi-layer architecture with LLM-based extraction.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Contract PDF                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Ingestion & Preprocessing                         │
│  - PDF text extraction (pdfplumber)                         │
│  - OCR fallback for scanned docs (pytesseract)              │
│  - Layout coordinate preservation                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Layout Detection & Clause Hierarchy               │
│  - Region classification (LayoutLLMv3 simulation)           │
│  - Heading/paragraph/table detection                         │
│  - Clause graph construction                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Orchestration (LangGraph)                         │
│  - State management                                          │
│  - Workflow coordination                                     │
│  - Retry and error handling                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Schema Management                                  │
│  - Contract schema definitions                               │
│  - Field specifications and constraints                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: LLM-based Extraction                              │
│  - Field-level extraction with LLaMA                        │
│  - Clause-scoped context                                     │
│  - JSON-constrained output                                   │
│  - Parallel execution support                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Output: Structured JSON Data                       │
└─────────────────────────────────────────────────────────────┘
```

## Features

### 1. **Ingestion and Preprocessing Layer**
- **Text Extraction**: Uses `pdfplumber` for high-quality text extraction with layout preservation
- **OCR Fallback**: Automatically detects low-quality/scanned PDFs and applies OCR using `pytesseract`
- **Layout Coordinates**: Preserves bounding box information for all extracted text regions
- **Unified Representation**: Outputs a consistent `ProcessedDocument` structure

### 2. **Layout Detection Layer**
- **Region Classification**: Identifies headings, paragraphs, tables, and lists
- **Clause Hierarchy**: Builds a logical tree structure of contract sections
- **Clause Graph**: Creates a navigable graph for context-aware extraction
- **Simulated LayoutLLMv3**: Demo uses heuristics; ready for actual model integration

### 3. **Orchestration Layer (LangGraph)**
- **State Management**: Maintains shared state across all processing steps
- **Conditional Execution**: Smart routing based on processing results
- **Retry Logic**: Automatic retry with configurable attempts
- **Error Recovery**: Graceful failure handling with detailed error reporting
- **Pipeline Visualization**: Clear workflow definition

### 4. **Schema Management Layer**
- **Hardcoded Demo Schema**: Includes common contract fields:
  - Contract metadata (title, date, type)
  - Parties (name, role, address)
  - Financial terms (value, payment terms, currency)
  - Key dates (start, end, renewal)
  - Obligations (description, responsible party)
  - Termination clauses (conditions, notice period)
- **Extensible Design**: Easy to add custom schemas for different contract types

### 5. **LLM-based Extraction Layer**
- **Schema-Guided Prompts**: Each field extracted with specific instructions
- **Clause-Scoped Context**: Uses only relevant sections for each extraction
- **JSON Constraints**: Enforces structured output format
- **Independent Field Extraction**: Allows parallel processing
- **Hallucination Reduction**: Focused context windows minimize errors
- **Mock LLaMA Integration**: Ready for actual LLaMA API connection

## Installation

### Prerequisites
```bash
# System dependencies for OCR
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils

# For macOS
brew install tesseract poppler
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Optional: For Production
```bash
# If using actual LayoutLLMv3
pip install transformers torch

# If using local LLaMA
pip install llama-cpp-python

# If using cloud-based LLM APIs
pip install openai anthropic
```

## Usage

### Basic Usage

```python
from contract_processing_system import OrchestrationLayer

# Initialize the orchestrator
orchestrator = OrchestrationLayer(max_retries=3)

# Process a contract
result = orchestrator.process_contract("path/to/contract.pdf")

# Access results
print(f"Status: {result['status']}")
print(f"Extracted Data: {result['extracted_data']}")
```

### Advanced Usage

```python
# Custom ingestion settings
from contract_processing_system import DocumentIngestionLayer

ingestion = DocumentIngestionLayer(ocr_threshold=100)
doc = ingestion.process_pdf("contract.pdf")

# Access layout analysis
from contract_processing_system import LayoutDetectionLayer

layout = LayoutDetectionLayer()
clause_graph = layout.analyze_layout(doc)

# Visualize clause hierarchy
import json
print(json.dumps(clause_graph.to_dict(), indent=2))

# Custom schema
from contract_processing_system import SchemaManagementLayer

schema_layer = SchemaManagementLayer()
schema = schema_layer.get_contract_schema()

# Modify schema for specific contract type
schema['custom_field'] = {
    "fields": [
        {
            "name": "special_clause",
            "type": "string",
            "description": "Custom clause description",
            "required": False
        }
    ]
}
```

## System Components

### ProcessedDocument
Unified document representation after preprocessing:
```python
@dataclass
class ProcessedDocument:
    filename: str
    num_pages: int
    regions: List[DocumentRegion]
    raw_text: str
    metadata: Dict[str, Any]
    ocr_applied: bool
```

### DocumentRegion
Represents a region with coordinates:
```python
@dataclass
class DocumentRegion:
    region_type: str  # 'heading', 'paragraph', 'table', 'list'
    text: str
    page_number: int
    bbox: Tuple[float, float, float, float]
    confidence: float
```

### ClauseGraph
Hierarchical structure of contract clauses:
```python
class ClauseGraph:
    root: ClauseNode
    nodes: Dict[str, ClauseNode]
    flat_list: List[ClauseNode]
    
    def get_clause_context(clause_id, context_window=2) -> str
    def to_dict() -> Dict
```

### ProcessingState
Shared state for LangGraph workflow:
```python
class ProcessingState(TypedDict):
    pdf_path: str
    document: Optional[ProcessedDocument]
    clause_graph: Optional[ClauseGraph]
    schema: Optional[Dict[str, Any]]
    extracted_data: Dict[str, Any]
    errors: List[str]
    retry_count: int
    status: str
```

## Workflow Diagram

```
START
  │
  ├─► Ingest Document (Layer 1)
  │      │
  │      ├─► Extract text with pdfplumber
  │      ├─► Check quality
  │      └─► Apply OCR if needed
  │
  ├─► Analyze Layout (Layer 2)
  │      │
  │      ├─► Classify regions
  │      └─► Build clause hierarchy
  │
  ├─► Load Schema (Layer 4)
  │
  ├─► Extract Fields (Layer 5)
  │      │
  │      ├─► For each schema section:
  │      │   ├─► Build context
  │      │   ├─► Create prompt
  │      │   └─► Call LLM
  │      │
  │      └─► Collect results
  │
  ├─► Validate
  │      │
  │      ├─► Check required fields
  │      ├─► Validate formats
  │      │
  │      └─► Retry if needed ──┐
  │                            │
  └─► END ◄──────────────────┘
```

## Production Integration

### Integrating Real LayoutLLMv3

```python
from transformers import AutoModel, AutoProcessor

class LayoutDetectionLayer:
    def __init__(self):
        self.model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")
        self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")
    
    def analyze_layout(self, doc: ProcessedDocument) -> ClauseGraph:
        # Convert document to model input
        # Run inference
        # Parse model output
        pass
```

### Integrating Real LLaMA

```python
import openai  # or use llama-cpp-python for local

class LLMExtractionLayer:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def _extract_section(self, section_name, section_schema, clause_graph):
        context = self._build_context(section_name, clause_graph)
        prompt = self._build_extraction_prompt(section_name, section_schema, context)
        
        response = self.client.chat.completions.create(
            model="llama-3.1-70b",
            messages=[
                {"role": "system", "content": "Extract contract information as JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
```

### Custom Schema Example

```python
# Create custom schema for employment contracts
employment_schema = {
    "employee_info": {
        "fields": [
            {"name": "employee_name", "type": "string", "required": True},
            {"name": "position", "type": "string", "required": True},
            {"name": "start_date", "type": "date", "required": True},
            {"name": "salary", "type": "number", "required": True}
        ]
    },
    "benefits": {
        "fields": [
            {"name": "health_insurance", "type": "boolean", "required": False},
            {"name": "vacation_days", "type": "number", "required": False},
            {"name": "retirement_plan", "type": "string", "required": False}
        ]
    }
}
```

## Performance Optimization

### Parallel Extraction
```python
from concurrent.futures import ThreadPoolExecutor

def extract_fields_parallel(self, clause_graph, schema):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        
        for section_name, section_schema in schema.items():
            future = executor.submit(
                self._extract_section,
                section_name,
                section_schema,
                clause_graph
            )
            futures[section_name] = future
        
        results = {}
        for section_name, future in futures.items():
            results[section_name] = future.result()
        
        return results
```

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_clause_context(self, clause_id: str) -> str:
    # Cached context retrieval
    pass
```

## Error Handling

The system includes comprehensive error handling:
- **Ingestion errors**: Fallback to OCR
- **Layout detection errors**: Use simple heuristics
- **Extraction errors**: Retry with modified prompts
- **Validation errors**: Report missing/invalid fields

```python
try:
    result = orchestrator.process_contract("contract.pdf")
except Exception as e:
    logger.error(f"Processing failed: {e}")
    # Handle error appropriately
```

## Testing

```python
import pytest
from contract_processing_system import *

def test_ingestion():
    ingestion = DocumentIngestionLayer()
    doc = ingestion.process_pdf("test_contract.pdf")
    assert doc.num_pages > 0
    assert len(doc.regions) > 0

def test_layout_detection():
    # Create mock document
    doc = ProcessedDocument(...)
    layout = LayoutDetectionLayer()
    graph = layout.analyze_layout(doc)
    assert len(graph.nodes) > 1

def test_orchestration():
    orchestrator = OrchestrationLayer()
    result = orchestrator.process_contract("test_contract.pdf")
    assert result['status'] in ['complete', 'failed']
```

## Limitations and Future Work

### Current Limitations
- LayoutLLMv3 is simulated with heuristics
- LLaMA extraction uses mock data
- Schema is hardcoded for demo
- No database persistence
- Limited error recovery strategies

### Planned Enhancements
1. Real LayoutLLMv3 integration
2. Production LLaMA API integration
3. Database-backed schema management
4. Advanced retry strategies
5. Streaming extraction for large documents
6. Multi-language support
7. Confidence scoring for extractions
8. Human-in-the-loop validation UI

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please submit PRs with:
- Unit tests
- Documentation updates
- Performance benchmarks

## Support

For issues and questions:
- GitHub Issues: [your-repo-url]
- Email: support@example.com
- Documentation: [docs-url]
