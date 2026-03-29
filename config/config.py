"""
Configuration management for the agentic ETL fabric.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for ML models."""
    # Layout / token classification
    layout_model: str = "models/layoutlmv3-nda/checkpoint_best"
    schema_recognition_layout_model: str = "models/schema_recognition/layoutlmv3"
    schema_recognition_layout_fallback_model: str = "microsoft/layoutlmv3-base"

    # Text LLM backend  ("ollama/<model>" or a HuggingFace model ID)
    llm_model:       str   = "ollama/llama3.2"
    llm_temperature: float = 0.1
    llm_max_tokens:  int   = 256

    # Donut OCR-free extractor
    donut_model: str = "naver-clova-ix/donut-base-finetuned-docvqa"
    schema_recognition_donut_model: str = "models/schema_recognition/donut"
    schema_recognition_donut_fallback_model: str = "naver-clova-ix/donut-base-finetuned-docvqa"

    # Device for local models
    device: str = "auto"   # "auto" | "cpu" | "cuda"

    # Ordered preference for local vision models used by Ollama extractor.
    vision_model_preference: List[str] = field(default_factory=lambda: [
        "moondream",
        "llama3.2-vision",
        "llava:7b",
        "minicpm-v",
    ])


@dataclass
class GroqConfig:
    """Configuration for the Groq-hosted LLM (schema-resolution agent)."""
    api_key:  str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model:    str = "llama-3.3-70b-versatile"
    # Fallback model for lower quota / rate-limit situations
    fallback_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.0
    max_tokens:  int   = 1024


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    ocr_threshold:        int   = 50    # min text chars before OCR kicks in
    max_page_size:        int   = 5000  # max image dimension in pixels
    confidence_threshold: float = 0.70  # min field confidence for acceptance
    # Minimum cosine similarity for schema registry hit
    schema_sim_threshold: float = 0.70
    # Run Donut + LayoutLM in parallel (False = LayoutLM only, faster)
    enable_parallel_extraction: bool = True


@dataclass
class PathConfig:
    """Configuration for file paths."""
    project_root:    Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir:        Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    raw_dir:         Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "raw")
    output_dir:      Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "outputs")
    schema_dir:      Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "schemas")
    test_dir:        Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "test")
    registry_dir:    Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "schema_registry")
    db_path:         Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "etl.db")

    def __post_init__(self):
        for p in [self.raw_dir, self.output_dir, self.schema_dir,
                  self.test_dir, self.registry_dir]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Root configuration object."""
    model:      ModelConfig      = field(default_factory=ModelConfig)
    groq:       GroqConfig       = field(default_factory=GroqConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    paths:      PathConfig       = field(default_factory=PathConfig)

    # Feature flags
    enable_validation:           bool = True
    enable_recovery:             bool = True
    enable_schema_agent:         bool = True   # Groq schema resolution
    enable_schema_recognition:   bool = True   # fine-tuned LayoutLMv3 / Donut schema classifier
    enable_db_population:        bool = True   # write to SQLite
    enable_parallel_extraction:  bool = True   # Donut + LayoutLM in parallel
    verbose:                     bool = False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config


def update_config(**kwargs):
    global _config
    cfg = get_config()
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
