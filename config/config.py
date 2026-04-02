"""
Configuration management for the agentic ETL fabric.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, List

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

    # LoRA adapter root  — each group lives at <adapter_root>/group_{1,2,3}/layoutlmv3/
    # Set to "" to disable adapter routing and fall back to legacy checkpoint.
    adapter_root: str = "models/adapters"

    # Text LLM backend
    llm_model:       str   = "google/flan-t5-base"
    llm_temperature: float = 0.1
    llm_max_tokens:  int   = 256

    # Donut OCR-free extractor
    donut_model: str = "naver-clova-ix/donut-base-finetuned-docvqa"
    schema_recognition_donut_model: str = "models/schema_recognition/donut"
    schema_recognition_donut_fallback_model: str = "naver-clova-ix/donut-base-finetuned-docvqa"

    # LoRA Donut adapter root — each group at <donut_adapter_root>/group_{1,2,3}/donut/
    donut_adapter_root: str = "models/adapters"

    # Device for local models
    device: str = "auto"

    vision_model_preference: List[str] = field(default_factory=lambda: [
        "moondream",
        "llava:7b",
        "minicpm-v",
    ])


@dataclass
class GroqConfig:
    """Configuration for the Groq-hosted LLM (schema-resolution agent)."""
    api_key:  str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    small_model: str = "llama-3.1-8b-instant"
    synthesis_model: str = "llama-3.3-70b-versatile"
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    temperature: float = 0.0
    max_tokens:  int   = 1024


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    ocr_threshold:        int   = 50
    max_page_size:        int   = 5000
    confidence_threshold: float = 0.70
    schema_sim_threshold: float = 0.70
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

    # LoRA adapter checkpoints root (written by train_lora.py)
    adapter_root:    Path = field(default_factory=lambda: Path(__file__).parent.parent / "models" / "adapters")

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
    enable_schema_agent:         bool = True
    enable_schema_recognition:   bool = True
    enable_db_population:        bool = True
    enable_parallel_extraction:  bool = True
    # When True, LayoutAnalyzer uses LoRA adapters if they exist at paths.adapter_root
    enable_lora_adapters:        bool = True
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