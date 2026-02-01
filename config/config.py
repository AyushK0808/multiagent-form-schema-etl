"""
Configuration management for the contract extraction system.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not available, use os.getenv only

@dataclass
class ModelConfig:
    """Configuration for ML models."""
    gemini_model: str = "gemini-2.0-flash"  # Use flash for speed and cost-efficiency
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    layout_model: str = "microsoft/layoutlmv3-base"
    llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 256
    device: str = "auto"  # "auto", "cpu", "cuda"
    
@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    ocr_threshold: int = 50  # Min text volume before OCR
    max_page_size: int = 5000  # Max image dimension
    confidence_threshold: float = 0.7  # Min confidence for predictions
    
@dataclass
class PathConfig:
    """Configuration for file paths."""
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    raw_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "raw")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "outputs")
    schema_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "schemas")
    test_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "test")
    
    def __post_init__(self):
        """Ensure directories exist."""
        for dir_path in [self.raw_dir, self.output_dir, self.schema_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class Config:
    """Main configuration object."""
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Feature flags
    enable_validation: bool = True
    enable_recovery: bool = True
    verbose: bool = False

# Global config instance
config = Config()

def get_config() -> Config:
    """Get global configuration instance."""
    return config

def update_config(**kwargs):
    """Update configuration from keyword arguments."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)