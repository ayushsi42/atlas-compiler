"""
Configuration management for Atlas.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LifterConfig:
    """Configuration for the Lifter layer."""
    cache_enabled: bool = True
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 86400  # 24 hours


@dataclass
class NeuralConfig:
    """Configuration for the Neural Optimizer layer."""
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_retries: int = 3
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class VerifierConfig:
    """Configuration for the Tribunal Verifier layer."""
    timeout_ms: int = 30000  # 30 seconds
    loop_unroll_depth: int = 10
    bit_width: int = 32


@dataclass
class ExecutorConfig:
    """Configuration for the Executor layer."""
    opt_level: int = 3  # LLVM optimization level (0-3)
    enable_fast_math: bool = False


@dataclass
class CEGARConfig:
    """Configuration for the CEGAR feedback loop."""
    max_iterations: int = 3
    safe_fallback: bool = True


@dataclass
class AOTGPTConfig:
    """Main configuration container."""
    lifter: LifterConfig = field(default_factory=LifterConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    cegar: CEGARConfig = field(default_factory=CEGARConfig)
    
    # Global settings
    verbose: bool = False
    debug: bool = False


# Global config instance
_config: Optional[AOTGPTConfig] = None


def get_config() -> AOTGPTConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AOTGPTConfig()
    return _config


def configure(**kwargs) -> AOTGPTConfig:
    """Update the global configuration."""
    global _config
    config = get_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Check nested configs
            for nested_name in ['lifter', 'neural', 'verifier', 'executor', 'cegar']:
                nested = getattr(config, nested_name)
                if hasattr(nested, key):
                    setattr(nested, key, value)
                    break
    
    return config
