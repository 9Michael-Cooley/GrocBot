"""grokbot — serving and agent runtime for the Grok model family.

Extracted from //research/serving. See NOTICE before redistributing.
"""

from .config import Config
from .errors import (
    ContextLengthExceeded,
    GrokBotError,
    RateLimited,
    SafetyBlocked,
    ToolError,
    WeightsNotFound,
)
from .inference import Completion, Engine, FinishReason, GenerationConfig, StreamChunk
from .model import TransformerConfig
from .tokenizer import Tokenizer, render_chat

__version__ = "0.4.2"

__all__ = [
    "Config",
    "Engine",
    "GenerationConfig",
    "Completion",
    "StreamChunk",
    "FinishReason",
    "TransformerConfig",
    "Tokenizer",
    "render_chat",
    "GrokBotError",
    "ContextLengthExceeded",
    "WeightsNotFound",
    "SafetyBlocked",
    "ToolError",
    "RateLimited",
    "__version__",
]
