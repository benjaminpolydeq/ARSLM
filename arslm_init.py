"""
ARSLM - Adaptive Reasoning Semantic Language Model
===================================================

A lightweight AI engine for intelligent response generation designed for
businesses worldwide requiring conversational capabilities without the
complexity of large-scale cloud solutions.

Key Features:
    - Lightweight and efficient
    - Privacy-first design
    - Multi-language support
    - Easy deployment
    - Modular architecture

Example:
    >>> from arslm import ARSLM
    >>> model = ARSLM()
    >>> response = model.generate("What is AI?")
    >>> print(response)

Author: Benjamin Amaad Kama
Email: benjokama@hotmail.fr
License: MIT
"""

import sys
import warnings
from typing import List

# Version information
__version__ = "0.1.0"
__author__ = "Benjamin Amaad Kama"
__email__ = "benjokama@hotmail.fr"
__license__ = "MIT"
__url__ = "https://github.com/benjaminpolydeq/ARSLM"

# Version check
if sys.version_info < (3, 8):
    raise RuntimeError(
        f"ARSLM requires Python 3.8 or later. "
        f"You are using Python {sys.version_info.major}.{sys.version_info.minor}."
    )

# Import core components
try:
    from arslm.core.model import ARSLM, ARSLMConfig
    from arslm.core.attention import (
        MultiHeadAttention,
        SelfAttention,
        CrossAttention,
    )
    from arslm.core.recurrent import AdaptiveRNN, AdaptiveLSTM, AdaptiveGRU
    from arslm.core.adaptive import AdaptiveLayer, DynamicRouter
    
    from arslm.utils.tokenizer import ARSLMTokenizer
    from arslm.utils.config import Config, load_config, save_config
    from arslm.utils.preprocessing import TextPreprocessor
    
    from arslm.api.client import ARSLMClient
    from arslm.api.schemas import (
        ChatRequest,
        ChatResponse,
        GenerationRequest,
        GenerationResponse,
    )
    
except ImportError as e:
    warnings.warn(
        f"Some components could not be imported: {e}. "
        f"This might be normal during initial setup."
    )
    # Define minimal exports for installation
    ARSLM = None
    ARSLMConfig = None
    ARSLMTokenizer = None
    ARSLMClient = None

# Define public API
__all__: List[str] = [
    # Core components
    "ARSLM",
    "ARSLMConfig",
    
    # Attention mechanisms
    "MultiHeadAttention",
    "SelfAttention",
    "CrossAttention",
    
    # Recurrent components
    "AdaptiveRNN",
    "AdaptiveLSTM",
    "AdaptiveGRU",
    
    # Adaptive components
    "AdaptiveLayer",
    "DynamicRouter",
    
    # Utilities
    "ARSLMTokenizer",
    "Config",
    "load_config",
    "save_config",
    "TextPreprocessor",
    
    # API
    "ARSLMClient",
    "ChatRequest",
    "ChatResponse",
    "GenerationRequest",
    "GenerationResponse",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__url__",
]

# Package-level configuration
def get_version() -> str:
    """Get the current version of ARSLM.
    
    Returns:
        str: Version string (e.g., "0.1.0")
    """
    return __version__

def get_info() -> dict:
    """Get package information.
    
    Returns:
        dict: Dictionary containing package metadata
    """
    return {
        "name": "arslm",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "url": __url__,
        "description": "Adaptive Reasoning Semantic Language Model",
    }

# Display welcome message on import (optional)
def _display_welcome():
    """Display welcome message when package is imported."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console()
        message = (
            f"[bold cyan]ARSLM v{__version__}[/bold cyan]\n"
            f"Adaptive Reasoning Semantic Language Model\n\n"
            f"[dim]Lightweight AI for everyone[/dim]"
        )
        
        # Only show on first import
        if not hasattr(_display_welcome, '_shown'):
            console.print(Panel(message, border_style="cyan"))
            _display_welcome._shown = True
    except ImportError:
        # Rich not available, skip welcome message
        pass

# Uncomment to enable welcome message
# _display_welcome()