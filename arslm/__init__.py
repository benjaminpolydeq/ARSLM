"""
ARSLM — Adaptive Recurrent State Language Model

A lightweight from-scratch hybrid language model combining
adaptive recurrent networks and attention mechanisms.
"""

from arslm.config import ARSLMConfig
from arslm.model import ARSLM

__version__ = "0.2.0"
__all__ = ["ARSLM", "ARSLMConfig", "__version__"]
