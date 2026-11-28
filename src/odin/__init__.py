"""ODIN: Orchestrated Debugging and Intelligent Navigation.

An agentic self-debugging framework for automated code repair using
Adaptive Retrieval-Augmented Generation (RAG).
"""
from odin.backend import OdinBackend
from odin.config import OdinConfig, config
from odin.state import ErrorCategory, RunHistory

__version__ = "0.1.0"

__all__ = [
    "OdinBackend",
    "OdinConfig",
    "config",
    "ErrorCategory",
    "RunHistory",
]
