"""Deterministic routing logic for adaptive RAG retrieval.

This module implements the core adaptive retrieval decision:
given an error category, should we retrieve external knowledge or skip it?

The routing is intentionally simple and deterministic (not LLM-based)
so it can be easily inspected, tuned, and overridden for experiments.
"""
import logging
from typing import Literal

from odin.state import ErrorCategory

logger = logging.getLogger(__name__)


# The core routing table: maps error categories to retrieval decisions
# This is the single source of truth for whether retrieval triggers
RETRIEVAL_ROUTING: dict[ErrorCategory, bool] = {
    "syntax": False,  # Syntax errors are self-contained, no docs needed
    "runtime": False,  # Standard runtime errors resolvable from traceback
    "logical": False,  # Logic bugs need reasoning, not external knowledge
    "library_api": True,  # API misuse benefits from official docs
    "version_specific": True,  # Version changes need version-specific docs
    "uncommon_pattern": True,  # Rare patterns benefit from curated examples
}


def should_retrieve(category: ErrorCategory) -> tuple[bool, str]:
    """Decide whether to trigger RAG retrieval for the given error category.
    
    Args:
        category: The error category assigned by the classifier
    
    Returns:
        A tuple of (should_retrieve, reason) explaining the decision
    
    This is a pure function with no side effects, making it easy to test
    and modify for different experimental configurations.
    """
    retrieve = RETRIEVAL_ROUTING.get(category, False)
    
    if retrieve:
        reason = f"Category '{category}' triggers retrieval per routing table"
    else:
        reason = f"Category '{category}' skips retrieval per routing table"
    
    logger.info(f"Retrieval decision for {category}: {retrieve} - {reason}")
    
    return retrieve, reason


def override_routing(
    custom_routing: dict[ErrorCategory, bool]
) -> None:
    """Override the default routing table with a custom configuration.
    
    This allows experiment harnesses to test different retrieval strategies
    without modifying the core routing logic.
    
    Args:
        custom_routing: A dictionary mapping categories to retrieval decisions
    """
    RETRIEVAL_ROUTING.update(custom_routing)
    logger.info(f"Routing table updated: {RETRIEVAL_ROUTING}")


def reset_routing() -> None:
    """Reset routing table to defaults."""
    RETRIEVAL_ROUTING.clear()
    RETRIEVAL_ROUTING.update({
        "syntax": False,
        "runtime": False,
        "logical": False,
        "library_api": True,
        "version_specific": True,
        "uncommon_pattern": True,
    })
    logger.info("Routing table reset to defaults")
