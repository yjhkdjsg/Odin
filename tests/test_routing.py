"""Unit tests for retrieval routing logic."""
import pytest

from odin.routing import should_retrieve, override_routing, reset_routing, RETRIEVAL_ROUTING


def test_syntax_skips_retrieval():
    """Test that syntax errors skip retrieval."""
    should_retrieve_flag, reason = should_retrieve("syntax")
    
    assert should_retrieve_flag is False
    assert "syntax" in reason.lower()


def test_runtime_skips_retrieval():
    """Test that runtime errors skip retrieval."""
    should_retrieve_flag, reason = should_retrieve("runtime")
    
    assert should_retrieve_flag is False
    assert "runtime" in reason.lower()


def test_logical_skips_retrieval():
    """Test that logical errors skip retrieval."""
    should_retrieve_flag, reason = should_retrieve("logical")
    
    assert should_retrieve_flag is False
    assert "logical" in reason.lower()


def test_library_api_triggers_retrieval():
    """Test that library_api errors trigger retrieval."""
    should_retrieve_flag, reason = should_retrieve("library_api")
    
    assert should_retrieve_flag is True
    assert "library_api" in reason.lower()


def test_version_specific_triggers_retrieval():
    """Test that version_specific errors trigger retrieval."""
    should_retrieve_flag, reason = should_retrieve("version_specific")
    
    assert should_retrieve_flag is True
    assert "version_specific" in reason.lower()


def test_uncommon_pattern_triggers_retrieval():
    """Test that uncommon_pattern errors trigger retrieval."""
    should_retrieve_flag, reason = should_retrieve("uncommon_pattern")
    
    assert should_retrieve_flag is True
    assert "uncommon_pattern" in reason.lower()


def test_override_routing():
    """Test that routing can be overridden for experiments."""
    # Save original
    original = dict(RETRIEVAL_ROUTING)
    
    try:
        # Override to always retrieve
        override_routing({
            "syntax": True,
            "runtime": True,
        })
        
        should_retrieve_flag, _ = should_retrieve("syntax")
        assert should_retrieve_flag is True
        
        should_retrieve_flag, _ = should_retrieve("runtime")
        assert should_retrieve_flag is True
    
    finally:
        # Restore
        reset_routing()
        
        # Verify restored
        should_retrieve_flag, _ = should_retrieve("syntax")
        assert should_retrieve_flag is False


def test_reset_routing():
    """Test that reset restores defaults."""
    # Override
    override_routing({"syntax": True})
    
    # Reset
    reset_routing()
    
    # Verify defaults
    should_retrieve_flag, _ = should_retrieve("syntax")
    assert should_retrieve_flag is False
    
    should_retrieve_flag, _ = should_retrieve("library_api")
    assert should_retrieve_flag is True
