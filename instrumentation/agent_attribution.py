"""
Agent Attribution for MAC Streaming Instrumentation

PURPOSE:
    Track which AutoGen agent is currently making an LLM call.
    This allows the trace collector to attribute token emissions
    to the correct agent.

DESIGN:
    - Uses thread-local storage for the current agent ID
    - Provides a simple context manager for setting/clearing
    - Does NOT modify agent behavior in any way

USAGE:
    with set_current_agent("Doctor0"):
        # LLM calls here will be attributed to Doctor0
        ...
"""

import threading
from contextlib import contextmanager
from typing import Optional


# Thread-local storage for current agent attribution
_attribution = threading.local()


def get_current_agent_id() -> str:
    """
    Get the current agent ID for attribution.
    
    Returns:
        The current agent ID, or "unknown" if not set
    """
    return getattr(_attribution, 'agent_id', 'unknown')


def set_current_agent_id(agent_id: str) -> None:
    """
    Set the current agent ID for attribution.
    
    Args:
        agent_id: The agent identifier to set
    """
    _attribution.agent_id = agent_id


def clear_current_agent_id() -> None:
    """Clear the current agent ID."""
    _attribution.agent_id = 'unknown'


@contextmanager
def agent_context(agent_id: str):
    """
    Context manager for setting agent attribution during a block.
    
    Args:
        agent_id: The agent identifier
        
    Usage:
        with agent_context("Doctor0"):
            # LLM calls here will be attributed to Doctor0
            agent.generate_reply(...)
    """
    previous = get_current_agent_id()
    set_current_agent_id(agent_id)
    try:
        yield
    finally:
        set_current_agent_id(previous)

