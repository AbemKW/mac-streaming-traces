"""
MAC Streaming Instrumentation Package

PURPOSE:
    Add transparent token-level streaming instrumentation to MAC without
    changing its behavior. This enables downstream systems (like EXAID)
    to replay realistic token-by-token timing.

WHAT THIS DOES:
    - Intercepts OpenAI client calls at the LLM boundary
    - Internally enables streaming to capture token timing
    - Reconstructs normal responses for AutoGen (transparent)
    - Stores timing data in memory for programmatic access

WHAT THIS DOES NOT DO:
    - Does NOT change MAC semantics
    - Does NOT modify speaker selection
    - Does NOT alter termination logic
    - Does NOT compute CTU or other metrics
    - Does NOT include EXAID-specific logic

USAGE:
    # CRITICAL: Must be called BEFORE importing autogen
    from instrumentation import install_instrumentation
    install_instrumentation()
    
    # Now safe to import autogen
    from autogen import AssistantAgent, GroupChat, ...
    
    # After MAC runs, access traces:
    from instrumentation import get_trace_collector
    traces = get_trace_collector().get_all_turns()

DESIGN PHILOSOPHY:
    "Adding a logic analyzer to a wire — not rewriting the circuit."
"""

from .trace_collector import (
    TraceCollector,
    TurnTrace,
    TokenEmission,
    get_trace_collector,
)

from .agent_attribution import (
    get_current_agent_id,
    set_current_agent_id,
    clear_current_agent_id,
    agent_context,
)

from .instrumented_client import (
    InstrumentedOpenAI,
    install_openai_patch,
    uninstall_openai_patch,
    is_patched,
)


def install_instrumentation() -> None:
    """
    Install all instrumentation hooks.
    
    CRITICAL: This must be called BEFORE importing autogen.
    
    This function:
    1. Patches openai.OpenAI to use our instrumented wrapper
    2. Enables token-level timing capture
    
    After calling this, all OpenAI client creations (including by AutoGen)
    will use the instrumented version that captures timing data while
    returning byte-identical responses.
    """
    install_openai_patch()


def uninstall_instrumentation() -> None:
    """
    Remove all instrumentation, restoring vanilla behavior.
    
    Useful for testing or when instrumentation should be disabled.
    Note: This does NOT affect already-created clients.
    """
    uninstall_openai_patch()


def is_instrumentation_active() -> bool:
    """Check if instrumentation is currently active."""
    return is_patched()


# Export key classes and functions
__all__ = [
    # Main entry points
    'install_instrumentation',
    'uninstall_instrumentation',
    'is_instrumentation_active',
    
    # Trace collector
    'TraceCollector',
    'TurnTrace',
    'TokenEmission',
    'get_trace_collector',
    
    # Agent attribution
    'get_current_agent_id',
    'set_current_agent_id',
    'clear_current_agent_id',
    'agent_context',
    
    # Low-level (for advanced use)
    'InstrumentedOpenAI',
    'install_openai_patch',
    'uninstall_openai_patch',
    'is_patched',
]

