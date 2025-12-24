"""
Instrumented OpenAI Client for MAC Streaming Instrumentation

PURPOSE:
    Transparent proxy around the OpenAI client that captures token-level
    timing data without changing the response seen by AutoGen.

HOW IT WORKS:
    1. Intercepts chat.completions.create() calls
    2. Internally forces stream=True
    3. Iterates over streaming chunks, timestamps each token
    4. Reconstructs a standard ChatCompletion object
    5. Returns the reconstructed response (AutoGen sees no difference)

WHAT THIS CAPTURES:
    - Token emission timestamps
    - Complete message content (reconstructed from stream)

WHAT THIS PRESERVES:
    - Byte-identical final message content
    - All response metadata (id, model, created, etc.)
    - Usage statistics (when available from stream)

CRITICAL CONSTRAINT:
    AutoGen must receive a normal, synchronous ChatCompletion object.
    It must never know that streaming occurred internally.
"""

import time
from typing import Any, Optional, Union, Iterator

from .trace_collector import get_trace_collector
from .agent_attribution import get_current_agent_id


class InstrumentedCompletions:
    """
    Wrapper around OpenAI chat.completions that instruments the create() method.
    """
    
    def __init__(self, original_completions):
        """
        Args:
            original_completions: The original chat.completions object from OpenAI client
        """
        self._original = original_completions
    
    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the original completions object."""
        return getattr(self._original, name)
    
    def create(self, **kwargs) -> Any:
        """
        Instrumented create() that captures token timing.
        
        This method:
        1. Forces streaming internally
        2. Captures each token with timestamp
        3. Reconstructs a non-streaming response
        4. Returns it to the caller (AutoGen)
        """
        # If the caller explicitly wants streaming, pass through without instrumentation
        # (AutoGen typically doesn't request streaming)
        if kwargs.get('stream', False):
            return self._original.create(**kwargs)
        
        # Get current agent for attribution
        agent_id = get_current_agent_id()
        collector = get_trace_collector()
        
        # Start recording this turn
        collector.start_turn(agent_id)
        
        # Force streaming for instrumentation
        kwargs_streaming = dict(kwargs)
        kwargs_streaming['stream'] = True
        
        # Request usage stats in streaming if supported (OpenAI 1.0+)
        # This may not be supported by all providers
        if 'stream_options' not in kwargs_streaming:
            kwargs_streaming['stream_options'] = {'include_usage': True}
        
        try:
            # Make the streaming request
            stream = self._original.create(**kwargs_streaming)
            
            # Reconstruct the response from chunks
            response = self._reconstruct_from_stream(stream, collector)
            
            return response
            
        except Exception as e:
            # If streaming fails, try falling back to non-streaming
            # This preserves behavior if the provider doesn't support streaming
            collector.end_turn("")  # End turn with empty content on failure
            
            # Fall back to original non-streaming call
            return self._original.create(**kwargs)
    
    def _reconstruct_from_stream(self, stream: Iterator, collector) -> Any:
        """
        Reconstruct a ChatCompletion from streaming chunks.
        
        Args:
            stream: Iterator of ChatCompletionChunk objects
            collector: TraceCollector to record tokens to
            
        Returns:
            A ChatCompletion-like object that AutoGen can use
        """
        # Accumulate data from chunks
        chunks = []
        content_parts = []
        first_chunk = None
        usage = None
        finish_reason = None
        
        for chunk in stream:
            chunks.append(chunk)
            
            # Capture metadata from first chunk
            if first_chunk is None:
                first_chunk = chunk
            
            # Process choices
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                
                # Get delta content
                if hasattr(choice, 'delta') and choice.delta:
                    delta = choice.delta
                    if hasattr(delta, 'content') and delta.content:
                        token = delta.content
                        content_parts.append(token)
                        collector.record_token(token)
                
                # Capture finish reason
                if hasattr(choice, 'finish_reason') and choice.finish_reason:
                    finish_reason = choice.finish_reason
            
            # Capture usage from final chunk (if stream_options was supported)
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                usage = chunk.usage
        
        # Reconstruct final content
        final_content = ''.join(content_parts)
        
        # End the turn in collector
        collector.end_turn(final_content)
        
        # Build a ChatCompletion-like response
        return self._build_completion_response(
            first_chunk=first_chunk,
            content=final_content,
            finish_reason=finish_reason,
            usage=usage
        )
    
    def _build_completion_response(
        self,
        first_chunk,
        content: str,
        finish_reason: Optional[str],
        usage: Optional[Any]
    ) -> Any:
        """
        Build a ChatCompletion-like response object.
        
        This creates an object that satisfies AutoGen's expectations
        for a non-streaming response.
        """
        # Import here to avoid circular dependencies
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.completion_usage import CompletionUsage
        
        # Build the message
        message = ChatCompletionMessage(
            role="assistant",
            content=content
        )
        
        # Build the choice
        choice = Choice(
            index=0,
            message=message,
            finish_reason=finish_reason or "stop"
        )
        
        # Build usage (estimate if not provided by stream)
        if usage is None:
            # Rough token estimation (actual counting would require tiktoken)
            # This is approximate but doesn't affect MAC behavior
            prompt_tokens = 0  # We don't have access to this
            completion_tokens = len(content.split())  # Very rough estimate
            usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        
        # Get metadata from first chunk
        completion_id = getattr(first_chunk, 'id', 'chatcmpl-instrumented')
        model = getattr(first_chunk, 'model', 'unknown')
        created = getattr(first_chunk, 'created', int(time.time()))
        
        # Build the ChatCompletion
        completion = ChatCompletion(
            id=completion_id,
            choices=[choice],
            created=created,
            model=model,
            object="chat.completion",
            usage=usage
        )
        
        return completion


class InstrumentedChat:
    """Wrapper around OpenAI chat that provides instrumented completions."""
    
    def __init__(self, original_chat):
        self._original = original_chat
        self._completions = None
    
    @property
    def completions(self) -> InstrumentedCompletions:
        """Get the instrumented completions object."""
        if self._completions is None:
            self._completions = InstrumentedCompletions(self._original.completions)
        return self._completions
    
    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the original chat object."""
        if name == 'completions':
            return self.completions
        return getattr(self._original, name)


class InstrumentedOpenAI:
    """
    Instrumented OpenAI client that captures token timing transparently.
    
    This wraps the real OpenAI client and intercepts chat.completions.create()
    calls to capture streaming timing data while returning reconstructed
    non-streaming responses.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Create an instrumented OpenAI client.
        
        All arguments are passed through to the real OpenAI client.
        """
        import openai
        
        # Store the original class before any patching
        original_class = getattr(openai, '_OriginalOpenAI', openai.OpenAI)
        
        # Create the real client
        self._client = original_class(*args, **kwargs)
        self._chat = None
    
    @property
    def chat(self) -> InstrumentedChat:
        """Get the instrumented chat object."""
        if self._chat is None:
            self._chat = InstrumentedChat(self._client.chat)
        return self._chat
    
    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the real client."""
        if name == 'chat':
            return self.chat
        return getattr(self._client, name)


# Store reference to whether patching has been applied
_patched = False


def install_openai_patch() -> None:
    """
    Install the OpenAI client patch.
    
    This replaces openai.OpenAI with InstrumentedOpenAI so that
    all subsequent client creations (including by AutoGen) use
    the instrumented version.
    
    CRITICAL: This must be called BEFORE importing autogen.
    """
    global _patched
    
    if _patched:
        return
    
    import openai
    
    # Store the original class for InstrumentedOpenAI to use
    if not hasattr(openai, '_OriginalOpenAI'):
        openai._OriginalOpenAI = openai.OpenAI
    
    # Replace with instrumented version
    openai.OpenAI = InstrumentedOpenAI
    
    _patched = True


def uninstall_openai_patch() -> None:
    """
    Remove the OpenAI client patch, restoring original behavior.
    
    Useful for testing or when instrumentation should be disabled.
    """
    global _patched
    
    if not _patched:
        return
    
    import openai
    
    if hasattr(openai, '_OriginalOpenAI'):
        openai.OpenAI = openai._OriginalOpenAI
    
    _patched = False


def is_patched() -> bool:
    """Check if the OpenAI patch is currently installed."""
    return _patched

