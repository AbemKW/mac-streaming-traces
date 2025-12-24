"""
Token-Level Trace Collector for MAC Streaming Instrumentation

PURPOSE:
    This module provides thread-safe collection of token emission timing data
    during LLM calls. It records when each token arrives during streaming,
    enabling downstream systems to replay realistic token-by-token timing.

WHAT THIS CAPTURES:
    - Per-turn timing (start, end, duration)
    - Per-token emission timestamps
    - Agent attribution for each turn
    - Sequential token ordering

WHAT THIS DOES NOT DO:
    - No CTU computation
    - No replay logic
    - No EXAID-specific schemas
    - No file I/O (that's handled by consumers)

USAGE:
    from instrumentation import get_trace_collector
    
    collector = get_trace_collector()
    traces = collector.get_all_turns()
"""

import threading
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class TokenEmission:
    """A single token emission event with timing."""
    token: str
    t_emitted_ms: int  # Timestamp in milliseconds since epoch
    seq: int           # Sequential index within the turn


@dataclass
class TurnTrace:
    """Complete trace for a single LLM turn/response."""
    turn_id: int
    agent_id: str
    content: str
    t_start_ms: int
    t_end_ms: int
    duration_ms: int
    token_emissions: List[TokenEmission] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "turn_id": self.turn_id,
            "agent_id": self.agent_id,
            "content": self.content,
            "t_start_ms": self.t_start_ms,
            "t_end_ms": self.t_end_ms,
            "duration_ms": self.duration_ms,
            "token_emissions": [asdict(te) for te in self.token_emissions]
        }


class TraceCollector:
    """
    Thread-safe singleton collector for token emission traces.
    
    This class accumulates timing data for each LLM turn without
    affecting the actual LLM behavior. Data is stored in memory
    and can be retrieved programmatically by downstream consumers.
    """
    
    _instance: Optional['TraceCollector'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'TraceCollector':
        """Singleton pattern - only one collector exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize collector state (only runs once due to singleton)."""
        if self._initialized:
            return
            
        self._data_lock = threading.Lock()
        self._turns: List[TurnTrace] = []
        self._turn_counter = 0
        
        # Current turn being recorded (per-thread would be cleaner,
        # but MAC runs sequentially so this is safe)
        self._current_turn: Optional[TurnTrace] = None
        self._current_seq = 0
        
        self._initialized = True
    
    def start_turn(self, agent_id: str) -> int:
        """
        Begin recording a new turn.
        
        Args:
            agent_id: Identifier for the agent making this LLM call
            
        Returns:
            The turn_id assigned to this turn
        """
        with self._data_lock:
            self._turn_counter += 1
            turn_id = self._turn_counter
            
            t_start = int(time.time() * 1000)
            
            self._current_turn = TurnTrace(
                turn_id=turn_id,
                agent_id=agent_id,
                content="",  # Will be set in end_turn
                t_start_ms=t_start,
                t_end_ms=0,  # Will be set in end_turn
                duration_ms=0,  # Will be computed in end_turn
                token_emissions=[]
            )
            self._current_seq = 0
            
            return turn_id
    
    def record_token(self, token: str) -> None:
        """
        Record a token emission during the current turn.
        
        Args:
            token: The token text that was emitted
        """
        if self._current_turn is None:
            return
            
        t_emitted = int(time.time() * 1000)
        
        with self._data_lock:
            emission = TokenEmission(
                token=token,
                t_emitted_ms=t_emitted,
                seq=self._current_seq
            )
            self._current_turn.token_emissions.append(emission)
            self._current_seq += 1
    
    def end_turn(self, content: str) -> Optional[TurnTrace]:
        """
        Finalize the current turn with the complete content.
        
        Args:
            content: The final reconstructed message content
            
        Returns:
            The completed TurnTrace, or None if no turn was active
        """
        with self._data_lock:
            if self._current_turn is None:
                return None
            
            t_end = int(time.time() * 1000)
            
            self._current_turn.content = content
            self._current_turn.t_end_ms = t_end
            self._current_turn.duration_ms = t_end - self._current_turn.t_start_ms
            
            completed_turn = self._current_turn
            self._turns.append(completed_turn)
            self._current_turn = None
            
            return completed_turn
    
    def get_all_turns(self) -> List[TurnTrace]:
        """
        Get all recorded turns.
        
        Returns:
            List of all TurnTrace objects recorded so far
        """
        with self._data_lock:
            return list(self._turns)
    
    def get_all_turns_as_dicts(self) -> List[Dict[str, Any]]:
        """
        Get all recorded turns as dictionaries (for JSON serialization).
        
        Returns:
            List of turn dictionaries
        """
        with self._data_lock:
            return [turn.to_dict() for turn in self._turns]
    
    def clear(self) -> None:
        """Reset the collector, clearing all recorded data."""
        with self._data_lock:
            self._turns.clear()
            self._turn_counter = 0
            self._current_turn = None
            self._current_seq = 0
    
    def get_turn_count(self) -> int:
        """Get the number of completed turns."""
        with self._data_lock:
            return len(self._turns)


# Module-level accessor for the singleton
def get_trace_collector() -> TraceCollector:
    """Get the global TraceCollector instance."""
    return TraceCollector()

