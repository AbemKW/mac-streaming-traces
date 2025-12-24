"""
Validation Script for MAC Streaming Instrumentation

This script verifies that:
1. Instrumentation can be installed without errors
2. Token timing is captured correctly
3. Responses are reconstructed properly
4. No behavior change occurs (content is preserved)

Run with: python test_instrumentation.py

NOTE: This is a unit test for the instrumentation layer itself.
It does NOT require API keys or network access - it uses mocks.
"""

import sys
import time
import unittest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Optional


class TestTraceCollector(unittest.TestCase):
    """Test the TraceCollector functionality."""
    
    def setUp(self):
        """Reset collector before each test."""
        from instrumentation import get_trace_collector
        self.collector = get_trace_collector()
        self.collector.clear()
    
    def test_singleton(self):
        """TraceCollector should be a singleton."""
        from instrumentation import get_trace_collector
        collector1 = get_trace_collector()
        collector2 = get_trace_collector()
        self.assertIs(collector1, collector2)
    
    def test_start_and_end_turn(self):
        """Should record a complete turn."""
        turn_id = self.collector.start_turn("TestAgent")
        self.assertEqual(turn_id, 1)
        
        # Record some tokens
        self.collector.record_token("Hello")
        time.sleep(0.01)  # Small delay
        self.collector.record_token(" world")
        
        # End the turn
        result = self.collector.end_turn("Hello world")
        
        self.assertIsNotNone(result)
        self.assertEqual(result.turn_id, 1)
        self.assertEqual(result.agent_id, "TestAgent")
        self.assertEqual(result.content, "Hello world")
        self.assertEqual(len(result.token_emissions), 2)
        self.assertEqual(result.token_emissions[0].token, "Hello")
        self.assertEqual(result.token_emissions[1].token, " world")
        self.assertEqual(result.token_emissions[0].seq, 0)
        self.assertEqual(result.token_emissions[1].seq, 1)
    
    def test_multiple_turns(self):
        """Should record multiple turns correctly."""
        # Turn 1
        self.collector.start_turn("Agent1")
        self.collector.record_token("First")
        self.collector.end_turn("First")
        
        # Turn 2
        self.collector.start_turn("Agent2")
        self.collector.record_token("Second")
        self.collector.end_turn("Second")
        
        turns = self.collector.get_all_turns()
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].agent_id, "Agent1")
        self.assertEqual(turns[1].agent_id, "Agent2")
    
    def test_clear(self):
        """Clear should reset all state."""
        self.collector.start_turn("Agent")
        self.collector.record_token("test")
        self.collector.end_turn("test")
        
        self.collector.clear()
        
        self.assertEqual(self.collector.get_turn_count(), 0)
        self.assertEqual(len(self.collector.get_all_turns()), 0)
    
    def test_to_dict(self):
        """Should serialize to dict correctly."""
        self.collector.start_turn("Agent")
        self.collector.record_token("test")
        self.collector.end_turn("test")
        
        dicts = self.collector.get_all_turns_as_dicts()
        self.assertEqual(len(dicts), 1)
        self.assertIn("turn_id", dicts[0])
        self.assertIn("agent_id", dicts[0])
        self.assertIn("content", dicts[0])
        self.assertIn("token_emissions", dicts[0])
    
    def test_timing_recorded(self):
        """Token emissions should have timestamps."""
        self.collector.start_turn("Agent")
        t1 = int(time.time() * 1000)
        self.collector.record_token("test")
        t2 = int(time.time() * 1000)
        self.collector.end_turn("test")
        
        turns = self.collector.get_all_turns()
        emission = turns[0].token_emissions[0]
        
        # Timestamp should be between t1 and t2
        self.assertGreaterEqual(emission.t_emitted_ms, t1)
        self.assertLessEqual(emission.t_emitted_ms, t2 + 1)  # +1 for timing variance


class TestAgentAttribution(unittest.TestCase):
    """Test the agent attribution system."""
    
    def test_default_agent(self):
        """Default agent should be 'unknown'."""
        from instrumentation import get_current_agent_id, clear_current_agent_id
        clear_current_agent_id()
        self.assertEqual(get_current_agent_id(), "unknown")
    
    def test_set_and_get(self):
        """Should be able to set and get agent ID."""
        from instrumentation import set_current_agent_id, get_current_agent_id
        set_current_agent_id("Doctor0")
        self.assertEqual(get_current_agent_id(), "Doctor0")
    
    def test_context_manager(self):
        """Context manager should set and restore agent ID."""
        from instrumentation import agent_context, get_current_agent_id, set_current_agent_id
        
        set_current_agent_id("Original")
        
        with agent_context("Temporary"):
            self.assertEqual(get_current_agent_id(), "Temporary")
        
        self.assertEqual(get_current_agent_id(), "Original")


class TestInstallation(unittest.TestCase):
    """Test the patching mechanism."""
    
    def test_install_and_uninstall(self):
        """Should be able to install and uninstall patch."""
        from instrumentation import (
            install_instrumentation,
            uninstall_instrumentation,
            is_instrumentation_active
        )
        
        # Install
        install_instrumentation()
        self.assertTrue(is_instrumentation_active())
        
        # Uninstall
        uninstall_instrumentation()
        self.assertFalse(is_instrumentation_active())
        
        # Re-install for other tests
        install_instrumentation()


class TestInstrumentedClient(unittest.TestCase):
    """Test the instrumented OpenAI client wrapper."""
    
    def test_stream_reconstruction(self):
        """Should reconstruct response from stream chunks."""
        from instrumentation import get_trace_collector
        from instrumentation.instrumented_client import InstrumentedCompletions
        
        collector = get_trace_collector()
        collector.clear()
        
        # Create mock streaming chunks
        @dataclass
        class MockDelta:
            content: Optional[str] = None
        
        @dataclass
        class MockChoice:
            delta: MockDelta
            finish_reason: Optional[str] = None
        
        @dataclass
        class MockChunk:
            id: str = "chatcmpl-test"
            model: str = "gpt-test"
            created: int = 1234567890
            choices: List[MockChoice] = None
            usage: Optional[object] = None
            
            def __post_init__(self):
                if self.choices is None:
                    self.choices = []
        
        # Simulate streaming chunks
        chunks = [
            MockChunk(choices=[MockChoice(delta=MockDelta(content="Hello"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(content=" "))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(content="world"))]),
            MockChunk(choices=[MockChoice(delta=MockDelta(content="!"), finish_reason="stop")]),
        ]
        
        # Create mock completions object
        mock_completions = Mock()
        mock_completions.create = Mock(return_value=iter(chunks))
        
        instrumented = InstrumentedCompletions(mock_completions)
        
        # Set up agent attribution
        from instrumentation import set_current_agent_id
        set_current_agent_id("TestAgent")
        
        # Start a turn manually (normally done in create())
        collector.start_turn("TestAgent")
        
        # Call the reconstruction method directly
        result = instrumented._reconstruct_from_stream(iter(chunks), collector)
        
        # Verify reconstruction
        self.assertEqual(result.choices[0].message.content, "Hello world!")
        self.assertEqual(result.id, "chatcmpl-test")
        self.assertEqual(result.model, "gpt-test")
        
        # Verify timing was captured
        turns = collector.get_all_turns()
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].content, "Hello world!")
        self.assertEqual(len(turns[0].token_emissions), 4)


def run_validation():
    """Run all validation tests."""
    print("=" * 60)
    print("MAC Streaming Instrumentation Validation")
    print("=" * 60)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTraceCollector))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentAttribution))
    suite.addTests(loader.loadTestsFromTestCase(TestInstallation))
    suite.addTests(loader.loadTestsFromTestCase(TestInstrumentedClient))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 60)
    if result.wasSuccessful():
        print("VALIDATION PASSED: All tests successful")
        print()
        print("The instrumentation is correctly:")
        print("  - Capturing token timing data")
        print("  - Tracking agent attribution")
        print("  - Reconstructing responses")
        print("  - Preserving content integrity")
    else:
        print("VALIDATION FAILED: Some tests failed")
        print("Please review the errors above")
    print("=" * 60)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_validation())

