"""Core LatencyTracker implementation."""
import time
import logging
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
from wisent.core.utils.infra_tools.tracking._latency_types import (
    TimingEvent, LatencyStats, GenerationMetrics, TrainingMetrics)
from wisent.core.utils.infra_tools.tracking._latency_tracker_reporting import LatencyReportingMixin
logger = logging.getLogger(__name__)

class LatencyTracker(LatencyReportingMixin):
    """
    Comprehensive latency tracker for wisent operations.
    
    Tracks timing for individual operations and provides aggregated statistics.
    Supports nested operation tracking and hierarchical timing analysis.
    """
    
    def __init__(self, auto_start: bool = True):
        """
        Initialize latency tracker.
        
        Args:
            auto_start: Whether to automatically start tracking
        """
        self.events: List[TimingEvent] = []
        self.active_operations: Dict[str, float] = {}
        self.operation_stack: List[str] = []
        self.is_tracking = auto_start
        self.start_time = time.time() if auto_start else None
    
    def start_tracking(self) -> None:
        """Start or resume latency tracking."""
        self.is_tracking = True
        if self.start_time is None:
            self.start_time = time.time()
    
    def stop_tracking(self) -> None:
        """Stop latency tracking."""
        self.is_tracking = False
    
    def start_operation(
        self, 
        name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start timing an operation.
        
        Args:
            name: Name of the operation
            metadata: Optional metadata to store with the event
            
        Returns:
            Operation ID for later reference
        """
        if not self.is_tracking:
            return name
        
        current_time = time.time()
        operation_id = f"{name}_{len(self.events)}"
        
        self.active_operations[operation_id] = current_time
        self.operation_stack.append(operation_id)
        
        return operation_id
    
    def end_operation(
        self, 
        operation_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[TimingEvent]:
        """
        End timing an operation.
        
        Args:
            operation_id: ID returned from start_operation
            metadata: Additional metadata to store
            
        Returns:
            TimingEvent if operation was found, None otherwise
        """
        if not self.is_tracking or operation_id not in self.active_operations:
            return None
        
        end_time = time.time()
        start_time = self.active_operations.pop(operation_id)
        duration = end_time - start_time
        
        # Extract operation name from ID
        name = operation_id.rsplit('_', 1)[0]
        
        # Determine parent operation
        parent = None
        if operation_id in self.operation_stack:
            stack_index = self.operation_stack.index(operation_id)
            if stack_index > 0:
                parent_id = self.operation_stack[stack_index - 1]
                parent = parent_id.rsplit('_', 1)[0]
            self.operation_stack.remove(operation_id)
        
        # Merge metadata
        combined_metadata = metadata or {}
        
        event = TimingEvent(
            name=name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            metadata=combined_metadata,
            parent=parent
        )
        
        self.events.append(event)
        return event
    
    @contextmanager
    def time_operation(
        self, 
        name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for timing operations.
        
        Args:
            name: Name of the operation
            metadata: Optional metadata to store
            
        Yields:
            TimingEvent that will be populated when context exits
        """
        operation_id = self.start_operation(name, metadata)
        event_placeholder = {"event": None}
        
        try:
            yield event_placeholder
        finally:
            event = self.end_operation(operation_id, metadata)
            event_placeholder["event"] = event
    
    @contextmanager
    def time_generation(self, name: str = "response_generation", prompt_length: int = 0):
        """
        Context manager for timing text generation with TTFT tracking.
        
        Args:
            name: Name of the generation operation
            prompt_length: Length of the input prompt in tokens
            
        Yields:
            Dict with methods to mark first token and update token count
        """
        start_time = time.time()
        operation_id = self.start_operation(name, {"prompt_length": prompt_length})
        
        generation_state = {
            "first_token_time": None,
            "token_count": 0
        }
        
        # Add methods that modify the dict
        generation_state["mark_first_token"] = lambda: generation_state.update({"first_token_time": time.time()})
        generation_state["update_tokens"] = lambda count: generation_state.update({"token_count": count})
        
        try:
            yield generation_state
        finally:
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Calculate TTFT
            ttft = generation_state["first_token_time"] - start_time if generation_state["first_token_time"] else 0.0
            
            # Calculate tokens per second
            tokens_per_sec = generation_state["token_count"] / total_duration if total_duration > 0 else 0.0
            
            metadata = {
                "prompt_length": prompt_length,
                "time_to_first_token": ttft,
                "token_count": generation_state["token_count"],
                "tokens_per_second": tokens_per_sec
            }
            
            self.end_operation(operation_id, metadata)
    
    def get_stats(self, operation_name: Optional[str] = None) -> Union[LatencyStats, Dict[str, LatencyStats]]:
        """
        Get latency statistics.
        
        Args:
            operation_name: Specific operation to get stats for, or None for all
            
        Returns:
            LatencyStats for specific operation or dict of all operation stats
        """
        if operation_name:
            events = [e for e in self.events if e.name == operation_name]
            if not events:
                raise InsufficientDataError(reason=f"No events found for operation: {operation_name}")
            return self._calculate_stats(operation_name, events)
        else:
            # Group events by operation name
            operation_events = defaultdict(list)
            for event in self.events:
                operation_events[event.name].append(event)
            
            return {
                name: self._calculate_stats(name, events)
                for name, events in operation_events.items()
            }
