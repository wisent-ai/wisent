"""Reporting and summary mixin for LatencyTracker."""
import statistics
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict
import time
from wisent.core.tracking._latency_types import (
    LatencyStats, GenerationMetrics, TrainingMetrics, TimingEvent)
from wisent.core import constants as _C

class LatencyReportingMixin:
    """Mixin providing reporting and summary methods."""

    def _calculate_stats(self, operation, durations, events=None):
        """Calculate latency statistics for an operation."""
        if events is None:
            events = []
        durations = sorted(durations)
        return LatencyStats(
            operation=operation,
            count=len(durations),
            total_time=sum(durations),
            mean_time=statistics.mean(durations),
            median_time=statistics.median(durations),
            min_time=min(durations),
            max_time=max(durations),
            std_dev=statistics.stdev(durations) if len(durations) > 1 else 0,
            percentile_95=self._percentile(durations, _C.PERCENTILE_HIGH),
            percentile_99=self._percentile(durations, _C.PERCENTILE_CRITICAL),
            events=events.copy()
        )
    
    def _percentile(self, sorted_data: List[float], percentile: float) -> float:
        """Calculate percentile from sorted data."""
        if not sorted_data:
            return 0
        
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def get_timeline(self) -> List[TimingEvent]:
        """Get chronological timeline of all events."""
        return sorted(self.events, key=lambda e: e.start_time)
    
    def get_hierarchy(self) -> Dict[str, List[TimingEvent]]:
        """Get hierarchical view of operations (parent -> children)."""
        hierarchy = defaultdict(list)
        
        for event in self.events:
            parent = event.parent or "root"
            hierarchy[parent].append(event)
        
        return dict(hierarchy)
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self.events.clear()
        self.active_operations.clear()
        self.operation_stack.clear()
        self.start_time = time.time() if self.is_tracking else None
    
    def get_generation_metrics(self, operation_name: str = "response_generation") -> Optional[GenerationMetrics]:
        """Get user-facing generation metrics."""
        events = [e for e in self.events if e.name == operation_name]
        if not events:
            return None
        
        # Use the most recent event
        latest_event = events[-1]
        metadata = latest_event.metadata
        
        return GenerationMetrics(
            time_to_first_token=metadata.get('time_to_first_token', 0.0),
            total_generation_time=latest_event.duration,
            token_count=metadata.get('token_count', 0),
            tokens_per_second=metadata.get('tokens_per_second', 0.0),
            prompt_length=metadata.get('prompt_length', 0)
        )
    
    def get_training_metrics(self, operation_name: str = "total_training_time") -> Optional[TrainingMetrics]:
        """Get user-facing training metrics."""
        events = [e for e in self.events if e.name == operation_name]
        if not events:
            return None
        
        latest_event = events[-1]
        metadata = latest_event.metadata
        
        return TrainingMetrics(
            total_training_time=latest_event.duration,
            training_samples=metadata.get('training_samples', 0),
            method=metadata.get('method', 'unknown'),
            success=metadata.get('success', True),
            error_message=metadata.get('error_message')
        )
    
    def format_user_metrics(self) -> str:
        """Format user-facing performance metrics."""
        lines = ["🚀 Performance Summary:"]
        
        # Training metrics
        training_metrics = self.get_training_metrics()
        if training_metrics:
            lines.extend([
                f"\n📚 Training:",
                f"  Method: {training_metrics.method}",
                f"  Total Time: {training_metrics.training_time_ms:.0f} ms",
                f"  Samples: {training_metrics.training_samples}",
                f"  Speed: {training_metrics.samples_per_second:.1f} samples/sec"
            ])
        
        # Generation metrics - check for both response_generation and individual generation events
        generation_metrics = self.get_generation_metrics("response_generation")
        if not generation_metrics:
            # Try to get metrics from steered_generation if response_generation doesn't exist
            generation_metrics = self.get_generation_metrics("steered_generation")
        
        if generation_metrics and generation_metrics.token_count > 0:
            lines.extend([
                f"\n🎭 Generation:",
                f"  Time to First Token: {generation_metrics.ttft_ms:.0f} ms",
                f"  Total Generation: {generation_metrics.total_time_ms:.0f} ms",
                f"  Tokens Generated: {generation_metrics.token_count}",
                f"  Speed: {generation_metrics.tokens_per_second:.1f} tokens/sec"
            ])
        
        # Steering overhead comparison
        steered_events = [e for e in self.events if e.name == "steered_generation"]
        unsteered_events = [e for e in self.events if e.name == "unsteered_generation"]
        
        if steered_events and unsteered_events:
            steered_avg = sum(e.duration for e in steered_events) / len(steered_events)
            unsteered_avg = sum(e.duration for e in unsteered_events) / len(unsteered_events)
            overhead = ((steered_avg - unsteered_avg) / unsteered_avg) * 100
            
            lines.extend([
                f"\n⚡ Steering Overhead:",
                f"  Unsteered Avg: {unsteered_avg * _C.MS_PER_SECOND:.0f} ms ({len(unsteered_events)} runs)",
                f"  Steered Avg: {steered_avg * _C.MS_PER_SECOND:.0f} ms ({len(steered_events)} runs)",
                f"  Overhead: {overhead:+.1f}%"
            ])
        elif steered_events:
            # Show steered performance even without comparison
            steered_avg = sum(e.duration for e in steered_events) / len(steered_events)
            lines.extend([
                f"\n🎯 Steered Generation:",
                f"  Average Time: {steered_avg * _C.MS_PER_SECOND:.0f} ms ({len(steered_events)} runs)"
            ])
        elif unsteered_events:
            # Show unsteered performance even without comparison
            unsteered_avg = sum(e.duration for e in unsteered_events) / len(unsteered_events)
            lines.extend([
                f"\n🔄 Unsteered Generation:",
                f"  Average Time: {unsteered_avg * _C.MS_PER_SECOND:.0f} ms ({len(unsteered_events)} runs)"
            ])
        
        # Show warning if no generation metrics found
        if not generation_metrics or generation_metrics.token_count == 0:
            lines.extend([
                f"\n⚠️ No generation metrics available",
                f"  (Responses may be empty or timing failed)"
            ])
        
        return '\n'.join(lines)

    def format_stats(
        self, 
        stats: Union[LatencyStats, Dict[str, LatencyStats]], 
        detailed: bool = False
    ) -> str:
        """Format latency statistics as a readable string."""
        if isinstance(stats, LatencyStats):
            return self._format_single_stats(stats, detailed)
        else:
            lines = ["Latency Statistics Summary:"]
            for operation, op_stats in stats.items():
                lines.append(f"\n{operation}:")
                lines.extend([f"  {line}" for line in self._format_single_stats(op_stats, detailed).split('\n')])
            return '\n'.join(lines)
    
    def _format_single_stats(self, stats: LatencyStats, detailed: bool) -> str:
        """Format statistics for a single operation."""
        lines = [
            f"Operation: {stats.operation}",
            f"Count: {stats.count}",
            f"Total Time: {stats.total_time_ms:.1f} ms",
            f"Mean Time: {stats.mean_time_ms:.1f} ms",
            f"Median Time: {stats.median_time_ms:.1f} ms",
            f"Min Time: {stats.min_time_ms:.1f} ms",
            f"Max Time: {stats.peak_time_ms:.1f} ms",
        ]
        
        if stats.count > 1:
            lines.extend([
                f"Std Dev: {stats.std_dev_ms:.1f} ms",
                f"95th Percentile: {stats.percentile_95_ms:.1f} ms",
                f"99th Percentile: {stats.percentile_99_ms:.1f} ms",
            ])
        
        if detailed and stats.events:
            lines.append(f"Recent Events:")
            for event in stats.events[-5:]:  # Show last 5 events
                lines.append(f"  {event.duration_ms:.1f} ms")
                if event.metadata:
                    lines.append(f"    Metadata: {event.metadata}")
        
        return '\n'.join(lines)
    
    def export_csv(self, filename: str) -> None:
        """Export timing events to CSV file."""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'operation', 'start_time', 'end_time', 'duration_ms', 
                'parent', 'metadata'
            ])
            
            for event in self.events:
                writer.writerow([
                    event.name,
                    event.start_time,
                    event.end_time,
                    event.duration_ms,
                    event.parent or '',
                    str(event.metadata) if event.metadata else ''
                ])


# Global latency tracker instance
_global_tracker: Optional[LatencyTracker] = None


