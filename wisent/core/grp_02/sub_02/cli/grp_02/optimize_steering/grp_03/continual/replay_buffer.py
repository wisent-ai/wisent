"""Experience replay buffer for continual learning forgetting detection."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from wisent.core.constants import REPLAY_BUFFER_MAX_SIZE


@dataclass
class ReplayEntry:
    """Single replay experience."""

    task: str
    enriched_pairs_path: str
    steering_path: str
    score: float
    cycle: int


class ReplayBuffer:
    """Circular buffer storing past optimization experiences for replay.

    Used to detect and correct catastrophic forgetting by periodically
    re-evaluating tasks and comparing against recorded scores.
    """

    def __init__(self, max_size: int = REPLAY_BUFFER_MAX_SIZE):
        self.max_size = max_size
        self.entries: List[ReplayEntry] = []
        self._task_index: Dict[str, List[int]] = defaultdict(list)

    def add(
        self,
        task: str,
        enriched_pairs_path: str,
        steering_path: str,
        score: float,
        cycle: int,
    ) -> None:
        """Store an experience. Evicts oldest entry when at capacity."""
        entry = ReplayEntry(
            task=task,
            enriched_pairs_path=enriched_pairs_path,
            steering_path=steering_path,
            score=score,
            cycle=cycle,
        )
        if len(self.entries) >= self.max_size:
            self.entries.pop(0)
            self._rebuild_index()
        idx = len(self.entries)
        self.entries.append(entry)
        self._task_index[task].append(idx)

    def sample(self, k: int) -> List[ReplayEntry]:
        """Sample k entries uniformly at random (without replacement)."""
        k = min(k, len(self.entries))
        if k <= 0:
            return []
        return random.sample(self.entries, k)

    def get_task_history(self, task: str) -> List[ReplayEntry]:
        """Return all entries for a given task, ordered by cycle."""
        indices = self._task_index.get(task, [])
        entries = [self.entries[i] for i in indices if i < len(self.entries)]
        return sorted(entries, key=lambda e: e.cycle)

    def get_best_score(self, task: str) -> Optional[float]:
        """Return the best recorded score for a task."""
        history = self.get_task_history(task)
        if not history:
            return None
        return max(e.score for e in history)

    def _rebuild_index(self) -> None:
        """Rebuild the task-to-index mapping from scratch."""
        self._task_index = defaultdict(list)
        for i, entry in enumerate(self.entries):
            self._task_index[entry.task].append(i)

    def save(self, path: str) -> None:
        """Serialize to JSON (paths and scores, no tensors)."""
        data = {
            "max_size": self.max_size,
            "entries": [asdict(e) for e in self.entries],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ReplayBuffer":
        """Deserialize from JSON."""
        with open(path) as f:
            data = json.load(f)
        buf = cls(max_size=data.get("max_size", REPLAY_BUFFER_MAX_SIZE))
        for ed in data.get("entries", []):
            buf.add(
                task=ed["task"],
                enriched_pairs_path=ed["enriched_pairs_path"],
                steering_path=ed["steering_path"],
                score=ed["score"],
                cycle=ed["cycle"],
            )
        return buf

    def __len__(self) -> int:
        return len(self.entries)


def detect_forgetting(
    replay_entries: List[ReplayEntry],
    current_scores: Dict[str, float],
    threshold: float,
) -> List[str]:
    """Identify tasks whose current score dropped below threshold * best.

    Args:
        replay_entries: All replay entries to extract per-task bests.
        current_scores: Mapping task -> current evaluation score.
        threshold: Ratio threshold (e.g. 0.9 = 10% degradation triggers).

    Returns:
        List of task names that have experienced forgetting.
    """
    best_per_task: Dict[str, float] = {}
    for entry in replay_entries:
        prev = best_per_task.get(entry.task, -float("inf"))
        best_per_task[entry.task] = max(prev, entry.score)

    degraded = []
    for task, best in best_per_task.items():
        current = current_scores.get(task)
        if current is None:
            continue
        if best > 0 and current < best * threshold:
            degraded.append(task)
    return degraded
