"""Core data models: Task, TaskDAG, ScheduleResult."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class Task:
    """A schedulable unit of work.

    Attributes
    ----------
    id:         Unique task identifier (string or int).
    duration:   Number of time-steps the task occupies (≥ 1).
    resources:  Resource requirements, e.g. {"cpu": 2, "gpu": 1}.
                Defaults to no resource requirements.
    """

    id: str
    duration: int = 1
    resources: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.duration < 1:
            raise ValueError(f"Task {self.id}: duration must be ≥ 1, got {self.duration}")
        for k, v in self.resources.items():
            if v < 0:
                raise ValueError(f"Task {self.id}: resource '{k}' must be ≥ 0, got {v}")

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Task) and self.id == other.id


class TaskDAG:
    """Directed Acyclic Graph of Tasks.

    Edges represent *precedence* constraints: if edge (u → v) exists,
    task u must **finish** before task v **starts**.

    Example
    -------
    >>> dag = TaskDAG()
    >>> dag.add_task(Task("A", duration=2))
    >>> dag.add_task(Task("B", duration=3))
    >>> dag.add_dependency("A", "B")   # A finishes before B starts
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}
        # successors[u] = set of v where u → v (u must precede v)
        self._successors: Dict[str, Set[str]] = {}
        # predecessors[v] = set of u where u → v
        self._predecessors: Dict[str, Set[str]] = {}

    # ------------------------------------------------------------------
    # Building the graph
    # ------------------------------------------------------------------

    def add_task(self, task: Task) -> None:
        if task.id in self._tasks:
            raise ValueError(f"Task '{task.id}' already in DAG")
        self._tasks[task.id] = task
        self._successors[task.id] = set()
        self._predecessors[task.id] = set()

    def add_dependency(self, predecessor_id: str, successor_id: str) -> None:
        """Add precedence edge: predecessor must finish before successor starts."""
        for tid in (predecessor_id, successor_id):
            if tid not in self._tasks:
                raise KeyError(f"Task '{tid}' not found in DAG")
        if predecessor_id == successor_id:
            raise ValueError("Self-loop not allowed")
        self._successors[predecessor_id].add(successor_id)
        self._predecessors[successor_id].add(predecessor_id)
        # Cycle check
        if self._has_cycle():
            self._successors[predecessor_id].discard(successor_id)
            self._predecessors[successor_id].discard(predecessor_id)
            raise ValueError(
                f"Adding edge {predecessor_id} → {successor_id} creates a cycle"
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def tasks(self) -> List[Task]:
        return list(self._tasks.values())

    @property
    def task_ids(self) -> List[str]:
        return list(self._tasks.keys())

    def get_task(self, task_id: str) -> Task:
        return self._tasks[task_id]

    def predecessors(self, task_id: str) -> Set[str]:
        return set(self._predecessors.get(task_id, set()))

    def successors(self, task_id: str) -> Set[str]:
        return set(self._successors.get(task_id, set()))

    def topological_order(self) -> List[str]:
        """Return tasks in topological order (Kahn's algorithm)."""
        in_degree = {tid: len(preds) for tid, preds in self._predecessors.items()}
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        order: List[str] = []
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in self._successors[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        if len(order) != len(self._tasks):
            raise RuntimeError("DAG has a cycle — cannot produce topological order")
        return order

    def __len__(self) -> int:
        return len(self._tasks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_cycle(self) -> bool:
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for nbr in self._successors[node]:
                if nbr not in visited:
                    if dfs(nbr):
                        return True
                elif nbr in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        for tid in self._tasks:
            if tid not in visited:
                if dfs(tid):
                    return True
        return False


@dataclass
class ScheduleResult:
    """Output of any scheduler.

    Attributes
    ----------
    schedule:       task_id → start_timestep mapping.
    makespan:       Total time to complete all tasks
                    (= max(start[t] + duration[t]) over all tasks).
    solver:         Name of the solver used.
    metadata:       Solver-specific extra info (iterations, energy, …).
    feasible:       Whether all precedence + resource constraints are satisfied.
    violations:     Human-readable list of violated constraints (if any).
    """

    schedule: Dict[str, int]
    makespan: int
    solver: str
    metadata: Dict = field(default_factory=dict)
    feasible: bool = True
    violations: List[str] = field(default_factory=list)

    def finish_time(self, task: Task) -> int:
        return self.schedule[task.id] + task.duration
