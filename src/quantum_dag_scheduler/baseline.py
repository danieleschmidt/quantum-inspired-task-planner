"""Critical Path Method (CPM) baseline scheduler.

Provides the classical lower-bound on makespan for comparison against
the QUBO-SA solver. Ignores resource constraints (pure precedence-based).

References
----------
- Kelley, J.E. (1961). "Critical-path planning and scheduling."
  Proceedings of the Eastern Joint Computer Conference.
"""

from __future__ import annotations

from typing import Dict, Optional

from .models import ScheduleResult, TaskDAG


class CriticalPathBaseline:
    """Compute schedule via Critical Path Method (CPM).

    The CPM schedule minimises makespan subject to *precedence* constraints
    only — it does not respect resource capacity. It therefore gives the
    theoretical lower-bound makespan for any precedence-constrained problem.

    Usage
    -----
    >>> baseline = CriticalPathBaseline()
    >>> result = baseline.schedule(dag)
    >>> print(result.makespan)
    """

    def schedule(
        self,
        dag: TaskDAG,
        resource_capacity: Optional[Dict[str, int]] = None,
    ) -> ScheduleResult:
        """Compute the earliest-start schedule respecting precedence only.

        Parameters
        ----------
        dag:                TaskDAG to schedule.
        resource_capacity:  Included for API compatibility with SA solver.
                            CPM ignores resource constraints.

        Returns
        -------
        ScheduleResult with CPM start times and makespan.
        """
        # Forward pass: compute Earliest Start Time (EST) for each task
        est: Dict[str, int] = {}
        for tid in dag.topological_order():
            task = dag.get_task(tid)
            preds = dag.predecessors(tid)
            if not preds:
                est[tid] = 0
            else:
                est[tid] = max(
                    est[pid] + dag.get_task(pid).duration for pid in preds
                )

        makespan = max(est[t.id] + t.duration for t in dag.tasks)

        # Backward pass: compute Latest Start Time (LST) and float
        lst: Dict[str, int] = {}
        for tid in reversed(dag.topological_order()):
            task = dag.get_task(tid)
            succs = dag.successors(tid)
            if not succs:
                lst[tid] = makespan - task.duration
            else:
                lst[tid] = min(
                    lst[sid] - task.duration for sid in succs
                )

        # Total float (slack) per task
        total_float = {tid: lst[tid] - est[tid] for tid in dag.task_ids}
        critical_path = [tid for tid, f in total_float.items() if f == 0]

        return ScheduleResult(
            schedule=est,
            makespan=makespan,
            solver="CPM",
            metadata={
                "critical_path": critical_path,
                "total_float": total_float,
                "note": "Resource constraints ignored (CPM lower bound)",
            },
            feasible=True,
            violations=[],
        )
