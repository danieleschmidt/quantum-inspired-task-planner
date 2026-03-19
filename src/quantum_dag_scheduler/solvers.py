"""Solvers for the QUBO-formulated DAG scheduling problem.

SimulatedAnnealingScheduler — the primary solver.
  Uses simulated annealing over the QUBO binary vector, then decodes
  the solution back into a start-time schedule.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import ScheduleResult, Task, TaskDAG
from .qubo import QUBOProblem, QUBOScheduler


class SimulatedAnnealingScheduler:
    """Solve DAG scheduling via QUBO + Simulated Annealing.

    The solver runs in two phases:
    1. **SA phase**: optimise the QUBO binary vector via simulated annealing,
       starting from a greedy-feasible initialisation.
    2. **Repair phase**: apply list scheduling to fix any residual precedence
       violations in the decoded schedule (a standard hybrid quantum-classical
       post-processing step).

    Parameters
    ----------
    max_timesteps:  Schedule horizon (passed to QUBOScheduler).
    T_init:         Initial annealing temperature.
    T_final:        Final annealing temperature.
    n_iterations:   Number of SA iterations.
    penalty_start:  QUBO one-hot penalty weight.
    penalty_prec:   QUBO precedence penalty weight.
    penalty_res:    QUBO resource penalty weight.
    alpha:          Makespan bias coefficient.
    repair:         If True (default), apply list-scheduling repair after SA
                    to guarantee precedence feasibility.
    seed:           Random seed for reproducibility.
    """

    def __init__(
        self,
        max_timesteps: int = 25,
        T_init: float = 10.0,
        T_final: float = 0.01,
        n_iterations: int = 50_000,
        penalty_start: float = 10.0,
        penalty_prec: float = 8.0,
        penalty_res: float = 6.0,
        alpha: float = 0.2,
        repair: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.max_timesteps = max_timesteps
        self.T_init = T_init
        self.T_final = T_final
        self.n_iterations = n_iterations
        self.penalty_start = penalty_start
        self.penalty_prec = penalty_prec
        self.penalty_res = penalty_res
        self.alpha = alpha
        self.repair = repair
        self.seed = seed

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def schedule(
        self,
        dag: TaskDAG,
        resource_capacity: Optional[Dict[str, int]] = None,
    ) -> ScheduleResult:
        """Build QUBO and run SA to find a schedule.

        Parameters
        ----------
        dag:                TaskDAG to schedule.
        resource_capacity:  {resource: total_capacity}.

        Returns
        -------
        ScheduleResult with start-time assignments.
        """
        t0 = time.perf_counter()

        qubo_builder = QUBOScheduler(
            max_timesteps=self.max_timesteps,
            penalty_start=self.penalty_start,
            penalty_prec=self.penalty_prec,
            penalty_res=self.penalty_res,
            alpha=self.alpha,
        )
        problem = qubo_builder.build(dag, resource_capacity)

        x_best, energy_best, energy_history = self._run_sa(problem)
        sa_schedule = problem.decode(x_best)

        # --- Phase 2: hybrid repair via resource-aware list scheduling ---
        # Use SA-decoded start times as task priorities, then run a
        # resource-aware list scheduler to produce a fully feasible schedule.
        # This is the standard hybrid quantum-classical post-processing step.
        if self.repair:
            schedule_map = _list_schedule(dag, sa_schedule, resource_capacity or {})
        else:
            schedule_map = sa_schedule

        elapsed = time.perf_counter() - t0

        # Compute makespan
        makespan = max(
            schedule_map[t.id] + t.duration for t in dag.tasks
        )

        result = ScheduleResult(
            schedule=schedule_map,
            makespan=makespan,
            solver="SimulatedAnnealing",
            metadata={
                "qubo_energy": float(energy_best),
                "n_vars": problem.n_vars,
                "n_iterations": self.n_iterations,
                "elapsed_s": round(elapsed, 4),
                "energy_history_len": len(energy_history),
                "repair_applied": self.repair,
            },
        )

        # Validate feasibility
        violations = _check_feasibility(dag, schedule_map, resource_capacity)
        result.feasible = len(violations) == 0
        result.violations = violations
        return result

    # ------------------------------------------------------------------
    # SA core
    # ------------------------------------------------------------------

    def _run_sa(
        self, problem: QUBOProblem
    ) -> Tuple[np.ndarray, float, List[float]]:
        rng = random.Random(self.seed)
        n = problem.n_vars
        Q = problem.Q

        # Initialise with a greedy feasible solution
        x = self._greedy_init(problem)
        energy = _qubo_energy(Q, x)

        x_best = x.copy()
        energy_best = energy

        T = self.T_init
        cooling = (self.T_final / self.T_init) ** (1.0 / max(self.n_iterations, 1))

        energy_history: List[float] = []
        log_interval = max(self.n_iterations // 20, 1)

        for step in range(self.n_iterations):
            # Flip a random bit
            flip_idx = rng.randrange(n)
            delta = _delta_energy(Q, x, flip_idx)

            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                x[flip_idx] ^= 1
                energy += delta

            if energy < energy_best:
                x_best = x.copy()
                energy_best = energy

            T *= cooling
            if step % log_interval == 0:
                energy_history.append(energy)

        return x_best, energy_best, energy_history

    def _greedy_init(self, problem: QUBOProblem) -> np.ndarray:
        """Greedy feasible initialisation using topological order.

        Assigns each task the earliest start time respecting precedence.
        Ignores resource constraints here (SA will fix them).
        """
        x = np.zeros(problem.n_vars, dtype=np.int8)
        dag = problem.dag
        earliest: Dict[str, int] = {}

        for tid in dag.topological_order():
            task = dag.get_task(tid)
            preds = dag.predecessors(tid)
            est = 0
            for pid in preds:
                pred_task = dag.get_task(pid)
                est = max(est, earliest.get(pid, 0) + pred_task.duration)
            # Clamp to valid range
            est = min(est, problem.max_timesteps - task.duration)
            est = max(est, 0)
            earliest[tid] = est
            key = (tid, est)
            if key in problem.var_index:
                x[problem.var_index[key]] = 1

        return x


# ------------------------------------------------------------------
# Energy helpers (standalone for speed)
# ------------------------------------------------------------------

def _qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    """Compute x^T Q x efficiently."""
    return float(x @ Q @ x)


def _delta_energy(Q: np.ndarray, x: np.ndarray, flip_idx: int) -> float:
    """Energy change when flipping x[flip_idx] (0↔1).

    ΔE = (1 - 2·x_i) · (Q_ii + Σ_{j≠i} (Q_ij + Q_ji) · x_j)
    Since Q is upper-triangular:
        row contribution = Q[flip_idx, :] · x
        col contribution = Q[:, flip_idx] · x (below diagonal, stored as 0)
    """
    xi = x[flip_idx]
    row = Q[flip_idx, :] @ x
    col = Q[:, flip_idx] @ x
    diag = Q[flip_idx, flip_idx] * xi  # counted in both row and col above
    linear = row + col - diag  # Σ Q[i,j] x_j + Σ Q[j,i] x_j  - double-count diag
    delta = (1 - 2 * xi) * linear
    return float(delta)


# ------------------------------------------------------------------
# Hybrid repair: resource-aware list scheduler
# ------------------------------------------------------------------

def _list_schedule(
    dag: TaskDAG,
    sa_schedule: Dict[str, int],
    resource_capacity: Dict[str, int],
) -> Dict[str, int]:
    """Resource-aware list scheduling using SA priorities.

    This is the classical post-processing step in hybrid quantum-classical
    scheduling. The SA QUBO solution provides a priority ordering for tasks;
    a list scheduler then assigns start times respecting both precedence
    and resource constraints.

    Priority rule: tasks with earlier SA-suggested start times are scheduled
    first (within topological feasibility).

    Algorithm
    ---------
    1. Compute topological levels (tasks at level 0 have no predecessors).
    2. At each time step t = 0, 1, …, schedule the highest-priority ready
       task that fits in remaining resource capacity.
    """
    # Sort tasks by SA-suggested start time as priority
    sa_priority: Dict[str, float] = {
        tid: sa_schedule.get(tid, 0) for tid in dag.task_ids
    }

    finish_time: Dict[str, int] = {}
    start_time: Dict[str, int] = {}
    scheduled: set = set()

    # Resource usage: resource → list of (start, end, usage) intervals
    # We track per-timestep usage via a simple dict
    res_usage: Dict[str, Dict[int, int]] = {r: {} for r in resource_capacity}

    def resource_available(t: int, task: Task) -> bool:
        for res, cap in resource_capacity.items():
            req = task.resources.get(res, 0)
            if req == 0:
                continue
            for ts in range(t, t + task.duration):
                if res_usage[res].get(ts, 0) + req > cap:
                    return False
        return True

    def reserve(t: int, task: Task) -> None:
        for res in resource_capacity:
            req = task.resources.get(res, 0)
            if req == 0:
                continue
            for ts in range(t, t + task.duration):
                res_usage[res][ts] = res_usage[res].get(ts, 0) + req

    t = 0
    max_t = sum(task.duration for task in dag.tasks) * 2 + 10  # upper bound

    while len(scheduled) < len(dag.task_ids) and t < max_t:
        # Ready tasks: not scheduled, all predecessors done by time t
        ready = [
            tid for tid in dag.task_ids
            if tid not in scheduled
            and all(
                finish_time.get(pid, max_t) <= t
                for pid in dag.predecessors(tid)
            )
        ]
        # Sort by SA priority (ascending = earlier SA start = higher priority)
        ready.sort(key=lambda tid: sa_priority[tid])

        for tid in ready:
            task = dag.get_task(tid)
            if resource_available(t, task):
                start_time[tid] = t
                finish_time[tid] = t + task.duration
                reserve(t, task)
                scheduled.add(tid)

        t += 1

    # Any unscheduled tasks (shouldn't happen) get placed at the end
    for tid in dag.task_ids:
        if tid not in start_time:
            start_time[tid] = t

    return start_time


# ------------------------------------------------------------------
# Feasibility checker
# ------------------------------------------------------------------

def _check_feasibility(
    dag: TaskDAG,
    schedule: Dict[str, int],
    resource_capacity: Optional[Dict[str, int]],
) -> List[str]:
    violations: List[str] = []

    # Precedence
    for uid in dag.task_ids:
        u = dag.get_task(uid)
        u_finish = schedule[uid] + u.duration
        for vid in dag.successors(uid):
            v_start = schedule[vid]
            if v_start < u_finish:
                violations.append(
                    f"Precedence: {uid} finishes at {u_finish}, "
                    f"but {vid} starts at {v_start}"
                )

    # Resource capacity
    if resource_capacity:
        max_t = max(schedule[t.id] + t.duration for t in dag.tasks)
        for res, cap in resource_capacity.items():
            for t in range(max_t):
                usage = 0
                for task in dag.tasks:
                    start = schedule[task.id]
                    if start <= t < start + task.duration:
                        usage += task.resources.get(res, 0)
                if usage > cap:
                    violations.append(
                        f"Resource '{res}' overloaded at t={t}: {usage} > {cap}"
                    )

    return violations
