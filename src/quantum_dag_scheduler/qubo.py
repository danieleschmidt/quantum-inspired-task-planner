"""QUBO formulation for DAG scheduling.

Decision variables
------------------
    x[i][t] ∈ {0, 1}  — task i starts at timestep t

QUBO penalty terms
------------------
1. **Single-start constraint** (one-hot per task):
       P_start · Σ_i (1 - Σ_t x[i][t])²
   Penalises tasks that start zero or more than once.

2. **Precedence constraints** (DAG edges u → v):
       P_prec · Σ_(u,v)∈E  Σ_{t_u,t_v: t_v < t_u + dur_u} x[u][t_u] · x[v][t_v]
   Penalises v starting before u has finished.

3. **Resource capacity** (per resource r, per timestep t):
       P_res  · Σ_r Σ_t ( Σ_{i,s: s≤t<s+dur_i} x[i][s] · req[i][r] - cap[r] )²
   Expanded into QUBO quadratic form — penalises over-capacity usage.

4. **Makespan minimisation**:
   Add a linear term that rewards early completion of each task:
       -α · Σ_i Σ_t (T - t) · x[i][t]
   where T = max_timesteps - 1. This biases the solver toward early starts.

Usage
-----
    qubo_prob = QUBOScheduler(max_timesteps=15, penalty_start=10,
                               penalty_prec=8, penalty_res=6, alpha=0.1)
    Q, var_index = qubo_prob.build(dag, resource_capacity={"cpu": 4})
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import Task, TaskDAG


@dataclass
class QUBOProblem:
    """Result of QUBO formulation.

    Attributes
    ----------
    Q:          Upper-triangular QUBO matrix (n×n numpy array).
                Minimise  x^T Q x  subject to x ∈ {0,1}^n.
    var_index:  Maps (task_id, timestep) → column index in Q.
    n_vars:     Total number of binary variables.
    dag:        Source DAG (reference, not copied).
    max_timesteps: Horizon used.
    """

    Q: np.ndarray
    var_index: Dict[Tuple[str, int], int]
    n_vars: int
    dag: TaskDAG
    max_timesteps: int

    def idx(self, task_id: str, t: int) -> int:
        return self.var_index[(task_id, t)]

    def decode(self, x: np.ndarray) -> Dict[str, int]:
        """Convert binary solution vector → {task_id: start_timestep}.

        If a task has no active bit (infeasible), assigns the latest
        timestep as a fallback.
        """
        schedule: Dict[str, int] = {}
        for tid in self.dag.task_ids:
            start = None
            for t in range(self.max_timesteps):
                key = (tid, t)
                if key in self.var_index and x[self.var_index[key]] > 0.5:
                    start = t
                    break
            if start is None:
                # Fallback: last valid timestep
                start = self.max_timesteps - 1
            schedule[tid] = start
        return schedule


class QUBOScheduler:
    """Converts a TaskDAG into a QUBO matrix.

    Parameters
    ----------
    max_timesteps:  Schedule horizon (number of discrete time slots).
    penalty_start:  Weight for one-hot-per-task constraint (P1).
    penalty_prec:   Weight for precedence constraint (P2).
    penalty_res:    Weight for resource-capacity constraint (P3).
    alpha:          Makespan bias coefficient (P4, smaller = lighter nudge).
    """

    def __init__(
        self,
        max_timesteps: int = 20,
        penalty_start: float = 10.0,
        penalty_prec: float = 8.0,
        penalty_res: float = 6.0,
        alpha: float = 0.2,
    ) -> None:
        self.max_timesteps = max_timesteps
        self.penalty_start = penalty_start
        self.penalty_prec = penalty_prec
        self.penalty_res = penalty_res
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        dag: TaskDAG,
        resource_capacity: Optional[Dict[str, int]] = None,
    ) -> QUBOProblem:
        """Build the QUBO problem for the given DAG.

        Parameters
        ----------
        dag:                TaskDAG to schedule.
        resource_capacity:  Dict[resource_name → total capacity].
                            Default: unlimited (resource penalties skipped).

        Returns
        -------
        QUBOProblem with populated Q matrix and variable index.
        """
        resource_capacity = resource_capacity or {}
        T = self.max_timesteps
        task_ids = dag.task_ids

        # Build variable index: (task_id, t) → integer index
        var_index: Dict[Tuple[str, int], int] = {}
        idx = 0
        for tid in task_ids:
            task = dag.get_task(tid)
            # task can only start at t ≤ T - duration
            for t in range(T - task.duration + 1):
                var_index[(tid, t)] = idx
                idx += 1
        n_vars = idx

        Q = np.zeros((n_vars, n_vars), dtype=np.float64)

        # --- P1: one-hot constraint (each task starts exactly once) ---
        self._add_one_hot(Q, var_index, task_ids, dag)

        # --- P2: precedence constraints ---
        self._add_precedence(Q, var_index, dag)

        # --- P3: resource capacity ---
        if resource_capacity:
            self._add_resource(Q, var_index, dag, resource_capacity)

        # --- P4: makespan minimisation bias ---
        self._add_makespan_bias(Q, var_index, dag)

        return QUBOProblem(
            Q=Q,
            var_index=var_index,
            n_vars=n_vars,
            dag=dag,
            max_timesteps=T,
        )

    # ------------------------------------------------------------------
    # Private penalty builders
    # ------------------------------------------------------------------

    def _add_one_hot(
        self,
        Q: np.ndarray,
        var_index: Dict[Tuple[str, int], int],
        task_ids: List[str],
        dag: TaskDAG,
    ) -> None:
        """P1: (1 - Σ_t x[i][t])² for each task i.

        Expands to:
            1 (ignored — constant)
            - 2·x[i][t]  (diagonal)
            + 2·x[i][t]·x[i][t'] for t ≠ t' (off-diagonal)
        """
        A = self.penalty_start
        for tid in task_ids:
            task = dag.get_task(tid)
            valid_ts = [t for t in range(self.max_timesteps) if (tid, t) in var_index]
            # Diagonal terms: -2·A per variable (from -2·x) + A·x² = A·x (binary)
            # Compact: diagonal += A * (1 - 2) = -A, but cross terms add +2A each pair
            # Correct expansion of (1 - Σx)²:
            #   = 1 - 2Σx + (Σx)²
            #   = 1 - 2Σx + Σx_i² + 2Σ_{i<j}x_i x_j
            # x_i² = x_i (binary), so diagonal contribution = -2A + A = -A per var
            for t in valid_ts:
                i = var_index[(tid, t)]
                Q[i, i] += -A  # (-2 + 1) * A
            # Off-diagonal: +2A per pair
            for si, ti in enumerate(valid_ts):
                for tj in valid_ts[si + 1:]:
                    i = var_index[(tid, ti)]
                    j = var_index[(tid, tj)]
                    a, b = min(i, j), max(i, j)
                    Q[a, b] += 2 * A

    def _add_precedence(
        self,
        Q: np.ndarray,
        var_index: Dict[Tuple[str, int], int],
        dag: TaskDAG,
    ) -> None:
        """P2: for each edge (u → v), penalise t_v < t_u + dur_u.

        Adds P_prec · x[u][t_u] · x[v][t_v] for every (t_u, t_v) pair
        where the constraint would be violated.
        """
        B = self.penalty_prec
        for uid in dag.task_ids:
            task_u = dag.get_task(uid)
            for vid in dag.successors(uid):
                for tu in range(self.max_timesteps):
                    if (uid, tu) not in var_index:
                        continue
                    i = var_index[(uid, tu)]
                    # v must start at tv >= tu + dur_u
                    min_tv = tu + task_u.duration
                    for tv in range(min_tv):
                        if (vid, tv) not in var_index:
                            continue
                        j = var_index[(vid, tv)]
                        a, b = min(i, j), max(i, j)
                        Q[a, b] += B

    def _add_resource(
        self,
        Q: np.ndarray,
        var_index: Dict[Tuple[str, int], int],
        dag: TaskDAG,
        resource_capacity: Dict[str, int],
    ) -> None:
        """P3: resource capacity constraint.

        For each resource r and each timestep t, the total usage must
        not exceed capacity[r].

        Penalty term: C · (Σ_i w_i · y_i(t) - cap)²
        where y_i(t) = Σ_{s≤t<s+dur_i} x[i][s] (task i is active at t).

        We expand this quadratic and add to Q.
        """
        C = self.penalty_res
        T = self.max_timesteps
        task_ids = dag.task_ids

        for res, cap in resource_capacity.items():
            for t in range(T):
                # Collect (var_index, weight) pairs for tasks active at time t
                active: List[Tuple[int, int]] = []
                for tid in task_ids:
                    task = dag.get_task(tid)
                    req = task.resources.get(res, 0)
                    if req == 0:
                        continue
                    # x[i][s] is active at t if s <= t < s + dur_i  ⟺  t-dur_i < s <= t
                    for s in range(max(0, t - task.duration + 1), t + 1):
                        if (tid, s) in var_index:
                            active.append((var_index[(tid, s)], req))

                if not active:
                    continue

                # (Σ w_i x_i - cap)² = Σ w_i² x_i² + 2Σ_{i<j} w_i w_j x_i x_j
                #                       - 2 cap Σ w_i x_i + cap²  (constant ignored)
                # For binary: x_i² = x_i → diagonal += C*(w_i² - 2*cap*w_i)
                for idx_i, wi in active:
                    Q[idx_i, idx_i] += C * (wi * wi - 2 * cap * wi)
                # Cross terms
                for si, (idx_i, wi) in enumerate(active):
                    for idx_j, wj in active[si + 1:]:
                        a, b = min(idx_i, idx_j), max(idx_i, idx_j)
                        Q[a, b] += 2 * C * wi * wj

    def _add_makespan_bias(
        self,
        Q: np.ndarray,
        var_index: Dict[Tuple[str, int], int],
        dag: TaskDAG,
    ) -> None:
        """P4: linear bias to favour early start times.

        Adds -alpha * (T - t - dur_i) to the diagonal of x[i][t],
        so later starts have higher cost.
        """
        T = self.max_timesteps
        for tid in dag.task_ids:
            task = dag.get_task(tid)
            for t in range(T):
                if (tid, t) not in var_index:
                    continue
                idx_it = var_index[(tid, t)]
                # finish time t + dur_i; later finish → higher penalty
                finish = t + task.duration
                Q[idx_it, idx_it] += self.alpha * finish
