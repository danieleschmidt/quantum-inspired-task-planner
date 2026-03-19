"""Tests for QUBO formulation."""

import numpy as np
import pytest
from quantum_dag_scheduler import Task, TaskDAG
from quantum_dag_scheduler.qubo import QUBOScheduler


def make_linear_dag():
    """Simple A → B → C chain."""
    dag = TaskDAG()
    dag.add_task(Task("A", duration=2, resources={"cpu": 1}))
    dag.add_task(Task("B", duration=2, resources={"cpu": 1}))
    dag.add_task(Task("C", duration=1, resources={"cpu": 1}))
    dag.add_dependency("A", "B")
    dag.add_dependency("B", "C")
    return dag


def test_qubo_builds():
    dag = make_linear_dag()
    builder = QUBOScheduler(max_timesteps=8)
    prob = builder.build(dag, resource_capacity={"cpu": 2})
    assert prob.Q.shape[0] == prob.Q.shape[1] == prob.n_vars
    assert prob.n_vars > 0


def test_qubo_upper_triangular():
    """Q matrix should be upper-triangular (no lower triangle entries)."""
    dag = make_linear_dag()
    builder = QUBOScheduler(max_timesteps=8)
    prob = builder.build(dag)
    lower = np.tril(prob.Q, k=-1)
    assert np.allclose(lower, 0), "Q matrix has non-zero lower-triangle entries"


def test_qubo_var_index_coverage():
    """Every valid (task, timestep) pair should be in var_index."""
    dag = make_linear_dag()
    T = 8
    builder = QUBOScheduler(max_timesteps=T)
    prob = builder.build(dag)
    for tid in dag.task_ids:
        task = dag.get_task(tid)
        max_start = T - task.duration
        for t in range(max_start + 1):
            assert (tid, t) in prob.var_index, f"Missing ({tid}, {t})"


def test_qubo_decode_valid():
    """A one-hot solution vector should decode to a proper schedule."""
    dag = make_linear_dag()
    builder = QUBOScheduler(max_timesteps=8)
    prob = builder.build(dag)

    x = np.zeros(prob.n_vars, dtype=np.int8)
    # Manually assign: A starts at 0, B at 2, C at 4
    x[prob.var_index[("A", 0)]] = 1
    x[prob.var_index[("B", 2)]] = 1
    x[prob.var_index[("C", 4)]] = 1

    schedule = prob.decode(x)
    assert schedule["A"] == 0
    assert schedule["B"] == 2
    assert schedule["C"] == 4


def test_greedy_init_respects_precedence():
    """Greedy init should produce a precedence-feasible solution."""
    from quantum_dag_scheduler.solvers import SimulatedAnnealingScheduler, _check_feasibility
    dag = make_linear_dag()
    sa = SimulatedAnnealingScheduler(max_timesteps=10, n_iterations=0, seed=0)
    from quantum_dag_scheduler.qubo import QUBOScheduler as QS
    prob = QS(max_timesteps=10).build(dag)
    x = sa._greedy_init(prob)
    schedule = prob.decode(x)
    violations = _check_feasibility(dag, schedule, None)
    prec_viols = [v for v in violations if "Precedence" in v]
    assert len(prec_viols) == 0, f"Greedy init violates precedence: {prec_viols}"
