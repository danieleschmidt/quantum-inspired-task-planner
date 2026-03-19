"""Tests for SA scheduler and CPM baseline."""

import pytest
from quantum_dag_scheduler import (
    Task, TaskDAG,
    SimulatedAnnealingScheduler,
    CriticalPathBaseline,
)


# ------------------------------------------------------------------
# Helper DAGs
# ------------------------------------------------------------------

def make_chain_dag(n: int = 4) -> TaskDAG:
    """Linear chain: T0 → T1 → ... → T(n-1)."""
    dag = TaskDAG()
    for i in range(n):
        dag.add_task(Task(f"T{i}", duration=2, resources={"cpu": 1}))
    for i in range(n - 1):
        dag.add_dependency(f"T{i}", f"T{i+1}")
    return dag


def make_parallel_dag() -> TaskDAG:
    """Three independent tasks, then a join."""
    dag = TaskDAG()
    dag.add_task(Task("A", duration=3, resources={"cpu": 2}))
    dag.add_task(Task("B", duration=2, resources={"cpu": 1}))
    dag.add_task(Task("C", duration=1, resources={"cpu": 1}))
    dag.add_task(Task("JOIN", duration=1, resources={"cpu": 1}))
    dag.add_dependency("A", "JOIN")
    dag.add_dependency("B", "JOIN")
    dag.add_dependency("C", "JOIN")
    return dag


# ------------------------------------------------------------------
# CPM Baseline
# ------------------------------------------------------------------

def test_cpm_chain():
    dag = make_chain_dag(4)
    cpm = CriticalPathBaseline()
    result = cpm.schedule(dag)
    assert result.makespan == 8   # 4 tasks × duration 2
    assert result.feasible


def test_cpm_parallel():
    dag = make_parallel_dag()
    cpm = CriticalPathBaseline()
    result = cpm.schedule(dag)
    # A is the longest (3), JOIN adds 1 → makespan = 4
    assert result.makespan == 4
    assert result.feasible


def test_cpm_critical_path():
    dag = make_parallel_dag()
    result = CriticalPathBaseline().schedule(dag)
    cp = result.metadata["critical_path"]
    # A and JOIN should be on critical path (A is longest predecessor)
    assert "A" in cp
    assert "JOIN" in cp


def test_cpm_precedence_satisfied():
    dag = make_chain_dag(4)
    result = CriticalPathBaseline().schedule(dag)
    sched = result.schedule
    for i in range(3):
        assert sched[f"T{i}"] + 2 <= sched[f"T{i+1}"]


# ------------------------------------------------------------------
# SA Scheduler
# ------------------------------------------------------------------

def test_sa_chain_feasible():
    dag = make_chain_dag(3)
    sa = SimulatedAnnealingScheduler(
        max_timesteps=10, n_iterations=20_000, seed=0
    )
    result = sa.schedule(dag, resource_capacity={"cpu": 2})
    assert result.makespan > 0
    # Precedence must be respected
    prec_viols = [v for v in result.violations if "Precedence" in v]
    assert len(prec_viols) == 0, f"SA violated precedence: {prec_viols}"


def test_sa_parallel_resources():
    """SA should produce a valid schedule on a parallel DAG with resources."""
    dag = make_parallel_dag()
    sa = SimulatedAnnealingScheduler(
        max_timesteps=14, n_iterations=60_000, seed=1
    )
    result = sa.schedule(dag, resource_capacity={"cpu": 2})
    assert result.makespan > 0
    prec_viols = [v for v in result.violations if "Precedence" in v]
    assert len(prec_viols) == 0


def test_sa_makespan_vs_cpm():
    """SA makespan should be >= CPM makespan (CPM is optimal for precedence)."""
    dag = make_chain_dag(4)
    cpm = CriticalPathBaseline()
    sa = SimulatedAnnealingScheduler(max_timesteps=12, n_iterations=30_000, seed=42)
    cpm_result = cpm.schedule(dag)
    sa_result = sa.schedule(dag, resource_capacity={"cpu": 4})
    assert sa_result.makespan >= cpm_result.makespan


def test_sa_no_resource_constraint():
    """Without resource constraints, SA should approach CPM makespan."""
    dag = make_parallel_dag()
    sa = SimulatedAnnealingScheduler(
        max_timesteps=10, n_iterations=50_000, seed=7
    )
    result = sa.schedule(dag)   # no resource_capacity
    assert result.makespan > 0
    prec_viols = [v for v in result.violations if "Precedence" in v]
    assert len(prec_viols) == 0


def test_sa_result_fields():
    dag = make_chain_dag(2)
    sa = SimulatedAnnealingScheduler(max_timesteps=8, n_iterations=5_000, seed=0)
    result = sa.schedule(dag)
    assert "qubo_energy" in result.metadata
    assert "elapsed_s" in result.metadata
    assert result.solver == "SimulatedAnnealing"
