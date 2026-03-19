"""Tests for TaskDAG and related models."""

import pytest
from quantum_dag_scheduler import Task, TaskDAG, ScheduleResult


# ------------------------------------------------------------------
# Task
# ------------------------------------------------------------------

def test_task_defaults():
    t = Task("A")
    assert t.duration == 1
    assert t.resources == {}


def test_task_invalid_duration():
    with pytest.raises(ValueError):
        Task("A", duration=0)


def test_task_invalid_resource():
    with pytest.raises(ValueError):
        Task("A", resources={"cpu": -1})


def test_task_equality():
    assert Task("A", 2) == Task("A", 3)   # id-based equality
    assert Task("A") != Task("B")


# ------------------------------------------------------------------
# TaskDAG
# ------------------------------------------------------------------

def make_simple_dag():
    dag = TaskDAG()
    dag.add_task(Task("A", duration=2))
    dag.add_task(Task("B", duration=3))
    dag.add_task(Task("C", duration=1))
    dag.add_dependency("A", "C")
    dag.add_dependency("B", "C")
    return dag


def test_dag_add_task():
    dag = TaskDAG()
    dag.add_task(Task("X"))
    assert "X" in dag.task_ids


def test_dag_duplicate_task():
    dag = TaskDAG()
    dag.add_task(Task("X"))
    with pytest.raises(ValueError):
        dag.add_task(Task("X"))


def test_dag_dependency():
    dag = make_simple_dag()
    assert "A" in dag.predecessors("C")
    assert "B" in dag.predecessors("C")
    assert "C" in dag.successors("A")


def test_dag_self_loop():
    dag = TaskDAG()
    dag.add_task(Task("X"))
    with pytest.raises(ValueError):
        dag.add_dependency("X", "X")


def test_dag_cycle_detection():
    dag = TaskDAG()
    dag.add_task(Task("A"))
    dag.add_task(Task("B"))
    dag.add_dependency("A", "B")
    with pytest.raises(ValueError):
        dag.add_dependency("B", "A")


def test_dag_topological_order():
    dag = make_simple_dag()
    order = dag.topological_order()
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("C")


def test_dag_missing_task_in_dependency():
    dag = TaskDAG()
    dag.add_task(Task("A"))
    with pytest.raises(KeyError):
        dag.add_dependency("A", "MISSING")
