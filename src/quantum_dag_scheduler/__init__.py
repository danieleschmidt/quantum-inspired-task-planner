"""Quantum-Inspired DAG Scheduler.

Formulates DAG scheduling (with precedence constraints) as a QUBO problem
and solves with simulated annealing. Includes a classical CPM baseline.

Core API
--------
    from quantum_dag_scheduler import TaskDAG, Task, QUBOScheduler, SimulatedAnnealingScheduler
    from quantum_dag_scheduler import CriticalPathBaseline

    dag = TaskDAG()
    dag.add_task(Task(id="A", duration=3, resources={"cpu": 2}))
    dag.add_task(Task(id="B", duration=2, resources={"cpu": 1}))
    dag.add_dependency("A", "B")   # A must finish before B starts

    scheduler = SimulatedAnnealingScheduler(max_timesteps=20)
    result = scheduler.schedule(dag, resource_capacity={"cpu": 4})
    print(result.makespan, result.schedule)
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"

from .models import Task, TaskDAG, ScheduleResult
from .qubo import QUBOScheduler, QUBOProblem
from .solvers import SimulatedAnnealingScheduler
from .baseline import CriticalPathBaseline

__all__ = [
    "Task",
    "TaskDAG",
    "ScheduleResult",
    "QUBOScheduler",
    "QUBOProblem",
    "SimulatedAnnealingScheduler",
    "CriticalPathBaseline",
]
