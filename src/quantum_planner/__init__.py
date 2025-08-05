"""Quantum Task Planner - A quantum-inspired task scheduling system."""

__version__ = "0.1.0"
__author__ = "Quantum Planning Team"

from .models import Agent, Task, TimeWindowTask, Solution
from .optimizer import (
    OptimizationBackend,
    OptimizationParams,
    OptimizerFactory,
    create_optimizer,
    optimize_tasks
)
from .planner import QuantumTaskPlanner, PlannerConfig

__all__ = [
    "Agent",
    "Task", 
    "TimeWindowTask",
    "Solution",
    "OptimizationBackend",
    "OptimizationParams",
    "OptimizerFactory",
    "create_optimizer",
    "optimize_tasks",
    "QuantumTaskPlanner",
    "PlannerConfig"
]