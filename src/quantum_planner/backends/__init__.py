"""Quantum and classical optimization backends."""

from .base import BaseBackend, BackendType, BackendInfo
from .dwave_backend import DWaveBackend
from .azure_backend import AzureQuantumBackend
from .ibm_backend import IBMQuantumBackend
from .classical_backends import (
    SimulatedAnnealingBackend,
    GeneticAlgorithmBackend,
    TabuSearchBackend
)
from .simulator_backend import QuantumSimulatorBackend

__all__ = [
    "BaseBackend",
    "BackendType", 
    "BackendInfo",
    "DWaveBackend",
    "AzureQuantumBackend",
    "IBMQuantumBackend",
    "SimulatedAnnealingBackend",
    "GeneticAlgorithmBackend",
    "TabuSearchBackend",
    "QuantumSimulatorBackend"
]