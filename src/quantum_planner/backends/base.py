"""Base backend interface for quantum and classical optimization."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Available backend types."""
    QUANTUM_ANNEALING = "quantum_annealing"
    GATE_BASED_QUANTUM = "gate_based_quantum"
    CLASSICAL_EXACT = "classical_exact"
    CLASSICAL_HEURISTIC = "classical_heuristic"
    SIMULATOR = "simulator"


@dataclass
class BackendInfo:
    """Information about a backend's capabilities."""
    name: str
    backend_type: BackendType
    max_variables: int
    connectivity: Optional[str] = None
    availability: bool = True
    cost_per_sample: float = 0.0
    typical_solve_time: float = 1.0
    supports_embedding: bool = False
    supports_constraints: bool = True
    
    def __str__(self) -> str:
        return (f"{self.name} ({self.backend_type.value}): "
                f"{self.max_variables} vars, "
                f"${self.cost_per_sample:.4f}/sample")


@dataclass
class OptimizationResult:
    """Result from backend optimization."""
    solution: Dict[int, int]  # variable_index -> value (0 or 1)
    energy: float
    num_samples: int
    timing: Dict[str, float]
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None


class BaseBackend(ABC):
    """Abstract base class for optimization backends."""
    
    def __init__(self, name: str, backend_type: BackendType):
        self.name = name
        self.backend_type = backend_type
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._is_available = True
        self._last_error: Optional[str] = None
    
    @abstractmethod
    def get_backend_info(self) -> BackendInfo:
        """Get information about this backend's capabilities."""
        pass
    
    @abstractmethod
    def solve_qubo(
        self,
        Q: np.ndarray,
        num_reads: int = 1000,
        **kwargs
    ) -> OptimizationResult:
        """Solve QUBO problem."""
        pass
    
    @abstractmethod
    def estimate_solve_time(self, problem_size: int) -> float:
        """Estimate solving time for given problem size."""
        pass
    
    def validate_problem(self, Q: np.ndarray) -> Tuple[bool, str]:
        """Validate if problem can be solved by this backend."""
        if Q.shape[0] != Q.shape[1]:
            return False, "QUBO matrix must be square"
        
        if Q.shape[0] == 0:
            return False, "Empty QUBO matrix"
        
        if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):
            return False, "QUBO matrix contains NaN or infinite values"
        
        backend_info = self.get_backend_info()
        if Q.shape[0] > backend_info.max_variables:
            return False, f"Problem size ({Q.shape[0]}) exceeds backend limit ({backend_info.max_variables})"
        
        return True, "Problem is valid"
    
    def is_available(self) -> bool:
        """Check if backend is currently available."""
        return self._is_available
    
    def set_availability(self, available: bool, error_message: Optional[str] = None):
        """Set backend availability status."""
        self._is_available = available
        self._last_error = error_message
        if not available and error_message:
            self.logger.warning(f"Backend {self.name} unavailable: {error_message}")
    
    def get_last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error
    
    def health_check(self) -> bool:
        """Perform a health check on the backend."""
        try:
            # Create a simple test problem
            test_Q = np.array([[1.0, -1.0], [-1.0, 1.0]])
            result = self.solve_qubo(test_Q, num_reads=1)
            self.set_availability(result.success)
            return result.success
        except Exception as e:
            self.set_availability(False, str(e))
            return False
    
    def preprocess_qubo(self, Q: np.ndarray) -> np.ndarray:
        """Preprocess QUBO matrix for this backend."""
        # Default preprocessing: ensure upper triangular form
        return np.triu(Q) + np.triu(Q, k=1).T
    
    def postprocess_solution(
        self,
        solution: Dict[int, int],
        Q: np.ndarray
    ) -> Dict[int, int]:
        """Postprocess solution from backend."""
        # Default: return as-is
        return solution
    
    def calculate_energy(self, solution: Dict[int, int], Q: np.ndarray) -> float:
        """Calculate energy of a solution."""
        energy = 0.0
        n = Q.shape[0]
        
        for i in range(n):
            for j in range(n):
                if Q[i, j] != 0:
                    xi = solution.get(i, 0)
                    xj = solution.get(j, 0)
                    energy += Q[i, j] * xi * xj
        
        return energy
    
    def __str__(self) -> str:
        info = self.get_backend_info()
        return str(info)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.backend_type})"


class BackendManager:
    """Manages multiple optimization backends."""
    
    def __init__(self):
        self.backends: Dict[str, BaseBackend] = {}
        self.logger = logging.getLogger(f"{__name__}.BackendManager")
    
    def register_backend(self, backend: BaseBackend) -> None:
        """Register a new backend."""
        self.backends[backend.name] = backend
        self.logger.info(f"Registered backend: {backend.name}")
    
    def get_backend(self, name: str) -> Optional[BaseBackend]:
        """Get backend by name."""
        return self.backends.get(name)
    
    def list_available_backends(self) -> List[str]:
        """List all available backend names."""
        return [name for name, backend in self.backends.items() 
                if backend.is_available()]
    
    def select_best_backend(
        self,
        problem_size: int,
        prefer_quantum: bool = True,
        max_cost: float = float('inf'),
        max_time: float = float('inf')
    ) -> Optional[BaseBackend]:
        """Select the best backend for given constraints."""
        
        candidates = []
        
        for backend in self.backends.values():
            if not backend.is_available():
                continue
            
            info = backend.get_backend_info()
            
            # Check constraints
            if problem_size > info.max_variables:
                continue
            
            if info.cost_per_sample * 1000 > max_cost:  # Assume 1000 samples
                continue
            
            if info.typical_solve_time > max_time:
                continue
            
            # Score based on preferences
            score = 0.0
            
            if prefer_quantum and info.backend_type in [BackendType.QUANTUM_ANNEALING, BackendType.GATE_BASED_QUANTUM]:
                score += 100.0
            
            # Prefer faster backends
            score += 10.0 / (info.typical_solve_time + 0.1)
            
            # Prefer cheaper backends
            score += 5.0 / (info.cost_per_sample + 0.001)
            
            # Prefer higher capacity backends
            score += info.max_variables / 1000.0
            
            candidates.append((score, backend))
        
        if not candidates:
            self.logger.warning("No suitable backend found for problem constraints")
            return None
        
        # Return highest scoring backend
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_backend = candidates[0][1]
        
        self.logger.info(f"Selected backend: {best_backend.name} (score: {candidates[0][0]:.2f})")
        return best_backend
    
    def health_check_all(self) -> Dict[str, bool]:
        """Run health checks on all backends."""
        results = {}
        
        for name, backend in self.backends.items():
            self.logger.info(f"Health checking backend: {name}")
            results[name] = backend.health_check()
        
        return results
    
    def get_backend_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all backends."""
        stats = {}
        
        for name, backend in self.backends.items():
            info = backend.get_backend_info()
            stats[name] = {
                "type": info.backend_type.value,
                "max_variables": info.max_variables,
                "available": backend.is_available(),
                "cost_per_sample": info.cost_per_sample,
                "typical_solve_time": info.typical_solve_time,
                "last_error": backend.get_last_error()
            }
        
        return stats