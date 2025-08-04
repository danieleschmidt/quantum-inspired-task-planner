"""D-Wave quantum annealing backend implementation."""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import time
import logging

from .base import BaseBackend, BackendType, BackendInfo, OptimizationResult

logger = logging.getLogger(__name__)


class DWaveBackend(BaseBackend):
    """D-Wave quantum annealing backend."""
    
    def __init__(
        self,
        token: Optional[str] = None,
        solver: Optional[str] = None,
        endpoint: Optional[str] = None
    ):
        super().__init__("dwave", BackendType.QUANTUM_ANNEALING)
        
        self.token = token
        self.solver_name = solver
        self.endpoint = endpoint
        self._sampler = None
        self._solver_info = None
        
        # Try to initialize D-Wave connection
        self._initialize_dwave()
    
    def _initialize_dwave(self) -> None:
        """Initialize D-Wave Ocean SDK connection."""
        try:
            import dimod
            from dwave.system import DWaveSampler, EmbeddingComposite
            from dwave.cloud import Client
            import dwave.inspector
            
            # Set up configuration
            if self.token:
                import os
                os.environ['DWAVE_API_TOKEN'] = self.token
            
            if self.endpoint:
                import os
                os.environ['DWAVE_API_ENDPOINT'] = self.endpoint
            
            # Initialize sampler
            try:
                if self.solver_name:
                    self._sampler = EmbeddingComposite(DWaveSampler(solver=self.solver_name))
                else:
                    self._sampler = EmbeddingComposite(DWaveSampler())
                
                # Get solver information
                self._solver_info = self._sampler.child.solver
                self.set_availability(True)
                self.logger.info(f"D-Wave backend initialized with solver: {self._solver_info.name}")
                
            except Exception as e:
                self.logger.warning(f"D-Wave hardware unavailable, using mock: {e}")
                self._setup_mock_sampler()
                
        except ImportError as e:
            self.logger.warning(f"D-Wave Ocean SDK not available: {e}")
            self._setup_mock_sampler()
    
    def _setup_mock_sampler(self) -> None:
        """Set up mock sampler for testing when D-Wave is unavailable."""
        self._sampler = MockDWaveSampler()
        self._solver_info = MockSolverInfo()
        self.set_availability(True, "Using mock D-Wave sampler")
    
    def get_backend_info(self) -> BackendInfo:
        """Get D-Wave backend information."""
        if self._solver_info:
            max_vars = getattr(self._solver_info, 'num_qubits', 2000)
            connectivity = getattr(self._solver_info, 'topology', {}).get('type', 'chimera')
        else:
            max_vars = 2000
            connectivity = "chimera"
        
        return BackendInfo(
            name=self.name,
            backend_type=self.backend_type,
            max_variables=max_vars,
            connectivity=connectivity,
            availability=self.is_available(),
            cost_per_sample=0.00025,  # Approximate D-Wave cost
            typical_solve_time=2.0,
            supports_embedding=True,
            supports_constraints=True
        )
    
    def solve_qubo(
        self,
        Q: np.ndarray,
        num_reads: int = 1000,
        chain_strength: Optional[float] = None,
        annealing_time: int = 20,
        **kwargs
    ) -> OptimizationResult:
        """Solve QUBO problem using D-Wave quantum annealer."""
        
        start_time = time.time()
        
        try:
            # Validate problem
            is_valid, error_msg = self.validate_problem(Q)
            if not is_valid:
                return OptimizationResult(
                    solution={},
                    energy=float('inf'),
                    num_samples=0,
                    timing={'total': time.time() - start_time},
                    metadata={'error': error_msg},
                    success=False,
                    error_message=error_msg
                )
            
            # Convert numpy array to dictionary format
            qubo_dict = self._numpy_to_qubo_dict(Q)
            
            # Set parameters
            sampler_params = {
                'num_reads': num_reads,
                'annealing_time': annealing_time,
                **kwargs
            }
            
            if chain_strength is not None:
                sampler_params['chain_strength'] = chain_strength
            else:
                # Auto-determine chain strength
                sampler_params['chain_strength'] = max(abs(v) for v in qubo_dict.values()) * 2
            
            self.logger.debug(f"Submitting QUBO with {len(qubo_dict)} terms, {num_reads} reads")
            
            # Submit to D-Wave
            submit_time = time.time()
            sampleset = self._sampler.sample_qubo(qubo_dict, **sampler_params)
            solve_time = time.time() - submit_time
            
            # Process results
            best_sample = sampleset.first
            solution = {i: int(best_sample.sample.get(i, 0)) for i in range(Q.shape[0])}
            energy = float(best_sample.energy)
            
            # Timing information
            timing = {
                'total': time.time() - start_time,
                'solve': solve_time,
                'preprocessing': submit_time - start_time,
                'postprocessing': time.time() - submit_time - solve_time
            }
            
            # Metadata
            metadata = {
                'chain_break_fraction': getattr(sampleset.data_vectors, 'chain_break_fraction', [0.0])[0],
                'num_chain_breaks': getattr(sampleset.data_vectors, 'num_chain_breaks', [0])[0],
                'timing': getattr(sampleset.info, 'timing', {}),
                'solver': getattr(self._solver_info, 'name', 'unknown'),
                'embedding_context': getattr(sampleset.info, 'embedding_context', {}),
                'num_samples': len(sampleset),
                'num_variables': Q.shape[0]
            }
            
            self.logger.info(
                f"D-Wave solve completed: energy={energy:.4f}, "
                f"chain_breaks={metadata.get('num_chain_breaks', 0)}, "
                f"time={solve_time:.2f}s"
            )
            
            return OptimizationResult(
                solution=solution,
                energy=energy,
                num_samples=len(sampleset),
                timing=timing,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            error_msg = f"D-Wave solve failed: {str(e)}"
            self.logger.error(error_msg)
            
            return OptimizationResult(
                solution={},
                energy=float('inf'),
                num_samples=0,
                timing={'total': time.time() - start_time},
                metadata={'error': error_msg},
                success=False,
                error_message=error_msg
            )
    
    def estimate_solve_time(self, problem_size: int) -> float:
        """Estimate D-Wave solve time based on problem size."""
        # Base time for D-Wave submission and embedding
        base_time = 2.0
        
        # Additional time based on problem complexity
        embedding_time = (problem_size / 100) * 0.5
        
        return base_time + embedding_time
    
    def _numpy_to_qubo_dict(self, Q: np.ndarray) -> Dict[Tuple[int, int], float]:
        """Convert numpy QUBO matrix to dictionary format."""
        qubo_dict = {}
        n = Q.shape[0]
        
        for i in range(n):
            for j in range(i, n):  # Upper triangular
                if Q[i, j] != 0:
                    qubo_dict[(i, j)] = float(Q[i, j])
        
        return qubo_dict
    
    def get_solver_properties(self) -> Dict[str, Any]:
        """Get properties of the D-Wave solver."""
        if self._solver_info:
            return {
                'name': getattr(self._solver_info, 'name', 'unknown'),
                'num_qubits': getattr(self._solver_info, 'num_qubits', 0),
                'topology': getattr(self._solver_info, 'topology', {}),
                'max_h': getattr(self._solver_info, 'h_range', [0, 0])[-1],
                'max_j': getattr(self._solver_info, 'j_range', [0, 0])[-1],
                'annealing_time_range': getattr(self._solver_info, 'annealing_time_range', [1, 2000]),
                'num_reads_range': getattr(self._solver_info, 'num_reads_range', [1, 10000])
            }
        
        return {'name': 'mock', 'num_qubits': 2000}
    
    def inspect_solution(self, sampleset, filename: Optional[str] = None) -> str:
        """Inspect D-Wave solution using D-Wave Inspector."""
        try:
            import dwave.inspector
            
            if filename:
                return dwave.inspector.show(sampleset, block=False)
            else:
                # Return inspection URL if available
                return dwave.inspector.show(sampleset, block=False)
                
        except ImportError:
            self.logger.warning("D-Wave Inspector not available")
            return "Inspector not available"
        except Exception as e:
            self.logger.error(f"Solution inspection failed: {e}")
            return f"Inspection failed: {e}"


class MockDWaveSampler:
    """Mock D-Wave sampler for testing without hardware access."""
    
    def __init__(self):
        self.child = self
        self.solver = MockSolverInfo()
    
    def sample_qubo(self, Q: Dict[Tuple[int, int], float], **kwargs) -> 'MockSampleSet':
        """Mock QUBO sampling using simulated annealing."""
        import random
        import time
        
        # Simulate annealing time
        time.sleep(0.1)
        
        # Get variables
        variables = set()
        for (i, j) in Q.keys():
            variables.add(i)
            variables.add(j)
        
        variables = sorted(variables)
        
        # Generate random solution
        solution = {var: random.choice([0, 1]) for var in variables}
        
        # Calculate energy
        energy = 0.0
        for (i, j), coeff in Q.items():
            if i == j:
                energy += coeff * solution.get(i, 0)
            else:
                energy += coeff * solution.get(i, 0) * solution.get(j, 0)
        
        return MockSampleSet(solution, energy, kwargs.get('num_reads', 1000))


class MockSolverInfo:
    """Mock solver info for testing."""
    
    def __init__(self):
        self.name = "mock-dwave-solver"
        self.num_qubits = 2000
        self.topology = {'type': 'chimera'}
        self.h_range = [-4.0, 4.0]
        self.j_range = [-1.0, 1.0]
        self.annealing_time_range = [1, 2000]
        self.num_reads_range = [1, 10000]


class MockSampleSet:
    """Mock sample set for testing."""
    
    def __init__(self, solution: Dict[int, int], energy: float, num_reads: int):
        self.first = MockSample(solution, energy)
        self.data_vectors = MockDataVectors()
        self.info = {
            'timing': {'qpu_access_time': 0.1, 'qpu_programming_time': 0.05},
            'embedding_context': {'num_chains': len(solution)}
        }
        self._num_samples = num_reads
    
    def __len__(self):
        return self._num_samples


class MockSample:
    """Mock sample for testing."""
    
    def __init__(self, solution: Dict[int, int], energy: float):
        self.sample = solution
        self.energy = energy


class MockDataVectors:
    """Mock data vectors for testing."""
    
    def __init__(self):
        self.chain_break_fraction = [0.0]
        self.num_chain_breaks = [0]