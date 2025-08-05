"""Enhanced quantum backends with robust error handling and fallbacks."""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .enhanced_base import (
    EnhancedQuantumBackend, 
    BackendCapabilities, 
    BackendStatus
)
from .enhanced_classical import EnhancedSimulatedAnnealingBackend

logger = logging.getLogger(__name__)


@dataclass
class QuantumDeviceInfo:
    """Information about quantum device capabilities."""
    name: str
    qubits: int
    connectivity: str
    coherence_time: float
    gate_fidelity: float
    availability: bool = True
    queue_length: int = 0


class EnhancedDWaveBackend(EnhancedQuantumBackend):
    """Enhanced D-Wave backend with robust error handling and smart fallbacks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced D-Wave backend."""
        default_config = {
            "solver": "Advantage_system6.1",
            "num_reads": 1000,
            "chain_strength": 2.0,
            "annealing_time": 20,
            "auto_scale": True,
            "use_embedding_cache": True,
            "fallback_enabled": True,
            "max_retries": 3,
            "timeout": 300
        }
        
        if config:
            default_config.update(config)
            
        super().__init__("enhanced_dwave", default_config)
        
        self._sampler = None
        self._embedding_cache: Dict[str, Any] = {}
        self._fallback_backend = None
    
    def _initialize_backend(self) -> None:
        """Initialize D-Wave connection with fallback."""
        try:
            # Try to import and initialize D-Wave
            import dwave.system
            from dwave.system import DWaveSampler, EmbeddingComposite
            
            # Initialize sampler
            solver_name = self.config.get("solver")
            if solver_name:
                self._sampler = EmbeddingComposite(DWaveSampler(solver=solver_name))
            else:
                self._sampler = EmbeddingComposite(DWaveSampler())
            
            logger.info(f"Successfully initialized D-Wave sampler: {self._sampler.child.solver.name}")
            
        except ImportError:
            logger.warning("D-Wave Ocean SDK not available, using mock sampler")
            self._sampler = self._create_mock_sampler()
            
        except Exception as e:
            logger.warning(f"Failed to initialize D-Wave sampler: {e}, using mock")
            self._sampler = self._create_mock_sampler()
        
        # Initialize fallback backend
        if self.config.get("fallback_enabled", True):
            self._fallback_backend = EnhancedSimulatedAnnealingBackend({
                "max_iterations": 5000,
                "name": "dwave_fallback"
            })
    
    def get_capabilities(self) -> BackendCapabilities:
        """Get D-Wave capabilities."""
        if self._is_real_dwave():
            return BackendCapabilities(
                max_variables=5000,  # Advantage system
                supports_constraints=True,
                supports_embedding=True,
                supports_async=True,
                supports_batching=True,
                max_batch_size=100,
                supported_objectives=["minimize"],
                constraint_types=["quadratic", "penalty"]
            )
        else:
            return BackendCapabilities(
                max_variables=100,  # Mock limitations
                supports_constraints=True,
                supports_embedding=False,
                supports_async=False,
                supports_batching=False,
                max_batch_size=1,
                supported_objectives=["minimize"],
                constraint_types=["quadratic"]
            )
    
    def _solve_qubo(self, Q: Any, **kwargs) -> Dict[int, int]:
        """Solve QUBO using D-Wave with robust error handling."""
        
        # Convert to D-Wave format
        if hasattr(Q, 'shape'):
            Q_dict = self._matrix_to_dict(Q)
        else:
            Q_dict = Q
        
        # Check if problem is too large
        capabilities = self.get_capabilities()
        num_vars = len(set([i for i, j in Q_dict.keys()] + [j for i, j in Q_dict.keys()]))
        
        if num_vars > capabilities.max_variables:
            if self._fallback_backend:
                logger.info(f"Problem too large for D-Wave ({num_vars} vars), using fallback")
                return self._fallback_backend._solve_qubo(Q, **kwargs)
            else:
                raise ValueError(f"Problem too large: {num_vars} > {capabilities.max_variables}")
        
        # Try D-Wave solving with retries
        max_retries = self.config.get("max_retries", 3)
        
        for attempt in range(max_retries):
            try:
                if self._is_real_dwave():
                    return self._solve_with_real_dwave(Q_dict, **kwargs)
                else:
                    return self._solve_with_mock_dwave(Q_dict, **kwargs)
                    
            except Exception as e:
                logger.warning(f"D-Wave attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    # Last attempt failed, use fallback
                    if self._fallback_backend:
                        logger.info("All D-Wave attempts failed, using fallback")
                        return self._fallback_backend._solve_qubo(Q, **kwargs)
                    else:
                        raise
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        raise RuntimeError("All D-Wave attempts exhausted")
    
    def _solve_with_real_dwave(self, Q: Dict[Tuple[int, int], float], **kwargs) -> Dict[int, int]:
        """Solve using real D-Wave hardware."""
        
        # Prepare parameters
        num_reads = kwargs.get("num_reads", self.config.get("num_reads", 1000))
        chain_strength = kwargs.get("chain_strength", self.config.get("chain_strength", 2.0))
        annealing_time = kwargs.get("annealing_time", self.config.get("annealing_time", 20))
        
        # Submit to D-Wave
        response = self._sampler.sample_qubo(
            Q,
            num_reads=num_reads,
            chain_strength=chain_strength,
            annealing_time=annealing_time,
            label=f"quantum_planner_{int(time.time())}"
        )
        
        # Get best solution
        best_sample = response.first.sample
        
        # Ensure all variables are present
        all_vars = set([i for i, j in Q.keys()] + [j for i, j in Q.keys()])
        solution = {var: best_sample.get(var, 0) for var in all_vars}
        
        return solution
    
    def _solve_with_mock_dwave(self, Q: Dict[Tuple[int, int], float], **kwargs) -> Dict[int, int]:
        """Solve using mock D-Wave (for testing/demo)."""
        import random
        
        # Simple random solution for mock
        all_vars = set([i for i, j in Q.keys()] + [j for i, j in Q.keys()])
        solution = {var: random.randint(0, 1) for var in all_vars}
        
        # Add some randomness to simulate quantum annealing
        time.sleep(0.1)  # Simulate network latency
        
        return solution
    
    def _is_real_dwave(self) -> bool:
        """Check if using real D-Wave hardware."""
        try:
            return (hasattr(self._sampler, 'child') and 
                   hasattr(self._sampler.child, 'solver') and
                   'mock' not in str(type(self._sampler)).lower())
        except:
            return False
    
    def _create_mock_sampler(self):
        """Create mock D-Wave sampler for testing."""
        class MockSampler:
            def sample_qubo(self, Q, **kwargs):
                import random
                from types import SimpleNamespace
                
                # Create mock response
                all_vars = set([i for i, j in Q.keys()] + [j for i, j in Q.keys()])
                sample = {var: random.randint(0, 1) for var in all_vars}
                
                response = SimpleNamespace()
                response.first = SimpleNamespace()
                response.first.sample = sample
                
                return response
        
        return MockSampler()
    
    def _matrix_to_dict(self, Q) -> Dict[Tuple[int, int], float]:
        """Convert matrix to D-Wave QUBO format."""
        Q_dict = {}
        
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                if Q[i, j] != 0:
                    Q_dict[(i, j)] = float(Q[i, j])
        
        return Q_dict
    
    def _check_connectivity(self) -> BackendStatus:
        """Check D-Wave connectivity."""
        try:
            if self._is_real_dwave():
                # Try a simple property access
                solver_name = self._sampler.child.solver.name
                return BackendStatus.AVAILABLE
            else:
                # Mock is always available
                return BackendStatus.AVAILABLE
                
        except Exception as e:
            logger.warning(f"D-Wave connectivity check failed: {e}")
            return BackendStatus.UNAVAILABLE
    
    def _base_time_estimation(self, num_variables: int) -> float:
        """Estimate D-Wave solve time."""
        if self._is_real_dwave():
            # Real D-Wave: mostly queue time
            base_time = 5.0  # 5 seconds base
            queue_factor = 1.0  # Could query actual queue
            annealing_time = self.config.get("annealing_time", 20) / 1000  # ms to s
            
            return base_time * queue_factor + annealing_time
        else:
            # Mock D-Wave
            return 0.5


class EnhancedAzureQuantumBackend(EnhancedQuantumBackend):
    """Enhanced Azure Quantum backend with multiple solver support."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Azure Quantum backend."""
        default_config = {
            "workspace": None,
            "resource_id": None,
            "location": "westus",
            "provider": "microsoft.simulatedannealing",
            "timeout": 600,
            "fallback_enabled": True
        }
        
        if config:
            default_config.update(config)
            
        super().__init__("enhanced_azure_quantum", default_config)
        
        self._workspace = None
        self._fallback_backend = None
    
    def _initialize_backend(self) -> None:
        """Initialize Azure Quantum workspace."""
        try:
            # Try to import Azure Quantum
            from azure.quantum import Workspace
            
            resource_id = self.config.get("resource_id")
            location = self.config.get("location", "westus")
            
            if resource_id:
                self._workspace = Workspace(
                    resource_id=resource_id,
                    location=location
                )
                logger.info(f"Initialized Azure Quantum workspace: {resource_id}")
            else:
                logger.warning("No Azure Quantum resource_id provided, using mock")
                self._workspace = None
                
        except ImportError:
            logger.warning("Azure Quantum SDK not available, using mock")
            self._workspace = None
            
        except Exception as e:
            logger.warning(f"Failed to initialize Azure Quantum: {e}")
            self._workspace = None
        
        # Initialize fallback
        if self.config.get("fallback_enabled", True):
            self._fallback_backend = EnhancedSimulatedAnnealingBackend({
                "max_iterations": 3000,
                "name": "azure_fallback"
            })
    
    def get_capabilities(self) -> BackendCapabilities:
        """Get Azure Quantum capabilities."""
        return BackendCapabilities(
            max_variables=2000,
            supports_constraints=True,
            supports_embedding=False,
            supports_async=True,
            supports_batching=False,
            max_batch_size=1,
            supported_objectives=["minimize"],
            constraint_types=["quadratic", "penalty"]
        )
    
    def _solve_qubo(self, Q: Any, **kwargs) -> Dict[int, int]:
        """Solve QUBO using Azure Quantum."""
        
        if self._workspace is None:
            if self._fallback_backend:
                logger.info("Azure Quantum not available, using fallback")
                return self._fallback_backend._solve_qubo(Q, **kwargs)
            else:
                raise RuntimeError("Azure Quantum not available and no fallback configured")
        
        # For now, use fallback (real implementation would use Azure Quantum)
        logger.info("Using Azure Quantum fallback (implementation pending)")
        return self._fallback_backend._solve_qubo(Q, **kwargs)
    
    def _check_connectivity(self) -> BackendStatus:
        """Check Azure Quantum connectivity."""
        if self._workspace is None:
            return BackendStatus.UNAVAILABLE
        
        try:
            # Could check workspace status here
            return BackendStatus.AVAILABLE
        except Exception:
            return BackendStatus.UNAVAILABLE
    
    def _base_time_estimation(self, num_variables: int) -> float:
        """Estimate Azure Quantum solve time."""
        return 10.0  # Base estimate