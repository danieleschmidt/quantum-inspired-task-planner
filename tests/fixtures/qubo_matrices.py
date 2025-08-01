"""Pre-computed QUBO matrices for testing."""

import numpy as np
from typing import Dict, Any

# Small QUBO matrices for unit tests
SMALL_QUBOS = {
    "2x2_simple": {
        "matrix": np.array([
            [1.0, -0.5],
            [-0.5, 1.0]
        ]),
        "optimal_solution": {0: 0, 1: 1},
        "optimal_energy": 0.5,
        "description": "Simple 2-variable QUBO with clear minimum"
    },
    
    "3x3_assignment": {
        "matrix": np.array([
            [2.0, -1.0, -0.5],
            [-1.0, 3.0, -1.5],
            [-0.5, -1.5, 2.5]
        ]),
        "optimal_solution": {0: 1, 1: 0, 2: 1},
        "optimal_energy": 0.5,
        "description": "3-variable task assignment QUBO"
    },
    
    "4x4_symmetric": {
        "matrix": np.array([
            [1.0, 0.5, 0.0, 0.2],
            [0.5, 2.0, 0.3, 0.0],
            [0.0, 0.3, 1.5, 0.4],
            [0.2, 0.0, 0.4, 1.8]
        ]),
        "description": "4-variable symmetric QUBO for general testing"
    }
}

# Medium-sized QUBO matrices
MEDIUM_QUBOS = {
    "8x8_scheduling": {
        "description": "8-variable scheduling problem QUBO",
        "num_variables": 8,
        "problem_type": "scheduling",
        "sparsity": 0.6
    },
    
    "10x10_assignment": {
        "description": "10-variable assignment problem QUBO",
        "num_variables": 10,
        "problem_type": "assignment",
        "sparsity": 0.4
    }
}

# Large QUBO matrices (generated on demand)
LARGE_QUBOS = {
    "20x20_sparse": {
        "description": "20-variable sparse QUBO",
        "num_variables": 20,
        "sparsity": 0.1,
        "random_seed": 42
    },
    
    "50x50_dense": {
        "description": "50-variable dense QUBO for performance testing",
        "num_variables": 50,
        "sparsity": 0.7,
        "random_seed": 123
    }
}

def generate_random_qubo(num_vars: int, sparsity: float = 0.5, seed: int = None) -> np.ndarray:
    """Generate a random symmetric QUBO matrix.
    
    Args:
        num_vars: Number of variables (matrix size)
        sparsity: Fraction of non-zero elements
        seed: Random seed for reproducibility
        
    Returns:
        Symmetric QUBO matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate upper triangular matrix
    upper = np.random.rand(num_vars, num_vars)
    upper = np.triu(upper)
    
    # Apply sparsity
    mask = np.random.rand(num_vars, num_vars) < sparsity
    mask = np.triu(mask)
    upper = upper * mask
    
    # Make symmetric
    qubo = upper + upper.T - np.diag(np.diag(upper))
    
    # Scale values to reasonable range
    qubo = qubo * 4 - 2  # Range [-2, 2]
    
    return qubo

def get_qubo_matrix(name: str) -> Dict[str, Any]:
    """Get QUBO matrix by name.
    
    Args:
        name: Name of the QUBO matrix
        
    Returns:
        Dictionary containing matrix and metadata
    """
    if name in SMALL_QUBOS:
        return SMALL_QUBOS[name].copy()
    
    elif name in MEDIUM_QUBOS:
        spec = MEDIUM_QUBOS[name]
        matrix = generate_random_qubo(
            spec["num_variables"], 
            spec["sparsity"],
            seed=hash(name) % 1000
        )
        return {
            "matrix": matrix,
            "description": spec["description"],
            "problem_type": spec.get("problem_type"),
            "sparsity": spec["sparsity"]
        }
    
    elif name in LARGE_QUBOS:
        spec = LARGE_QUBOS[name]
        matrix = generate_random_qubo(
            spec["num_variables"],
            spec["sparsity"],
            spec.get("random_seed")
        )
        return {
            "matrix": matrix,
            "description": spec["description"],
            "sparsity": spec["sparsity"],
            "num_variables": spec["num_variables"]
        }
    
    else:
        raise ValueError(f"Unknown QUBO matrix: {name}")

def validate_qubo_matrix(Q: np.ndarray) -> Dict[str, Any]:
    """Validate and analyze a QUBO matrix.
    
    Args:
        Q: QUBO matrix to validate
        
    Returns:
        Dictionary with validation results and properties
    """
    n = Q.shape[0]
    
    # Check if square
    is_square = Q.shape[0] == Q.shape[1]
    
    # Check if symmetric
    is_symmetric = np.allclose(Q, Q.T, rtol=1e-10)
    
    # Check for NaN or infinite values
    has_nan = np.isnan(Q).any()
    has_inf = np.isinf(Q).any()
    
    # Calculate properties
    sparsity = np.count_nonzero(Q) / (n * n)
    condition_number = np.linalg.cond(Q) if is_square else float('inf')
    eigenvalues = np.linalg.eigvals(Q) if is_square else None
    
    # Estimate problem difficulty
    if eigenvalues is not None:
        min_eigenval = np.min(eigenvalues)
        max_eigenval = np.max(eigenvalues)
        spectral_gap = max_eigenval - min_eigenval
        difficulty = "easy" if spectral_gap < 5 else "medium" if spectral_gap < 20 else "hard"
    else:
        difficulty = "unknown"
    
    return {
        "is_valid": is_square and is_symmetric and not has_nan and not has_inf,
        "is_square": is_square,
        "is_symmetric": is_symmetric,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "size": n,
        "sparsity": sparsity,
        "condition_number": condition_number,
        "min_eigenvalue": np.min(eigenvalues) if eigenvalues is not None else None,
        "max_eigenvalue": np.max(eigenvalues) if eigenvalues is not None else None,
        "difficulty": difficulty
    }

# Known difficult QUBO instances for stress testing
STRESS_TEST_QUBOS = {
    "degenerate": {
        "description": "QUBO with many degenerate solutions",
        "matrix": np.array([
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0]
        ]),
        "num_optimal_solutions": 8,
        "difficulty": "high"
    },
    
    "frustrated": {
        "description": "Frustrated QUBO with competing constraints",
        "matrix": np.array([
            [1.0, -2.0, -2.0],
            [-2.0, 1.0, -2.0],
            [-2.0, -2.0, 1.0]
        ]),
        "difficulty": "high",
        "note": "All pairwise interactions want different states"
    }
}