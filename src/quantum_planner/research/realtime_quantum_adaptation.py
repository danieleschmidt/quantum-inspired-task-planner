"""
Real-Time Quantum Adaptation Framework

Implements advanced real-time learning and adaptation capabilities for
quantum optimization systems. Features online learning from solution quality
feedback, dynamic algorithm selection, and autonomous performance optimization.

This represents a breakthrough in autonomous quantum optimization systems.

Author: Terragon Labs Quantum Research Team  
Version: 1.0.0
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import time
from collections import deque, defaultdict
import warnings
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Using simplified learning models.")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Deep learning features will use simplified implementations.")

logger = logging.getLogger(__name__)


class OptimizationContext(Enum):
    """Optimization context types for adaptation."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    CONVERGENCE = "convergence"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


class AdaptationTrigger(Enum):
    """Triggers for adaptation decisions."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CONVERGENCE_STALL = "convergence_stall"
    NOISE_CHANGE = "noise_change"
    RESOURCE_CONSTRAINT = "resource_constraint"
    TIME_LIMIT = "time_limit"
    USER_INTERVENTION = "user_intervention"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for optimization runs."""
    
    energy: float
    solution_quality: float
    convergence_rate: float
    time_to_solution: float
    resource_utilization: float
    noise_resilience: float
    
    # Context information
    problem_size: int
    algorithm_used: str
    backend_type: str
    noise_level: float
    timestamp: float = field(default_factory=time.time)
    
    # Derived metrics
    efficiency: float = field(init=False)
    robustness: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.efficiency = self.solution_quality / max(self.time_to_solution, 0.001)
        self.robustness = self.solution_quality * (1.0 - self.noise_level)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for machine learning."""
        return np.array([
            self.energy,
            self.solution_quality,
            self.convergence_rate,
            self.time_to_solution,
            self.resource_utilization,
            self.noise_resilience,
            self.problem_size,
            self.noise_level,
            self.efficiency,
            self.robustness
        ])


@dataclass
class AdaptationAction:
    """Represents an adaptation action to be taken."""
    
    action_type: str
    parameters: Dict[str, Any]
    expected_improvement: float
    confidence: float
    resource_cost: float
    
    timestamp: float = field(default_factory=time.time)
    execution_status: str = "pending"
    actual_improvement: Optional[float] = None


class PerformancePredictor(ABC):
    """Abstract base class for performance prediction models."""
    
    @abstractmethod
    def predict(self, context: Dict[str, Any]) -> Tuple[float, float]:
        """Predict performance (mean, uncertainty)."""
        pass
    
    @abstractmethod
    def update(self, context: Dict[str, Any], performance: float):
        """Update model with new data."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass


class GaussianProcessPredictor(PerformancePredictor):
    """Gaussian Process-based performance predictor with uncertainty quantification."""
    
    def __init__(self, kernel=None, alpha=1e-6, n_restarts_optimizer=5):
        """Initialize GP predictor."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for GaussianProcessPredictor")
        
        if kernel is None:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-3)
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=True
        )
        
        self.scaler = StandardScaler()
        self.training_data = []
        self.training_targets = []
        self.is_fitted = False
        
        logger.info("Initialized GaussianProcessPredictor")
    
    def predict(self, context: Dict[str, Any]) -> Tuple[float, float]:
        """Predict performance with uncertainty."""
        if not self.is_fitted:
            # Return prior estimate
            return 0.5, 1.0
        
        features = self._context_to_features(context)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        mean, std = self.gp.predict(features_scaled, return_std=True)
        return float(mean[0]), float(std[0])
    
    def update(self, context: Dict[str, Any], performance: float):
        """Update GP with new observation."""
        features = self._context_to_features(context)
        
        self.training_data.append(features)
        self.training_targets.append(performance)
        
        # Retrain periodically
        if len(self.training_data) % 10 == 0:
            self._retrain()
    
    def _retrain(self):
        """Retrain the Gaussian Process."""
        if len(self.training_data) < 5:
            return
        
        X = np.array(self.training_data)
        y = np.array(self.training_targets)
        
        # Fit scaler and transform data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit GP
        self.gp.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Retrained GP with {len(self.training_data)} samples")
    
    def _context_to_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Convert context to feature vector."""
        features = [
            context.get('problem_size', 0),
            context.get('noise_level', 0),
            context.get('time_limit', 100),
            context.get('resource_limit', 1.0),
            len(context.get('constraints', [])),
            hash(context.get('algorithm', 'default')) % 1000 / 1000.0,  # Normalized hash
            context.get('backend_load', 0.5),
            context.get('temperature', 1.0)  # Optimization temperature
        ]
        return np.array(features)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from GP kernel."""
        if not self.is_fitted:
            return {}
        
        feature_names = [
            'problem_size', 'noise_level', 'time_limit', 'resource_limit',
            'num_constraints', 'algorithm_hash', 'backend_load', 'temperature'
        ]
        
        # Extract length scales from RBF kernel
        try:
            kernel_params = self.gp.kernel_.get_params()
            length_scales = kernel_params.get('k2__length_scale', [1.0] * len(feature_names))
            
            if np.isscalar(length_scales):
                length_scales = [length_scales] * len(feature_names)
            
            # Higher length scale means less important
            importances = 1.0 / np.array(length_scales)
            importances = importances / np.sum(importances)  # Normalize
            
            return dict(zip(feature_names, importances))
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {name: 1.0/len(feature_names) for name in feature_names}


class DeepQLearningPredictor(PerformancePredictor):
    """Deep Q-Learning based performance predictor for action-value estimation."""
    
    def __init__(self, state_dim=8, action_dim=5, hidden_dims=[64, 32], lr=0.001):
        """Initialize Deep Q-Learning predictor."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Using simplified Q-learning.")
            self.use_torch = False
            self.q_table = defaultdict(lambda: defaultdict(float))
            return
        
        self.use_torch = True
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build neural network
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.q_network = nn.Sequential(*layers)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        
        logger.info("Initialized DeepQLearningPredictor")
    
    def predict(self, context: Dict[str, Any]) -> Tuple[float, float]:
        """Predict Q-values for actions."""
        if not self.use_torch:
            return self._simple_q_predict(context)
        
        state = self._context_to_state(context)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        # Return max Q-value and uncertainty estimate
        max_q = float(torch.max(q_values))
        uncertainty = float(torch.std(q_values))
        
        return max_q, uncertainty
    
    def update(self, context: Dict[str, Any], performance: float):
        """Update Q-network with experience."""
        if not self.use_torch:
            return self._simple_q_update(context, performance)
        
        state = self._context_to_state(context)
        action = context.get('action_taken', 0)
        reward = performance
        
        # Store experience
        self.memory.append((state, action, reward))
        
        # Train if enough samples
        if len(self.memory) >= self.batch_size:
            self._train_step()
    
    def _train_step(self):
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values (simplified - using reward as target)
        target_q_values = rewards.unsqueeze(1)
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _context_to_state(self, context: Dict[str, Any]) -> np.ndarray:
        """Convert context to state vector."""
        state = [
            context.get('problem_size', 0) / 1000.0,  # Normalize
            context.get('noise_level', 0),
            context.get('time_limit', 100) / 100.0,
            context.get('resource_limit', 1.0),
            min(len(context.get('constraints', [])), 10) / 10.0,
            context.get('backend_load', 0.5),
            context.get('temperature', 1.0),
            context.get('iteration', 0) / 100.0
        ]
        return np.array(state)
    
    def _simple_q_predict(self, context: Dict[str, Any]) -> Tuple[float, float]:
        """Simple Q-table prediction when PyTorch unavailable."""
        state_key = self._context_to_key(context)
        q_values = list(self.q_table[state_key].values())
        
        if not q_values:
            return 0.5, 1.0
        
        return max(q_values), np.std(q_values) if len(q_values) > 1 else 1.0
    
    def _simple_q_update(self, context: Dict[str, Any], performance: float):
        """Simple Q-table update when PyTorch unavailable."""
        state_key = self._context_to_key(context)
        action = context.get('action_taken', 0)
        
        # Simple Q-learning update
        alpha = 0.1  # Learning rate
        current_q = self.q_table[state_key][action]
        self.q_table[state_key][action] = current_q + alpha * (performance - current_q)
    
    def _context_to_key(self, context: Dict[str, Any]) -> str:
        """Convert context to hashable key."""
        key_parts = [
            int(context.get('problem_size', 0) / 10),  # Discretize
            int(context.get('noise_level', 0) * 100),
            int(context.get('time_limit', 100) / 10),
            len(context.get('constraints', [])),
            context.get('algorithm', 'default')
        ]
        return str(tuple(key_parts))
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (simplified)."""
        features = ['problem_size', 'noise_level', 'time_limit', 'resource_limit',
                   'num_constraints', 'backend_load', 'temperature', 'iteration']
        
        if not self.use_torch:
            return {name: 1.0/len(features) for name in features}
        
        # Analyze network weights (simplified)
        first_layer_weights = self.q_network[0].weight.data.abs().mean(dim=0).numpy()
        importance = first_layer_weights / first_layer_weights.sum()
        
        return dict(zip(features, importance))


class RealTimeAdaptationEngine:
    """
    Advanced real-time adaptation engine for quantum optimization.
    
    Features:
    - Online learning from solution quality feedback
    - Dynamic algorithm selection and parameter tuning
    - Contextual bandits for exploration-exploitation balance
    - Multi-objective adaptation with resource constraints
    """
    
    def __init__(
        self,
        predictor: Optional[PerformancePredictor] = None,
        adaptation_rate: float = 0.1,
        exploration_rate: float = 0.15,
        memory_size: int = 1000,
        adaptation_threshold: float = 0.05
    ):
        """Initialize real-time adaptation engine."""
        
        # Initialize predictor
        if predictor is None:
            try:
                self.predictor = GaussianProcessPredictor()
            except ImportError:
                self.predictor = DeepQLearningPredictor()
        else:
            self.predictor = predictor
        
        self.adaptation_rate = adaptation_rate
        self.exploration_rate = exploration_rate
        self.adaptation_threshold = adaptation_threshold
        
        # Performance tracking
        self.performance_history = deque(maxlen=memory_size)
        self.context_history = deque(maxlen=memory_size)
        self.action_history = deque(maxlen=memory_size)
        
        # Algorithm portfolio
        self.algorithm_portfolio = {
            'simulated_annealing': {'temperature': 1.0, 'cooling_rate': 0.95},
            'genetic_algorithm': {'population_size': 50, 'mutation_rate': 0.1},
            'quantum_annealing': {'annealing_time': 20, 'num_reads': 1000},
            'qaoa': {'p_layers': 3, 'optimizer': 'COBYLA'},
            'vqe': {'ansatz': 'efficient_su2', 'shots': 8192}
        }
        
        # Adaptation statistics
        self.adaptation_stats = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'exploration_actions': 0,
            'exploitation_actions': 0
        }
        
        # Threading for async adaptation
        self.adaptation_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info("Initialized RealTimeAdaptationEngine")
    
    def observe_performance(
        self,
        context: Dict[str, Any],
        metrics: PerformanceMetrics,
        action_taken: Optional[str] = None
    ):
        """Observe performance and update learning models."""
        
        with self.adaptation_lock:
            # Store observations
            self.performance_history.append(metrics)
            self.context_history.append(context.copy())
            self.action_history.append(action_taken)
            
            # Update predictor
            performance_score = self._calculate_performance_score(metrics)
            context_with_action = context.copy()
            if action_taken:
                context_with_action['action_taken'] = self._encode_action(action_taken)
            
            self.predictor.update(context_with_action, performance_score)
            
            logger.debug(f"Observed performance: {performance_score:.3f}, action: {action_taken}")
    
    def recommend_adaptation(
        self,
        current_context: Dict[str, Any],
        trigger: AdaptationTrigger = AdaptationTrigger.PERFORMANCE_DEGRADATION
    ) -> Optional[AdaptationAction]:
        """Recommend adaptation action based on current context."""
        
        # Analyze current situation
        situation_analysis = self._analyze_situation(current_context, trigger)
        
        if not self._should_adapt(situation_analysis):
            return None
        
        # Generate candidate actions
        candidate_actions = self._generate_candidate_actions(current_context, situation_analysis)
        
        if not candidate_actions:
            return None
        
        # Select best action using predictor
        best_action = self._select_best_action(candidate_actions, current_context)
        
        if best_action:
            self.adaptation_stats['total_adaptations'] += 1
            if best_action.action_type == 'explore':
                self.adaptation_stats['exploration_actions'] += 1
            else:
                self.adaptation_stats['exploitation_actions'] += 1
        
        return best_action
    
    def execute_adaptation(
        self,
        action: AdaptationAction,
        optimization_system: Any
    ) -> bool:
        """Execute adaptation action on optimization system."""
        
        try:
            success = False
            
            if action.action_type == 'algorithm_switch':
                success = self._execute_algorithm_switch(action, optimization_system)
            elif action.action_type == 'parameter_tune':
                success = self._execute_parameter_tune(action, optimization_system)
            elif action.action_type == 'resource_reallocation':
                success = self._execute_resource_reallocation(action, optimization_system)
            elif action.action_type == 'backend_switch':
                success = self._execute_backend_switch(action, optimization_system)
            elif action.action_type == 'problem_decomposition':
                success = self._execute_problem_decomposition(action, optimization_system)
            else:
                logger.warning(f"Unknown action type: {action.action_type}")
                return False
            
            # Update action status
            action.execution_status = "success" if success else "failed"
            
            if success:
                self.adaptation_stats['successful_adaptations'] += 1
            
            logger.info(f"Executed adaptation: {action.action_type}, success: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Error executing adaptation: {e}")
            action.execution_status = "error"
            return False
    
    def get_adaptation_insights(self) -> Dict[str, Any]:
        """Get insights about adaptation performance and patterns."""
        
        if len(self.performance_history) < 10:
            return {"status": "insufficient_data"}
        
        # Performance trends
        recent_performance = [self._calculate_performance_score(m) for m in list(self.performance_history)[-20:]]
        performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        
        # Algorithm effectiveness
        algorithm_performance = defaultdict(list)
        for i, action in enumerate(list(self.action_history)[-50:]):
            if action and i < len(self.performance_history):
                metrics = list(self.performance_history)[-(50-i)]
                performance = self._calculate_performance_score(metrics)
                algorithm_performance[action].append(performance)
        
        algorithm_rankings = {
            alg: np.mean(perfs) for alg, perfs in algorithm_performance.items()
            if len(perfs) > 2
        }
        
        # Feature importance
        feature_importance = self.predictor.get_feature_importance()
        
        insights = {
            "status": "ready",
            "performance_trend": performance_trend,
            "total_observations": len(self.performance_history),
            "adaptation_stats": self.adaptation_stats.copy(),
            "algorithm_rankings": algorithm_rankings,
            "feature_importance": feature_importance,
            "exploration_exploitation_ratio": (
                self.adaptation_stats['exploration_actions'] / 
                max(self.adaptation_stats['total_adaptations'], 1)
            ),
            "success_rate": (
                self.adaptation_stats['successful_adaptations'] /
                max(self.adaptation_stats['total_adaptations'], 1)
            )
        }
        
        return insights
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate unified performance score."""
        # Weighted combination of metrics
        score = (
            0.4 * metrics.solution_quality +
            0.2 * metrics.efficiency +
            0.2 * metrics.robustness +
            0.1 * (1.0 - metrics.noise_level) +
            0.1 * min(metrics.convergence_rate, 1.0)
        )
        return max(0.0, min(1.0, score))
    
    def _analyze_situation(
        self, 
        context: Dict[str, Any], 
        trigger: AdaptationTrigger
    ) -> Dict[str, Any]:
        """Analyze current optimization situation."""
        
        analysis = {
            'trigger': trigger,
            'context_type': self._classify_context(context),
            'performance_trend': self._get_performance_trend(),
            'resource_utilization': context.get('resource_utilization', 0.5),
            'noise_level': context.get('noise_level', 0.0),
            'time_remaining': context.get('time_remaining', float('inf')),
            'convergence_status': context.get('convergence_status', 'unknown')
        }
        
        # Add urgency assessment
        urgency_factors = [
            trigger == AdaptationTrigger.PERFORMANCE_DEGRADATION,
            analysis['performance_trend'] < -0.1,
            analysis['resource_utilization'] > 0.9,
            analysis['time_remaining'] < 60.0,  # seconds
            analysis['convergence_status'] == 'stalled'
        ]
        
        analysis['urgency'] = sum(urgency_factors) / len(urgency_factors)
        
        return analysis
    
    def _classify_context(self, context: Dict[str, Any]) -> OptimizationContext:
        """Classify current optimization context."""
        
        iteration = context.get('iteration', 0)
        convergence_rate = context.get('convergence_rate', 0.0)
        stagnation_count = context.get('stagnation_count', 0)
        
        if iteration < 10:
            return OptimizationContext.EXPLORATION
        elif convergence_rate > 0.1:
            return OptimizationContext.CONVERGENCE
        elif stagnation_count > 5:
            return OptimizationContext.RECOVERY
        else:
            return OptimizationContext.EXPLOITATION
    
    def _get_performance_trend(self) -> float:
        """Get recent performance trend."""
        if len(self.performance_history) < 5:
            return 0.0
        
        recent_scores = [
            self._calculate_performance_score(m) 
            for m in list(self.performance_history)[-10:]
        ]
        
        if len(recent_scores) < 2:
            return 0.0
        
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        return trend
    
    def _should_adapt(self, analysis: Dict[str, Any]) -> bool:
        """Determine if adaptation should be triggered."""
        
        # Don't adapt too frequently
        if self.adaptation_stats['total_adaptations'] > 0:
            recent_adaptations = sum(1 for action in list(self.action_history)[-5:] if action)
            if recent_adaptations > 2:  # More than 2 adaptations in last 5 steps
                return False
        
        # Adapt if performance is degrading significantly
        if analysis['performance_trend'] < -self.adaptation_threshold:
            return True
        
        # Adapt if high urgency
        if analysis['urgency'] > 0.7:
            return True
        
        # Probabilistic adaptation based on exploration rate
        if np.random.random() < self.exploration_rate:
            return True
        
        return False
    
    def _generate_candidate_actions(
        self,
        context: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> List[AdaptationAction]:
        """Generate candidate adaptation actions."""
        
        candidates = []
        
        # Algorithm switching actions
        current_algorithm = context.get('current_algorithm', 'unknown')
        for algorithm, params in self.algorithm_portfolio.items():
            if algorithm != current_algorithm:
                candidates.append(AdaptationAction(
                    action_type='algorithm_switch',
                    parameters={'new_algorithm': algorithm, 'params': params},
                    expected_improvement=self._estimate_algorithm_improvement(algorithm, context),
                    confidence=0.7,
                    resource_cost=self._estimate_switch_cost(algorithm, current_algorithm)
                ))
        
        # Parameter tuning actions
        if current_algorithm in self.algorithm_portfolio:
            current_params = self.algorithm_portfolio[current_algorithm]
            for param_name, param_value in current_params.items():
                # Generate parameter variations
                variations = self._generate_parameter_variations(param_name, param_value)
                for new_value in variations:
                    new_params = current_params.copy()
                    new_params[param_name] = new_value
                    candidates.append(AdaptationAction(
                        action_type='parameter_tune',
                        parameters={'param_name': param_name, 'new_value': new_value},
                        expected_improvement=self._estimate_parameter_improvement(
                            param_name, new_value, context
                        ),
                        confidence=0.6,
                        resource_cost=0.1  # Parameter changes are low cost
                    ))
        
        # Resource reallocation actions
        if context.get('resource_utilization', 0.5) < 0.8:
            candidates.append(AdaptationAction(
                action_type='resource_reallocation',
                parameters={'action': 'increase_resources', 'factor': 1.5},
                expected_improvement=0.2,
                confidence=0.8,
                resource_cost=0.5
            ))
        
        # Backend switching actions
        available_backends = context.get('available_backends', ['simulator'])
        current_backend = context.get('current_backend', 'simulator')
        for backend in available_backends:
            if backend != current_backend:
                candidates.append(AdaptationAction(
                    action_type='backend_switch',
                    parameters={'new_backend': backend},
                    expected_improvement=self._estimate_backend_improvement(backend, context),
                    confidence=0.5,
                    resource_cost=self._estimate_backend_switch_cost(backend, current_backend)
                ))
        
        # Problem decomposition actions
        problem_size = context.get('problem_size', 0)
        if problem_size > 20:
            candidates.append(AdaptationAction(
                action_type='problem_decomposition',
                parameters={'strategy': 'spectral_clustering', 'num_subproblems': 2},
                expected_improvement=0.3,
                confidence=0.6,
                resource_cost=0.8
            ))
        
        return candidates
    
    def _select_best_action(
        self,
        candidates: List[AdaptationAction],
        context: Dict[str, Any]
    ) -> Optional[AdaptationAction]:
        """Select best action using predictor and multi-criteria decision making."""
        
        if not candidates:
            return None
        
        # Score each candidate
        scored_candidates = []
        
        for action in candidates:
            # Predict performance improvement
            action_context = context.copy()
            action_context.update(action.parameters)
            action_context['action_taken'] = self._encode_action(action.action_type)
            
            predicted_improvement, uncertainty = self.predictor.predict(action_context)
            
            # Multi-criteria scoring
            score = (
                0.4 * predicted_improvement +
                0.2 * action.expected_improvement +
                0.2 * action.confidence +
                0.1 * (1.0 - action.resource_cost) +
                0.1 * (1.0 - uncertainty)  # Prefer lower uncertainty
            )
            
            scored_candidates.append((score, action))
        
        # Select best action
        if not scored_candidates:
            return None
        
        # Add exploration: sometimes select randomly
        if np.random.random() < self.exploration_rate:
            return np.random.choice([a for _, a in scored_candidates])
        
        # Exploitation: select best scored action
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[0][1]
    
    def _encode_action(self, action: str) -> int:
        """Encode action string to integer for learning."""
        action_map = {
            'algorithm_switch': 0,
            'parameter_tune': 1,
            'resource_reallocation': 2,
            'backend_switch': 3,
            'problem_decomposition': 4
        }
        return action_map.get(action, 0)
    
    def _estimate_algorithm_improvement(self, algorithm: str, context: Dict[str, Any]) -> float:
        """Estimate improvement from algorithm switch."""
        # Simple heuristic based on problem characteristics
        problem_size = context.get('problem_size', 0)
        noise_level = context.get('noise_level', 0.0)
        
        improvements = {
            'simulated_annealing': 0.1 + 0.2 * (noise_level < 0.05),
            'genetic_algorithm': 0.15 + 0.3 * (problem_size > 50),
            'quantum_annealing': 0.2 + 0.4 * (problem_size < 30) * (noise_level < 0.1),
            'qaoa': 0.25 + 0.3 * (10 < problem_size < 25),
            'vqe': 0.3 + 0.2 * (problem_size < 15)
        }
        
        return improvements.get(algorithm, 0.1)
    
    def _estimate_switch_cost(self, new_algorithm: str, current_algorithm: str) -> float:
        """Estimate cost of switching algorithms."""
        # Quantum algorithms have higher switch costs
        quantum_algorithms = {'quantum_annealing', 'qaoa', 'vqe'}
        
        cost = 0.2  # Base cost
        
        if new_algorithm in quantum_algorithms:
            cost += 0.3
        if current_algorithm in quantum_algorithms and new_algorithm not in quantum_algorithms:
            cost += 0.2  # Cost of quantum -> classical switch
        
        return min(cost, 1.0)
    
    def _generate_parameter_variations(self, param_name: str, current_value: Any) -> List[Any]:
        """Generate parameter variations for tuning."""
        if isinstance(current_value, (int, float)):
            # Numeric parameters
            variations = [
                current_value * 0.5,
                current_value * 0.8,
                current_value * 1.2,
                current_value * 2.0
            ]
            
            if isinstance(current_value, int):
                variations = [max(1, int(v)) for v in variations]
            
            return variations[:2]  # Limit to 2 variations
        
        elif isinstance(current_value, str):
            # Categorical parameters
            categorical_options = {
                'optimizer': ['COBYLA', 'SLSQP', 'L-BFGS-B', 'SPSA'],
                'ansatz': ['efficient_su2', 'real_amplitudes', 'two_local']
            }
            
            options = categorical_options.get(param_name, [current_value])
            return [opt for opt in options if opt != current_value][:2]
        
        return []
    
    def _estimate_parameter_improvement(
        self, param_name: str, new_value: Any, context: Dict[str, Any]
    ) -> float:
        """Estimate improvement from parameter change."""
        # Heuristic estimates based on parameter type
        improvements = {
            'temperature': 0.1,
            'cooling_rate': 0.08,
            'population_size': 0.12,
            'mutation_rate': 0.06,
            'annealing_time': 0.15,
            'num_reads': 0.1,
            'p_layers': 0.2,
            'shots': 0.05
        }
        
        base_improvement = improvements.get(param_name, 0.05)
        
        # Adjust based on current performance
        if len(self.performance_history) > 5:
            recent_trend = self._get_performance_trend()
            if recent_trend < 0:  # Performance degrading
                base_improvement *= 1.5
        
        return min(base_improvement, 0.5)
    
    def _estimate_backend_improvement(self, backend: str, context: Dict[str, Any]) -> float:
        """Estimate improvement from backend switch."""
        problem_size = context.get('problem_size', 0)
        
        backend_improvements = {
            'simulator': 0.05,  # Low but reliable
            'ibm_quantum': 0.3 if problem_size < 20 else 0.1,
            'dwave': 0.4 if problem_size > 10 else 0.15,
            'azure_quantum': 0.25,
            'rigetti': 0.2
        }
        
        return backend_improvements.get(backend, 0.1)
    
    def _estimate_backend_switch_cost(self, new_backend: str, current_backend: str) -> float:
        """Estimate cost of backend switch."""
        # Switching to quantum hardware is more expensive
        quantum_backends = {'ibm_quantum', 'dwave', 'rigetti', 'azure_quantum'}
        
        base_cost = 0.3
        
        if new_backend in quantum_backends:
            base_cost += 0.4
        
        if current_backend != new_backend:
            base_cost += 0.2  # General switch cost
        
        return min(base_cost, 1.0)
    
    # Adaptation execution methods
    def _execute_algorithm_switch(self, action: AdaptationAction, system: Any) -> bool:
        """Execute algorithm switch adaptation."""
        try:
            new_algorithm = action.parameters.get('new_algorithm')
            params = action.parameters.get('params', {})
            
            if hasattr(system, 'set_algorithm'):
                system.set_algorithm(new_algorithm, **params)
                return True
            else:
                logger.warning("System does not support algorithm switching")
                return False
        except Exception as e:
            logger.error(f"Error switching algorithm: {e}")
            return False
    
    def _execute_parameter_tune(self, action: AdaptationAction, system: Any) -> bool:
        """Execute parameter tuning adaptation."""
        try:
            param_name = action.parameters.get('param_name')
            new_value = action.parameters.get('new_value')
            
            if hasattr(system, 'set_parameter'):
                system.set_parameter(param_name, new_value)
                return True
            else:
                logger.warning("System does not support parameter tuning")
                return False
        except Exception as e:
            logger.error(f"Error tuning parameter: {e}")
            return False
    
    def _execute_resource_reallocation(self, action: AdaptationAction, system: Any) -> bool:
        """Execute resource reallocation adaptation."""
        try:
            reallocation_action = action.parameters.get('action')
            factor = action.parameters.get('factor', 1.0)
            
            if hasattr(system, 'adjust_resources'):
                system.adjust_resources(reallocation_action, factor)
                return True
            else:
                logger.warning("System does not support resource reallocation")
                return False
        except Exception as e:
            logger.error(f"Error reallocating resources: {e}")
            return False
    
    def _execute_backend_switch(self, action: AdaptationAction, system: Any) -> bool:
        """Execute backend switch adaptation."""
        try:
            new_backend = action.parameters.get('new_backend')
            
            if hasattr(system, 'switch_backend'):
                system.switch_backend(new_backend)
                return True
            else:
                logger.warning("System does not support backend switching")
                return False
        except Exception as e:
            logger.error(f"Error switching backend: {e}")
            return False
    
    def _execute_problem_decomposition(self, action: AdaptationAction, system: Any) -> bool:
        """Execute problem decomposition adaptation."""
        try:
            strategy = action.parameters.get('strategy')
            num_subproblems = action.parameters.get('num_subproblems', 2)
            
            if hasattr(system, 'decompose_problem'):
                system.decompose_problem(strategy, num_subproblems)
                return True
            else:
                logger.warning("System does not support problem decomposition")
                return False
        except Exception as e:
            logger.error(f"Error decomposing problem: {e}")
            return False
    
    def save_state(self, filepath: str):
        """Save adaptation engine state."""
        state = {
            'adaptation_stats': self.adaptation_stats,
            'algorithm_portfolio': self.algorithm_portfolio,
            'performance_history': list(self.performance_history),
            'context_history': list(self.context_history),
            'action_history': list(self.action_history)
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Saved adaptation engine state to {filepath}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self, filepath: str):
        """Load adaptation engine state."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.adaptation_stats = state.get('adaptation_stats', self.adaptation_stats)
            self.algorithm_portfolio = state.get('algorithm_portfolio', self.algorithm_portfolio)
            
            # Restore histories
            for item in state.get('performance_history', []):
                self.performance_history.append(item)
            for item in state.get('context_history', []):
                self.context_history.append(item)
            for item in state.get('action_history', []):
                self.action_history.append(item)
            
            logger.info(f"Loaded adaptation engine state from {filepath}")
        except Exception as e:
            logger.error(f"Error loading state: {e}")


# Example usage and demonstration
def demonstrate_realtime_adaptation():
    """Demonstrate real-time adaptation capabilities."""
    print("Real-Time Quantum Adaptation Demonstration")
    print("=" * 55)
    
    # Initialize adaptation engine
    try:
        predictor = GaussianProcessPredictor()
        print("✓ Using Gaussian Process predictor")
    except ImportError:
        predictor = DeepQLearningPredictor()
        print("✓ Using Deep Q-Learning predictor (fallback)")
    
    engine = RealTimeAdaptationEngine(predictor=predictor)
    
    # Simulate optimization run with adaptation
    print(f"\nSimulating optimization with real-time adaptation:")
    print("-" * 50)
    
    contexts = []
    performances = []
    adaptations = []
    
    for iteration in range(20):
        # Simulate changing context
        context = {
            'iteration': iteration,
            'problem_size': 25 + np.random.randint(-5, 5),
            'noise_level': max(0, 0.05 + 0.02 * np.sin(iteration / 5)),
            'resource_utilization': min(1.0, 0.3 + iteration * 0.03),
            'current_algorithm': np.random.choice(['simulated_annealing', 'qaoa', 'vqe']),
            'convergence_rate': max(0, 0.2 - iteration * 0.01 + np.random.normal(0, 0.05)),
            'time_remaining': max(10, 300 - iteration * 10)
        }
        
        # Simulate performance (degrading over time with noise)
        base_performance = 0.8 - iteration * 0.02 + np.random.normal(0, 0.1)
        performance = max(0.1, min(1.0, base_performance))
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            energy=-performance * 100,  # Lower energy is better
            solution_quality=performance,
            convergence_rate=context['convergence_rate'],
            time_to_solution=10 + np.random.exponential(5),
            resource_utilization=context['resource_utilization'],
            noise_resilience=1.0 - context['noise_level'],
            problem_size=context['problem_size'],
            algorithm_used=context['current_algorithm'],
            backend_type='simulator',
            noise_level=context['noise_level']
        )
        
        contexts.append(context)
        performances.append(performance)
        
        # Observe performance
        engine.observe_performance(context, metrics, context['current_algorithm'])
        
        # Check for adaptation opportunity
        if iteration > 5 and np.random.random() < 0.4:  # 40% chance of adaptation check
            
            # Determine trigger
            if performance < 0.3:
                trigger = AdaptationTrigger.PERFORMANCE_DEGRADATION
            elif context['convergence_rate'] < 0.05:
                trigger = AdaptationTrigger.CONVERGENCE_STALL
            else:
                trigger = AdaptationTrigger.NOISE_CHANGE
            
            # Get recommendation
            recommendation = engine.recommend_adaptation(context, trigger)
            
            if recommendation:
                adaptations.append((iteration, recommendation))
                print(f"Iter {iteration:2d}: Adapted - {recommendation.action_type} "
                      f"(expected: +{recommendation.expected_improvement:.2f}, "
                      f"confidence: {recommendation.confidence:.2f})")
                
                # Simulate improvement from adaptation
                if np.random.random() < recommendation.confidence:
                    performances[-1] = min(1.0, performance + recommendation.expected_improvement * 0.5)
            else:
                print(f"Iter {iteration:2d}: No adaptation needed (performance: {performance:.2f})")
        else:
            print(f"Iter {iteration:2d}: Continuing (performance: {performance:.2f})")
    
    # Get insights
    insights = engine.get_adaptation_insights()
    
    print(f"\nAdaptation Summary:")
    print(f"{'='*30}")
    print(f"Total observations: {insights['total_observations']}")
    print(f"Total adaptations: {insights['adaptation_stats']['total_adaptations']}")
    print(f"Success rate: {insights['success_rate']:.1%}")
    print(f"Exploration rate: {insights['exploration_exploitation_ratio']:.1%}")
    print(f"Performance trend: {insights['performance_trend']:+.3f}")
    
    if insights.get('algorithm_rankings'):
        print(f"\nAlgorithm Performance Rankings:")
        for alg, perf in sorted(insights['algorithm_rankings'].items(), 
                               key=lambda x: x[1], reverse=True):
            print(f"  {alg}: {perf:.3f}")
    
    if insights.get('feature_importance'):
        print(f"\nTop Important Features:")
        sorted_features = sorted(insights['feature_importance'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]
        for feature, importance in sorted_features:
            print(f"  {feature}: {importance:.3f}")
    
    return engine, contexts, performances, adaptations


if __name__ == "__main__":
    # Run demonstration
    engine, contexts, performances, adaptations = demonstrate_realtime_adaptation()
    
    # Additional validation
    print(f"\nValidation Tests:")
    print("-" * 20)
    
    # Test predictor initialization
    try:
        gp_predictor = GaussianProcessPredictor()
        print("✓ GaussianProcessPredictor initialized")
    except ImportError:
        print("⚠ GaussianProcessPredictor requires scikit-learn")
    
    # Test deep learning predictor
    dl_predictor = DeepQLearningPredictor()
    print("✓ DeepQLearningPredictor initialized")
    
    # Test adaptation action creation
    action = AdaptationAction(
        action_type='algorithm_switch',
        parameters={'new_algorithm': 'qaoa'},
        expected_improvement=0.2,
        confidence=0.8,
        resource_cost=0.3
    )
    print(f"✓ AdaptationAction created: {action.action_type}")
    
    # Test performance metrics
    metrics = PerformanceMetrics(
        energy=10.0,
        solution_quality=0.8,
        convergence_rate=0.1,
        time_to_solution=30.0,
        resource_utilization=0.6,
        noise_resilience=0.9,
        problem_size=20,
        algorithm_used='qaoa',
        backend_type='simulator',
        noise_level=0.05
    )
    print(f"✓ PerformanceMetrics created: efficiency={metrics.efficiency:.2f}")
    
    print(f"\nReal-time adaptation module ready for integration!")