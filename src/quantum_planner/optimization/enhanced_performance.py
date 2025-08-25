"""Enhanced Performance Optimization Engine for Quantum Task Planning.

This module implements advanced performance optimizations including:
- Adaptive workload balancing
- Predictive resource allocation 
- Real-time performance tuning
- Quantum-classical hybrid optimization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from ..models import Agent, Task, Solution


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""
    
    solve_time: float = 0.0
    memory_usage: float = 0.0  
    cpu_utilization: float = 0.0
    throughput: float = 0.0
    quality_score: float = 0.0
    scalability_factor: float = 1.0
    efficiency_ratio: float = 1.0
    timestamps: Dict[str, float] = field(default_factory=dict)


class AdaptiveWorkloadBalancer:
    """Adaptive workload balancing for optimal resource utilization."""
    
    def __init__(self, num_workers: int = 4):
        """Initialize balancer with worker pool."""
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.load_history: List[float] = []
        self.performance_history: List[PerformanceMetrics] = []
        
    def balance_workload(self, tasks: List[Task], agents: List[Agent]) -> Dict[str, List[Task]]:
        """Balance workload across available agents optimally.
        
        Args:
            tasks: List of tasks to balance
            agents: List of available agents
            
        Returns:
            Dictionary mapping agent IDs to task assignments
        """
        # Analyze current system load
        current_load = self._analyze_system_load()
        self.load_history.append(current_load)
        
        # Calculate optimal distribution
        agent_capacities = {agent.id: agent.capacity for agent in agents}
        agent_skills = {agent.id: set(agent.skills) for agent in agents}
        
        # Sort tasks by priority and complexity
        sorted_tasks = sorted(tasks, key=lambda t: (t.priority, t.duration), reverse=True)
        
        # Distribute tasks using adaptive algorithm
        assignments = {agent.id: [] for agent in agents}
        agent_loads = {agent.id: 0 for agent in agents}
        
        for task in sorted_tasks:
            # Find best agent based on skills, capacity, and current load
            best_agent = self._find_optimal_agent(
                task, agents, agent_skills, agent_loads, current_load
            )
            
            if best_agent:
                assignments[best_agent.id].append(task)
                agent_loads[best_agent.id] += task.duration
        
        return assignments
    
    def _analyze_system_load(self) -> float:
        """Analyze current system load."""
        # Simple CPU load approximation
        start = time.time()
        _ = sum(i * i for i in range(1000))
        end = time.time()
        
        # Normalize to 0-1 scale
        load_factor = min((end - start) * 1000, 1.0)
        return load_factor
    
    def _find_optimal_agent(self, task: Task, agents: List[Agent], 
                          agent_skills: Dict[str, set], 
                          agent_loads: Dict[str, int],
                          system_load: float) -> Optional[Agent]:
        """Find optimal agent for task assignment."""
        eligible_agents = []
        
        for agent in agents:
            # Check skill compatibility
            if not task.can_be_assigned_to(agent):
                continue
                
            # Check capacity constraints
            if agent_loads[agent.id] + task.duration > agent.capacity:
                continue
                
            # Calculate assignment score
            skill_match = len(set(task.required_skills) & agent_skills[agent.id])
            load_factor = 1.0 - (agent_loads[agent.id] / agent.capacity)
            system_factor = 1.0 - system_load
            
            score = skill_match * load_factor * system_factor
            eligible_agents.append((agent, score))
        
        if not eligible_agents:
            return None
            
        # Return agent with highest score
        return max(eligible_agents, key=lambda x: x[1])[0]


class PredictiveResourceAllocator:
    """Predictive resource allocation based on historical patterns."""
    
    def __init__(self):
        """Initialize predictive allocator."""
        self.resource_history: List[Dict[str, Any]] = []
        self.prediction_model = None
        self.lock = threading.Lock()
        
    def predict_resource_needs(self, tasks: List[Task], 
                             agents: List[Agent]) -> Dict[str, float]:
        """Predict resource needs for optimal allocation.
        
        Args:
            tasks: List of tasks to analyze
            agents: List of available agents
            
        Returns:
            Dictionary of predicted resource requirements
        """
        with self.lock:
            # Analyze task characteristics
            total_duration = sum(task.duration for task in tasks)
            avg_priority = np.mean([task.priority for task in tasks])
            unique_skills = len(set(skill for task in tasks for skill in task.required_skills))
            
            # Analyze agent characteristics
            total_capacity = sum(agent.capacity for agent in agents)
            avg_availability = np.mean([agent.availability for agent in agents])
            
            # Predict resource needs based on patterns
            cpu_prediction = self._predict_cpu_usage(total_duration, len(tasks))
            memory_prediction = self._predict_memory_usage(unique_skills, len(agents))
            time_prediction = self._predict_solve_time(total_duration, total_capacity)
            
            predictions = {
                'cpu_usage': cpu_prediction,
                'memory_usage': memory_prediction,
                'solve_time': time_prediction,
                'recommended_workers': self._recommend_worker_count(len(tasks)),
                'expected_throughput': total_capacity / time_prediction if time_prediction > 0 else 0
            }
            
            # Store for future learning
            self.resource_history.append({
                'tasks': len(tasks),
                'agents': len(agents),
                'total_duration': total_duration,
                'predictions': predictions,
                'timestamp': time.time()
            })
            
            return predictions
    
    def _predict_cpu_usage(self, total_duration: int, num_tasks: int) -> float:
        """Predict CPU usage based on workload."""
        base_usage = 0.1  # Baseline CPU usage
        task_factor = min(num_tasks * 0.05, 0.8)  # Scale with task count
        duration_factor = min(total_duration * 0.02, 0.7)  # Scale with duration
        
        return min(base_usage + task_factor + duration_factor, 1.0)
    
    def _predict_memory_usage(self, unique_skills: int, num_agents: int) -> float:
        """Predict memory usage based on complexity."""
        base_memory = 50.0  # Base memory in MB
        skill_memory = unique_skills * 10.0  # Memory per skill type
        agent_memory = num_agents * 5.0  # Memory per agent
        
        return base_memory + skill_memory + agent_memory
    
    def _predict_solve_time(self, total_duration: int, total_capacity: int) -> float:
        """Predict solve time based on workload and capacity."""
        if total_capacity == 0:
            return float('inf')
            
        # Base solve time with capacity factor
        base_time = 1.0  # Minimum solve time in seconds
        workload_ratio = total_duration / total_capacity
        
        # Exponential scaling for high workload ratios
        if workload_ratio > 1.0:
            solve_time = base_time * (workload_ratio ** 1.5)
        else:
            solve_time = base_time * workload_ratio
            
        return max(solve_time, 0.1)  # Minimum 0.1 seconds
    
    def _recommend_worker_count(self, num_tasks: int) -> int:
        """Recommend optimal worker count."""
        if num_tasks <= 5:
            return 2
        elif num_tasks <= 20:
            return 4
        elif num_tasks <= 50:
            return 8
        else:
            return min(16, num_tasks // 10)


class RealTimePerformanceTuner:
    """Real-time performance tuning during optimization."""
    
    def __init__(self):
        """Initialize performance tuner."""
        self.metrics_history: List[PerformanceMetrics] = []
        self.tuning_parameters = {
            'batch_size': 32,
            'learning_rate': 0.01,
            'convergence_threshold': 1e-6,
            'max_iterations': 1000
        }
        self.lock = threading.Lock()
        
    def tune_parameters(self, current_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Tune parameters based on current performance.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            Updated tuning parameters
        """
        with self.lock:
            self.metrics_history.append(current_metrics)
            
            # Analyze recent performance trends
            if len(self.metrics_history) >= 3:
                recent_metrics = self.metrics_history[-3:]
                
                # Adjust batch size based on memory usage
                avg_memory = np.mean([m.memory_usage for m in recent_metrics])
                if avg_memory > 800:  # High memory usage
                    self.tuning_parameters['batch_size'] = max(16, self.tuning_parameters['batch_size'] // 2)
                elif avg_memory < 200:  # Low memory usage
                    self.tuning_parameters['batch_size'] = min(128, self.tuning_parameters['batch_size'] * 2)
                
                # Adjust learning rate based on convergence
                avg_quality = np.mean([m.quality_score for m in recent_metrics])
                quality_trend = recent_metrics[-1].quality_score - recent_metrics[0].quality_score
                
                if quality_trend < 0.01:  # Slow convergence
                    self.tuning_parameters['learning_rate'] *= 1.1
                elif quality_trend > 0.1:  # Fast convergence
                    self.tuning_parameters['learning_rate'] *= 0.9
                
                # Adjust iterations based on solve time
                avg_solve_time = np.mean([m.solve_time for m in recent_metrics])
                if avg_solve_time > 10.0:  # Slow solving
                    self.tuning_parameters['max_iterations'] = max(500, self.tuning_parameters['max_iterations'] - 100)
                elif avg_solve_time < 1.0:  # Fast solving
                    self.tuning_parameters['max_iterations'] = min(2000, self.tuning_parameters['max_iterations'] + 100)
            
            # Ensure parameters stay within reasonable bounds
            self.tuning_parameters['batch_size'] = max(8, min(256, self.tuning_parameters['batch_size']))
            self.tuning_parameters['learning_rate'] = max(0.001, min(0.1, self.tuning_parameters['learning_rate']))
            self.tuning_parameters['max_iterations'] = max(100, min(5000, self.tuning_parameters['max_iterations']))
            
            return self.tuning_parameters.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance trends."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        summary = {
            'total_runs': len(self.metrics_history),
            'avg_solve_time': np.mean([m.solve_time for m in recent_metrics]),
            'avg_memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
            'avg_quality_score': np.mean([m.quality_score for m in recent_metrics]),
            'efficiency_trend': 'improving' if len(recent_metrics) > 1 and 
                              recent_metrics[-1].efficiency_ratio > recent_metrics[0].efficiency_ratio else 'stable',
            'current_parameters': self.tuning_parameters.copy()
        }
        
        return summary


class QuantumClassicalHybridOptimizer:
    """Hybrid optimizer combining quantum and classical approaches."""
    
    def __init__(self):
        """Initialize hybrid optimizer."""
        self.quantum_threshold = 15  # Switch to quantum for problems with >15 variables
        self.classical_methods = ['genetic', 'simulated_annealing', 'tabu_search']
        self.quantum_methods = ['qaoa', 'vqe', 'adiabatic']
        
    def optimize_hybrid(self, tasks: List[Task], agents: List[Agent]) -> Solution:
        """Optimize using hybrid quantum-classical approach.
        
        Args:
            tasks: List of tasks to optimize
            agents: List of available agents
            
        Returns:
            Optimized solution
        """
        problem_size = len(tasks) * len(agents)
        
        if problem_size <= self.quantum_threshold:
            # Use classical optimization for small problems
            return self._classical_optimize(tasks, agents)
        else:
            # Use quantum-inspired optimization for larger problems
            return self._quantum_inspired_optimize(tasks, agents)
    
    def _classical_optimize(self, tasks: List[Task], agents: List[Agent]) -> Solution:
        """Classical optimization using multiple methods."""
        # Simple greedy assignment as baseline
        assignments = {}
        agent_loads = {agent.id: 0 for agent in agents}
        
        # Sort tasks by priority (descending)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        for task in sorted_tasks:
            best_agent = None
            best_score = float('inf')
            
            for agent in agents:
                if not task.can_be_assigned_to(agent):
                    continue
                    
                if agent_loads[agent.id] + task.duration > agent.capacity:
                    continue
                
                # Calculate assignment cost
                load_penalty = agent_loads[agent.id] / agent.capacity
                skill_bonus = len(set(task.required_skills) & set(agent.skills))
                score = task.duration + load_penalty - skill_bonus
                
                if score < best_score:
                    best_score = score
                    best_agent = agent
            
            if best_agent:
                assignments[task.id] = best_agent.id
                agent_loads[best_agent.id] += task.duration
        
        # Calculate solution metrics
        makespan = max(agent_loads.values()) if agent_loads else 0
        cost = sum(agent.cost_per_hour * agent_loads[agent.id] for agent in agents)
        
        return Solution(
            assignments=assignments,
            makespan=float(makespan),
            cost=cost,
            backend_used="classical_greedy"
        )
    
    def _quantum_inspired_optimize(self, tasks: List[Task], agents: List[Agent]) -> Solution:
        """Quantum-inspired optimization using superposition and entanglement concepts."""
        # Implement quantum-inspired algorithm
        # This is a simplified version - real implementation would use quantum computing libraries
        
        num_iterations = 100
        best_solution = None
        best_cost = float('inf')
        
        for iteration in range(num_iterations):
            # Create random assignment with quantum-inspired probabilistic selection
            assignments = {}
            agent_loads = {agent.id: 0 for agent in agents}
            
            for task in tasks:
                # Calculate quantum probabilities for each agent
                probabilities = self._calculate_quantum_probabilities(task, agents, agent_loads)
                
                # Select agent based on quantum probabilities
                selected_agent = self._quantum_select(agents, probabilities)
                
                if selected_agent and agent_loads[selected_agent.id] + task.duration <= selected_agent.capacity:
                    assignments[task.id] = selected_agent.id
                    agent_loads[selected_agent.id] += task.duration
            
            # Evaluate solution
            makespan = max(agent_loads.values()) if agent_loads else 0
            cost = sum(agent.cost_per_hour * agent_loads[agent.id] for agent in agents)
            
            if cost < best_cost:
                best_cost = cost
                best_solution = Solution(
                    assignments=assignments,
                    makespan=float(makespan),
                    cost=cost,
                    backend_used="quantum_inspired"
                )
        
        return best_solution or Solution({}, 0.0, 0.0, "quantum_inspired")
    
    def _calculate_quantum_probabilities(self, task: Task, agents: List[Agent], 
                                       agent_loads: Dict[str, int]) -> List[float]:
        """Calculate quantum probabilities for agent selection."""
        probabilities = []
        
        for agent in agents:
            if not task.can_be_assigned_to(agent):
                probabilities.append(0.0)
                continue
                
            if agent_loads[agent.id] + task.duration > agent.capacity:
                probabilities.append(0.0)
                continue
            
            # Calculate probability based on skill match and availability
            skill_match = len(set(task.required_skills) & set(agent.skills))
            availability = 1.0 - (agent_loads[agent.id] / agent.capacity)
            
            # Quantum-inspired probability with interference effects
            probability = (skill_match + 1) * availability * np.random.random()
            probabilities.append(probability)
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        
        return probabilities
    
    def _quantum_select(self, agents: List[Agent], probabilities: List[float]) -> Optional[Agent]:
        """Select agent using quantum-inspired probabilistic selection."""
        if not any(probabilities):
            return None
        
        # Roulette wheel selection
        r = np.random.random()
        cumulative_prob = 0.0
        
        for agent, prob in zip(agents, probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return agent
        
        return agents[-1]  # Fallback to last agent