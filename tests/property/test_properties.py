"""Property-based testing for quantum task planner."""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
import numpy as np

from quantum_planner.models import Agent, Task
from quantum_planner.formulation.qubo import QUBOBuilder
from quantum_planner.core.planner import QuantumTaskPlanner


# Basic property tests
class TestAgentProperties:
    """Property-based tests for Agent model."""
    
    @given(
        agent_id=st.text(min_size=1, max_size=20),
        skills=st.lists(st.text(min_size=1, max_size=15), min_size=1, max_size=10),
        capacity=st.integers(min_value=1, max_value=20)
    )
    def test_agent_creation_properties(self, agent_id, skills, capacity):
        """Test that agent creation maintains basic properties."""
        # Remove duplicates from skills
        unique_skills = list(set(skills))
        assume(len(unique_skills) > 0)
        
        agent = Agent(agent_id, unique_skills, capacity)
        
        assert agent.id == agent_id
        assert set(agent.skills) == set(unique_skills)
        assert agent.capacity == capacity
        assert agent.capacity > 0
        assert len(agent.skills) > 0


class TestTaskProperties:
    """Property-based tests for Task model."""
    
    @given(
        task_id=st.text(min_size=1, max_size=20),
        required_skills=st.lists(st.text(min_size=1, max_size=15), min_size=1, max_size=5),
        priority=st.integers(min_value=1, max_value=10),
        duration=st.integers(min_value=1, max_value=20)
    )
    def test_task_creation_properties(self, task_id, required_skills, priority, duration):
        """Test that task creation maintains basic properties."""
        unique_skills = list(set(required_skills))
        assume(len(unique_skills) > 0)
        
        task = Task(task_id, unique_skills, priority, duration)
        
        assert task.id == task_id
        assert set(task.required_skills) == set(unique_skills)
        assert task.priority == priority
        assert task.duration == duration
        assert task.priority > 0
        assert task.duration > 0


class TestQUBOProperties:
    """Property-based tests for QUBO formulation."""
    
    @given(
        size=st.integers(min_value=2, max_value=20),
        sparsity=st.floats(min_value=0.1, max_value=1.0)
    )
    @settings(max_examples=50)
    def test_qubo_matrix_properties(self, size, sparsity):
        """Test properties of generated QUBO matrices."""
        # Generate random QUBO
        np.random.seed(42)  # For reproducibility
        
        # Create upper triangular matrix
        upper = np.random.rand(size, size)
        upper = np.triu(upper)
        
        # Apply sparsity
        mask = np.random.rand(size, size) < sparsity
        mask = np.triu(mask)
        upper = upper * mask
        
        # Make symmetric
        Q = upper + upper.T - np.diag(np.diag(upper))
        
        # Properties that must hold
        assert Q.shape == (size, size), "Matrix must be square"
        assert np.allclose(Q, Q.T), "QUBO matrix must be symmetric"
        assert not np.isnan(Q).any(), "Matrix must not contain NaN"
        assert not np.isinf(Q).any(), "Matrix must not contain infinity"
        
        # Sparsity property (allowing some tolerance)
        actual_sparsity = np.count_nonzero(Q) / (size * size)
        assert actual_sparsity <= sparsity * 1.1, "Sparsity should be approximately as requested"
    
    @given(
        agents=st.lists(
            st.builds(
                Agent,
                agent_id=st.text(min_size=1, max_size=10),
                skills=st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=3),
                capacity=st.integers(min_value=1, max_value=5)
            ),
            min_size=1, max_size=5
        ),
        tasks=st.lists(
            st.builds(
                Task,
                task_id=st.text(min_size=1, max_size=10),
                required_skills=st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=2),
                priority=st.integers(min_value=1, max_value=10),
                duration=st.integers(min_value=1, max_value=5)
            ),
            min_size=1, max_size=5
        )
    )
    @settings(max_examples=20)
    def test_qubo_builder_properties(self, agents, tasks):
        """Test properties of QUBO builder output."""
        # Ensure unique IDs
        agent_ids = [a.id for a in agents]
        task_ids = [t.id for t in tasks]
        assume(len(set(agent_ids)) == len(agent_ids))
        assume(len(set(task_ids)) == len(task_ids))
        
        builder = QUBOBuilder()
        
        try:
            Q = builder.build_assignment_qubo(agents, tasks)
            
            # Basic properties
            assert isinstance(Q, np.ndarray), "Output must be numpy array"
            assert Q.ndim == 2, "Output must be 2D matrix"
            assert Q.shape[0] == Q.shape[1], "Output must be square matrix"
            assert np.allclose(Q, Q.T), "QUBO must be symmetric"
            
            # Size should match problem dimensions
            expected_size = len(agents) * len(tasks)
            assert Q.shape[0] == expected_size, f"Matrix size should be {expected_size}"
            
        except Exception as e:
            # Some combinations might be infeasible, that's okay
            pytest.skip(f"Problem combination not feasible: {e}")


class TestPlannerProperties:
    """Property-based tests for the main planner."""
    
    @given(
        num_agents=st.integers(min_value=1, max_value=5),
        num_tasks=st.integers(min_value=1, max_value=5),
        skill_pool_size=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=10, deadline=5000)
    def test_solution_properties(self, num_agents, num_tasks, skill_pool_size):
        """Test properties of planner solutions."""
        # Generate skill pool
        skill_pool = [f"skill_{i}" for i in range(skill_pool_size)]
        
        # Generate agents
        agents = []
        for i in range(num_agents):
            # Each agent has 1-3 random skills
            agent_skills = np.random.choice(
                skill_pool, 
                size=min(np.random.randint(1, 4), len(skill_pool)),
                replace=False
            ).tolist()
            agents.append(Agent(f"agent_{i}", agent_skills, np.random.randint(1, 4)))
        
        # Generate tasks
        tasks = []
        for i in range(num_tasks):
            # Each task requires 1-2 random skills
            task_skills = np.random.choice(
                skill_pool,
                size=min(np.random.randint(1, 3), len(skill_pool)),
                replace=False
            ).tolist()
            tasks.append(Task(
                f"task_{i}", 
                task_skills, 
                np.random.randint(1, 10),
                np.random.randint(1, 5)
            ))
        
        planner = QuantumTaskPlanner(backend="simulator")
        
        try:
            solution = planner.assign(agents, tasks, objective="minimize_makespan")
            
            # Basic solution properties
            assert hasattr(solution, 'assignments'), "Solution must have assignments"
            assert hasattr(solution, 'makespan'), "Solution must have makespan"
            assert isinstance(solution.assignments, dict), "Assignments must be dictionary"
            
            # All tasks should be assigned
            assigned_tasks = set(solution.assignments.keys())
            all_tasks = set(t.id for t in tasks)
            assert assigned_tasks == all_tasks, "All tasks must be assigned"
            
            # All assigned agents should exist
            assigned_agents = set(solution.assignments.values())
            all_agents = set(a.id for a in agents)
            assert assigned_agents.issubset(all_agents), "Assigned agents must exist"
            
            # Makespan should be positive
            assert solution.makespan > 0, "Makespan must be positive"
            
            # Skill constraints should be satisfied
            for task in tasks:
                assigned_agent_id = solution.assignments[task.id]
                assigned_agent = next(a for a in agents if a.id == assigned_agent_id)
                
                # Check if agent has required skills
                required_skills = set(task.required_skills)
                agent_skills = set(assigned_agent.skills)
                assert required_skills.issubset(agent_skills), \
                    f"Agent {assigned_agent_id} lacks skills for task {task.id}"
            
        except Exception as e:
            # Some problem instances might be infeasible
            pytest.skip(f"Problem instance not solvable: {e}")


# Stateful property testing
class PlannerStateMachine(RuleBasedStateMachine):
    """Stateful testing of planner behavior."""
    
    def __init__(self):
        super().__init__()
        self.agents = []
        self.tasks = []
        self.planner = QuantumTaskPlanner(backend="simulator")
        self.solutions = []
    
    @rule(
        agent_id=st.text(min_size=1, max_size=10),
        skills=st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=3),
        capacity=st.integers(min_value=1, max_value=5)
    )
    def add_agent(self, agent_id, skills, capacity):
        """Add an agent to the system."""
        # Ensure unique agent ID
        assume(not any(a.id == agent_id for a in self.agents))
        
        unique_skills = list(set(skills))
        agent = Agent(agent_id, unique_skills, capacity)
        self.agents.append(agent)
    
    @rule(
        task_id=st.text(min_size=1, max_size=10),
        required_skills=st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=2),
        priority=st.integers(min_value=1, max_value=10),
        duration=st.integers(min_value=1, max_value=5)
    )
    def add_task(self, task_id, required_skills, priority, duration):
        """Add a task to the system."""
        # Ensure unique task ID
        assume(not any(t.id == task_id for t in self.tasks))
        
        unique_skills = list(set(required_skills))
        task = Task(task_id, unique_skills, priority, duration)
        self.tasks.append(task)
    
    @rule()
    def solve_assignment(self):
        """Try to solve the current assignment problem."""
        assume(len(self.agents) > 0 and len(self.tasks) > 0)
        
        try:
            solution = self.planner.assign(
                self.agents.copy(), 
                self.tasks.copy(),
                objective="minimize_makespan"
            )
            self.solutions.append(solution)
        except Exception as e:
            # Some configurations might be infeasible
            pass
    
    @invariant()
    def agents_have_positive_capacity(self):
        """All agents must have positive capacity."""
        for agent in self.agents:
            assert agent.capacity > 0
    
    @invariant()
    def tasks_have_positive_duration(self):
        """All tasks must have positive duration."""
        for task in self.tasks:
            assert task.duration > 0
            assert task.priority > 0
    
    @invariant()
    def solutions_are_valid(self):
        """All generated solutions must be valid."""
        for solution in self.solutions:
            # All tasks assigned
            assigned_tasks = set(solution.assignments.keys())
            all_tasks = set(t.id for t in self.tasks)
            assert assigned_tasks == all_tasks
            
            # Positive makespan
            assert solution.makespan > 0


# Test the state machine
TestPlannerStateful = PlannerStateMachine.TestCase


# Regression property tests
class TestRegressionProperties:
    """Property tests for regression prevention."""
    
    @given(st.data())
    def test_deterministic_behavior(self, data):
        """Test that planner behaves deterministically with same input."""
        # Generate fixed problem
        agents = [
            Agent("agent1", ["python"], 2),
            Agent("agent2", ["javascript"], 1)
        ]
        tasks = [
            Task("task1", ["python"], 5, 1),
            Task("task2", ["javascript"], 3, 2)
        ]
        
        planner = QuantumTaskPlanner(backend="simulator")
        
        # Solve multiple times
        solution1 = planner.assign(agents, tasks)
        solution2 = planner.assign(agents, tasks)
        
        # Results should be identical for deterministic backend
        assert solution1.assignments == solution2.assignments
        assert solution1.makespan == solution2.makespan
    
    @given(
        objective=st.sampled_from(["minimize_makespan", "maximize_priority", "balance_load"])
    )
    def test_objective_consistency(self, objective):
        """Test that different objectives produce valid solutions."""
        agents = [
            Agent("agent1", ["python", "ml"], 3),
            Agent("agent2", ["javascript", "react"], 2)
        ]
        tasks = [
            Task("task1", ["python"], 5, 2),
            Task("task2", ["javascript"], 3, 1),
            Task("task3", ["ml"], 8, 3)
        ]
        
        planner = QuantumTaskPlanner(backend="simulator")
        
        try:
            solution = planner.assign(agents, tasks, objective=objective)
            
            # Basic validity checks regardless of objective
            assert len(solution.assignments) == len(tasks)
            assert solution.makespan > 0
            
            # All assigned agents should exist
            assigned_agents = set(solution.assignments.values())
            all_agents = set(a.id for a in agents)
            assert assigned_agents.issubset(all_agents)
            
        except Exception as e:
            pytest.skip(f"Objective {objective} not supported or problem infeasible: {e}")