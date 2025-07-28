"""Unit tests for core data models."""

import pytest
from hypothesis import given, strategies as st

from quantum_planner.models import Agent, Task, TimeWindowTask, Solution


class TestAgent:
    """Test cases for the Agent model."""

    def test_agent_creation(self):
        """Test basic agent creation."""
        agent = Agent("agent1", skills=["python", "ml"], capacity=3)
        
        assert agent.agent_id == "agent1"
        assert agent.skills == ["python", "ml"]
        assert agent.capacity == 3
        assert agent.availability == 1.0  # Default
        assert agent.preferences == {}    # Default

    def test_agent_with_optional_params(self):
        """Test agent creation with optional parameters."""
        agent = Agent(
            "agent2",
            skills=["javascript"],
            capacity=2,
            availability=0.8,
            preferences={"priority_tasks": True}
        )
        
        assert agent.availability == 0.8
        assert agent.preferences == {"priority_tasks": True}

    def test_agent_validation(self):
        """Test agent parameter validation."""
        # Test empty skills
        with pytest.raises(ValueError, match="Skills cannot be empty"):
            Agent("agent1", skills=[], capacity=1)
        
        # Test invalid capacity
        with pytest.raises(ValueError, match="Capacity must be positive"):
            Agent("agent1", skills=["python"], capacity=0)
        
        # Test invalid availability
        with pytest.raises(ValueError, match="Availability must be between 0 and 1"):
            Agent("agent1", skills=["python"], capacity=1, availability=1.5)

    def test_agent_equality(self):
        """Test agent equality comparison."""
        agent1 = Agent("agent1", skills=["python"], capacity=1)
        agent2 = Agent("agent1", skills=["python"], capacity=1)
        agent3 = Agent("agent2", skills=["python"], capacity=1)
        
        assert agent1 == agent2
        assert agent1 != agent3

    def test_agent_hash(self):
        """Test agent hashing for use in sets/dicts."""
        agent1 = Agent("agent1", skills=["python"], capacity=1)
        agent2 = Agent("agent1", skills=["python"], capacity=1)
        
        assert hash(agent1) == hash(agent2)
        assert {agent1, agent2} == {agent1}  # Should be same in set

    def test_agent_repr(self):
        """Test agent string representation."""
        agent = Agent("agent1", skills=["python", "ml"], capacity=3)
        repr_str = repr(agent)
        
        assert "agent1" in repr_str
        assert "python" in repr_str
        assert "ml" in repr_str
        assert "3" in repr_str

    @given(
        agent_id=st.text(min_size=1, max_size=20),
        skills=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=10),
        capacity=st.integers(min_value=1, max_value=100)
    )
    def test_agent_property_based(self, agent_id, skills, capacity):
        """Property-based test for agent creation."""
        agent = Agent(agent_id, skills=skills, capacity=capacity)
        
        assert agent.agent_id == agent_id
        assert agent.skills == skills
        assert agent.capacity == capacity
        assert 0 <= agent.availability <= 1
        assert isinstance(agent.preferences, dict)


class TestTask:
    """Test cases for the Task model."""

    def test_task_creation(self):
        """Test basic task creation."""
        task = Task("task1", required_skills=["python"], priority=5, duration=2)
        
        assert task.task_id == "task1"
        assert task.required_skills == ["python"]
        assert task.priority == 5
        assert task.duration == 2
        assert task.dependencies == []  # Default

    def test_task_with_dependencies(self):
        """Test task creation with dependencies."""
        task = Task(
            "task1",
            required_skills=["python"],
            priority=5,
            duration=2,
            dependencies=["task0"]
        )
        
        assert task.dependencies == ["task0"]

    def test_task_validation(self):
        """Test task parameter validation."""
        # Test empty required skills
        with pytest.raises(ValueError, match="Required skills cannot be empty"):
            Task("task1", required_skills=[], priority=1, duration=1)
        
        # Test invalid priority
        with pytest.raises(ValueError, match="Priority must be positive"):
            Task("task1", required_skills=["python"], priority=0, duration=1)
        
        # Test invalid duration
        with pytest.raises(ValueError, match="Duration must be positive"):
            Task("task1", required_skills=["python"], priority=1, duration=0)

    def test_task_equality(self):
        """Test task equality comparison."""
        task1 = Task("task1", required_skills=["python"], priority=1, duration=1)
        task2 = Task("task1", required_skills=["python"], priority=1, duration=1)
        task3 = Task("task2", required_skills=["python"], priority=1, duration=1)
        
        assert task1 == task2
        assert task1 != task3

    def test_task_skill_matching(self):
        """Test task skill matching logic."""
        task = Task("task1", required_skills=["python", "ml"], priority=1, duration=1)
        
        # Agent with all required skills
        agent1 = Agent("agent1", skills=["python", "ml", "devops"], capacity=1)
        assert task.can_be_assigned_to(agent1)
        
        # Agent with partial skills
        agent2 = Agent("agent2", skills=["python"], capacity=1)
        assert not task.can_be_assigned_to(agent2)
        
        # Agent with no matching skills
        agent3 = Agent("agent3", skills=["javascript"], capacity=1)
        assert not task.can_be_assigned_to(agent3)


class TestTimeWindowTask:
    """Test cases for the TimeWindowTask model."""

    def test_time_window_task_creation(self):
        """Test time window task creation."""
        task = TimeWindowTask(
            "task1",
            required_skills=["python"],
            earliest_start=0,
            latest_finish=10,
            duration=3
        )
        
        assert task.earliest_start == 0
        assert task.latest_finish == 10
        assert task.duration == 3

    def test_time_window_validation(self):
        """Test time window validation."""
        # Test invalid time window (impossible to complete)
        with pytest.raises(ValueError, match="Task cannot be completed within time window"):
            TimeWindowTask(
                "task1",
                required_skills=["python"],
                earliest_start=5,
                latest_finish=6,
                duration=3
            )
        
        # Test negative earliest start
        with pytest.raises(ValueError, match="Earliest start must be non-negative"):
            TimeWindowTask(
                "task1",
                required_skills=["python"],
                earliest_start=-1,
                latest_finish=10,
                duration=3
            )

    def test_time_window_feasibility(self):
        """Test time window feasibility checks."""
        task = TimeWindowTask(
            "task1",
            required_skills=["python"],
            earliest_start=2,
            latest_finish=8,
            duration=3
        )
        
        # Valid time slots
        assert task.is_feasible_at_time(2)  # Start at earliest
        assert task.is_feasible_at_time(5)  # Finish exactly at latest
        
        # Invalid time slots
        assert not task.is_feasible_at_time(1)  # Too early
        assert not task.is_feasible_at_time(6)  # Would finish too late


class TestSolution:
    """Test cases for the Solution model."""

    def test_solution_creation(self):
        """Test solution creation."""
        assignments = {"task1": "agent1", "task2": "agent2"}
        solution = Solution(
            assignments=assignments,
            makespan=10.0,
            cost=5.0,
            backend_used="quantum"
        )
        
        assert solution.assignments == assignments
        assert solution.makespan == 10.0
        assert solution.cost == 5.0
        assert solution.backend_used == "quantum"

    def test_solution_validation(self):
        """Test solution validation."""
        # Test empty assignments
        with pytest.raises(ValueError, match="Assignments cannot be empty"):
            Solution(assignments={}, makespan=0, cost=0, backend_used="test")
        
        # Test negative makespan
        with pytest.raises(ValueError, match="Makespan must be non-negative"):
            Solution(
                assignments={"task1": "agent1"},
                makespan=-1,
                cost=0,
                backend_used="test"
            )

    def test_solution_metrics(self):
        """Test solution metric calculations."""
        assignments = {
            "task1": "agent1",
            "task2": "agent1", 
            "task3": "agent2"
        }
        solution = Solution(
            assignments=assignments,
            makespan=10.0,
            cost=5.0,
            backend_used="test"
        )
        
        # Test load distribution
        load_distribution = solution.get_load_distribution()
        assert load_distribution["agent1"] == 2
        assert load_distribution["agent2"] == 1
        
        # Test assigned agents
        assigned_agents = solution.get_assigned_agents()
        assert assigned_agents == {"agent1", "agent2"}
        
        # Test task count
        assert solution.get_task_count() == 3

    def test_solution_quality_score(self):
        """Test solution quality scoring."""
        solution = Solution(
            assignments={"task1": "agent1"},
            makespan=10.0,
            cost=5.0,
            backend_used="test"
        )
        
        # Test quality score calculation
        quality = solution.calculate_quality_score(
            weight_makespan=0.5,
            weight_cost=0.3,
            weight_balance=0.2
        )
        
        assert isinstance(quality, float)
        assert 0 <= quality <= 1

    def test_solution_serialization(self):
        """Test solution serialization/deserialization."""
        solution = Solution(
            assignments={"task1": "agent1"},
            makespan=10.0,
            cost=5.0,
            backend_used="test"
        )
        
        # Test to dict
        solution_dict = solution.to_dict()
        assert isinstance(solution_dict, dict)
        assert "assignments" in solution_dict
        assert "makespan" in solution_dict
        
        # Test from dict
        restored_solution = Solution.from_dict(solution_dict)
        assert restored_solution.assignments == solution.assignments
        assert restored_solution.makespan == solution.makespan