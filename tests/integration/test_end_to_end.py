"""End-to-end integration tests for the quantum task planner."""

import pytest
from unittest.mock import patch, Mock

from quantum_planner import QuantumTaskPlanner
from quantum_planner.models import Agent, Task, TimeWindowTask


class TestBasicAssignment:
    """Test basic task assignment functionality."""

    def test_simple_assignment_with_mock_backend(self, sample_agents, sample_tasks):
        """Test simple assignment with mocked quantum backend."""
        # Mock the backend to return a valid assignment
        with patch('quantum_planner.backends.get_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.solve_qubo.return_value = {
                0: 1, 1: 0, 2: 0, 3: 0,  # task1 -> agent1
                4: 0, 5: 1, 6: 0, 7: 0,  # task2 -> agent2
                8: 0, 9: 0, 10: 1, 11: 0, # task3 -> agent3
                12: 0, 13: 0, 14: 0, 15: 1, # task4 -> agent4
            }
            mock_get_backend.return_value = mock_backend
            
            planner = QuantumTaskPlanner(backend="mock")
            
            solution = planner.assign(
                agents=sample_agents[:4],
                tasks=sample_tasks[:4],
                objective="minimize_makespan",
                constraints={"skill_match": True, "capacity_limit": True}
            )
            
            assert solution is not None
            assert len(solution.assignments) == 4
            assert all(task in solution.assignments for task in [t.task_id for t in sample_tasks[:4]])

    def test_infeasible_problem_handling(self):
        """Test handling of infeasible problems."""
        # Create agents and tasks that cannot be satisfied
        agents = [Agent("agent1", skills=["python"], capacity=1)]
        tasks = [
            Task("task1", required_skills=["javascript"], priority=1, duration=1),
            Task("task2", required_skills=["go"], priority=1, duration=1),
        ]
        
        planner = QuantumTaskPlanner(backend="simulator")
        
        with pytest.raises(ValueError, match="No feasible solution found"):
            planner.assign(
                agents=agents,
                tasks=tasks,
                objective="minimize_makespan",
                constraints={"skill_match": True}
            )

    def test_multi_objective_optimization(self, sample_agents, sample_tasks):
        """Test multi-objective optimization."""
        with patch('quantum_planner.backends.get_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.solve_qubo.return_value = {i: i % 2 for i in range(20)}
            mock_get_backend.return_value = mock_backend
            
            planner = QuantumTaskPlanner(backend="mock")
            
            objectives = [
                {"type": "minimize_makespan", "weight": 0.4},
                {"type": "maximize_skill_utilization", "weight": 0.3},
                {"type": "balance_workload", "weight": 0.3}
            ]
            
            solution = planner.solve_multi_objective(
                agents=sample_agents,
                tasks=sample_tasks,
                objectives=objectives
            )
            
            assert solution is not None
            assert hasattr(solution, 'pareto_front')
            assert len(solution.pareto_front) > 0


class TestTimeWindowConstraints:
    """Test time window constraint handling."""

    def test_time_window_assignment(self, sample_agents, sample_time_window_tasks):
        """Test assignment with time window constraints."""
        with patch('quantum_planner.backends.get_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.solve_qubo.return_value = {
                0: 1, 1: 0, 2: 0, 3: 0,  # urgent_fix -> agent1 at time 0
                4: 0, 5: 0, 6: 1, 7: 0,  # maintenance -> agent3 at time 10
            }
            mock_get_backend.return_value = mock_backend
            
            planner = QuantumTaskPlanner(backend="mock")
            
            solution = planner.assign_with_time(
                agents=sample_agents,
                tasks=sample_time_window_tasks,
                time_horizon=20
            )
            
            assert solution is not None
            assert hasattr(solution, 'schedule')
            
            # Check that tasks are scheduled within their time windows
            for task in sample_time_window_tasks:
                if task.task_id in solution.schedule:
                    start_time, end_time = solution.schedule[task.task_id]
                    assert start_time >= task.earliest_start
                    assert end_time <= task.latest_finish

    def test_impossible_time_window(self, sample_agents):
        """Test handling of impossible time window constraints."""
        # Create a task with impossible time window
        impossible_task = TimeWindowTask(
            "impossible",
            required_skills=["python"],
            earliest_start=10,
            latest_finish=11,
            duration=5  # Cannot fit in 1-hour window
        )
        
        planner = QuantumTaskPlanner(backend="simulator")
        
        with pytest.raises(ValueError, match="infeasible time window"):
            planner.assign_with_time(
                agents=sample_agents,
                tasks=[impossible_task],
                time_horizon=20
            )


class TestFrameworkIntegration:
    """Test integration with different agent frameworks."""

    @pytest.mark.integration
    def test_crewai_integration(self, sample_agents, sample_tasks):
        """Test CrewAI framework integration."""
        pytest.importorskip("crewai")
        
        from quantum_planner.integrations import CrewAIScheduler
        
        with patch('crewai.Crew') as mock_crew:
            scheduler = CrewAIScheduler(backend="simulator")
            
            # Mock crew setup
            mock_crew_instance = Mock()
            mock_crew.return_value = mock_crew_instance
            
            crew = scheduler.create_optimized_crew(
                agents=sample_agents,
                tasks=sample_tasks,
                objective="minimize_time"
            )
            
            assert crew is not None
            mock_crew.assert_called_once()

    @pytest.mark.integration
    def test_autogen_integration(self, sample_agents, sample_tasks):
        """Test AutoGen framework integration."""
        pytest.importorskip("autogen")
        
        from quantum_planner.integrations import AutoGenScheduler
        
        scheduler = AutoGenScheduler()
        
        # Create mock AutoGen agents
        mock_agents = [Mock() for _ in sample_agents]
        
        assignment = scheduler.assign_tasks(
            agents=mock_agents,
            tasks=[t.task_id for t in sample_tasks],
            dependencies={"task3": ["task1"]}
        )
        
        assert assignment is not None
        assert len(assignment) <= len(mock_agents)

    @pytest.mark.integration  
    def test_langchain_integration(self, sample_agents, sample_tasks):
        """Test LangChain framework integration."""
        pytest.importorskip("langchain")
        
        from quantum_planner.integrations import LangChainScheduler
        
        scheduler = LangChainScheduler(backend="simulator")
        
        # Create mock LangChain executors
        mock_executors = [Mock() for _ in sample_agents]
        
        execution_plan = scheduler.build_plan(
            agents=mock_executors,
            tasks=[t.task_id for t in sample_tasks],
            constraints={"skill_match": True}
        )
        
        assert execution_plan is not None
        assert hasattr(execution_plan, 'execute')


class TestBackendFallback:
    """Test backend fallback mechanisms."""

    def test_quantum_to_classical_fallback(self, sample_agents, sample_tasks):
        """Test fallback from quantum to classical backend."""
        # Mock quantum backend failure
        with patch('quantum_planner.backends.get_backend') as mock_get_backend:
            def side_effect(backend_type):
                if backend_type == "quantum":
                    raise ConnectionError("Quantum backend unavailable")
                else:
                    mock_backend = Mock()
                    mock_backend.solve_qubo.return_value = {i: i % 2 for i in range(20)}
                    return mock_backend
            
            mock_get_backend.side_effect = side_effect
            
            planner = QuantumTaskPlanner(
                backend="quantum",
                fallback="classical"
            )
            
            solution = planner.assign(
                agents=sample_agents,
                tasks=sample_tasks,
                objective="minimize_makespan"
            )
            
            assert solution is not None
            assert solution.backend_used == "classical"

    def test_multiple_fallback_chain(self, sample_agents, sample_tasks):
        """Test multiple fallback backends."""
        with patch('quantum_planner.backends.get_backend') as mock_get_backend:
            call_count = 0
            
            def side_effect(backend_type):
                nonlocal call_count
                call_count += 1
                
                if call_count <= 2:  # First two backends fail
                    raise ConnectionError(f"{backend_type} backend unavailable")
                else:  # Third backend succeeds
                    mock_backend = Mock()
                    mock_backend.solve_qubo.return_value = {i: i % 2 for i in range(20)}
                    return mock_backend
            
            mock_get_backend.side_effect = side_effect
            
            planner = QuantumTaskPlanner(
                backend="quantum",
                fallback_chain=["quantum_simulator", "classical", "heuristic"]
            )
            
            solution = planner.assign(
                agents=sample_agents,
                tasks=sample_tasks,
                objective="minimize_makespan"
            )
            
            assert solution is not None
            assert call_count == 3  # Tried three backends


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""

    @pytest.mark.slow
    def test_large_problem_handling(self):
        """Test handling of large problems."""
        # Create a larger problem
        num_agents = 50
        num_tasks = 75
        
        agents = [
            Agent(f"agent_{i}", skills=["python", "ml"], capacity=3)
            for i in range(num_agents)
        ]
        
        tasks = [
            Task(f"task_{i}", required_skills=["python"], priority=1, duration=1)
            for i in range(num_tasks)
        ]
        
        with patch('quantum_planner.backends.get_backend') as mock_get_backend:
            mock_backend = Mock()
            # Mock a valid assignment
            assignment = {}
            for i in range(min(num_tasks, num_agents * 3)):  # Respect capacity
                assignment[i * 4] = 1  # Assign every 4th variable
            mock_backend.solve_qubo.return_value = assignment
            mock_get_backend.return_value = mock_backend
            
            planner = QuantumTaskPlanner(backend="mock")
            
            solution = planner.assign(
                agents=agents,
                tasks=tasks,
                objective="minimize_makespan"
            )
            
            assert solution is not None
            assert len(solution.assignments) <= num_tasks

    @pytest.mark.benchmark
    def test_assignment_performance(self, benchmark, sample_agents, sample_tasks):
        """Benchmark assignment performance."""
        with patch('quantum_planner.backends.get_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.solve_qubo.return_value = {i: i % 2 for i in range(20)}
            mock_get_backend.return_value = mock_backend
            
            planner = QuantumTaskPlanner(backend="mock")
            
            def assignment_task():
                return planner.assign(
                    agents=sample_agents,
                    tasks=sample_tasks,
                    objective="minimize_makespan"
                )
            
            result = benchmark(assignment_task)
            assert result is not None

    def test_memory_usage(self, memory_profiler, sample_agents, sample_tasks):
        """Test memory usage during assignment."""
        with patch('quantum_planner.backends.get_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.solve_qubo.return_value = {i: i % 2 for i in range(20)}
            mock_get_backend.return_value = mock_backend
            
            planner = QuantumTaskPlanner(backend="mock")
            
            initial_memory = memory_profiler()
            
            solution = planner.assign(
                agents=sample_agents,
                tasks=sample_tasks,
                objective="minimize_makespan"
            )
            
            final_memory = memory_profiler()
            
            # Check that memory usage is reasonable
            memory_used = final_memory["current"] - initial_memory["current"]
            assert memory_used < 50 * 1024 * 1024  # Less than 50MB
            assert solution is not None


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        planner = QuantumTaskPlanner(backend="simulator")
        
        # Empty agents
        with pytest.raises(ValueError, match="No agents provided"):
            planner.assign(agents=[], tasks=[], objective="minimize_makespan")
        
        # Empty tasks  
        agents = [Agent("agent1", skills=["python"], capacity=1)]
        with pytest.raises(ValueError, match="No tasks provided"):
            planner.assign(agents=agents, tasks=[], objective="minimize_makespan")

    def test_invalid_constraints(self, sample_agents, sample_tasks):
        """Test handling of invalid constraint specifications."""
        planner = QuantumTaskPlanner(backend="simulator")
        
        # Invalid constraint type
        with pytest.raises(ValueError, match="Unknown constraint"):
            planner.assign(
                agents=sample_agents,
                tasks=sample_tasks,
                objective="minimize_makespan",
                constraints={"invalid_constraint": True}
            )

    def test_backend_timeout(self, sample_agents, sample_tasks):
        """Test handling of backend timeouts."""
        with patch('quantum_planner.backends.get_backend') as mock_get_backend:
            mock_backend = Mock()
            mock_backend.solve_qubo.side_effect = TimeoutError("Backend timeout")
            mock_get_backend.return_value = mock_backend
            
            planner = QuantumTaskPlanner(backend="mock", timeout=1)
            
            with pytest.raises(TimeoutError):
                planner.assign(
                    agents=sample_agents,
                    tasks=sample_tasks,
                    objective="minimize_makespan"
                )