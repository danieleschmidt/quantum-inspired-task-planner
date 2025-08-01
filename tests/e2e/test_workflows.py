"""End-to-end workflow tests."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from quantum_planner import QuantumTaskPlanner, Agent, Task
from quantum_planner.integrations import CrewAIScheduler, AutoGenScheduler


class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_basic_workflow_from_json(self, tmp_path):
        """Test loading problem from JSON and solving."""
        # Create problem definition
        problem_data = {
            "agents": [
                {"id": "dev1", "skills": ["python", "ml"], "capacity": 3},
                {"id": "dev2", "skills": ["javascript", "react"], "capacity": 2}
            ],
            "tasks": [
                {"id": "api", "required_skills": ["python"], "priority": 5, "duration": 2},
                {"id": "ui", "required_skills": ["javascript", "react"], "priority": 3, "duration": 3}
            ],
            "objective": "minimize_makespan",
            "constraints": {
                "skill_match": True,
                "capacity_limit": True
            }
        }
        
        # Save to file
        problem_file = tmp_path / "problem.json"
        with open(problem_file, 'w') as f:
            json.dump(problem_data, f)
        
        # Load and solve
        with open(problem_file) as f:
            data = json.load(f)
        
        agents = [Agent(**agent_data) for agent_data in data["agents"]]
        tasks = [Task(**task_data) for task_data in data["tasks"]]
        
        planner = QuantumTaskPlanner(backend="simulator")
        solution = planner.assign(agents, tasks, objective=data["objective"])
        
        # Verify solution
        assert len(solution.assignments) == 2
        assert solution.assignments["api"] == "dev1"
        assert solution.assignments["ui"] == "dev2"
        assert solution.makespan == 3  # max(2, 3)
    
    def test_multi_objective_workflow(self):
        """Test multi-objective optimization workflow."""
        agents = [
            Agent("specialist", ["python", "ml"], capacity=2),
            Agent("generalist", ["python", "javascript", "ml"], capacity=3),
            Agent("frontend", ["javascript", "react"], capacity=2)
        ]
        
        tasks = [
            Task("ml_model", ["python", "ml"], priority=9, duration=4),
            Task("api_server", ["python"], priority=7, duration=2),
            Task("web_ui", ["javascript", "react"], priority=5, duration=3),
            Task("integration", ["python", "javascript"], priority=6, duration=2)
        ]
        
        planner = QuantumTaskPlanner(backend="simulator")
        
        # Test different objectives
        objectives = ["minimize_makespan", "maximize_priority", "balance_load"]
        solutions = {}
        
        for objective in objectives:
            solution = planner.assign(agents, tasks, objective=objective)
            solutions[objective] = solution
            
            # All solutions should be valid
            assert len(solution.assignments) == len(tasks)
            assert all(task.id in solution.assignments for task in tasks)
        
        # Solutions might differ based on objective
        makespans = [sol.makespan for sol in solutions.values()]
        assert all(ms > 0 for ms in makespans)
    
    def test_constraint_handling_workflow(self):
        """Test workflow with various constraints."""
        agents = [
            Agent("dev1", ["python"], capacity=2),
            Agent("dev2", ["python"], capacity=2),
            Agent("tester", ["testing"], capacity=3)
        ]
        
        tasks = [
            Task("implement", ["python"], priority=8, duration=3),
            Task("review", ["python"], priority=6, duration=1),
            Task("test", ["testing"], priority=5, duration=2),
            Task("deploy", ["python"], priority=4, duration=1)
        ]
        
        planner = QuantumTaskPlanner(backend="simulator")
        
        # Test with precedence constraints
        constraints = {
            "precedence": {
                "review": ["implement"],
                "test": ["implement"],
                "deploy": ["review", "test"]
            }
        }
        
        solution = planner.assign(agents, tasks, constraints=constraints)
        
        # Verify precedence is respected in solution
        assert solution.assignments["implement"] in ["dev1", "dev2"]
        assert solution.assignments["review"] in ["dev1", "dev2"] 
        assert solution.assignments["test"] == "tester"
        assert solution.assignments["deploy"] in ["dev1", "dev2"]
    
    def test_scalability_workflow(self):
        """Test workflow with larger problem sizes."""
        # Generate larger problem
        num_agents = 10
        num_tasks = 15
        skills = ["python", "javascript", "java", "go", "rust"]
        
        agents = []
        for i in range(num_agents):
            agent_skills = skills[:2 + (i % 3)]  # Each agent has 2-4 skills
            agents.append(Agent(f"agent_{i}", agent_skills, 3 + (i % 3)))
        
        tasks = []
        for i in range(num_tasks):
            required_skills = [skills[i % len(skills)]]
            tasks.append(Task(f"task_{i}", required_skills, 1 + (i % 10), 1 + (i % 5)))
        
        planner = QuantumTaskPlanner(backend="simulator")
        
        # Should solve without errors
        solution = planner.assign(agents, tasks)
        
        assert len(solution.assignments) == num_tasks
        assert solution.makespan > 0
        
        # Verify all skill constraints
        for task in tasks:
            assigned_agent_id = solution.assignments[task.id]
            assigned_agent = next(a for a in agents if a.id == assigned_agent_id)
            
            required_skills = set(task.required_skills)
            agent_skills = set(assigned_agent.skills)
            assert required_skills.issubset(agent_skills)
    
    @pytest.mark.integration
    def test_backend_fallback_workflow(self):
        """Test workflow with backend fallback."""
        agents = [Agent("dev", ["python"], 2)]
        tasks = [Task("task", ["python"], 5, 1)]
        
        # Mock quantum backend that fails
        mock_quantum = Mock()
        mock_quantum.solve_qubo.side_effect = Exception("Quantum backend unavailable")
        
        # Mock classical backend that works
        mock_classical = Mock()
        mock_classical.solve_qubo.return_value = {0: 1}
        
        with patch('quantum_planner.backends.get_backend') as mock_get_backend:
            def get_backend_side_effect(name):
                if name == "quantum":
                    return mock_quantum
                elif name == "classical":
                    return mock_classical
                else:
                    return Mock()
            
            mock_get_backend.side_effect = get_backend_side_effect
            
            planner = QuantumTaskPlanner(
                backend="quantum", 
                fallback="classical"
            )
            
            solution = planner.assign(agents, tasks)
            
            # Should have fallen back to classical
            assert solution is not None
            assert len(solution.assignments) == 1


class TestFrameworkIntegrations:
    """Test integrations with AI agent frameworks."""
    
    @pytest.mark.integration 
    def test_crewai_integration_workflow(self):
        """Test integration with CrewAI framework."""
        # Mock CrewAI components
        mock_crew_agent = Mock()
        mock_crew_agent.id = "dev_agent"
        mock_crew_agent.capabilities = ["python", "ml"]
        mock_crew_agent.max_tasks = 3
        
        mock_crew_task = Mock()
        mock_crew_task.id = "ml_task"
        mock_crew_task.required_capabilities = ["python", "ml"]
        mock_crew_task.priority = 8
        mock_crew_task.estimated_duration = 4
        
        # Test scheduler integration
        planner = QuantumTaskPlanner(backend="simulator")
        scheduler = CrewAIScheduler(planner)
        
        with patch.object(scheduler, '_convert_agent') as mock_convert_agent, \
             patch.object(scheduler, '_convert_task') as mock_convert_task, \
             patch.object(scheduler, '_convert_solution') as mock_convert_solution:
            
            mock_convert_agent.return_value = Agent("dev_agent", ["python", "ml"], 3)
            mock_convert_task.return_value = Task("ml_task", ["python", "ml"], 8, 4)
            mock_convert_solution.return_value = {"ml_task": "dev_agent"}
            
            result = scheduler.integrate_with_framework([mock_crew_agent], [mock_crew_task])
            
            # Verify integration worked
            assert result is not None
            mock_convert_agent.assert_called_once()
            mock_convert_task.assert_called_once()
            mock_convert_solution.assert_called_once()
    
    @pytest.mark.integration
    def test_autogen_integration_workflow(self):
        """Test integration with AutoGen framework."""
        # Mock AutoGen components
        mock_autogen_agent = Mock()
        mock_autogen_agent.id = "coder"
        mock_autogen_agent.skills = ["python", "testing"]
        mock_autogen_agent.capacity = 2
        
        # Test scheduler
        planner = QuantumTaskPlanner(backend="simulator")  
        scheduler = AutoGenScheduler(planner)
        
        tasks = ["implement_feature", "write_tests", "code_review"]
        dependencies = {
            "write_tests": ["implement_feature"],
            "code_review": ["implement_feature"]
        }
        
        with patch.object(scheduler, 'assign_tasks') as mock_assign:
            mock_assign.return_value = {
                mock_autogen_agent: ["implement_feature", "write_tests"]
            }
            
            result = scheduler.assign_tasks(
                agents=[mock_autogen_agent],
                tasks=tasks,
                dependencies=dependencies
            )
            
            assert result is not None
            mock_assign.assert_called_once()


class TestRealWorldScenarios:
    """Test realistic scenarios from various domains."""
    
    def test_software_development_scenario(self):
        """Test realistic software development team scenario."""
        # Development team
        agents = [
            Agent("senior_dev", ["python", "architecture", "mentoring"], 4),
            Agent("junior_dev", ["python", "testing"], 3),
            Agent("frontend_dev", ["javascript", "react", "css"], 3),
            Agent("devops", ["docker", "kubernetes", "monitoring"], 2),
            Agent("qa_engineer", ["testing", "automation"], 4)
        ]
        
        # Sprint tasks
        tasks = [
            Task("system_design", ["architecture"], 9, 3),
            Task("auth_backend", ["python"], 8, 4),
            Task("user_dashboard", ["react", "css"], 7, 5),
            Task("api_tests", ["python", "testing"], 6, 2),
            Task("ui_tests", ["javascript", "testing"], 5, 3),
            Task("deployment", ["docker", "kubernetes"], 6, 2),
            Task("monitoring_setup", ["monitoring"], 4, 1),
            Task("code_review", ["python", "mentoring"], 5, 1)
        ]
        
        planner = QuantumTaskPlanner(backend="simulator")
        solution = planner.assign(agents, tasks, objective="minimize_makespan")
        
        # Verify realistic assignments
        assert solution.assignments["system_design"] == "senior_dev"
        assert solution.assignments["auth_backend"] in ["senior_dev", "junior_dev"]
        assert solution.assignments["user_dashboard"] == "frontend_dev"
        assert solution.assignments["deployment"] == "devops"
        
        # Should have reasonable makespan for team of 5
        assert solution.makespan <= 8  # Reasonable for this team size
    
    def test_research_team_scenario(self):
        """Test research team collaboration scenario."""
        agents = [
            Agent("ml_researcher", ["pytorch", "nlp", "publications"], 3),
            Agent("data_scientist", ["pandas", "visualization", "statistics"], 4),
            Agent("research_engineer", ["optimization", "deployment"], 3),
            Agent("phd_student", ["pytorch", "experimentation"], 2)
        ]
        
        tasks = [
            Task("literature_review", ["publications"], 7, 2),
            Task("data_analysis", ["pandas", "statistics"], 8, 3),
            Task("model_development", ["pytorch", "nlp"], 9, 5),
            Task("experiments", ["pytorch", "experimentation"], 7, 4),
            Task("optimization", ["optimization"], 6, 2),
            Task("visualization", ["visualization"], 5, 1),
            Task("deployment", ["deployment"], 4, 2)
        ]
        
        planner = QuantumTaskPlanner(backend="simulator")
        solution = planner.assign(agents, tasks, objective="maximize_priority")
        
        # High priority tasks should be assigned appropriately
        assert solution.assignments["model_development"] == "ml_researcher"
        assert solution.assignments["data_analysis"] == "data_scientist"
        assert solution.assignments["optimization"] == "research_engineer"
    
    def test_content_creation_scenario(self):
        """Test content creation team scenario."""
        agents = [
            Agent("content_writer", ["writing", "research", "seo"], 4),
            Agent("copy_editor", ["editing", "proofreading"], 5),
            Agent("graphic_designer", ["design", "illustration"], 3),
            Agent("video_editor", ["video", "motion_graphics"], 2),
            Agent("social_media", ["social", "scheduling"], 3)
        ]
        
        tasks = [
            Task("blog_post", ["writing", "research"], 8, 4),
            Task("editing", ["editing"], 7, 1),
            Task("proofreading", ["proofreading"], 6, 1),
            Task("featured_image", ["design"], 5, 2),
            Task("social_graphics", ["design", "illustration"], 4, 1),
            Task("promo_video", ["video"], 6, 3),
            Task("social_posts", ["social"], 3, 1),
            Task("scheduling", ["scheduling"], 2, 1)
        ]
        
        planner = QuantumTaskPlanner(backend="simulator")
        solution = planner.assign(agents, tasks, objective="balance_load")
        
        # Verify content workflow assignments
        assert solution.assignments["blog_post"] == "content_writer"
        assert solution.assignments["editing"] == "copy_editor"
        assert solution.assignments["promo_video"] == "video_editor"
        assert solution.assignments["social_posts"] == "social_media"


class TestErrorScenarios:
    """Test error handling and edge cases."""
    
    def test_infeasible_problem_workflow(self):
        """Test handling of infeasible problems."""
        # Agent can't do required task
        agents = [Agent("dev", ["python"], 1)]
        tasks = [Task("task", ["javascript"], 5, 1)]
        
        planner = QuantumTaskPlanner(backend="simulator")
        
        with pytest.raises(Exception):  # Should raise infeasible problem error
            planner.assign(agents, tasks)
    
    def test_overconstrained_workflow(self):
        """Test handling of overconstrained problems."""
        agents = [Agent("dev", ["python"], 1)]  # Capacity 1
        tasks = [
            Task("task1", ["python"], 5, 1),
            Task("task2", ["python"], 5, 1),
            Task("task3", ["python"], 5, 1)  # Too many tasks for capacity
        ]
        
        planner = QuantumTaskPlanner(backend="simulator")
        
        # Should still find solution (tasks run sequentially)
        solution = planner.assign(agents, tasks)
        assert len(solution.assignments) == 3
        assert solution.makespan >= 3  # At least 3 time units
    
    def test_empty_problem_workflow(self):
        """Test handling of empty problems."""
        planner = QuantumTaskPlanner(backend="simulator")
        
        # Empty tasks
        with pytest.raises(ValueError):
            planner.assign([Agent("dev", ["python"], 1)], [])
        
        # Empty agents  
        with pytest.raises(ValueError):
            planner.assign([], [Task("task", ["python"], 5, 1)])
    
    def test_malformed_input_workflow(self):
        """Test handling of malformed inputs."""
        planner = QuantumTaskPlanner(backend="simulator")
        
        # Invalid agent capacity
        with pytest.raises(ValueError):
            Agent("dev", ["python"], 0)  # Zero capacity
        
        # Invalid task duration
        with pytest.raises(ValueError):
            Task("task", ["python"], 5, 0)  # Zero duration
        
        # Invalid priority
        with pytest.raises(ValueError):
            Task("task", ["python"], 0, 1)  # Zero priority