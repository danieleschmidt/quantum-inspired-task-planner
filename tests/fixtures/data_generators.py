"""Test data generators for comprehensive testing scenarios."""

import random
from typing import List, Dict, Any
from dataclasses import dataclass

from quantum_planner.models import Agent, Task, TimeWindowTask


@dataclass
class GenerationConfig:
    """Configuration for test data generation."""
    num_agents: int
    num_tasks: int
    skill_diversity: int = 10
    max_capacity: int = 5
    max_priority: int = 10
    max_duration: int = 10
    seed: int = 42


class TestDataGenerator:
    """Generates realistic test data for various scenarios."""
    
    COMMON_SKILLS = [
        "python", "javascript", "java", "cpp", "go", "rust",
        "react", "vue", "angular", "node", "django", "flask",
        "docker", "kubernetes", "aws", "azure", "gcp",
        "postgres", "mysql", "mongodb", "redis",
        "ml", "nlp", "cv", "rl", "pytorch", "tensorflow",
        "data_analysis", "visualization", "etl", "spark",
        "frontend", "backend", "fullstack", "devops",
        "testing", "qa", "automation", "performance"
    ]
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        random.seed(config.seed)
    
    def generate_agents(self) -> List[Agent]:
        """Generate diverse agents with realistic skill combinations."""
        agents = []
        
        for i in range(self.config.num_agents):
            # Number of skills per agent (1-4)
            num_skills = random.randint(1, min(4, len(self.COMMON_SKILLS)))
            skills = random.sample(self.COMMON_SKILLS[:self.config.skill_diversity], num_skills)
            capacity = random.randint(1, self.config.max_capacity)
            
            agents.append(Agent(
                agent_id=f"agent_{i+1:03d}",
                skills=skills,
                capacity=capacity
            ))
        
        return agents
    
    def generate_tasks(self, available_skills: List[str]) -> List[Task]:
        """Generate tasks that match available agent skills."""
        tasks = []
        
        for i in range(self.config.num_tasks):
            # Number of required skills per task (1-3)
            num_required = random.randint(1, min(3, len(available_skills)))
            required_skills = random.sample(available_skills, num_required)
            priority = random.randint(1, self.config.max_priority)
            duration = random.randint(1, self.config.max_duration)
            
            tasks.append(Task(
                task_id=f"task_{i+1:03d}",
                required_skills=required_skills,
                priority=priority,
                duration=duration
            ))
        
        return tasks
    
    def generate_time_window_tasks(self, available_skills: List[str], 
                                 time_horizon: int = 50) -> List[TimeWindowTask]:
        """Generate tasks with time window constraints."""
        tasks = []
        
        for i in range(self.config.num_tasks):
            num_required = random.randint(1, min(3, len(available_skills)))
            required_skills = random.sample(available_skills, num_required)
            duration = random.randint(1, self.config.max_duration)
            
            # Generate time windows
            earliest_start = random.randint(0, time_horizon // 3)
            latest_finish = random.randint(
                earliest_start + duration, 
                time_horizon
            )
            
            tasks.append(TimeWindowTask(
                task_id=f"timed_task_{i+1:03d}",
                required_skills=required_skills,
                earliest_start=earliest_start,
                latest_finish=latest_finish,
                duration=duration
            ))
        
        return tasks
    
    def generate_problem_set(self) -> Dict[str, Any]:
        """Generate a complete problem set with agents and tasks."""
        agents = self.generate_agents()
        
        # Collect all available skills
        all_skills = set()
        for agent in agents:
            all_skills.update(agent.skills)
        available_skills = list(all_skills)
        
        tasks = self.generate_tasks(available_skills)
        
        return {
            "agents": agents,
            "tasks": tasks,
            "metadata": {
                "num_agents": len(agents),
                "num_tasks": len(tasks),
                "skill_diversity": len(available_skills),
                "available_skills": available_skills,
                "config": self.config
            }
        }


class ScenarioGenerator:
    """Generates specific test scenarios for different use cases."""
    
    @staticmethod
    def software_development_team(team_size: int = 8) -> Dict[str, Any]:
        """Generate a software development team scenario."""
        agents = [
            Agent("tech_lead", ["python", "architecture", "mentoring"], capacity=4),
            Agent("senior_backend", ["python", "postgres", "redis"], capacity=3),
            Agent("senior_frontend", ["react", "typescript", "testing"], capacity=3),
            Agent("fullstack_dev", ["python", "react", "docker"], capacity=3),
            Agent("junior_backend", ["python", "testing"], capacity=2),
            Agent("junior_frontend", ["react", "css"], capacity=2),
            Agent("devops_engineer", ["docker", "kubernetes", "aws"], capacity=3),
            Agent("qa_engineer", ["testing", "automation", "performance"], capacity=2),
        ][:team_size]
        
        tasks = [
            Task("api_design", ["python", "architecture"], priority=9, duration=3),
            Task("database_schema", ["postgres"], priority=8, duration=2),
            Task("auth_service", ["python", "redis"], priority=7, duration=4),
            Task("user_interface", ["react", "typescript"], priority=6, duration=5),
            Task("integration_tests", ["testing", "automation"], priority=5, duration=3),
            Task("deployment_pipeline", ["docker", "kubernetes"], priority=4, duration=2),
            Task("performance_testing", ["performance"], priority=4, duration=2),
            Task("documentation", ["python", "react"], priority=3, duration=1),
        ]
        
        return {"agents": agents, "tasks": tasks}
    
    @staticmethod
    def research_lab(lab_size: int = 10) -> Dict[str, Any]:
        """Generate a research lab scenario."""
        agents = [
            Agent("principal_investigator", ["ml", "nlp", "mentoring"], capacity=2),
            Agent("senior_researcher_1", ["pytorch", "nlp", "cv"], capacity=3),
            Agent("senior_researcher_2", ["tensorflow", "rl", "optimization"], capacity=3),
            Agent("postdoc_1", ["pytorch", "data_analysis"], capacity=4),
            Agent("postdoc_2", ["tensorflow", "visualization"], capacity=4),
            Agent("phd_student_1", ["ml", "etl"], capacity=5),
            Agent("phd_student_2", ["nlp", "data_analysis"], capacity=5),
            Agent("masters_student", ["visualization", "testing"], capacity=3),
            Agent("data_engineer", ["spark", "etl", "aws"], capacity=4),
            Agent("ml_engineer", ["deployment", "optimization", "docker"], capacity=3),
        ][:lab_size]
        
        tasks = [
            Task("literature_review", ["nlp"], priority=9, duration=5),
            Task("dataset_creation", ["etl", "data_analysis"], priority=8, duration=6),
            Task("baseline_model", ["pytorch"], priority=7, duration=4),
            Task("novel_architecture", ["tensorflow", "optimization"], priority=8, duration=8),
            Task("experiments", ["ml", "data_analysis"], priority=6, duration=10),
            Task("ablation_studies", ["pytorch", "visualization"], priority=5, duration=6),
            Task("paper_writing", ["ml", "visualization"], priority=4, duration=8),
            Task("model_deployment", ["deployment", "docker"], priority=3, duration=3),
        ]
        
        return {"agents": agents, "tasks": tasks}


def generate_benchmark_suite() -> Dict[str, Dict[str, Any]]:
    """Generate a comprehensive benchmark suite."""
    benchmarks = {}
    
    # Small problems (quick validation)
    for i, size in enumerate([5, 10, 15]):
        config = GenerationConfig(
            num_agents=size,
            num_tasks=size + 3,
            skill_diversity=5,
            max_capacity=3,
            seed=42 + i
        )
        generator = TestDataGenerator(config)
        benchmarks[f"small_{size}"] = generator.generate_problem_set()
    
    # Medium problems (realistic scenarios)
    for i, size in enumerate([25, 40, 60]):
        config = GenerationConfig(
            num_agents=size,
            num_tasks=int(size * 1.5),
            skill_diversity=15,
            max_capacity=5,
            seed=100 + i
        )
        generator = TestDataGenerator(config)
        benchmarks[f"medium_{size}"] = generator.generate_problem_set()
    
    # Large problems (scalability testing)
    for i, size in enumerate([100, 200, 500]):
        config = GenerationConfig(
            num_agents=size,
            num_tasks=int(size * 1.2),
            skill_diversity=25,
            max_capacity=8,
            seed=200 + i
        )
        generator = TestDataGenerator(config)
        benchmarks[f"large_{size}"] = generator.generate_problem_set()
    
    return benchmarks


if __name__ == "__main__":
    # Example usage
    config = GenerationConfig(num_agents=10, num_tasks=15)
    generator = TestDataGenerator(config)
    problem = generator.generate_problem_set()
    
    print(f"Generated {problem['metadata']['num_agents']} agents")
    print(f"Generated {problem['metadata']['num_tasks']} tasks")
    print(f"Skills available: {problem['metadata']['available_skills']}")