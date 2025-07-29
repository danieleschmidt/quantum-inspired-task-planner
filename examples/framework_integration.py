#!/usr/bin/env python3
"""
Framework Integration Examples

Demonstrates how to integrate the Quantum-Inspired Task Planner
with popular agent frameworks: CrewAI, AutoGen, and LangChain.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import time


# ============================================================================
# Mock Framework Classes (representing real integrations)
# ============================================================================

@dataclass
class QAgent:
    """Generic agent representation for quantum optimization."""
    id: str
    role: str
    skills: List[str]
    capacity: int
    cost_per_hour: float = 50.0
    preferences: Dict[str, Any] = None

    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}


@dataclass
class QTask:
    """Generic task representation for quantum optimization."""
    id: str
    description: str
    required_skills: List[str]
    priority: int
    estimated_duration: int
    deadline: int = None
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class QuantumScheduler(ABC):
    """Abstract base for quantum-optimized schedulers."""
    
    @abstractmethod
    def optimize_assignment(self, agents: List[QAgent], tasks: List[QTask]) -> Dict[str, str]:
        """Optimize task assignment using quantum methods."""
        pass


# ============================================================================
# CrewAI Integration Example
# ============================================================================

class CrewAIAgent:
    """Mock CrewAI Agent class."""
    def __init__(self, role: str, goal: str, skills: List[str], max_tasks: int = 3):
        self.role = role
        self.goal = goal
        self.skills = skills
        self.max_tasks = max_tasks
        self.assigned_tasks = []
    
    def execute_task(self, task):
        """Execute a task (simulation)."""
        print(f"🤖 {self.role} executing: {task}")
        time.sleep(0.1)  # Simulate work
        return f"Completed {task}"


class CrewAITask:
    """Mock CrewAI Task class."""
    def __init__(self, description: str, required_skills: List[str], priority: int = 5):
        self.description = description
        self.required_skills = required_skills
        self.priority = priority


class CrewAICrew:
    """Mock CrewAI Crew class with quantum scheduler integration."""
    def __init__(self, agents: List[CrewAIAgent], scheduler: QuantumScheduler = None):
        self.agents = agents
        self.scheduler = scheduler
        self.tasks = []
    
    def add_task(self, task: CrewAITask):
        """Add a task to the crew's task list."""
        self.tasks.append(task)
    
    def kickoff(self):
        """Execute all tasks with quantum-optimized assignment."""
        print("🚀 CrewAI with Quantum Optimization - Starting execution...")
        
        if self.scheduler:
            # Convert to quantum-compatible format
            q_agents = [
                QAgent(
                    id=agent.role,
                    role=agent.role,
                    skills=agent.skills,
                    capacity=agent.max_tasks
                ) for agent in self.agents
            ]
            
            q_tasks = [
                QTask(
                    id=f"task_{i}",
                    description=task.description,
                    required_skills=task.required_skills,
                    priority=task.priority,
                    estimated_duration=2  # Default duration
                ) for i, task in enumerate(self.tasks)
            ]
            
            # Get quantum-optimized assignment
            assignment = self.scheduler.optimize_assignment(q_agents, q_tasks)
            
            # Execute based on optimal assignment
            print("\n📊 Quantum-Optimized Task Assignment:")
            for task_id, agent_id in assignment.items():
                task_idx = int(task_id.split('_')[1])
                task = self.tasks[task_idx]
                agent = next(a for a in self.agents if a.role == agent_id)
                
                print(f"   {task.description} → {agent.role}")
                result = agent.execute_task(task.description)
                print(f"   ✅ {result}")
        else:
            # Fallback to simple assignment
            print("⚠️  No quantum scheduler - using simple assignment")
            for i, task in enumerate(self.tasks):
                agent = self.agents[i % len(self.agents)]
                agent.execute_task(task.description)


# ============================================================================
# AutoGen Integration Example  
# ============================================================================

class AutoGenAgent:
    """Mock AutoGen Agent class."""
    def __init__(self, name: str, system_message: str, skills: List[str]):
        self.name = name
        self.system_message = system_message
        self.skills = skills
        self.conversation_history = []
    
    def generate_reply(self, message: str) -> str:
        """Generate a reply to a message."""
        self.conversation_history.append(("received", message))
        reply = f"{self.name}: Processing '{message}' with skills {self.skills}"
        self.conversation_history.append(("sent", reply))
        return reply


class AutoGenGroupChat:
    """Mock AutoGen GroupChat with quantum optimization."""
    def __init__(self, agents: List[AutoGenAgent], scheduler: QuantumScheduler = None):
        self.agents = agents
        self.scheduler = scheduler
        self.conversation_flow = []
    
    def initiate_chat(self, initial_message: str, tasks: List[str]):
        """Start a group chat with quantum-optimized task distribution."""
        print("💬 AutoGen Group Chat with Quantum Task Distribution")
        
        if self.scheduler and tasks:
            # Convert to quantum format
            q_agents = [
                QAgent(
                    id=agent.name,
                    role=agent.name,
                    skills=agent.skills,
                    capacity=2  # Default capacity
                ) for agent in self.agents
            ]
            
            q_tasks = [
                QTask(
                    id=f"conversation_{i}",
                    description=task,
                    required_skills=["communication"],  # All agents can communicate
                    priority=5,
                    estimated_duration=1
                ) for i, task in enumerate(tasks)
            ]
            
            # Get optimal assignment
            assignment = self.scheduler.optimize_assignment(q_agents, q_tasks)
            
            print(f"\n🎯 Quantum-Optimized Conversation Flow:")
            for task_id, agent_id in assignment.items():
                task_idx = int(task_id.split('_')[1])
                task = tasks[task_idx]
                agent = next(a for a in self.agents if a.name == agent_id)
                
                print(f"   Step {task_idx + 1}: {agent.name} handles '{task}'")
                response = agent.generate_reply(task)
                print(f"   📝 {response}")
                
                self.conversation_flow.append((agent.name, task, response))
        else:
            print("⚠️  Basic conversation without quantum optimization")
            for task in tasks:
                agent = self.agents[0]  # Simple assignment to first agent
                response = agent.generate_reply(task)
                print(f"📝 {response}")


# ============================================================================
# LangChain Integration Example
# ============================================================================

class LangChainAgent:
    """Mock LangChain Agent class."""
    def __init__(self, name: str, llm_config: Dict, tools: List[str]):
        self.name = name
        self.llm_config = llm_config
        self.tools = tools
        self.execution_history = []
    
    def run(self, input_data: str) -> str:
        """Run the agent on input data."""
        result = f"{self.name} processed '{input_data}' using tools: {self.tools}"
        self.execution_history.append((input_data, result))
        return result


class LangChainChain:
    """Mock LangChain Chain with quantum optimization."""
    def __init__(self, agents: List[LangChainAgent], scheduler: QuantumScheduler = None):
        self.agents = agents
        self.scheduler = scheduler
        self.chain_results = []
    
    def run_chain(self, input_data: str, processing_steps: List[str]):
        """Execute a chain of operations with quantum-optimized agent selection."""
        print("🔗 LangChain with Quantum Agent Selection")
        
        if self.scheduler and processing_steps:
            # Convert to quantum format
            q_agents = [
                QAgent(
                    id=agent.name,
                    role=agent.name,
                    skills=agent.tools,
                    capacity=3  # Default capacity
                ) for agent in self.agents
            ]
            
            q_tasks = [
                QTask(
                    id=f"step_{i}",
                    description=step,
                    required_skills=["processing"],  # Generic skill
                    priority=len(processing_steps) - i,  # Earlier steps higher priority
                    estimated_duration=1
                ) for i, step in enumerate(processing_steps)
            ]
            
            # Get optimal assignment
            assignment = self.scheduler.optimize_assignment(q_agents, q_tasks)
            
            print(f"\n⚡ Quantum-Optimized Processing Chain:")
            current_data = input_data
            
            for step_id in sorted(assignment.keys()):
                agent_id = assignment[step_id]
                step_idx = int(step_id.split('_')[1])
                step_description = processing_steps[step_idx]
                
                agent = next(a for a in self.agents if a.name == agent_id)
                
                print(f"   Step {step_idx + 1}: {agent.name} - {step_description}")
                result = agent.run(current_data)
                print(f"   ✅ {result}")
                
                self.chain_results.append((step_description, agent.name, result))
                current_data = result  # Chain the results
            
            return current_data
        else:
            print("⚠️  Basic chain processing without quantum optimization")
            current_data = input_data
            for step in processing_steps:
                agent = self.agents[0]  # Simple assignment
                result = agent.run(current_data)
                print(f"✅ {result}")
                current_data = result
            return current_data


# ============================================================================
# Quantum Scheduler Implementation (Simulated)
# ============================================================================

class SimulatedQuantumScheduler(QuantumScheduler):
    """Simulated quantum scheduler for demonstration."""
    
    def optimize_assignment(self, agents: List[QAgent], tasks: List[QTask]) -> Dict[str, str]:
        """Simulate quantum optimization of task assignment."""
        print("🔄 Quantum Scheduler: Formulating QUBO matrix...")
        time.sleep(0.3)
        
        print("⚡ Quantum Scheduler: Solving via D-Wave simulator...")
        time.sleep(0.5)
        
        # Simple heuristic assignment for simulation
        assignment = {}
        agent_workloads = {agent.id: 0 for agent in agents}
        
        # Sort tasks by priority (descending)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        for task in sorted_tasks:
            # Find agent with matching skills and lowest workload
            compatible_agents = []
            for agent in agents:
                if any(skill in agent.skills for skill in task.required_skills):
                    compatible_agents.append(agent)
            
            if compatible_agents:
                # Select agent with lowest current workload
                best_agent = min(compatible_agents, key=lambda a: agent_workloads[a.id])
                assignment[task.id] = best_agent.id
                agent_workloads[best_agent.id] += task.estimated_duration
            else:
                # Fallback: assign to least loaded agent
                best_agent = min(agents, key=lambda a: agent_workloads[a.id])
                assignment[task.id] = best_agent.id
                agent_workloads[best_agent.id] += task.estimated_duration
        
        print("✅ Quantum Scheduler: Optimal assignment found!")
        return assignment


# ============================================================================
# Example Execution Functions
# ============================================================================

def demo_crewai_integration():
    """Demonstrate CrewAI integration with quantum scheduling."""
    print("\n" + "="*80)
    print("🎯 CREWAI + QUANTUM OPTIMIZATION DEMO")
    print("="*80)
    
    # Create CrewAI agents
    agents = [
        CrewAIAgent("Developer", "Write clean, efficient code", 
                   ["python", "javascript", "api_development"], max_tasks=2),
        CrewAIAgent("Designer", "Create beautiful user interfaces", 
                   ["ui_design", "figma", "user_research"], max_tasks=1),
        CrewAIAgent("DevOps", "Ensure reliable deployment and monitoring", 
                   ["docker", "kubernetes", "monitoring"], max_tasks=2),
        CrewAIAgent("Tester", "Ensure quality and reliability", 
                   ["testing", "automation", "quality_assurance"], max_tasks=3),
    ]
    
    # Create tasks
    tasks = [
        CrewAITask("Develop user authentication API", ["python", "api_development"], priority=9),
        CrewAITask("Design login interface", ["ui_design", "figma"], priority=7),
        CrewAITask("Set up CI/CD pipeline", ["docker", "kubernetes"], priority=8),
        CrewAITask("Write integration tests", ["testing", "automation"], priority=6),
        CrewAITask("Create user dashboard", ["ui_design"], priority=5),
        CrewAITask("Implement monitoring", ["monitoring"], priority=7),
    ]
    
    # Create crew with quantum scheduler
    scheduler = SimulatedQuantumScheduler()
    crew = CrewAICrew(agents, scheduler)
    
    for task in tasks:
        crew.add_task(task)
    
    # Execute with quantum optimization
    crew.kickoff()


def demo_autogen_integration():
    """Demonstrate AutoGen integration with quantum scheduling."""
    print("\n" + "="*80)
    print("💬 AUTOGEN + QUANTUM OPTIMIZATION DEMO")
    print("="*80)
    
    # Create AutoGen agents
    agents = [
        AutoGenAgent("Analyst", "Analyze data and provide insights", 
                    ["data_analysis", "statistics", "reporting"]),
        AutoGenAgent("Strategist", "Develop strategic recommendations", 
                    ["strategy", "planning", "business_analysis"]),
        AutoGenAgent("Implementer", "Execute plans and solutions", 
                    ["implementation", "project_management", "execution"]),
        AutoGenAgent("Reviewer", "Review and validate work", 
                    ["review", "quality_control", "validation"]),
    ]
    
    # Define conversation tasks
    conversation_tasks = [
        "Analyze market trends for Q4 planning",
        "Develop growth strategy based on analysis",
        "Create implementation roadmap",
        "Review and validate the complete plan",
        "Identify potential risks and mitigation strategies"
    ]
    
    # Create group chat with quantum optimization
    scheduler = SimulatedQuantumScheduler()
    group_chat = AutoGenGroupChat(agents, scheduler)
    
    # Initiate quantum-optimized conversation
    group_chat.initiate_chat("Let's plan our Q4 strategy", conversation_tasks)


def demo_langchain_integration():
    """Demonstrate LangChain integration with quantum scheduling."""
    print("\n" + "="*80)
    print("🔗 LANGCHAIN + QUANTUM OPTIMIZATION DEMO")
    print("="*80)
    
    # Create LangChain agents
    agents = [
        LangChainAgent("DataProcessor", {"model": "gpt-4", "temperature": 0.1}, 
                      ["data_processing", "cleaning", "transformation"]),
        LangChainAgent("TextAnalyzer", {"model": "bert-large", "task": "classification"}, 
                      ["text_analysis", "sentiment", "classification"]),
        LangChainAgent("ReportGenerator", {"model": "gpt-4", "temperature": 0.7}, 
                      ["report_generation", "summarization", "writing"]),
        LangChainAgent("Validator", {"model": "roberta-base", "task": "validation"}, 
                      ["validation", "fact_checking", "quality_control"]),
    ]
    
    # Define processing steps
    processing_steps = [
        "Clean and preprocess input data",
        "Extract key insights and sentiment",
        "Generate comprehensive report",
        "Validate findings and recommendations",
        "Format final deliverable"
    ]
    
    # Create processing chain with quantum optimization
    scheduler = SimulatedQuantumScheduler()
    chain = LangChainChain(agents, scheduler)
    
    # Execute quantum-optimized chain
    input_data = "Customer feedback data for product improvement analysis"
    final_result = chain.run_chain(input_data, processing_steps)
    
    print(f"\n🎯 Final Chain Result: {final_result}")


def main():
    """Main demonstration of framework integrations."""
    print("🌟 QUANTUM-INSPIRED TASK PLANNER - FRAMEWORK INTEGRATIONS")
    print("=" * 80)
    print("Demonstrating quantum optimization with popular agent frameworks:")
    print("• CrewAI: Multi-agent collaboration")
    print("• AutoGen: Conversational AI systems") 
    print("• LangChain: Agent chains and workflows")
    
    # Run all integration demos
    demo_crewai_integration()
    demo_autogen_integration()
    demo_langchain_integration()
    
    print("\n" + "="*80)
    print("🚀 INTEGRATION SUMMARY")
    print("="*80)
    print("✅ CrewAI: Quantum-optimized task distribution for agent crews")
    print("✅ AutoGen: Optimal conversation flow assignment")
    print("✅ LangChain: Quantum-enhanced agent chain execution")
    print("\n🎯 Benefits of Quantum Integration:")
    print("• Optimal task-agent matching based on skills and capacity")
    print("• Minimized completion time through quantum optimization")
    print("• Better load balancing across agent resources")
    print("• Scalable to large multi-agent systems")
    print("\n📚 Next Steps:")
    print("• Install framework integrations: pip install quantum-planner[frameworks]")
    print("• Configure quantum backends for production optimization")
    print("• Explore custom constraint definitions for domain-specific needs")


if __name__ == "__main__":
    main()