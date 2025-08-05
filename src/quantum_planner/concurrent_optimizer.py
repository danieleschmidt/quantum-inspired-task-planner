"""Concurrent optimization features for quantum task planner."""

import time
import threading
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
import queue
import multiprocessing

from .models import Agent, Task, Solution
from .optimizer import OptimizationBackend, OptimizationParams, BaseOptimizer

logger = logging.getLogger(__name__)


@dataclass
class OptimizationJob:
    """Represents an optimization job."""
    job_id: str
    agents: List[Agent]
    tasks: List[Task]
    objective: str
    constraints: Dict[str, Any]
    params: OptimizationParams
    submitted_at: float
    priority: int = 0
    callback: Optional[Callable[[Solution], None]] = None


@dataclass
class JobResult:
    """Result of an optimization job."""
    job_id: str
    solution: Optional[Solution]
    error: Optional[str]
    completed_at: float
    execution_time: float
    backend_used: str


class ConcurrentOptimizer:
    """Concurrent optimization manager for handling multiple optimization requests."""
    
    def __init__(self, max_workers: int = None, queue_size: int = 100):
        self.max_workers = max_workers or min(4, multiprocessing.cpu_count())
        self.queue_size = queue_size
        
        # Job management
        self.job_queue = queue.PriorityQueue(maxsize=queue_size)
        self.active_jobs: Dict[str, OptimizationJob] = {}
        self.completed_jobs: Dict[str, JobResult] = {}
        
        # Thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="quantum-optimizer"
        )
        
        # Worker threads
        self.workers = []
        self.running = False
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_execution_time': 0.0,
            'avg_execution_time': 0.0,
            'queue_peak': 0
        }
    
    def start(self) -> None:
        """Start the concurrent optimizer."""
        if self.running:
            return
        
        self.running = True
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"optimizer-worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Concurrent optimizer started with {self.max_workers} workers")
    
    def stop(self, timeout: float = 10.0) -> None:
        """Stop the concurrent optimizer."""
        if not self.running:
            return
        
        self.running = False
        
        # Signal all workers to stop
        for _ in range(self.max_workers):
            try:
                self.job_queue.put_nowait((0, None))  # Sentinel value
            except queue.Full:
                pass
        
        # Wait for workers to complete
        for worker in self.workers:
            worker.join(timeout=timeout / len(self.workers))
        
        # Shutdown executor (timeout parameter added in Python 3.9)
        try:
            self.executor.shutdown(wait=True, timeout=timeout)
        except TypeError:
            # Fallback for older Python versions
            self.executor.shutdown(wait=True)
        
        logger.info("Concurrent optimizer stopped")
    
    def submit_job(self,
                  agents: List[Agent],
                  tasks: List[Task],
                  objective: str = "minimize_makespan",
                  constraints: Optional[Dict[str, Any]] = None,
                  priority: int = 0,
                  callback: Optional[Callable[[Solution], None]] = None,
                  **params) -> str:
        """Submit an optimization job."""
        
        if not self.running:
            raise RuntimeError("Concurrent optimizer is not running")
        
        job_id = f"job_{int(time.time() * 1000000)}"
        
        job = OptimizationJob(
            job_id=job_id,
            agents=agents,
            tasks=tasks,
            objective=objective,
            constraints=constraints or {},
            params=OptimizationParams(**params),
            submitted_at=time.time(),
            priority=priority,
            callback=callback
        )
        
        try:
            # Higher priority jobs get negative priority for min-heap behavior
            self.job_queue.put_nowait((-priority, job))
            
            with self._lock:
                self.stats['total_submitted'] += 1
                current_queue_size = self.job_queue.qsize()
                if current_queue_size > self.stats['queue_peak']:
                    self.stats['queue_peak'] = current_queue_size
            
            logger.debug(f"Job {job_id} submitted with priority {priority}")
            return job_id
            
        except queue.Full:
            raise RuntimeError("Optimization queue is full")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a specific job."""
        
        # Check if job is completed
        if job_id in self.completed_jobs:
            result = self.completed_jobs[job_id]
            return {
                'status': 'completed',
                'job_id': job_id,
                'completed_at': result.completed_at,
                'execution_time': result.execution_time,
                'backend_used': result.backend_used,
                'has_solution': result.solution is not None,
                'error': result.error
            }
        
        # Check if job is active
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                'status': 'running',
                'job_id': job_id,
                'submitted_at': job.submitted_at,
                'priority': job.priority,
                'running_time': time.time() - job.submitted_at
            }
        
        # Check if job is queued
        with self.job_queue.mutex:
            for priority, job in self.job_queue.queue:
                if job and job.job_id == job_id:
                    return {
                        'status': 'queued',
                        'job_id': job_id,
                        'submitted_at': job.submitted_at,
                        'priority': job.priority,
                        'queue_position': list(self.job_queue.queue).index((priority, job)) + 1
                    }
        
        return {'status': 'not_found', 'job_id': job_id}
    
    def get_result(self, job_id: str, timeout: Optional[float] = None) -> Optional[Solution]:
        """Get result of a completed job."""
        
        start_time = time.time()
        
        while True:
            if job_id in self.completed_jobs:
                result = self.completed_jobs[job_id]
                if result.error:
                    raise RuntimeError(f"Job {job_id} failed: {result.error}")
                return result.solution
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
            
            time.sleep(0.1)  # Poll every 100ms
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job."""
        
        # Remove from queue if present
        temp_jobs = []
        cancelled = False
        
        try:
            while not self.job_queue.empty():
                priority, job = self.job_queue.get_nowait()
                if job and job.job_id == job_id:
                    cancelled = True
                    logger.info(f"Job {job_id} cancelled")
                else:
                    temp_jobs.append((priority, job))
        except queue.Empty:
            pass
        
        # Put remaining jobs back
        for priority, job in temp_jobs:
            try:
                self.job_queue.put_nowait((priority, job))
            except queue.Full:
                logger.error("Queue overflow during job cancellation")
        
        return cancelled
    
    def _worker_loop(self) -> None:
        """Main worker loop."""
        worker_name = threading.current_thread().name
        
        while self.running:
            try:
                # Get next job from queue
                priority, job = self.job_queue.get(timeout=1.0)
                
                # Check for sentinel value (shutdown signal)
                if job is None:
                    break
                
                # Process the job
                self._process_job(job, worker_name)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
    
    def _process_job(self, job: OptimizationJob, worker_name: str) -> None:
        """Process a single optimization job."""
        
        start_time = time.time()
        
        # Mark job as active
        with self._lock:
            self.active_jobs[job.job_id] = job
        
        logger.info(f"Worker {worker_name} processing job {job.job_id}")
        
        try:
            # Import here to avoid circular imports
            from .optimizer import SimulatorOptimizer
            
            # Create optimizer
            optimizer = SimulatorOptimizer()
            
            # Run optimization
            solution = optimizer.optimize(job.agents, job.tasks, job.params)
            
            execution_time = time.time() - start_time
            
            # Create result
            result = JobResult(
                job_id=job.job_id,
                solution=solution,
                error=None,
                completed_at=time.time(),
                execution_time=execution_time,
                backend_used=optimizer.backend.value
            )
            
            # Store result
            with self._lock:
                self.completed_jobs[job.job_id] = result
                if job.job_id in self.active_jobs:
                    del self.active_jobs[job.job_id]
                
                # Update statistics
                self.stats['total_completed'] += 1
                self.stats['total_execution_time'] += execution_time
                self.stats['avg_execution_time'] = (
                    self.stats['total_execution_time'] / self.stats['total_completed']
                )
            
            # Call callback if provided
            if job.callback and solution:
                try:
                    job.callback(solution)
                except Exception as e:
                    logger.error(f"Callback error for job {job.job_id}: {e}")
            
            logger.info(f"Job {job.job_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Create error result
            result = JobResult(
                job_id=job.job_id,
                solution=None,
                error=str(e),
                completed_at=time.time(),
                execution_time=execution_time,
                backend_used="unknown"
            )
            
            # Store result
            with self._lock:
                self.completed_jobs[job.job_id] = result
                if job.job_id in self.active_jobs:
                    del self.active_jobs[job.job_id]
                
                self.stats['total_failed'] += 1
            
            logger.error(f"Job {job.job_id} failed after {execution_time:.2f}s: {e}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue and worker statistics."""
        with self._lock:
            return {
                'queue_size': self.job_queue.qsize(),
                'active_jobs': len(self.active_jobs),
                'completed_jobs': len(self.completed_jobs),
                'max_workers': self.max_workers,
                'running': self.running,
                'statistics': self.stats.copy()
            }
    
    def cleanup_completed_jobs(self, max_age_seconds: float = 3600) -> int:
        """Clean up old completed jobs."""
        cutoff_time = time.time() - max_age_seconds
        
        with self._lock:
            jobs_to_remove = [
                job_id for job_id, result in self.completed_jobs.items()
                if result.completed_at < cutoff_time
            ]
            
            for job_id in jobs_to_remove:
                del self.completed_jobs[job_id]
        
        logger.info(f"Cleaned up {len(jobs_to_remove)} old completed jobs")
        return len(jobs_to_remove)


# Parallel problem decomposition
class ParallelDecomposer:
    """Decompose large problems for parallel processing."""
    
    @staticmethod
    def can_decompose(agents: List[Agent], tasks: List[Task]) -> bool:
        """Check if problem can be effectively decomposed."""
        # Simple heuristics for decomposition
        if len(agents) < 4 or len(tasks) < 6:
            return False
        
        # Check skill diversity
        skill_groups = ParallelDecomposer._group_by_skills(agents, tasks)
        return len(skill_groups) > 1
    
    @staticmethod
    def decompose(agents: List[Agent], tasks: List[Task]) -> List[Tuple[List[Agent], List[Task]]]:
        """Decompose problem into smaller subproblems."""
        
        skill_groups = ParallelDecomposer._group_by_skills(agents, tasks)
        
        subproblems = []
        for skill_set, (group_agents, group_tasks) in skill_groups.items():
            if group_agents and group_tasks:
                subproblems.append((group_agents, group_tasks))
        
        return subproblems
    
    @staticmethod
    def _group_by_skills(agents: List[Agent], tasks: List[Task]) -> Dict[str, Tuple[List[Agent], List[Task]]]:
        """Group agents and tasks by skill compatibility."""
        
        # Find all skill combinations
        skill_combinations = set()
        for task in tasks:
            skill_combinations.add(tuple(sorted(task.required_skills)))
        
        groups = {}
        
        for skill_combo in skill_combinations:
            # Find agents with these skills
            compatible_agents = [
                agent for agent in agents
                if all(skill in agent.skills for skill in skill_combo)
            ]
            
            # Find tasks requiring these skills
            compatible_tasks = [
                task for task in tasks
                if tuple(sorted(task.required_skills)) == skill_combo
            ]
            
            if compatible_agents and compatible_tasks:
                skill_key = "_".join(skill_combo)
                groups[skill_key] = (compatible_agents, compatible_tasks)
        
        return groups
    
    @staticmethod
    def merge_solutions(subproblems: List[Tuple[List[Agent], List[Task]]], 
                       solutions: List[Solution]) -> Solution:
        """Merge solutions from subproblems."""
        
        merged_assignments = {}
        total_cost = 0.0
        max_makespan = 0.0
        backends_used = set()
        
        for solution in solutions:
            if solution:
                merged_assignments.update(solution.assignments)
                total_cost += solution.cost
                max_makespan = max(max_makespan, solution.makespan)
                backends_used.add(solution.backend_used)
        
        return Solution(
            assignments=merged_assignments,
            makespan=max_makespan,
            cost=total_cost,
            backend_used=f"parallel_{'+'.join(backends_used)}",
            metadata={
                'decomposed': True,
                'subproblems': len(subproblems),
                'merge_timestamp': time.time()
            }
        )


# Global concurrent optimizer instance
concurrent_optimizer = ConcurrentOptimizer()


def optimize_concurrently(agents: List[Agent], 
                         tasks: List[Task],
                         objective: str = "minimize_makespan",
                         **kwargs) -> str:
    """Submit concurrent optimization job."""
    if not concurrent_optimizer.running:
        concurrent_optimizer.start()
    
    return concurrent_optimizer.submit_job(
        agents=agents,
        tasks=tasks,
        objective=objective,
        **kwargs
    )