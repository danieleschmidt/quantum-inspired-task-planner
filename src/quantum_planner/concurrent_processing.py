"""Concurrent processing module for quantum task planner scalability."""

import asyncio
import concurrent.futures
import threading
import time
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import multiprocessing as mp
from functools import partial

from .models import Agent, Task, Solution
from .caching import cache_manager, cached

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing mode options."""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    ASYNC = "async"
    MULTIPROCESS = "multiprocess"
    HYBRID = "hybrid"


@dataclass
class ProcessingJob:
    """Represents a processing job."""
    job_id: str
    agents: List[Agent]
    tasks: List[Task]
    objective: str
    constraints: Dict[str, Any]
    priority: int = 1
    timeout: Optional[float] = None
    callback: Optional[Callable] = None


@dataclass
class ProcessingResult:
    """Result of a processing job."""
    job_id: str
    solution: Optional[Solution]
    execution_time: float
    processing_mode: ProcessingMode
    success: bool
    error: Optional[str] = None
    worker_id: Optional[str] = None


class WorkerPool:
    """Intelligent worker pool for concurrent processing."""
    
    def __init__(self, max_workers: int = None, mode: ProcessingMode = ProcessingMode.THREADED):
        self.mode = mode
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.completed_jobs: Dict[str, ProcessingResult] = {}
        self.stats = {
            "jobs_submitted": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0
        }
        
        # Initialize executors based on mode
        if mode == ProcessingMode.THREADED:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        elif mode == ProcessingMode.MULTIPROCESS:
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = None
        
        self.lock = threading.RLock()
    
    def submit_job(self, job: ProcessingJob) -> concurrent.futures.Future:
        """Submit a job for processing."""
        with self.lock:
            self.active_jobs[job.job_id] = job
            self.stats["jobs_submitted"] += 1
        
        if self.mode == ProcessingMode.SEQUENTIAL:
            return self._process_sequential(job)
        elif self.mode in [ProcessingMode.THREADED, ProcessingMode.MULTIPROCESS]:
            return self.executor.submit(self._process_job, job)
        elif self.mode == ProcessingMode.ASYNC:
            return asyncio.create_task(self._process_async(job))
        else:
            return self._process_hybrid(job)
    
    def _process_job(self, job: ProcessingJob) -> ProcessingResult:
        """Process a single job."""
        start_time = time.time()
        worker_id = threading.current_thread().name
        
        try:
            # Import here to avoid circular imports
            from .planner import QuantumTaskPlanner
            
            planner = QuantumTaskPlanner()
            solution = planner.assign(
                agents=job.agents,
                tasks=job.tasks,
                objective=job.objective,
                constraints=job.constraints
            )
            
            execution_time = time.time() - start_time
            result = ProcessingResult(
                job_id=job.job_id,
                solution=solution,
                execution_time=execution_time,
                processing_mode=self.mode,
                success=True,
                worker_id=worker_id
            )
            
            # Execute callback if provided
            if job.callback:
                try:
                    job.callback(result)
                except Exception as e:
                    logger.warning(f"Callback failed for job {job.job_id}: {e}")
            
            self._record_completion(result)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = ProcessingResult(
                job_id=job.job_id,
                solution=None,
                execution_time=execution_time,
                processing_mode=self.mode,
                success=False,
                error=str(e),
                worker_id=worker_id
            )
            
            self._record_failure(result)
            return result
    
    async def _process_async(self, job: ProcessingJob) -> ProcessingResult:
        """Process job asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Run CPU-bound task in thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(executor, self._process_job, job)
            return await future
    
    def _process_sequential(self, job: ProcessingJob) -> ProcessingResult:
        """Process job sequentially (for testing/debugging)."""
        return self._process_job(job)
    
    def _process_hybrid(self, job: ProcessingJob) -> concurrent.futures.Future:
        """Process job using hybrid approach."""
        # Use threaded executor to avoid serialization issues with complex objects
        problem_size = len(job.agents) * len(job.tasks)
        
        # Use threaded processing for all jobs to avoid pickling issues
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            return executor.submit(self._process_job, job)
    
    def _record_completion(self, result: ProcessingResult) -> None:
        """Record job completion."""
        with self.lock:
            self.completed_jobs[result.job_id] = result
            if result.job_id in self.active_jobs:
                del self.active_jobs[result.job_id]
            
            self.stats["jobs_completed"] += 1
            self.stats["total_execution_time"] += result.execution_time
            self.stats["avg_execution_time"] = (
                self.stats["total_execution_time"] / self.stats["jobs_completed"]
            )
    
    def _record_failure(self, result: ProcessingResult) -> None:
        """Record job failure."""
        with self.lock:
            self.completed_jobs[result.job_id] = result
            if result.job_id in self.active_jobs:
                del self.active_jobs[result.job_id]
            
            self.stats["jobs_failed"] += 1
    
    def get_job_status(self, job_id: str) -> Optional[str]:
        """Get status of a job."""
        with self.lock:
            if job_id in self.active_jobs:
                return "running"
            elif job_id in self.completed_jobs:
                result = self.completed_jobs[job_id]
                return "completed" if result.success else "failed"
            else:
                return "not_found"
    
    def get_result(self, job_id: str) -> Optional[ProcessingResult]:
        """Get result of a completed job."""
        with self.lock:
            return self.completed_jobs.get(job_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        with self.lock:
            return {
                "mode": self.mode.value,
                "max_workers": self.max_workers,
                "active_jobs": len(self.active_jobs),
                "completed_jobs": len(self.completed_jobs),
                **self.stats
            }
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the worker pool."""
        if self.executor:
            self.executor.shutdown(wait=wait)


class LoadBalancer:
    """Intelligent load balancer for distributing work."""
    
    def __init__(self):
        self.worker_pools: Dict[str, WorkerPool] = {
            "high_priority": WorkerPool(max_workers=8, mode=ProcessingMode.THREADED),
            "normal_priority": WorkerPool(max_workers=16, mode=ProcessingMode.THREADED),
            "batch_processing": WorkerPool(max_workers=4, mode=ProcessingMode.THREADED)
        }
        self.routing_stats = {
            "high_priority_jobs": 0,
            "normal_priority_jobs": 0,
            "batch_jobs": 0,
            "load_balance_decisions": 0
        }
    
    def route_job(self, job: ProcessingJob) -> Tuple[str, concurrent.futures.Future]:
        """Route job to appropriate worker pool."""
        pool_name = self._select_pool(job)
        pool = self.worker_pools[pool_name]
        future = pool.submit_job(job)
        
        # Update routing stats
        self.routing_stats[f"{pool_name.replace('_priority', '_priority_jobs').replace('processing', 'jobs')}"] += 1
        self.routing_stats["load_balance_decisions"] += 1
        
        return pool_name, future
    
    def _select_pool(self, job: ProcessingJob) -> str:
        """Select appropriate worker pool based on job characteristics."""
        # High priority jobs
        if job.priority >= 8:
            return "high_priority"
        
        # Large batch jobs
        problem_size = len(job.agents) * len(job.tasks)
        if problem_size > 1000:
            return "batch_processing"
        
        # Normal jobs
        return "normal_priority"
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all pools."""
        pool_stats = {}
        total_active = 0
        total_completed = 0
        total_failed = 0
        
        for name, pool in self.worker_pools.items():
            stats = pool.get_stats()
            pool_stats[name] = stats
            total_active += stats["active_jobs"]
            total_completed += stats["jobs_completed"]
            total_failed += stats["jobs_failed"]
        
        return {
            "pool_stats": pool_stats,
            "routing_stats": self.routing_stats,
            "global_totals": {
                "active_jobs": total_active,
                "completed_jobs": total_completed,
                "failed_jobs": total_failed,
                "success_rate": total_completed / (total_completed + total_failed) if (total_completed + total_failed) > 0 else 0
            }
        }
    
    def shutdown_all(self) -> None:
        """Shutdown all worker pools."""
        for pool in self.worker_pools.values():
            pool.shutdown()


class ConcurrentOptimizer:
    """High-level concurrent optimizer with intelligent scheduling."""
    
    def __init__(self):
        self.load_balancer = LoadBalancer()
        self.job_counter = 0
        self.job_futures: Dict[str, concurrent.futures.Future] = {}
        self.lock = threading.RLock()
    
    def optimize_concurrent(
        self,
        agents: List[Agent],
        tasks: List[Task],
        objective: str = "minimize_makespan",
        constraints: Optional[Dict[str, Any]] = None,
        priority: int = 1,
        timeout: Optional[float] = None,
        callback: Optional[Callable] = None
    ) -> str:
        """Submit optimization job for concurrent processing."""
        with self.lock:
            self.job_counter += 1
            job_id = f"job_{self.job_counter}_{int(time.time())}"
        
        job = ProcessingJob(
            job_id=job_id,
            agents=agents,
            tasks=tasks,
            objective=objective,
            constraints=constraints or {},
            priority=priority,
            timeout=timeout,
            callback=callback
        )
        
        pool_name, future = self.load_balancer.route_job(job)
        
        with self.lock:
            self.job_futures[job_id] = future
        
        logger.info(f"Submitted job {job_id} to pool {pool_name}")
        return job_id
    
    def get_result(self, job_id: str, timeout: Optional[float] = None) -> Optional[ProcessingResult]:
        """Get result of optimization job."""
        future = self.job_futures.get(job_id)
        if not future:
            return None
        
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"Job {job_id} timed out")
            return None
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            return None
    
    def wait_for_completion(self, job_ids: List[str], timeout: Optional[float] = None) -> Dict[str, ProcessingResult]:
        """Wait for multiple jobs to complete."""
        futures = {job_id: self.job_futures[job_id] for job_id in job_ids if job_id in self.job_futures}
        
        results = {}
        completed_futures = concurrent.futures.as_completed(futures.values(), timeout=timeout)
        
        for future in completed_futures:
            # Find the job_id for this future
            job_id = next(jid for jid, f in futures.items() if f is future)
            try:
                results[job_id] = future.result()
            except Exception as e:
                logger.error(f"Job {job_id} failed during wait: {e}")
                results[job_id] = ProcessingResult(
                    job_id=job_id,
                    solution=None,
                    execution_time=0.0,
                    processing_mode=ProcessingMode.THREADED,
                    success=False,
                    error=str(e)
                )
        
        return results
    
    @cached("problem_analysis", ttl=3600, priority="high")
    def analyze_problem_complexity(self, agents: List[Agent], tasks: List[Task]) -> Dict[str, Any]:
        """Analyze problem complexity for optimization decisions."""
        num_agents = len(agents)
        num_tasks = len(tasks)
        problem_size = num_agents * num_tasks
        
        # Calculate skill diversity
        all_skills = set()
        for agent in agents:
            all_skills.update(agent.skills)
        skill_diversity = len(all_skills)
        
        # Calculate task complexity
        total_duration = sum(task.duration for task in tasks)
        avg_duration = total_duration / num_tasks if num_tasks > 0 else 0
        
        # Determine complexity score
        complexity_score = (
            problem_size * 0.4 +
            skill_diversity * 0.3 +
            avg_duration * 0.3
        )
        
        return {
            "num_agents": num_agents,
            "num_tasks": num_tasks,
            "problem_size": problem_size,
            "skill_diversity": skill_diversity,
            "avg_task_duration": avg_duration,
            "complexity_score": complexity_score,
            "recommended_mode": self._recommend_processing_mode(complexity_score),
            "estimated_time": self._estimate_processing_time(complexity_score)
        }
    
    def _recommend_processing_mode(self, complexity_score: float) -> ProcessingMode:
        """Recommend processing mode based on complexity."""
        if complexity_score < 50:
            return ProcessingMode.SEQUENTIAL
        elif complexity_score < 200:
            return ProcessingMode.THREADED
        elif complexity_score < 1000:
            return ProcessingMode.HYBRID
        else:
            return ProcessingMode.MULTIPROCESS
    
    def _estimate_processing_time(self, complexity_score: float) -> float:
        """Estimate processing time in seconds."""
        # Simple linear model - would be improved with real data
        return max(0.1, complexity_score * 0.01)
    
    def batch_optimize(
        self,
        job_configs: List[Dict[str, Any]],
        max_concurrent: int = 10
    ) -> Dict[str, ProcessingResult]:
        """Optimize multiple problems concurrently with batching."""
        job_ids = []
        
        # Submit jobs in batches
        for i in range(0, len(job_configs), max_concurrent):
            batch = job_configs[i:i + max_concurrent]
            batch_job_ids = []
            
            for config in batch:
                job_id = self.optimize_concurrent(**config)
                job_ids.append(job_id)
                batch_job_ids.append(job_id)
            
            # Wait for current batch to complete before submitting next
            if i + max_concurrent < len(job_configs):
                self.wait_for_completion(batch_job_ids)
        
        # Wait for all jobs to complete
        return self.wait_for_completion(job_ids)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "load_balancer": self.load_balancer.get_global_stats(),
            "cache_stats": cache_manager.get_global_cache_stats(),
            "active_jobs": len(self.job_futures),
            "job_counter": self.job_counter
        }
    
    def shutdown(self) -> None:
        """Shutdown the concurrent optimizer."""
        self.load_balancer.shutdown_all()


# Global concurrent optimizer instance
concurrent_optimizer = ConcurrentOptimizer()


def parallel_solve(problems: List[Dict[str, Any]], max_workers: int = 4) -> List[ProcessingResult]:
    """Solve multiple problems in parallel using simple interface."""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = []
        for i, problem in enumerate(problems):
            job_id = f"parallel_{i}"
            job = ProcessingJob(job_id=job_id, **problem)
            future = executor.submit(concurrent_optimizer.load_balancer.worker_pools["normal_priority"]._process_job, job)
            futures.append(future)
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel solve failed: {e}")
                results.append(ProcessingResult(
                    job_id="unknown",
                    solution=None,
                    execution_time=0.0,
                    processing_mode=ProcessingMode.THREADED,
                    success=False,
                    error=str(e)
                ))
    
    return results