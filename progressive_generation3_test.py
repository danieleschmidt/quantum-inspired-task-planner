#!/usr/bin/env python3
"""Generation 3: MAKE IT SCALE - Performance optimization and caching testing."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import asyncio
import logging
from concurrent.futures import as_completed
from quantum_planner import QuantumTaskPlanner, Agent, Task
from quantum_planner.caching import cache_manager, cached, LRUCache, AdaptiveCache
from quantum_planner.concurrent_processing import (
    concurrent_optimizer, ConcurrentOptimizer, WorkerPool, ProcessingMode, ProcessingJob
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_caching_system():
    """Test advanced caching system."""
    logger.info("=== Generation 3: Caching System ===")
    
    try:
        # Test LRU Cache
        lru_cache = LRUCache(max_size=5, default_ttl=1)
        
        # Fill cache
        for i in range(10):
            lru_cache.put(f"key_{i}", f"value_{i}")
        
        # Should only have 5 items (max_size)
        if lru_cache.get_stats()["size"] == 5:
            logger.info("✅ LRU cache respects max_size")
        else:
            logger.error(f"❌ LRU cache size issue: {lru_cache.get_stats()['size']}")
            return False
        
        # Test cache hit/miss
        if lru_cache.get("key_9") == "value_9":  # Should hit
            logger.info("✅ LRU cache hit works")
        else:
            logger.error("❌ LRU cache hit failed")
            return False
        
        if lru_cache.get("key_0") is None:  # Should miss (evicted)
            logger.info("✅ LRU cache eviction works")
        else:
            logger.error("❌ LRU cache eviction failed")
            return False
        
        # Test TTL expiration
        time.sleep(1.1)  # Wait for TTL to expire
        if lru_cache.get("key_9") is None:  # Should be expired
            logger.info("✅ TTL expiration works")
        else:
            logger.error("❌ TTL expiration failed")
            return False
        
        # Test Adaptive Cache
        adaptive_cache = AdaptiveCache()
        
        # Test different priority levels
        adaptive_cache.put("high_key", "high_value", priority="high")
        adaptive_cache.put("normal_key", "normal_value", priority="normal")
        adaptive_cache.put("low_key", "low_value", priority="low")
        
        # Should be able to retrieve all
        if (adaptive_cache.get("high_key") == "high_value" and
            adaptive_cache.get("normal_key") == "normal_value" and
            adaptive_cache.get("low_key") == "low_value"):
            logger.info("✅ Adaptive cache multi-level storage works")
        else:
            logger.error("❌ Adaptive cache multi-level storage failed")
            return False
        
        # Test cache manager
        stats = cache_manager.get_global_cache_stats()
        if "cache_details" in stats and "global_stats" in stats:
            logger.info("✅ Cache manager statistics work")
        else:
            logger.error("❌ Cache manager statistics failed")
            return False
        
    except Exception as e:
        logger.error(f"❌ Caching test failed: {e}")
        return False
    
    return True


@cached("solution", ttl=300)
def cached_optimization_function(agents, tasks, objective):
    """Test function for caching decorator."""
    time.sleep(0.1)  # Simulate work
    return f"result_for_{len(agents)}_{len(tasks)}_{objective}"


def test_caching_decorator():
    """Test caching decorator functionality."""
    logger.info("=== Generation 3: Caching Decorator ===")
    
    try:
        agents = [Agent("agent1", skills=["python"], capacity=1)]
        tasks = [Task("task1", required_skills=["python"], priority=1, duration=1)]
        
        # First call - should be slow
        start_time = time.time()
        result1 = cached_optimization_function(agents, tasks, "minimize_makespan")
        first_call_time = time.time() - start_time
        
        # Second call - should be fast (cached)
        start_time = time.time()
        result2 = cached_optimization_function(agents, tasks, "minimize_makespan")
        second_call_time = time.time() - start_time
        
        if result1 == result2:
            logger.info("✅ Cached function returns same result")
        else:
            logger.error("❌ Cached function result mismatch")
            return False
        
        if second_call_time < first_call_time * 0.5:  # Should be much faster
            logger.info(f"✅ Caching provides speedup: {first_call_time:.3f}s -> {second_call_time:.3f}s")
        else:
            logger.error(f"❌ Caching no speedup: {first_call_time:.3f}s -> {second_call_time:.3f}s")
            return False
        
    except Exception as e:
        logger.error(f"❌ Caching decorator test failed: {e}")
        return False
    
    return True


def test_concurrent_processing():
    """Test concurrent processing system."""
    logger.info("=== Generation 3: Concurrent Processing ===")
    
    try:
        # Test WorkerPool
        worker_pool = WorkerPool(max_workers=4, mode=ProcessingMode.THREADED)
        
        # Create test job
        agents = [Agent(f"agent_{i}", skills=["python"], capacity=2) for i in range(3)]
        tasks = [Task(f"task_{i}", required_skills=["python"], priority=1, duration=1) for i in range(5)]
        
        job = ProcessingJob(
            job_id="test_job_1",
            agents=agents,
            tasks=tasks,
            objective="minimize_makespan",
            constraints={}
        )
        
        # Submit job
        future = worker_pool.submit_job(job)
        result = future.result(timeout=10)
        
        if result.success and result.solution:
            logger.info("✅ Worker pool job execution successful")
        else:
            logger.error(f"❌ Worker pool job failed: {result.error}")
            return False
        
        # Test stats
        stats = worker_pool.get_stats()
        if stats["jobs_completed"] >= 1:
            logger.info("✅ Worker pool statistics tracking works")
        else:
            logger.error("❌ Worker pool statistics failed")
            return False
        
        worker_pool.shutdown()
        
    except Exception as e:
        logger.error(f"❌ Concurrent processing test failed: {e}")
        return False
    
    return True


def test_load_balancing():
    """Test intelligent load balancing."""
    logger.info("=== Generation 3: Load Balancing ===")
    
    try:
        optimizer = ConcurrentOptimizer()
        
        # Create different types of jobs
        small_agents = [Agent(f"agent_{i}", skills=["python"], capacity=1) for i in range(2)]
        small_tasks = [Task(f"task_{i}", required_skills=["python"], priority=1, duration=1) for i in range(3)]
        
        large_agents = [Agent(f"agent_{i}", skills=["python"], capacity=2) for i in range(10)]
        large_tasks = [Task(f"task_{i}", required_skills=["python"], priority=1, duration=1) for i in range(20)]
        
        # Submit different priority jobs
        high_priority_job = optimizer.optimize_concurrent(
            agents=small_agents,
            tasks=small_tasks,
            priority=9,  # High priority
            objective="minimize_makespan"
        )
        
        normal_priority_job = optimizer.optimize_concurrent(
            agents=small_agents,
            tasks=small_tasks,
            priority=5,  # Normal priority
            objective="minimize_makespan"
        )
        
        batch_job = optimizer.optimize_concurrent(
            agents=large_agents,
            tasks=large_tasks,
            priority=1,  # Low priority, large problem
            objective="minimize_makespan"
        )
        
        # Wait for completion
        results = optimizer.wait_for_completion([high_priority_job, normal_priority_job, batch_job], timeout=15)
        
        if len(results) == 3 and all(r.success for r in results.values()):
            logger.info("✅ Load balancing successfully processed different job types")
        else:
            logger.error(f"❌ Load balancing failed: {len(results)} results, success: {[r.success for r in results.values()]}")
            return False
        
        # Check stats
        stats = optimizer.get_comprehensive_stats()
        if "load_balancer" in stats and stats["load_balancer"]["global_totals"]["completed_jobs"] >= 3:
            logger.info("✅ Load balancer statistics tracking works")
        else:
            logger.error("❌ Load balancer statistics failed")
            return False
        
        optimizer.shutdown()
        
    except Exception as e:
        logger.error(f"❌ Load balancing test failed: {e}")
        return False
    
    return True


def test_performance_optimization():
    """Test performance optimization features."""
    logger.info("=== Generation 3: Performance Optimization ===")
    
    try:
        optimizer = ConcurrentOptimizer()
        
        # Test problem analysis
        agents = [Agent(f"agent_{i}", skills=["python", "ml"], capacity=3) for i in range(5)]
        tasks = [Task(f"task_{i}", required_skills=["python"], priority=i+1, duration=2) for i in range(10)]
        
        analysis = optimizer.analyze_problem_complexity(agents, tasks)
        
        required_fields = ["num_agents", "num_tasks", "complexity_score", "recommended_mode", "estimated_time"]
        if all(field in analysis for field in required_fields):
            logger.info("✅ Problem complexity analysis works")
            logger.info(f"   Complexity: {analysis['complexity_score']:.1f}, Mode: {analysis['recommended_mode']}")
        else:
            logger.error(f"❌ Problem analysis missing fields: {required_fields}")
            return False
        
        # Test batch optimization
        job_configs = []
        for i in range(3):
            job_configs.append({
                "agents": [Agent(f"agent_{j}", skills=["python"], capacity=1) for j in range(2)],
                "tasks": [Task(f"task_{j}", required_skills=["python"], priority=1, duration=1) for j in range(3)],
                "objective": "minimize_makespan",
                "priority": 1
            })
        
        batch_results = optimizer.batch_optimize(job_configs, max_concurrent=2)
        
        if len(batch_results) == 3 and all(r.success for r in batch_results.values()):
            logger.info("✅ Batch optimization works")
        else:
            logger.error(f"❌ Batch optimization failed: {len(batch_results)} results")
            return False
        
        optimizer.shutdown()
        
    except Exception as e:
        logger.error(f"❌ Performance optimization test failed: {e}")
        return False
    
    return True


def test_scalability_benchmarks():
    """Test scalability under load."""
    logger.info("=== Generation 3: Scalability Benchmarks ===")
    
    try:
        # Test increasing problem sizes
        problem_sizes = [(5, 10), (10, 20), (20, 40)]
        execution_times = []
        
        for num_agents, num_tasks in problem_sizes:
            agents = [Agent(f"agent_{i}", skills=["python"], capacity=2) for i in range(num_agents)]
            tasks = [Task(f"task_{i}", required_skills=["python"], priority=1, duration=1) for i in range(num_tasks)]
            
            start_time = time.time()
            
            # Use concurrent optimization
            job_id = concurrent_optimizer.optimize_concurrent(
                agents=agents,
                tasks=tasks,
                objective="minimize_makespan",
                priority=1
            )
            
            result = concurrent_optimizer.get_result(job_id, timeout=10)
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            if result and result.success:
                logger.info(f"✅ Solved {num_agents}x{num_tasks} problem in {execution_time:.3f}s")
            else:
                logger.error(f"❌ Failed to solve {num_agents}x{num_tasks} problem")
                return False
        
        # Check if scaling is reasonable (should not grow exponentially)
        if len(execution_times) >= 2:
            scaling_factor = execution_times[-1] / execution_times[0]
            if scaling_factor < 10:  # Should scale reasonably
                logger.info(f"✅ Reasonable scaling factor: {scaling_factor:.2f}x")
            else:
                logger.warning(f"⚠️  High scaling factor: {scaling_factor:.2f}x")
        
        # Test cache effectiveness
        cache_stats = cache_manager.get_global_cache_stats()
        global_stats = cache_stats["global_stats"]
        if global_stats["total_hits"] > 0:
            hit_rate = global_stats["global_hit_rate"]
            logger.info(f"✅ Cache hit rate: {hit_rate:.2%}")
        else:
            logger.info("ℹ️  No cache hits recorded (may be expected for diverse problems)")
        
    except Exception as e:
        logger.error(f"❌ Scalability benchmark failed: {e}")
        return False
    
    return True


def test_memory_efficiency():
    """Test memory efficiency and resource management."""
    logger.info("=== Generation 3: Memory Efficiency ===")
    
    try:
        # Test cache memory management
        cache_stats = cache_manager.get_global_cache_stats()
        initial_memory = cache_stats["global_stats"]["total_memory_mb"]
        
        # Generate some cached data
        for i in range(100):
            cached_optimization_function(
                [Agent(f"agent_{i}", skills=["python"], capacity=1)],
                [Task(f"task_{i}", required_skills=["python"], priority=1, duration=1)],
                "minimize_makespan"
            )
        
        # Check memory usage
        cache_stats = cache_manager.get_global_cache_stats()
        final_memory = cache_stats["global_stats"]["total_memory_mb"]
        
        if final_memory < 100:  # Should stay under 100MB for test data
            logger.info(f"✅ Memory usage reasonable: {final_memory:.2f}MB")
        else:
            logger.warning(f"⚠️  High memory usage: {final_memory:.2f}MB")
        
        # Test cache cleanup
        cache_manager.clear_all_caches()
        cache_stats = cache_manager.get_global_cache_stats()
        cleared_memory = cache_stats["global_stats"]["total_memory_mb"]
        
        if cleared_memory < initial_memory:
            logger.info("✅ Cache cleanup reduces memory usage")
        else:
            logger.error("❌ Cache cleanup failed")
            return False
        
    except Exception as e:
        logger.error(f"❌ Memory efficiency test failed: {e}")
        return False
    
    return True


def main():
    """Run all Generation 3 tests."""
    logger.info("🚀 Generation 3: MAKE IT SCALE")
    logger.info("📋 Performance Optimization, Caching, Concurrency & Auto-scaling")
    
    tests = [
        ("Caching System", test_caching_system),
        ("Caching Decorator", test_caching_decorator),
        ("Concurrent Processing", test_concurrent_processing),
        ("Load Balancing", test_load_balancing),
        ("Performance Optimization", test_performance_optimization),
        ("Scalability Benchmarks", test_scalability_benchmarks),
        ("Memory Efficiency", test_memory_efficiency),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ FAIL - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("GENERATION 3 TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:.<30} {status}")
    
    logger.info(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 Generation 3 complete! System now SCALES efficiently")
        logger.info("📈 Ready for Quality Gates and Production Deployment")
        
        # Show final performance stats
        stats = concurrent_optimizer.get_comprehensive_stats()
        logger.info(f"\n📊 Final Performance Summary:")
        logger.info(f"   Cache hit rate: {stats['cache_stats']['global_stats']['global_hit_rate']:.2%}")
        logger.info(f"   Total memory usage: {stats['cache_stats']['global_stats']['total_memory_mb']:.2f}MB")
        logger.info(f"   Jobs processed: {stats['load_balancer']['global_totals']['completed_jobs']}")
        logger.info(f"   Success rate: {stats['load_balancer']['global_totals']['success_rate']:.2%}")
        
        return True
    else:
        logger.error("💥 Generation 3 failed! Fix performance issues before proceeding")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)