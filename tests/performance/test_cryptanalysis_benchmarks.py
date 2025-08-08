"""Performance benchmarks for cryptanalysis operations."""

import pytest
import torch
import numpy as np
import time
import psutil
import statistics
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path

try:
    from quantum_planner.research.enhanced_neural_cryptanalysis import (
        create_enhanced_cryptanalysis_framework,
        analyze_cipher_securely
    )
    from quantum_planner.research.cryptanalysis_performance import (
        create_performance_optimizer,
        optimize_tensor_operations
    )
    from quantum_planner.research.cryptanalysis_security import SecurityLevel
    BENCHMARKS_AVAILABLE = True
except ImportError:
    BENCHMARKS_AVAILABLE = False
    pytest.skip("Benchmark dependencies not available", allow_module_level=True)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    operation_name: str
    data_size: int
    execution_time: float
    memory_usage_mb: float
    throughput_mb_per_sec: float
    iterations: int
    gpu_used: bool = False
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class CryptanalysisBenchmarkSuite:
    """Comprehensive benchmark suite for cryptanalysis operations."""
    
    def __init__(self):
        self.results = []
        self.baseline_results = {}
        
    def benchmark_operation(
        self,
        operation_func,
        operation_name: str,
        data_sizes: List[int],
        iterations: int = 3,
        **kwargs
    ) -> List[BenchmarkResult]:
        """Benchmark an operation across different data sizes."""
        
        results = []
        
        for data_size in data_sizes:
            print(f"Benchmarking {operation_name} with {data_size} bytes...")
            
            # Generate test data
            test_data = torch.randint(0, 256, (data_size,), dtype=torch.uint8)
            
            execution_times = []
            memory_usages = []
            
            for iteration in range(iterations):
                # Measure memory before
                process = psutil.Process()
                memory_before = process.memory_info().rss
                
                # Clear caches
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Benchmark execution
                start_time = time.time()
                
                try:
                    result = operation_func(test_data, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Measure memory after
                    memory_after = process.memory_info().rss
                    memory_usage = (memory_after - memory_before) / 1e6  # MB
                    
                    execution_times.append(execution_time)
                    memory_usages.append(memory_usage)
                    
                except Exception as e:
                    print(f"Benchmark failed for {operation_name} at size {data_size}: {e}")
                    execution_times.append(float('inf'))
                    memory_usages.append(0)
            
            # Calculate statistics
            if execution_times and all(t != float('inf') for t in execution_times):
                avg_execution_time = statistics.mean(execution_times)
                avg_memory_usage = statistics.mean(memory_usages)
                throughput = (data_size / 1e6) / avg_execution_time  # MB/s
                
                benchmark_result = BenchmarkResult(
                    operation_name=operation_name,
                    data_size=data_size,
                    execution_time=avg_execution_time,
                    memory_usage_mb=avg_memory_usage,
                    throughput_mb_per_sec=throughput,
                    iterations=iterations,
                    gpu_used=torch.cuda.is_available()
                )
                
                results.append(benchmark_result)
                self.results.append(benchmark_result)
            
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Group results by operation
        operations = {}
        for result in self.results:
            if result.operation_name not in operations:
                operations[result.operation_name] = []
            operations[result.operation_name].append(result)
        
        report = {
            "summary": {
                "total_operations": len(operations),
                "total_benchmarks": len(self.results),
                "gpu_available": torch.cuda.is_available(),
                "timestamp": time.time()
            },
            "operations": {},
            "performance_analysis": {}
        }
        
        # Analyze each operation
        for op_name, op_results in operations.items():
            execution_times = [r.execution_time for r in op_results]
            throughputs = [r.throughput_mb_per_sec for r in op_results]
            memory_usages = [r.memory_usage_mb for r in op_results]
            
            report["operations"][op_name] = {
                "sample_count": len(op_results),
                "execution_time": {
                    "mean": statistics.mean(execution_times),
                    "median": statistics.median(execution_times),
                    "min": min(execution_times),
                    "max": max(execution_times),
                    "std": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                },
                "throughput_mb_per_sec": {
                    "mean": statistics.mean(throughputs),
                    "median": statistics.median(throughputs),
                    "min": min(throughputs),
                    "max": max(throughputs)
                },
                "memory_usage_mb": {
                    "mean": statistics.mean(memory_usages),
                    "max": max(memory_usages)
                },
                "scalability": self._analyze_scalability(op_results)
            }
        
        # Overall performance analysis
        all_throughputs = [r.throughput_mb_per_sec for r in self.results]
        all_execution_times = [r.execution_time for r in self.results]
        
        report["performance_analysis"] = {
            "overall_throughput_mb_per_sec": statistics.mean(all_throughputs),
            "overall_avg_execution_time": statistics.mean(all_execution_times),
            "performance_grade": self._calculate_performance_grade(all_throughputs),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _analyze_scalability(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze scalability characteristics."""
        if len(results) < 2:
            return {"insufficient_data": True}
        
        # Sort by data size
        sorted_results = sorted(results, key=lambda r: r.data_size)
        
        # Calculate scaling factor
        first = sorted_results[0]
        last = sorted_results[-1]
        
        size_ratio = last.data_size / first.data_size
        time_ratio = last.execution_time / first.execution_time
        
        scaling_efficiency = size_ratio / time_ratio
        
        return {
            "size_range": f"{first.data_size} - {last.data_size} bytes",
            "scaling_efficiency": scaling_efficiency,
            "scaling_grade": "excellent" if scaling_efficiency > 0.8 else 
                            "good" if scaling_efficiency > 0.6 else
                            "fair" if scaling_efficiency > 0.4 else "poor"
        }
    
    def _calculate_performance_grade(self, throughputs: List[float]) -> str:
        """Calculate overall performance grade."""
        if not throughputs:
            return "unknown"
        
        avg_throughput = statistics.mean(throughputs)
        
        # Grading thresholds (MB/s)
        if avg_throughput > 100:
            return "excellent"
        elif avg_throughput > 50:
            return "good"
        elif avg_throughput > 20:
            return "fair"
        else:
            return "needs_improvement"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Analyze GPU usage
        gpu_results = [r for r in self.results if r.gpu_used]
        cpu_results = [r for r in self.results if not r.gpu_used]
        
        if torch.cuda.is_available() and not gpu_results:
            recommendations.append("Consider enabling GPU acceleration for better performance")
        
        # Analyze memory usage
        high_memory_ops = [r for r in self.results if r.memory_usage_mb > 100]
        if high_memory_ops:
            recommendations.append("High memory usage detected - consider memory optimization")
        
        # Analyze execution times
        slow_ops = [r for r in self.results if r.execution_time > 5.0]
        if slow_ops:
            recommendations.append("Slow operations detected - consider performance optimization")
        
        # Cache analysis
        cache_enabled_ops = [r for r in self.results if r.cache_hits + r.cache_misses > 0]
        if cache_enabled_ops:
            avg_hit_ratio = statistics.mean([r.cache_hit_ratio for r in cache_enabled_ops])
            if avg_hit_ratio < 0.5:
                recommendations.append("Low cache hit ratio - consider cache optimization")
        
        return recommendations


class TestCryptanalysisBenchmarks:
    """Benchmark tests for cryptanalysis operations."""
    
    @pytest.fixture(scope="class")
    def benchmark_suite(self):
        """Create benchmark suite."""
        return CryptanalysisBenchmarkSuite()
    
    @pytest.fixture(scope="class")
    def data_sizes(self):
        """Standard data sizes for benchmarking."""
        return [128, 512, 1024, 2048, 4096]
    
    @pytest.mark.benchmark
    def test_basic_cryptanalysis_performance(self, benchmark_suite, data_sizes):
        """Benchmark basic cryptanalysis operations."""
        
        def basic_analysis(cipher_data, **kwargs):
            return analyze_cipher_securely(
                cipher_data=cipher_data,
                analysis_types=["frequency"],
                security_level=SecurityLevel.LOW
            )
        
        results = benchmark_suite.benchmark_operation(
            operation_func=basic_analysis,
            operation_name="basic_cryptanalysis",
            data_sizes=data_sizes,
            iterations=3
        )
        
        # Verify results
        assert len(results) > 0
        
        # Check performance thresholds
        for result in results:
            # Should complete within reasonable time
            assert result.execution_time < 30.0, f"Too slow: {result.execution_time}s for {result.data_size} bytes"
            
            # Should have reasonable throughput
            assert result.throughput_mb_per_sec > 0.001, f"Too low throughput: {result.throughput_mb_per_sec}"
    
    @pytest.mark.benchmark
    def test_enhanced_framework_performance(self, benchmark_suite, data_sizes):
        """Benchmark enhanced framework performance."""
        
        def enhanced_analysis(cipher_data, **kwargs):
            framework = create_enhanced_cryptanalysis_framework(
                security_level=SecurityLevel.LOW,
                enable_caching=True,
                enable_parallel_processing=False  # For consistent benchmarking
            )
            
            try:
                return framework.analyze_cipher_comprehensive(
                    cipher_data=cipher_data,
                    analysis_types=["frequency"]
                )
            finally:
                framework.shutdown()
        
        results = benchmark_suite.benchmark_operation(
            operation_func=enhanced_analysis,
            operation_name="enhanced_framework",
            data_sizes=data_sizes[:3],  # Smaller range for framework tests
            iterations=2
        )
        
        # Verify results
        assert len(results) > 0
        
        # Enhanced framework should be reasonably fast
        for result in results:
            assert result.execution_time < 60.0, f"Enhanced framework too slow: {result.execution_time}s"
    
    @pytest.mark.benchmark
    def test_performance_optimizer_benchmarks(self, benchmark_suite):
        """Benchmark performance optimizer components."""
        
        def optimizer_test(cipher_data, **kwargs):
            optimizer = create_performance_optimizer(
                enable_gpu=False,  # CPU for consistent testing
                enable_caching=True
            )
            
            try:
                # Test tensor optimization
                optimized = optimizer.gpu_accelerator.optimize_tensor(cipher_data.float())
                
                # Test optimized operation
                def simple_op(x):
                    return torch.sum(x)
                
                result = optimizer.execute_optimized_operation(
                    operation=simple_op,
                    operation_name="tensor_sum",
                    optimized
                )
                
                return {"result": result, "optimizer": optimizer}
            finally:
                optimizer.cleanup()
        
        results = benchmark_suite.benchmark_operation(
            operation_func=optimizer_test,
            operation_name="performance_optimizer",
            data_sizes=[512, 1024, 2048],
            iterations=2
        )
        
        # Verify optimizer performance
        assert len(results) > 0
        
        for result in results:
            # Optimizer should add minimal overhead
            assert result.execution_time < 5.0, f"Optimizer overhead too high: {result.execution_time}s"
    
    @pytest.mark.benchmark
    def test_batch_processing_performance(self, benchmark_suite):
        """Benchmark batch processing performance."""
        
        def batch_analysis(cipher_data, **kwargs):
            # Split data into batches
            batch_size = len(cipher_data) // 4
            batches = [
                cipher_data[i:i+batch_size] 
                for i in range(0, len(cipher_data), batch_size)
                if i + batch_size <= len(cipher_data)
            ]
            
            framework = create_enhanced_cryptanalysis_framework(
                enable_parallel_processing=True,
                max_workers=2
            )
            
            try:
                results = framework.batch_analyze_ciphers(
                    cipher_datasets=batches,
                    analysis_types=["frequency"],
                    max_workers=2
                )
                return results
            finally:
                framework.shutdown()
        
        results = benchmark_suite.benchmark_operation(
            operation_func=batch_analysis,
            operation_name="batch_processing",
            data_sizes=[1024, 2048],  # Larger sizes for batch testing
            iterations=2
        )
        
        # Verify batch processing performance
        assert len(results) > 0
    
    @pytest.mark.benchmark
    def test_caching_performance_impact(self, benchmark_suite):
        """Benchmark caching performance impact."""
        
        test_data = torch.randint(0, 256, (1024,), dtype=torch.uint8)
        
        # Test without caching
        def no_cache_analysis(cipher_data, **kwargs):
            framework = create_enhanced_cryptanalysis_framework(
                enable_caching=False
            )
            
            try:
                return framework.analyze_cipher_comprehensive(
                    cipher_data=cipher_data,
                    analysis_types=["frequency"]
                )
            finally:
                framework.shutdown()
        
        # Test with caching
        def cached_analysis(cipher_data, **kwargs):
            framework = create_enhanced_cryptanalysis_framework(
                enable_caching=True
            )
            
            try:
                # First call (cache miss)
                result1 = framework.analyze_cipher_comprehensive(
                    cipher_data=cipher_data,
                    analysis_types=["frequency"],
                    cache_key="benchmark_cache_test"
                )
                
                # Second call (cache hit)
                result2 = framework.analyze_cipher_comprehensive(
                    cipher_data=cipher_data,
                    analysis_types=["frequency"],
                    cache_key="benchmark_cache_test"
                )
                
                return {"first": result1, "second": result2}
            finally:
                framework.shutdown()
        
        # Benchmark both approaches
        no_cache_results = benchmark_suite.benchmark_operation(
            operation_func=no_cache_analysis,
            operation_name="no_caching",
            data_sizes=[1024],
            iterations=3
        )
        
        cached_results = benchmark_suite.benchmark_operation(
            operation_func=cached_analysis,
            operation_name="with_caching",
            data_sizes=[1024],
            iterations=3
        )
        
        # Analyze caching impact
        if no_cache_results and cached_results:
            no_cache_time = no_cache_results[0].execution_time
            cached_time = cached_results[0].execution_time
            
            print(f"No cache: {no_cache_time:.3f}s, With cache: {cached_time:.3f}s")
    
    @pytest.mark.benchmark
    def test_memory_usage_benchmarks(self, benchmark_suite):
        """Benchmark memory usage patterns."""
        
        def memory_intensive_analysis(cipher_data, **kwargs):
            framework = create_enhanced_cryptanalysis_framework(
                enable_performance_monitoring=True
            )
            
            try:
                result = framework.analyze_cipher_comprehensive(
                    cipher_data=cipher_data,
                    analysis_types=["differential", "linear", "frequency"]
                )
                
                # Get memory stats
                status = framework.get_system_status()
                return {"analysis": result, "status": status}
            finally:
                framework.shutdown()
        
        results = benchmark_suite.benchmark_operation(
            operation_func=memory_intensive_analysis,
            operation_name="memory_intensive",
            data_sizes=[512, 1024],
            iterations=2
        )
        
        # Check memory usage is reasonable
        for result in results:
            # Memory usage should be reasonable (less than 500MB)
            assert result.memory_usage_mb < 500, f"Memory usage too high: {result.memory_usage_mb}MB"
    
    @pytest.mark.benchmark
    def test_generate_performance_report(self, benchmark_suite):
        """Test performance report generation."""
        
        # Ensure we have some benchmark data
        if not benchmark_suite.results:
            # Run a quick benchmark
            test_data = torch.randint(0, 256, (512,), dtype=torch.uint8)
            
            def quick_test(cipher_data, **kwargs):
                return analyze_cipher_securely(
                    cipher_data=cipher_data,
                    analysis_types=["frequency"],
                    security_level=SecurityLevel.LOW
                )
            
            benchmark_suite.benchmark_operation(
                operation_func=quick_test,
                operation_name="quick_test",
                data_sizes=[512],
                iterations=1
            )
        
        # Generate report
        report = benchmark_suite.generate_report()
        
        # Verify report structure
        assert "summary" in report
        assert "operations" in report
        assert "performance_analysis" in report
        
        summary = report["summary"]
        assert "total_operations" in summary
        assert "total_benchmarks" in summary
        assert "gpu_available" in summary
        
        # Verify operations data
        if report["operations"]:
            for op_name, op_data in report["operations"].items():
                assert "sample_count" in op_data
                assert "execution_time" in op_data
                assert "throughput_mb_per_sec" in op_data
                assert "memory_usage_mb" in op_data
        
        # Verify performance analysis
        perf_analysis = report["performance_analysis"]
        assert "overall_throughput_mb_per_sec" in perf_analysis
        assert "performance_grade" in perf_analysis
        assert "recommendations" in perf_analysis
        
        # Print report for manual inspection
        print("\nPerformance Report:")
        print(f"Total Operations: {summary['total_operations']}")
        print(f"Total Benchmarks: {summary['total_benchmarks']}")
        print(f"Performance Grade: {perf_analysis['performance_grade']}")
        print(f"Overall Throughput: {perf_analysis['overall_throughput_mb_per_sec']:.2f} MB/s")
        
        if perf_analysis['recommendations']:
            print("Recommendations:")
            for rec in perf_analysis['recommendations']:
                print(f"  - {rec}")


@pytest.mark.benchmark
class TestScalabilityBenchmarks:
    """Test scalability characteristics."""
    
    def test_data_size_scalability(self):
        """Test how performance scales with data size."""
        data_sizes = [256, 512, 1024, 2048, 4096, 8192]
        results = {}
        
        for size in data_sizes:
            cipher_data = torch.randint(0, 256, (size,), dtype=torch.uint8)
            
            start_time = time.time()
            
            result = analyze_cipher_securely(
                cipher_data=cipher_data,
                analysis_types=["frequency"],
                security_level=SecurityLevel.LOW
            )
            
            execution_time = time.time() - start_time
            results[size] = execution_time
            
            print(f"Size {size}: {execution_time:.3f}s")
        
        # Analyze scaling characteristics
        sizes = list(results.keys())
        times = list(results.values())
        
        # Calculate scaling factor between smallest and largest
        size_ratio = max(sizes) / min(sizes)
        time_ratio = max(times) / min(times)
        scaling_efficiency = size_ratio / time_ratio
        
        print(f"Scaling efficiency: {scaling_efficiency:.2f}")
        
        # Should scale reasonably well (efficiency > 0.1)
        assert scaling_efficiency > 0.1, f"Poor scaling efficiency: {scaling_efficiency}"
    
    def test_concurrent_load_scalability(self):
        """Test performance under concurrent load."""
        import threading
        import concurrent.futures
        
        cipher_data = torch.randint(0, 256, (1024,), dtype=torch.uint8)
        worker_counts = [1, 2, 4]
        
        def analysis_worker():
            return analyze_cipher_securely(
                cipher_data=cipher_data,
                analysis_types=["frequency"],
                security_level=SecurityLevel.LOW
            )
        
        for worker_count in worker_counts:
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(analysis_worker) for _ in range(worker_count)]
                results = [future.result() for future in futures]
            
            total_time = time.time() - start_time
            avg_time_per_worker = total_time / worker_count
            
            print(f"Workers: {worker_count}, Total time: {total_time:.3f}s, Avg per worker: {avg_time_per_worker:.3f}s")
            
            # Verify all results are valid
            for result in results:
                assert "overall" in result
        
        print("Concurrent load scalability test completed")


if __name__ == "__main__":
    # Run benchmarks
    pytest.main([__file__, "-v", "-m", "benchmark"])
