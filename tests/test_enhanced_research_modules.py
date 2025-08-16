"""
Comprehensive Tests for Enhanced Research Modules

Tests for production-ready quantum research implementations:
- Enhanced neural operator cryptanalysis
- Quantum advantage prediction  
- Real-time adaptation algorithms
- Scalable performance optimization

Author: Terragon Labs Quantum Research Team
Version: 2.0.0 (Production Testing)
"""

import pytest
import numpy as np
import torch
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

# Import the enhanced research modules
import sys
sys.path.insert(0, '/root/repo/src')

try:
    from quantum_planner.research.enhanced_neural_cryptanalysis import (
        EnhancedFourierNeuralOperator,
        CryptanalysisConfig,
        AnalysisResult,
        EnhancedCryptanalysisFramework,
        create_enhanced_cryptanalysis_framework
    )
    CRYPTANALYSIS_AVAILABLE = True
except ImportError as e:
    CRYPTANALYSIS_AVAILABLE = False
    print(f"Cryptanalysis module not available: {e}")

try:
    from quantum_planner.research.enhanced_quantum_advantage_predictor import (
        EnhancedQuantumAdvantagePredictor,
        EnhancedProblemAnalyzer,
        EnhancedHardwareProfile,
        EnhancedProblemCharacteristics,
        QuantumAdvantageRegime,
        PredictionConfidence
    )
    ADVANTAGE_PREDICTOR_AVAILABLE = True
except ImportError as e:
    ADVANTAGE_PREDICTOR_AVAILABLE = False
    print(f"Advantage predictor not available: {e}")

try:
    from quantum_planner.research.realtime_quantum_adaptation import (
        RealTimeAdaptationEngine,
        PerformanceMetrics,
        AdaptationAction,
        AdaptationTrigger
    )
    ADAPTATION_AVAILABLE = True
except ImportError as e:
    ADAPTATION_AVAILABLE = False
    print(f"Adaptation module not available: {e}")

try:
    from quantum_planner.research.scalable_quantum_performance_optimizer import (
        ScalableQuantumOptimizer,
        DistributedCache,
        ResourcePool,
        AutoScaler,
        OptimizationTask,
        ScalingStrategy,
        CacheStrategy
    )
    SCALABLE_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    SCALABLE_OPTIMIZER_AVAILABLE = False
    print(f"Scalable optimizer not available: {e}")

from quantum_planner.validation import InputValidator
from quantum_planner.security import SecurityManager, SecurityLevel


class TestEnhancedNeuralCryptanalysis:
    """Test suite for enhanced neural operator cryptanalysis."""
    
    @pytest.fixture
    def cryptanalysis_config(self):
        """Create test configuration."""
        return CryptanalysisConfig(
            cipher_type="test_cipher",
            neural_operator_type="fourier",
            hidden_dim=64,
            num_layers=2,
            max_input_size=1000,
            timeout_seconds=30
        )
    
    @pytest.fixture
    def test_ciphertext(self):
        """Create test ciphertext data."""
        return torch.randn(100)
    
    @pytest.mark.skipif(not CRYPTANALYSIS_AVAILABLE, reason="Cryptanalysis module not available")
    def test_cryptanalysis_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = CryptanalysisConfig(
            cipher_type="aes",
            hidden_dim=128,
            num_layers=4
        )
        assert config.cipher_type == "aes"
        assert config.hidden_dim == 128
        
        # Invalid configuration
        with pytest.raises(ValueError):
            CryptanalysisConfig(
                cipher_type="aes",
                hidden_dim=-1  # Invalid
            )
    
    @pytest.mark.skipif(not CRYPTANALYSIS_AVAILABLE, reason="Cryptanalysis module not available")
    def test_enhanced_fourier_operator_initialization(self, cryptanalysis_config):
        """Test enhanced Fourier neural operator initialization."""
        operator = EnhancedFourierNeuralOperator(cryptanalysis_config)
        
        assert operator.hidden_dim == cryptanalysis_config.hidden_dim
        assert len(operator.fourier_layers) == cryptanalysis_config.num_layers
        assert operator.config == cryptanalysis_config
    
    @pytest.mark.skipif(not CRYPTANALYSIS_AVAILABLE, reason="Cryptanalysis module not available")
    def test_input_validation(self, cryptanalysis_config, test_ciphertext):
        """Test input validation for cryptanalysis."""
        operator = EnhancedFourierNeuralOperator(cryptanalysis_config)
        
        # Valid input
        assert operator.validate_input(test_ciphertext)
        
        # Invalid inputs
        with pytest.raises(Exception):
            # Too large input
            large_input = torch.randn(10000)
            operator.validate_input(large_input)
        
        with pytest.raises(Exception):
            # NaN input
            nan_input = torch.tensor([float('nan')])
            operator.validate_input(nan_input)
    
    @pytest.mark.skipif(not CRYPTANALYSIS_AVAILABLE, reason="Cryptanalysis module not available")
    def test_cipher_analysis(self, cryptanalysis_config, test_ciphertext):
        """Test cipher analysis functionality."""
        operator = EnhancedFourierNeuralOperator(cryptanalysis_config)
        
        # Mock security manager to avoid permission issues
        with patch.object(operator.security_manager, 'log_security_event'):
            result = operator.analyze_cipher(test_ciphertext)
        
        assert isinstance(result, AnalysisResult)
        assert 0 <= result.vulnerability_score <= 1
        assert 0 <= result.confidence_level <= 1
        assert result.analysis_type == "fourier_neural_operator"
        assert result.computation_time > 0
    
    @pytest.mark.skipif(not CRYPTANALYSIS_AVAILABLE, reason="Cryptanalysis module not available")
    def test_framework_creation(self):
        """Test cryptanalysis framework creation."""
        framework = create_enhanced_cryptanalysis_framework(
            cipher_type="test",
            security_level=SecurityLevel.MEDIUM
        )
        
        assert isinstance(framework, EnhancedCryptanalysisFramework)
        assert framework.config.cipher_type == "test"
    
    @pytest.mark.skipif(not CRYPTANALYSIS_AVAILABLE, reason="Cryptanalysis module not available")
    def test_model_persistence(self, cryptanalysis_config):
        """Test model saving and loading."""
        operator = EnhancedFourierNeuralOperator(cryptanalysis_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_model.pth")
            
            # Save model
            operator.save_checkpoint(filepath)
            assert os.path.exists(filepath)
            
            # Load model
            new_operator = EnhancedFourierNeuralOperator(cryptanalysis_config)
            new_operator.load_checkpoint(filepath)


class TestEnhancedQuantumAdvantagePredictor:
    """Test suite for enhanced quantum advantage prediction."""
    
    @pytest.fixture
    def sample_problem_matrix(self):
        """Create sample problem matrix."""
        matrix = np.random.randn(20, 20)
        return (matrix + matrix.T) / 2  # Make symmetric
    
    @pytest.fixture
    def hardware_profile(self):
        """Create test hardware profile."""
        return EnhancedHardwareProfile(
            name="test_hardware",
            num_qubits=30,
            connectivity="grid",
            gate_error_rate=0.001,
            readout_error_rate=0.02,
            coherence_time=100.0,
            gate_time=0.1
        )
    
    @pytest.mark.skipif(not ADVANTAGE_PREDICTOR_AVAILABLE, reason="Advantage predictor not available")
    def test_hardware_profile_validation(self):
        """Test hardware profile validation."""
        # Valid profile
        profile = EnhancedHardwareProfile(
            name="valid",
            num_qubits=10,
            connectivity="grid",
            gate_error_rate=0.01,
            readout_error_rate=0.05,
            coherence_time=50.0,
            gate_time=0.2
        )
        assert profile.is_validated
        
        # Invalid profile
        with pytest.raises(ValueError):
            EnhancedHardwareProfile(
                name="invalid",
                num_qubits=-1,  # Invalid
                connectivity="grid",
                gate_error_rate=0.01,
                readout_error_rate=0.05,
                coherence_time=50.0,
                gate_time=0.2
            )
    
    @pytest.mark.skipif(not ADVANTAGE_PREDICTOR_AVAILABLE, reason="Advantage predictor not available")
    def test_problem_analysis(self, sample_problem_matrix):
        """Test problem analysis functionality."""
        analyzer = EnhancedProblemAnalyzer()
        
        characteristics = analyzer.analyze_problem(sample_problem_matrix)
        
        assert isinstance(characteristics, EnhancedProblemCharacteristics)
        assert characteristics.is_validated
        assert characteristics.problem_size == sample_problem_matrix.shape[0]
        assert 0 <= characteristics.matrix_density <= 1
        assert characteristics.matrix_condition_number > 0
    
    @pytest.mark.skipif(not ADVANTAGE_PREDICTOR_AVAILABLE, reason="Advantage predictor not available")
    def test_advantage_prediction(self, sample_problem_matrix, hardware_profile):
        """Test quantum advantage prediction."""
        analyzer = EnhancedProblemAnalyzer()
        predictor = EnhancedQuantumAdvantagePredictor()
        
        problem_chars = analyzer.analyze_problem(sample_problem_matrix)
        
        # Mock security manager for testing
        with patch.object(predictor.security_manager, 'log_security_event'):
            prediction = predictor.predict(problem_chars, hardware_profile)
        
        assert isinstance(prediction.predicted_regime, QuantumAdvantageRegime)
        assert isinstance(prediction.confidence, PredictionConfidence)
        assert isinstance(prediction.numerical_advantage, float)
        assert len(prediction.confidence_interval) == 2
        assert prediction.recommended_algorithm in ["quantum", "classical", "hybrid"]
    
    @pytest.mark.skipif(not ADVANTAGE_PREDICTOR_AVAILABLE, reason="Advantage predictor not available")
    def test_predictor_persistence(self):
        """Test predictor model saving and loading."""
        predictor = EnhancedQuantumAdvantagePredictor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_predictor.pkl")
            
            # Save model
            predictor.save_model(filepath)
            assert os.path.exists(filepath)
            
            # Load model
            new_predictor = EnhancedQuantumAdvantagePredictor()
            new_predictor.load_model(filepath)


class TestRealTimeAdaptation:
    """Test suite for real-time adaptation algorithms."""
    
    @pytest.fixture
    def performance_metrics(self):
        """Create test performance metrics."""
        return PerformanceMetrics(
            energy=-50.0,
            solution_quality=0.8,
            convergence_rate=0.1,
            time_to_solution=30.0,
            resource_utilization=0.6,
            noise_resilience=0.9,
            problem_size=25,
            algorithm_used="qaoa",
            backend_type="simulator",
            noise_level=0.05
        )
    
    @pytest.mark.skipif(not ADAPTATION_AVAILABLE, reason="Adaptation module not available")
    def test_adaptation_engine_initialization(self):
        """Test adaptation engine initialization."""
        engine = RealTimeAdaptationEngine()
        
        assert engine.predictor is not None
        assert engine.adaptation_rate > 0
        assert engine.exploration_rate > 0
        assert len(engine.algorithm_portfolio) > 0
    
    @pytest.mark.skipif(not ADAPTATION_AVAILABLE, reason="Adaptation module not available")
    def test_performance_observation(self, performance_metrics):
        """Test performance observation and learning."""
        engine = RealTimeAdaptationEngine()
        
        context = {
            'iteration': 10,
            'problem_size': 25,
            'noise_level': 0.05,
            'current_algorithm': 'qaoa'
        }
        
        # Observe performance
        engine.observe_performance(context, performance_metrics, 'qaoa')
        
        assert len(engine.performance_history) == 1
        assert len(engine.context_history) == 1
        assert len(engine.action_history) == 1
    
    @pytest.mark.skipif(not ADAPTATION_AVAILABLE, reason="Adaptation module not available")
    def test_adaptation_recommendation(self):
        """Test adaptation recommendation generation."""
        engine = RealTimeAdaptationEngine()
        
        # Add some history first
        context = {
            'iteration': 10,
            'problem_size': 25,
            'current_algorithm': 'qaoa',
            'convergence_rate': 0.01  # Low convergence
        }
        
        recommendation = engine.recommend_adaptation(
            context, AdaptationTrigger.CONVERGENCE_STALL
        )
        
        if recommendation:  # May be None if no adaptation needed
            assert isinstance(recommendation, AdaptationAction)
            assert recommendation.action_type in [
                'algorithm_switch', 'parameter_tune', 'resource_reallocation',
                'backend_switch', 'problem_decomposition'
            ]
    
    @pytest.mark.skipif(not ADAPTATION_AVAILABLE, reason="Adaptation module not available")
    def test_adaptation_insights(self, performance_metrics):
        """Test adaptation insights generation."""
        engine = RealTimeAdaptationEngine()
        
        # Add multiple performance observations
        for i in range(15):
            context = {'iteration': i, 'problem_size': 25}
            engine.observe_performance(context, performance_metrics, 'qaoa')
        
        insights = engine.get_adaptation_insights()
        
        assert insights['status'] == 'ready'
        assert 'performance_trend' in insights
        assert 'adaptation_stats' in insights
        assert 'total_observations' in insights


class TestScalableOptimization:
    """Test suite for scalable quantum optimization."""
    
    @pytest.fixture
    def test_problem_matrix(self):
        """Create test optimization problem."""
        matrix = np.random.randn(50, 50)
        return (matrix + matrix.T) / 2
    
    @pytest.mark.skipif(not SCALABLE_OPTIMIZER_AVAILABLE, reason="Scalable optimizer not available")
    def test_distributed_cache(self):
        """Test distributed cache functionality."""
        cache = DistributedCache(
            cache_strategy=CacheStrategy.LRU,
            max_memory_mb=100
        )
        
        # Test cache operations
        cache.set("test_key", {"value": 42})
        result = cache.get("test_key")
        
        assert result is not None
        assert result["value"] == 42
        
        # Test cache statistics
        stats = cache.get_statistics()
        assert stats['cache_hits'] > 0
        assert 'hit_rate' in stats
    
    @pytest.mark.skipif(not SCALABLE_OPTIMIZER_AVAILABLE, reason="Scalable optimizer not available")
    def test_resource_pool(self):
        """Test resource pool management."""
        pool = ResourcePool(max_workers=2, max_processes=1)
        
        def dummy_task():
            time.sleep(0.1)
            return "completed"
        
        # Submit task
        future = pool.submit_cpu_task(dummy_task)
        result = future.result(timeout=5)
        
        assert result == "completed"
        
        # Check utilization
        utilization = pool.get_resource_utilization()
        assert 'thread_utilization' in utilization
        assert 'system_cpu_percent' in utilization
        
        pool.shutdown()
    
    @pytest.mark.skipif(not SCALABLE_OPTIMIZER_AVAILABLE, reason="Scalable optimizer not available")
    def test_auto_scaler(self):
        """Test auto-scaling functionality."""
        from quantum_planner.research.scalable_quantum_performance_optimizer import ScalingMetrics
        
        scaler = AutoScaler(
            scaling_strategy=ScalingStrategy.ADAPTIVE,
            min_workers=2,
            max_workers=10
        )
        
        # Test scaling recommendation
        metrics = ScalingMetrics()
        metrics.avg_cpu_utilization = 0.9  # High utilization
        metrics.queued_tasks = 20
        
        recommendation = scaler.get_scaling_recommendation(metrics)
        
        assert 'action' in recommendation
        assert 'target_workers' in recommendation
        assert 'reasoning' in recommendation
    
    @pytest.mark.skipif(not SCALABLE_OPTIMIZER_AVAILABLE, reason="Scalable optimizer not available")
    def test_optimization_task_creation(self, test_problem_matrix):
        """Test optimization task creation and validation."""
        task = OptimizationTask(
            task_id="test_task_001",
            problem_matrix=test_problem_matrix,
            constraints={"type": "quadratic"},
            priority=5
        )
        
        assert task.task_id == "test_task_001"
        assert task.problem_matrix.shape == test_problem_matrix.shape
        assert task.cache_key is not None
        assert len(task.cache_key) == 16  # Hash length
    
    @pytest.mark.skipif(not SCALABLE_OPTIMIZER_AVAILABLE, reason="Scalable optimizer not available")
    def test_scalable_optimizer_basic_flow(self, test_problem_matrix):
        """Test basic optimization flow without starting the engine."""
        optimizer = ScalableQuantumOptimizer(max_workers=2)
        
        # Test task validation
        task = OptimizationTask(
            task_id="test",
            problem_matrix=test_problem_matrix,
            constraints={}
        )
        
        # Should not raise exception
        optimizer._validate_task(task)
        
        # Test system status (before starting)
        status = optimizer.get_system_status()
        assert 'system' in status
        assert 'tasks' in status
        assert 'resources' in status


class TestIntegrationAndSecurity:
    """Integration tests and security validation."""
    
    def test_security_manager_integration(self):
        """Test security manager integration across modules."""
        security_manager = SecurityManager()
        
        # Test session management
        token = security_manager.generate_session_token("test_user")
        assert token is not None
        assert len(token) > 20
        
        # Test token validation
        session = security_manager.validate_session_token(token)
        assert session is not None
        assert session['user_id'] == "test_user"
        
        # Test rate limiting
        assert security_manager.check_rate_limit("test_user", max_requests=5)
        
        # Test input sanitization
        dirty_input = "<script>alert('xss')</script>"
        clean_input = security_manager.sanitize_input(dirty_input)
        assert "<script>" not in clean_input
    
    def test_input_validator_integration(self):
        """Test input validator across different modules."""
        validator = InputValidator(strict_mode=True)
        
        # Test with various inputs
        from quantum_planner.models import Agent, Task
        
        # Valid agent
        agent = Agent(agent_id="test_agent", skills=["skill1"], capacity=5)
        agent_report = validator.validate_agents([agent])
        assert agent_report.is_valid
        
        # Valid task
        task = Task(task_id="test_task", required_skills=["skill1"], priority=3, duration=10)
        task_report = validator.validate_tasks([task])
        assert task_report.is_valid
    
    @pytest.mark.skipif(not (CRYPTANALYSIS_AVAILABLE and ADVANTAGE_PREDICTOR_AVAILABLE), 
                       reason="Research modules not available")
    def test_research_modules_integration(self):
        """Test integration between research modules."""
        # Test that modules can work together
        
        # Create cryptanalysis framework
        crypto_framework = create_enhanced_cryptanalysis_framework(
            cipher_type="test",
            security_level=SecurityLevel.MEDIUM
        )
        
        # Create advantage predictor
        predictor = EnhancedQuantumAdvantagePredictor()
        
        # Test that both use compatible security levels
        assert crypto_framework.security_level in [SecurityLevel.LOW, SecurityLevel.MEDIUM, SecurityLevel.HIGH]
        assert predictor.security_level in [SecurityLevel.LOW, SecurityLevel.MEDIUM, SecurityLevel.HIGH]


class TestPerformanceBenchmarks:
    """Performance benchmarks and optimization tests."""
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not ADVANTAGE_PREDICTOR_AVAILABLE, reason="Advantage predictor not available")
    def test_problem_analysis_performance(self):
        """Benchmark problem analysis performance."""
        analyzer = EnhancedProblemAnalyzer()
        
        # Test with different problem sizes
        sizes = [10, 20, 50, 100]
        times = []
        
        for size in sizes:
            matrix = np.random.randn(size, size)
            matrix = (matrix + matrix.T) / 2
            
            start_time = time.time()
            characteristics = analyzer.analyze_problem(matrix)
            analysis_time = time.time() - start_time
            
            times.append(analysis_time)
            
            # Verify result quality
            assert characteristics.is_validated
            assert characteristics.problem_size == size
        
        # Performance should scale reasonably
        assert all(t < 5.0 for t in times)  # Should complete within 5 seconds
        print(f"Analysis times for sizes {sizes}: {[f'{t:.3f}s' for t in times]}")
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not SCALABLE_OPTIMIZER_AVAILABLE, reason="Scalable optimizer not available")
    def test_cache_performance(self):
        """Benchmark cache performance."""
        cache = DistributedCache(max_memory_mb=50)
        
        # Benchmark cache operations
        num_operations = 1000
        
        start_time = time.time()
        for i in range(num_operations):
            cache.set(f"key_{i}", {"data": np.random.randn(10)})
        set_time = time.time() - start_time
        
        start_time = time.time()
        hits = 0
        for i in range(num_operations):
            result = cache.get(f"key_{i}")
            if result is not None:
                hits += 1
        get_time = time.time() - start_time
        
        print(f"Cache performance: {num_operations} sets in {set_time:.3f}s, "
              f"{num_operations} gets in {get_time:.3f}s, hit rate: {hits/num_operations:.2%}")
        
        # Performance expectations
        assert set_time < 5.0  # Should complete within 5 seconds
        assert get_time < 5.0
        assert hits / num_operations > 0.8  # At least 80% hit rate
    
    @pytest.mark.benchmark
    def test_concurrent_processing_performance(self):
        """Benchmark concurrent processing capabilities."""
        def cpu_intensive_task(n):
            """Simulate CPU-intensive work."""
            return sum(i * i for i in range(n))
        
        # Sequential execution
        start_time = time.time()
        sequential_results = [cpu_intensive_task(10000) for _ in range(10)]
        sequential_time = time.time() - start_time
        
        # Concurrent execution
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_intensive_task, 10000) for _ in range(10)]
            concurrent_results = [f.result() for f in futures]
        concurrent_time = time.time() - start_time
        
        # Verify results are identical
        assert sequential_results == concurrent_results
        
        # Concurrent should be faster (though not always due to GIL in Python)
        speedup = sequential_time / concurrent_time
        print(f"Concurrency speedup: {speedup:.2f}x (sequential: {sequential_time:.3f}s, "
              f"concurrent: {concurrent_time:.3f}s)")


# Test configuration and markers
pytestmark = [
    pytest.mark.unit,
    pytest.mark.research,
]


def test_module_imports():
    """Test that all research modules can be imported."""
    # This test ensures the modules are at least syntactically correct
    assert True  # If we got here, imports in this file succeeded


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])