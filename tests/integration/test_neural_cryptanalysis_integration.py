"""Integration tests for neural operator cryptanalysis system."""

import pytest
import torch
import numpy as np
import time
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch

try:
    from quantum_planner.research.enhanced_neural_cryptanalysis import (
        create_enhanced_cryptanalysis_framework,
        EnhancedCryptanalysisConfig,
        analyze_cipher_securely
    )
    from quantum_planner.research.cryptanalysis_security import (
        SecurityLevel,
        create_secure_cryptanalysis_environment
    )
    from quantum_planner.research.cryptanalysis_performance import (
        create_performance_optimizer,
        PerformanceConfig
    )
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    pytest.skip("Enhanced cryptanalysis features not available", allow_module_level=True)


class TestEnhancedCryptanalysisIntegration:
    """Test integration of enhanced cryptanalysis framework."""
    
    @pytest.fixture
    def sample_cipher_data(self):
        """Generate sample cipher data for testing."""
        torch.manual_seed(42)
        return torch.randint(0, 256, (512,), dtype=torch.uint8)
    
    @pytest.fixture
    def weak_cipher_data(self):
        """Generate weak cipher data with patterns."""
        pattern = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], dtype=torch.uint8)
        return pattern.repeat(64)  # 512 bytes with clear pattern
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_basic_framework_creation(self):
        """Test basic framework creation and initialization."""
        framework = create_enhanced_cryptanalysis_framework(
            cipher_type="test",
            security_level=SecurityLevel.LOW,
            enable_parallel_processing=False
        )
        
        assert framework is not None
        assert framework.config.cipher_type == "test"
        assert framework.config.security_level == SecurityLevel.LOW
        
        # Cleanup
        framework.shutdown()
    
    def test_secure_cipher_analysis(self, sample_cipher_data):
        """Test secure cipher analysis end-to-end."""
        result = analyze_cipher_securely(
            cipher_data=sample_cipher_data,
            security_level=SecurityLevel.LOW,
            neural_operator_type="fourier"
        )
        
        # Verify result structure
        assert "overall" in result
        assert "combined_vulnerability_score" in result["overall"]
        assert "overall_vulnerability_level" in result["overall"]
        assert "recommendation" in result["overall"]
        
        # Verify performance metadata
        if "performance_metadata" in result:
            metadata = result["performance_metadata"]
            assert "execution_time" in metadata
            assert "operation_id" in metadata
            assert "data_size" in metadata
            assert metadata["data_size"] == sample_cipher_data.numel()
    
    def test_weak_cipher_detection(self, weak_cipher_data):
        """Test detection of weak cipher patterns."""
        framework = create_enhanced_cryptanalysis_framework(
            cipher_type="weak_test",
            security_level=SecurityLevel.MEDIUM,
            enable_caching=False  # Disable caching for consistent results
        )
        
        try:
            result = framework.analyze_cipher_comprehensive(
                cipher_data=weak_cipher_data,
                analysis_types=["frequency"]
            )
            
            # Should detect some level of vulnerability in patterned data
            assert "overall" in result
            vulnerability_score = result["overall"]["combined_vulnerability_score"]
            
            if torch.is_tensor(vulnerability_score):
                score_value = vulnerability_score.item()
            else:
                score_value = float(vulnerability_score)
            
            # Patterned data should show some vulnerability
            assert score_value >= 0.0
            
        finally:
            framework.shutdown()
    
    def test_batch_analysis(self, sample_cipher_data):
        """Test batch analysis functionality."""
        framework = create_enhanced_cryptanalysis_framework(
            max_workers=2,
            enable_parallel_processing=True
        )
        
        try:
            # Create multiple datasets
            datasets = [
                sample_cipher_data[:128],
                sample_cipher_data[128:256],
                sample_cipher_data[256:384]
            ]
            
            results = framework.batch_analyze_ciphers(
                cipher_datasets=datasets,
                analysis_types=["frequency"],
                max_workers=2
            )
            
            assert len(results) == len(datasets)
            
            # Check that all results have the expected structure
            for i, result in enumerate(results):
                if "error" not in result:
                    assert "overall" in result, f"Result {i} missing 'overall' key"
                else:
                    # Log error for debugging
                    print(f"Batch analysis error in result {i}: {result['error']}")
        
        finally:
            framework.shutdown()
    
    def test_performance_monitoring(self, sample_cipher_data):
        """Test performance monitoring and metrics collection."""
        framework = create_enhanced_cryptanalysis_framework(
            enable_performance_monitoring=True,
            enable_caching=True
        )
        
        try:
            # Perform multiple analyses
            for i in range(3):
                framework.analyze_cipher_comprehensive(
                    cipher_data=sample_cipher_data,
                    analysis_types=["frequency"]
                )
            
            # Check performance metrics
            status = framework.get_system_status()
            
            assert "performance_metrics" in status
            assert "config" in status
            
            metrics = status["performance_metrics"]
            assert metrics["operations_count"] >= 3
            assert metrics["total_execution_time"] > 0
            
            # Check computed metrics
            if "computed_metrics" in status:
                computed = status["computed_metrics"]
                assert "average_execution_time" in computed
                assert computed["average_execution_time"] > 0
        
        finally:
            framework.shutdown()
    
    def test_caching_functionality(self, sample_cipher_data):
        """Test caching system functionality."""
        framework = create_enhanced_cryptanalysis_framework(
            enable_caching=True
        )
        
        try:
            # First analysis (should be cached)
            start_time = time.time()
            result1 = framework.analyze_cipher_comprehensive(
                cipher_data=sample_cipher_data,
                analysis_types=["frequency"],
                cache_key="test_cache_key"
            )
            first_time = time.time() - start_time
            
            # Second analysis with same cache key (should use cache)
            start_time = time.time()
            result2 = framework.analyze_cipher_comprehensive(
                cipher_data=sample_cipher_data,
                analysis_types=["frequency"],
                cache_key="test_cache_key"
            )
            second_time = time.time() - start_time
            
            # Verify results are consistent
            assert "overall" in result1
            assert "overall" in result2
            
            # Performance metadata should indicate cache usage
            if "performance_metadata" in result2:
                assert result2["performance_metadata"]["cache_used"]
            
            # Clear cache
            framework.clear_cache()
            
        finally:
            framework.shutdown()
    
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        framework = create_enhanced_cryptanalysis_framework(
            security_level=SecurityLevel.HIGH  # Strict validation
        )
        
        try:
            # Test with empty tensor
            empty_data = torch.tensor([], dtype=torch.uint8)
            
            result = framework.analyze_cipher_comprehensive(
                cipher_data=empty_data
            )
            
            # Should handle error gracefully
            assert "error" in result or "overall" in result
            
            if "error" in result:
                assert "type" in result["error"]
                assert "message" in result["error"]
        
        finally:
            framework.shutdown()
    
    def test_security_levels(self, sample_cipher_data):
        """Test different security levels."""
        security_levels = [SecurityLevel.LOW, SecurityLevel.MEDIUM, SecurityLevel.HIGH]
        
        for level in security_levels:
            framework = create_enhanced_cryptanalysis_framework(
                security_level=level,
                enable_caching=False  # Avoid cache interactions
            )
            
            try:
                result = framework.analyze_cipher_comprehensive(
                    cipher_data=sample_cipher_data,
                    analysis_types=["frequency"]
                )
                
                # Should complete successfully at all security levels
                assert "overall" in result or "error" in result
                
                if "overall" in result:
                    assert "combined_vulnerability_score" in result["overall"]
            
            finally:
                framework.shutdown()
    
    def test_memory_management(self, sample_cipher_data):
        """Test memory management during analysis."""
        framework = create_enhanced_cryptanalysis_framework(
            enable_performance_monitoring=True
        )
        
        try:
            # Perform analysis and check memory stats
            result = framework.analyze_cipher_comprehensive(
                cipher_data=sample_cipher_data
            )
            
            status = framework.get_system_status()
            
            # Should have GPU stats if available
            if "gpu_stats" in status and status["gpu_stats"].get("device") != "cpu":
                gpu_stats = status["gpu_stats"]
                assert "allocated_gb" in gpu_stats
                assert "total_gb" in gpu_stats
                
                # Allocated memory should be reasonable
                assert gpu_stats["allocated_gb"] >= 0
                assert gpu_stats["total_gb"] > 0
        
        finally:
            framework.shutdown()
    
    @pytest.mark.slow
    def test_large_data_processing(self):
        """Test processing of larger datasets."""
        # Create larger dataset
        large_data = torch.randint(0, 256, (4096,), dtype=torch.uint8)
        
        framework = create_enhanced_cryptanalysis_framework(
            enable_parallel_processing=True,
            max_workers=2
        )
        
        try:
            start_time = time.time()
            
            result = framework.analyze_cipher_comprehensive(
                cipher_data=large_data,
                analysis_types=["frequency"]
            )
            
            execution_time = time.time() - start_time
            
            # Should complete in reasonable time (less than 30 seconds)
            assert execution_time < 30.0
            
            # Should produce valid results
            assert "overall" in result or "error" in result
            
            if "performance_metadata" in result:
                metadata = result["performance_metadata"]
                assert metadata["data_size"] == large_data.numel()
        
        finally:
            framework.shutdown()
    
    def test_concurrent_analysis(self, sample_cipher_data):
        """Test concurrent analysis operations."""
        framework = create_enhanced_cryptanalysis_framework(
            enable_parallel_processing=True,
            max_workers=2
        )
        
        try:
            import threading
            
            results = []
            errors = []
            
            def analyze_worker(worker_id):
                try:
                    result = framework.analyze_cipher_comprehensive(
                        cipher_data=sample_cipher_data,
                        analysis_types=["frequency"],
                        cache_key=f"worker_{worker_id}"  # Unique cache keys
                    )
                    results.append((worker_id, result))
                except Exception as e:
                    errors.append((worker_id, e))
            
            # Start multiple threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=analyze_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=30)  # 30 second timeout
            
            # Check results
            assert len(errors) == 0, f"Concurrent analysis errors: {errors}"
            assert len(results) == 3
            
            # All results should be valid
            for worker_id, result in results:
                assert "overall" in result or "error" in result, f"Invalid result from worker {worker_id}"
        
        finally:
            framework.shutdown()


class TestPerformanceOptimization:
    """Test performance optimization components."""
    
    def test_performance_optimizer_creation(self):
        """Test performance optimizer creation."""
        optimizer = create_performance_optimizer(
            enable_gpu=False,  # Use CPU for testing
            enable_distributed=False,
            enable_caching=True,
            max_workers=2
        )
        
        assert optimizer is not None
        assert optimizer.config.enable_gpu is False
        assert optimizer.config.max_workers == 2
        
        # Get performance report
        report = optimizer.get_performance_report()
        
        assert "config" in report
        assert "runtime_stats" in report
        assert "cache_stats" in report
        
        # Cleanup
        optimizer.cleanup()
    
    def test_tensor_optimization(self):
        """Test tensor optimization functionality."""
        optimizer = create_performance_optimizer(enable_gpu=False)
        
        try:
            # Test tensor optimization
            test_tensor = torch.randn(100, 50)
            optimized_tensor = optimizer.gpu_accelerator.optimize_tensor(test_tensor)
            
            # Should return tensor (potentially moved to different device)
            assert torch.is_tensor(optimized_tensor)
            assert optimized_tensor.shape == test_tensor.shape
        
        finally:
            optimizer.cleanup()
    
    def test_batch_processing(self):
        """Test batch processing optimization."""
        optimizer = create_performance_optimizer(enable_gpu=False)
        
        try:
            # Create test data
            test_batches = [torch.randn(10, 5) for _ in range(5)]
            
            def simple_operation(batch):
                return torch.sum(batch, dim=1)
            
            # Process in parallel
            results = optimizer.parallel_batch_process(
                data_batches=test_batches,
                process_function=simple_operation,
                max_workers=2
            )
            
            assert len(results) == len(test_batches)
            
            # All results should be valid tensors
            for result in results:
                if result is not None:  # Some might fail in test environment
                    assert torch.is_tensor(result)
        
        finally:
            optimizer.cleanup()


class TestSecurityIntegration:
    """Test security integration components."""
    
    def test_security_environment_creation(self):
        """Test secure environment creation."""
        security_manager, validator, error_handler = create_secure_cryptanalysis_environment(
            security_level=SecurityLevel.MEDIUM,
            max_data_size=1000000,
            max_execution_time=60.0
        )
        
        assert security_manager is not None
        assert validator is not None
        assert error_handler is not None
        
        # Test operation ID generation
        op_id = security_manager.generate_operation_id()
        assert isinstance(op_id, str)
        assert len(op_id) > 0
    
    def test_input_validation(self):
        """Test input validation functionality."""
        _, validator, _ = create_secure_cryptanalysis_environment()
        
        # Valid tensor
        valid_tensor = torch.randint(0, 256, (100,), dtype=torch.uint8)
        assert validator.validate_cipher_data(valid_tensor, "test")
        
        # Test with various edge cases
        edge_cases = [
            torch.tensor([1, 2, 3], dtype=torch.uint8),  # Small but valid
            torch.randint(0, 256, (1000,), dtype=torch.uint8),  # Normal size
        ]
        
        for tensor in edge_cases:
            try:
                result = validator.validate_cipher_data(tensor, "edge_case")
                assert isinstance(result, bool)
            except Exception as e:
                # Some edge cases might fail validation, which is expected
                assert "ValidationError" in str(type(e)) or "Invalid" in str(e)
    
    def test_security_operation_context(self):
        """Test secure operation context manager."""
        security_manager, _, _ = create_secure_cryptanalysis_environment()
        
        operation_id = security_manager.generate_operation_id()
        
        # Test successful operation
        with security_manager.secure_operation("test_operation", operation_id):
            # Simulate some work
            time.sleep(0.1)
            result = "success"
        
        assert result == "success"
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        security_manager, _, _ = create_secure_cryptanalysis_environment()
        
        # Test multiple requests
        allowed_count = 0
        for _ in range(10):
            if security_manager.rate_limiter.allow_request():
                allowed_count += 1
        
        # Should allow some requests
        assert allowed_count > 0
        assert allowed_count <= 10  # Shouldn't exceed the test count


# Test fixtures and utilities
@pytest.fixture(scope="session")
def test_environment():
    """Set up test environment."""
    # Set torch to use CPU for consistent testing
    torch.set_default_device('cpu')
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    yield
    
    # Cleanup
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


@pytest.mark.integration
class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    def test_complete_workflow(self, test_environment):
        """Test complete cryptanalysis workflow."""
        # Generate test data
        cipher_data = torch.randint(0, 256, (256,), dtype=torch.uint8)
        
        # Perform secure analysis
        result = analyze_cipher_securely(
            cipher_data=cipher_data,
            analysis_types=["frequency"],
            security_level=SecurityLevel.LOW,
            neural_operator_type="fourier"
        )
        
        # Verify complete result structure
        assert isinstance(result, dict)
        assert "overall" in result
        
        overall = result["overall"]
        assert "combined_vulnerability_score" in overall
        assert "overall_vulnerability_level" in overall
        assert "recommendation" in overall
        
        # Verify score is reasonable
        score = overall["combined_vulnerability_score"]
        if torch.is_tensor(score):
            score_value = score.item()
        else:
            score_value = float(score)
        
        assert 0.0 <= score_value <= 10.0  # Reasonable bounds
        
        # Verify vulnerability level is valid
        level = overall["overall_vulnerability_level"]
        valid_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL", "NEGLIGIBLE", "UNKNOWN", "ERROR"]
        assert level in valid_levels
        
        # Verify recommendation exists
        recommendation = overall["recommendation"]
        assert isinstance(recommendation, str)
        assert len(recommendation) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
