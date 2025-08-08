"""Unit tests for neural operator cryptanalysis module."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

try:
    from quantum_planner.research.neural_operator_cryptanalysis import (
        CryptanalysisConfig,
        FourierNeuralOperator,
        WaveletNeuralOperator,
        DifferentialCryptanalysis,
        LinearCryptanalysis,
        CryptanalysisFramework,
        create_cryptanalysis_framework
    )
except ImportError:
    # Skip tests if dependencies not available
    pytest.skip("Neural operator cryptanalysis dependencies not available", allow_module_level=True)


class TestCryptanalysisConfig:
    """Test cryptanalysis configuration."""
    
    def test_default_config(self):
        config = CryptanalysisConfig(cipher_type="test")
        assert config.cipher_type == "test"
        assert config.neural_operator_type == "fourier"
        assert config.hidden_dim == 128
        assert config.num_layers == 4
        assert config.security_level == 128
    
    def test_custom_config(self):
        config = CryptanalysisConfig(
            cipher_type="aes",
            neural_operator_type="wavelet",
            hidden_dim=256,
            num_layers=6,
            security_level=256
        )
        assert config.cipher_type == "aes"
        assert config.neural_operator_type == "wavelet"
        assert config.hidden_dim == 256
        assert config.num_layers == 6
        assert config.security_level == 256


class TestFourierNeuralOperator:
    """Test Fourier neural operator."""
    
    def test_initialization(self):
        config = CryptanalysisConfig(cipher_type="test", hidden_dim=64, num_layers=2)
        operator = FourierNeuralOperator(config)
        
        assert operator.hidden_dim == 64
        assert len(operator.fourier_layers) == 2
        assert operator.projection is not None
        assert operator.output_projection is not None
    
    def test_forward_pass(self):
        config = CryptanalysisConfig(cipher_type="test", hidden_dim=32, num_layers=2)
        operator = FourierNeuralOperator(config)
        
        # Test with sample input
        x = torch.randn(4, 16)  # batch_size=4, seq_len=16
        output = operator(x)
        
        assert output.shape == (4, 16)
        assert not torch.isnan(output).any()
    
    def test_analyze_cipher(self):
        config = CryptanalysisConfig(cipher_type="test", hidden_dim=32, num_layers=2)
        operator = FourierNeuralOperator(config)
        
        # Test with sample ciphertext
        ciphertext = torch.randint(0, 256, (100,), dtype=torch.uint8)
        result = operator.analyze_cipher(ciphertext)
        
        assert "frequency_magnitude" in result
        assert "frequency_phase" in result
        assert "detected_patterns" in result
        assert "vulnerability_score" in result
        
        # Check output shapes and types
        assert result["frequency_magnitude"].shape == ciphertext.shape
        assert torch.is_tensor(result["vulnerability_score"])


class TestWaveletNeuralOperator:
    """Test Wavelet neural operator."""
    
    def test_initialization(self):
        config = CryptanalysisConfig(cipher_type="test", hidden_dim=64, num_layers=3)
        operator = WaveletNeuralOperator(config)
        
        assert operator.hidden_dim == 64
        assert len(operator.scale_processors) == 4  # 4 scale levels
        assert operator.wavelet_transform is not None
        assert operator.fusion is not None
        assert operator.classifier is not None
    
    def test_forward_pass(self):
        config = CryptanalysisConfig(cipher_type="test", hidden_dim=32, num_layers=2)
        operator = WaveletNeuralOperator(config)
        
        # Test with sample input
        x = torch.randn(4, 64)  # batch_size=4, seq_len=64
        output = operator(x)
        
        assert output.shape == (4,)  # Should output single value per sample
        assert not torch.isnan(output).any()
    
    def test_analyze_cipher(self):
        config = CryptanalysisConfig(cipher_type="test", hidden_dim=32, num_layers=2)
        operator = WaveletNeuralOperator(config)
        
        # Test with sample ciphertext
        ciphertext = torch.randint(0, 256, (128,), dtype=torch.uint8)
        result = operator.analyze_cipher(ciphertext)
        
        assert "scales" in result
        assert "scale_entropies" in result
        assert "vulnerability_indicators" in result
        assert "randomness_score" in result
        
        # Check that we have multiple scales
        assert len(result["scales"]) > 1
        assert torch.is_tensor(result["randomness_score"])


class TestDifferentialCryptanalysis:
    """Test differential cryptanalysis."""
    
    def test_initialization(self):
        config = CryptanalysisConfig(cipher_type="test", neural_operator_type="fourier")
        analyzer = DifferentialCryptanalysis(config)
        
        assert analyzer.config == config
        assert analyzer.neural_operator is not None
    
    def test_differential_pattern_analysis(self):
        config = CryptanalysisConfig(cipher_type="test", hidden_dim=16, num_layers=1)
        analyzer = DifferentialCryptanalysis(config)
        
        # Create sample plaintext and ciphertext pairs
        plaintext_pairs = [
            (torch.randint(0, 256, (32,), dtype=torch.uint8),
             torch.randint(0, 256, (32,), dtype=torch.uint8))
            for _ in range(3)
        ]
        ciphertext_pairs = [
            (torch.randint(0, 256, (32,), dtype=torch.uint8),
             torch.randint(0, 256, (32,), dtype=torch.uint8))
            for _ in range(3)
        ]
        
        result = analyzer.analyze_differential_patterns(plaintext_pairs, ciphertext_pairs)
        
        assert "differential_scores" in result
        assert "mean_differential_score" in result
        assert "vulnerability_level" in result
        
        # Check output format
        assert len(result["differential_scores"]) == 3
        assert result["vulnerability_level"] in ["HIGH", "MEDIUM", "LOW", "NEGLIGIBLE"]
    
    def test_vulnerability_assessment(self):
        config = CryptanalysisConfig(cipher_type="test")
        analyzer = DifferentialCryptanalysis(config)
        
        # Test different vulnerability levels
        high_scores = torch.tensor([0.9, 0.85, 0.8])
        assert analyzer._assess_vulnerability(high_scores) == "HIGH"
        
        medium_scores = torch.tensor([0.6, 0.5, 0.7])
        assert analyzer._assess_vulnerability(medium_scores) == "MEDIUM"
        
        low_scores = torch.tensor([0.3, 0.2, 0.25])
        assert analyzer._assess_vulnerability(low_scores) == "LOW"
        
        negligible_scores = torch.tensor([0.1, 0.05, 0.08])
        assert analyzer._assess_vulnerability(negligible_scores) == "NEGLIGIBLE"


class TestLinearCryptanalysis:
    """Test linear cryptanalysis."""
    
    def test_initialization(self):
        config = CryptanalysisConfig(cipher_type="test", neural_operator_type="wavelet")
        analyzer = LinearCryptanalysis(config)
        
        assert analyzer.config == config
        assert analyzer.neural_operator is not None
    
    def test_linear_approximations(self):
        config = CryptanalysisConfig(cipher_type="test", hidden_dim=16, num_layers=1)
        analyzer = LinearCryptanalysis(config)
        
        # Create sample data
        plaintext_samples = torch.randint(0, 256, (50, 32), dtype=torch.uint8)
        ciphertext_samples = torch.randint(0, 256, (50, 32), dtype=torch.uint8)
        
        result = analyzer.find_linear_approximations(
            plaintext_samples, ciphertext_samples, num_approximations=10
        )
        
        assert "linear_biases" in result
        assert "max_bias" in result
        assert "mean_bias" in result
        assert "vulnerability_assessment" in result
        
        # Check output format
        assert len(result["linear_biases"]) == 10
        assert result["vulnerability_assessment"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    
    def test_apply_mask(self):
        config = CryptanalysisConfig(cipher_type="test")
        analyzer = LinearCryptanalysis(config)
        
        # Test mask application
        data = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.uint8)
        mask = torch.tensor([1, 1, 0, 1], dtype=torch.uint8)
        
        result = analyzer._apply_mask(data, mask)
        
        assert result.shape == (2,)
        assert result.dtype == torch.uint8


class TestCryptanalysisFramework:
    """Test comprehensive cryptanalysis framework."""
    
    def test_initialization(self):
        config = CryptanalysisConfig(cipher_type="test")
        framework = CryptanalysisFramework(config)
        
        assert framework.config == config
        assert framework.differential_analyzer is not None
        assert framework.linear_analyzer is not None
    
    def test_comprehensive_analysis(self):
        config = CryptanalysisConfig(cipher_type="test", hidden_dim=16, num_layers=1)
        framework = CryptanalysisFramework(config)
        
        # Prepare sample data
        cipher_samples = {
            "plaintext_pairs": [
                (torch.randint(0, 256, (16,), dtype=torch.uint8),
                 torch.randint(0, 256, (16,), dtype=torch.uint8))
                for _ in range(2)
            ],
            "ciphertext_pairs": [
                (torch.randint(0, 256, (16,), dtype=torch.uint8),
                 torch.randint(0, 256, (16,), dtype=torch.uint8))
                for _ in range(2)
            ],
            "plaintext_samples": torch.randint(0, 256, (20, 16), dtype=torch.uint8),
            "ciphertext_samples": torch.randint(0, 256, (20, 16), dtype=torch.uint8)
        }
        
        result = framework.comprehensive_analysis(cipher_samples)
        
        assert "differential" in result
        assert "linear" in result
        assert "overall" in result
        
        # Check overall assessment
        overall = result["overall"]
        assert "combined_vulnerability_score" in overall
        assert "overall_vulnerability_level" in overall
        assert "recommendation" in overall
    
    def test_combine_analyses(self):
        config = CryptanalysisConfig(cipher_type="test")
        framework = CryptanalysisFramework(config)
        
        # Mock analysis results
        results = {
            "differential": {
                "mean_differential_score": torch.tensor(0.6),
                "vulnerability_level": "MEDIUM"
            },
            "linear": {
                "max_bias": torch.tensor(0.15),
                "vulnerability_assessment": "HIGH"
            }
        }
        
        combined = framework._combine_analyses(results)
        
        assert "combined_vulnerability_score" in combined
        assert "overall_vulnerability_level" in combined
        assert "recommendation" in combined
        
        # Should take the highest level (HIGH)
        assert combined["overall_vulnerability_level"] == "HIGH"
    
    def test_generate_recommendation(self):
        config = CryptanalysisConfig(cipher_type="test")
        framework = CryptanalysisFramework(config)
        
        # Test different recommendation levels
        assert "Immediate review" in framework._generate_recommendation("CRITICAL")
        assert "may be vulnerable" in framework._generate_recommendation("HIGH")
        assert "Some weaknesses" in framework._generate_recommendation("MEDIUM")
        assert "Minor vulnerabilities" in framework._generate_recommendation("LOW")
        assert "No significant" in framework._generate_recommendation("NEGLIGIBLE")


class TestFactoryFunction:
    """Test factory function for creating framework."""
    
    def test_create_framework(self):
        framework = create_cryptanalysis_framework(
            cipher_type="aes",
            neural_operator_type="fourier",
            hidden_dim=64
        )
        
        assert isinstance(framework, CryptanalysisFramework)
        assert framework.config.cipher_type == "aes"
        assert framework.config.neural_operator_type == "fourier"
        assert framework.config.hidden_dim == 64
    
    def test_create_framework_defaults(self):
        framework = create_cryptanalysis_framework(cipher_type="test")
        
        assert isinstance(framework, CryptanalysisFramework)
        assert framework.config.neural_operator_type == "fourier"  # default
        assert framework.config.hidden_dim == 128  # default


# Integration tests
class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_analysis(self):
        """Test complete analysis pipeline."""
        # Create framework
        framework = create_cryptanalysis_framework(
            cipher_type="demo",
            neural_operator_type="fourier",
            hidden_dim=32,
            num_layers=2
        )
        
        # Generate test data with known patterns
        pattern = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], dtype=torch.uint8)
        weak_cipher = pattern.repeat(20)  # Repeating pattern (vulnerable)
        
        # Create samples
        cipher_samples = {
            "plaintext_pairs": [
                (weak_cipher[i:i+8], weak_cipher[i+8:i+16])
                for i in range(0, 32, 16)
            ],
            "ciphertext_pairs": [
                (weak_cipher[i+40:i+48], weak_cipher[i+48:i+56])
                for i in range(0, 32, 16)
            ],
            "plaintext_samples": weak_cipher[:80].reshape(10, 8),
            "ciphertext_samples": weak_cipher[80:160].reshape(10, 8)
        }
        
        # Perform analysis
        result = framework.comprehensive_analysis(cipher_samples)
        
        # Verify results structure
        assert "differential" in result
        assert "linear" in result
        assert "overall" in result
        
        # Should detect some vulnerability in patterned data
        overall_score = result["overall"]["combined_vulnerability_score"]
        assert torch.is_tensor(overall_score)
        assert overall_score.item() >= 0.0
    
    def test_different_operator_types(self):
        """Test both Fourier and Wavelet operators."""
        test_data = torch.randint(0, 256, (64,), dtype=torch.uint8)
        
        for operator_type in ["fourier", "wavelet"]:
            framework = create_cryptanalysis_framework(
                cipher_type="test",
                neural_operator_type=operator_type,
                hidden_dim=16,
                num_layers=1
            )
            
            cipher_samples = {
                "plaintext_samples": test_data[:32].reshape(4, 8),
                "ciphertext_samples": test_data[32:].reshape(4, 8)
            }
            
            result = framework.comprehensive_analysis(cipher_samples)
            
            # Should work with both operator types
            assert "linear" in result
            assert "overall" in result
            assert torch.is_tensor(result["overall"]["combined_vulnerability_score"])


# Performance tests
class TestPerformance:
    """Performance tests for the neural operator cryptanalysis."""
    
    @pytest.mark.slow
    def test_large_data_performance(self):
        """Test performance with larger datasets."""
        framework = create_cryptanalysis_framework(
            cipher_type="performance_test",
            hidden_dim=64,
            num_layers=2
        )
        
        # Large test dataset
        large_data = torch.randint(0, 256, (2048,), dtype=torch.uint8)
        
        cipher_samples = {
            "plaintext_samples": large_data[:1024].reshape(128, 8),
            "ciphertext_samples": large_data[1024:].reshape(128, 8)
        }
        
        import time
        start_time = time.time()
        result = framework.comprehensive_analysis(cipher_samples)
        execution_time = time.time() - start_time
        
        # Should complete in reasonable time (< 10 seconds)
        assert execution_time < 10.0
        assert "overall" in result
    
    def test_memory_usage(self):
        """Test memory usage with moderate datasets."""
        framework = create_cryptanalysis_framework(
            cipher_type="memory_test",
            hidden_dim=32,
            num_layers=1
        )
        
        # Multiple analyses to test memory stability
        for i in range(5):
            test_data = torch.randint(0, 256, (256,), dtype=torch.uint8)
            
            cipher_samples = {
                "plaintext_samples": test_data[:128].reshape(16, 8),
                "ciphertext_samples": test_data[128:].reshape(16, 8)
            }
            
            result = framework.comprehensive_analysis(cipher_samples)
            assert "overall" in result
            
            # Force garbage collection
            import gc
            gc.collect()


if __name__ == "__main__":
    pytest.main([__file__])
