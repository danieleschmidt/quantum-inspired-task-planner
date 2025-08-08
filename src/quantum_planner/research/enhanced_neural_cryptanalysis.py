"""Enhanced Neural Operator Cryptanalysis with Security and Reliability.

Integrates neural operator cryptanalysis with comprehensive security measures,
reliability patterns, and production-ready error handling.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from loguru import logger
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .neural_operator_cryptanalysis import (
        CryptanalysisFramework,
        CryptanalysisConfig,
        create_cryptanalysis_framework
    )
    from .cryptanalysis_security import (
        secure_cryptanalysis_operation,
        SecurityLevel,
        create_secure_cryptanalysis_environment,
        SecurityManager,
        CryptanalysisValidator
    )
    from .cryptanalysis_reliability import (
        ReliabilityManager,
        create_reliable_cryptanalysis_environment,
        OperationStatus,
        OperationResult
    )
except ImportError:
    # Fallback for testing environments
    logger.warning("Some cryptanalysis modules not available - running in fallback mode")


@dataclass
class EnhancedCryptanalysisConfig:
    """Enhanced configuration with security and reliability settings."""
    
    # Core cryptanalysis settings
    cipher_type: str = "generic"
    neural_operator_type: str = "fourier"
    hidden_dim: int = 128
    num_layers: int = 4
    
    # Security settings
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    enable_input_validation: bool = True
    enable_output_sanitization: bool = True
    max_data_size: int = 10_000_000
    
    # Reliability settings
    enable_retry_mechanism: bool = True
    enable_circuit_breaker: bool = True
    enable_caching: bool = True
    max_execution_time: float = 300.0
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    batch_processing_threshold: int = 1000
    
    # Monitoring settings
    enable_performance_monitoring: bool = True
    enable_health_monitoring: bool = True
    log_level: str = "INFO"


class EnhancedCryptanalysisFramework:
    """Production-ready cryptanalysis framework with security and reliability."""
    
    def __init__(self, config: EnhancedCryptanalysisConfig):
        self.config = config
        self.logger = logger.bind(component="enhanced_cryptanalysis")
        
        # Initialize core components
        self._initialize_components()
        
        # Initialize security and reliability
        self._initialize_security_reliability()
        
        # Performance tracking
        self.performance_metrics = {
            "operations_count": 0,
            "total_execution_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors_count": 0
        }
    
    def _initialize_components(self):
        """Initialize core cryptanalysis components."""
        try:
            # Create base cryptanalysis framework
            base_config = CryptanalysisConfig(
                cipher_type=self.config.cipher_type,
                neural_operator_type=self.config.neural_operator_type,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers
            )
            
            self.base_framework = create_cryptanalysis_framework(
                cipher_type=self.config.cipher_type,
                neural_operator_type=self.config.neural_operator_type,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers
            )
            
            self.logger.info("Core cryptanalysis components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize core components: {e}")
            # Create fallback framework
            self.base_framework = self._create_fallback_framework()
    
    def _initialize_security_reliability(self):
        """Initialize security and reliability components."""
        try:
            # Security manager
            self.security_manager, self.validator, self.error_handler = \
                create_secure_cryptanalysis_environment(
                    security_level=self.config.security_level,
                    max_data_size=self.config.max_data_size,
                    max_execution_time=self.config.max_execution_time
                )
            
            # Reliability manager
            self.reliability_manager = create_reliable_cryptanalysis_environment()
            
            # Register fallback strategies
            self._register_fallback_strategies()
            
            self.logger.info("Security and reliability components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize security/reliability: {e}")
            self.security_manager = None
            self.validator = None
            self.reliability_manager = None
    
    def _create_fallback_framework(self):
        """Create simple fallback framework when dependencies unavailable."""
        class FallbackFramework:
            def comprehensive_analysis(self, cipher_samples):
                return {
                    "overall": {
                        "combined_vulnerability_score": torch.tensor(0.0),
                        "overall_vulnerability_level": "UNKNOWN",
                        "recommendation": "Analysis not available - dependencies missing"
                    },
                    "fallback_mode": True
                }
        
        return FallbackFramework()
    
    def _register_fallback_strategies(self):
        """Register fallback strategies for different operation types."""
        if not self.reliability_manager:
            return
            
        def simple_frequency_analysis(cipher_data, *args, **kwargs):
            """Simple frequency analysis fallback."""
            if torch.is_tensor(cipher_data):
                unique_values, counts = torch.unique(cipher_data, return_counts=True)
                entropy = -torch.sum((counts.float() / counts.sum()) * 
                                   torch.log2(counts.float() / counts.sum() + 1e-10))
                
                return {
                    "overall": {
                        "combined_vulnerability_score": entropy / 8.0,  # Normalize
                        "overall_vulnerability_level": "LOW" if entropy > 6 else "MEDIUM",
                        "recommendation": "Fallback frequency analysis completed"
                    },
                    "frequency_analysis": {
                        "entropy": entropy,
                        "unique_values": len(unique_values)
                    }
                }
            else:
                return {"error": "Invalid input for fallback analysis"}
        
        self.reliability_manager.fallback_strategy.register_fallback(
            "comprehensive_analysis", simple_frequency_analysis
        )
    
    @secure_cryptanalysis_operation(SecurityLevel.MEDIUM)
    def analyze_cipher_comprehensive(
        self,
        cipher_data: torch.Tensor,
        analysis_types: Optional[List[str]] = None,
        use_parallel: bool = None,
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive cipher analysis with full security and reliability."""
        
        # Use config defaults if not specified
        if use_parallel is None:
            use_parallel = self.config.enable_parallel_processing
        
        if analysis_types is None:
            analysis_types = ["differential", "linear", "frequency"]
        
        operation_id = self._generate_operation_id(cipher_data)
        
        self.logger.info(
            f"Starting comprehensive analysis {operation_id} "
            f"for {cipher_data.numel()} data points"
        )
        
        start_time = time.time()
        
        try:
            # Generate cache key if not provided
            if cache_key is None and self.config.enable_caching:
                cache_key = self._generate_cache_key(cipher_data, analysis_types)
            
            # Prepare cipher samples
            cipher_samples = self._prepare_cipher_samples(cipher_data, analysis_types)
            
            # Execute analysis with reliability management
            if self.reliability_manager:
                result = self._execute_with_reliability(
                    cipher_samples, operation_id, cache_key
                )
            else:
                result = self._execute_direct_analysis(cipher_samples)
            
            # Add performance metadata
            execution_time = time.time() - start_time
            result["performance_metadata"] = {
                "execution_time": execution_time,
                "operation_id": operation_id,
                "data_size": cipher_data.numel(),
                "parallel_processing": use_parallel,
                "cache_used": cache_key is not None
            }
            
            # Update performance metrics
            self._update_performance_metrics(execution_time, cache_key is not None)
            
            self.logger.info(f"Analysis {operation_id} completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.performance_metrics["errors_count"] += 1
            self.logger.error(f"Analysis {operation_id} failed: {e}")
            
            # Return error result
            return {
                "overall": {
                    "combined_vulnerability_score": torch.tensor(0.0),
                    "overall_vulnerability_level": "ERROR",
                    "recommendation": f"Analysis failed: {str(e)}"
                },
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "operation_id": operation_id
                }
            }
    
    def _generate_operation_id(self, cipher_data: torch.Tensor) -> str:
        """Generate unique operation ID."""
        data_hash = hashlib.md5(cipher_data.numpy().tobytes()).hexdigest()[:8]
        timestamp = int(time.time() * 1000)
        return f"crypto_{timestamp}_{data_hash}"
    
    def _generate_cache_key(self, cipher_data: torch.Tensor, analysis_types: List[str]) -> str:
        """Generate cache key for operation."""
        data_hash = hashlib.md5(cipher_data.numpy().tobytes()).hexdigest()
        analysis_hash = hashlib.md5("-".join(sorted(analysis_types)).encode()).hexdigest()[:8]
        return f"analysis_{data_hash}_{analysis_hash}"
    
    def _prepare_cipher_samples(
        self, 
        cipher_data: torch.Tensor, 
        analysis_types: List[str]
    ) -> Dict[str, Any]:
        """Prepare cipher samples for different analysis types."""
        samples = {}
        
        try:
            data_size = cipher_data.numel()
            
            if "differential" in analysis_types and data_size >= 64:
                # Prepare differential analysis samples
                batch_size = min(50, data_size // 4)
                samples["plaintext_pairs"] = [
                    (cipher_data[i], cipher_data[i+1])
                    for i in range(0, batch_size*2, 2)
                ]
                samples["ciphertext_pairs"] = [
                    (cipher_data[i+batch_size*2], cipher_data[i+batch_size*2+1])
                    for i in range(0, batch_size*2, 2)
                    if i+batch_size*2+1 < data_size
                ]
            
            if "linear" in analysis_types and data_size >= 32:
                # Prepare linear analysis samples
                mid_point = data_size // 2
                if mid_point > 0:
                    samples["plaintext_samples"] = cipher_data[:mid_point]
                    samples["ciphertext_samples"] = cipher_data[mid_point:2*mid_point]
            
            if "frequency" in analysis_types:
                # Prepare frequency analysis samples
                samples["ciphertext_samples"] = cipher_data
            
            return samples
            
        except Exception as e:
            self.logger.error(f"Failed to prepare cipher samples: {e}")
            return {"ciphertext_samples": cipher_data}  # Minimal fallback
    
    def _execute_with_reliability(
        self, 
        cipher_samples: Dict[str, Any], 
        operation_id: str, 
        cache_key: Optional[str]
    ) -> Dict[str, Any]:
        """Execute analysis with reliability management."""
        
        def analysis_operation():
            return self.base_framework.comprehensive_analysis(cipher_samples)
        
        result = self.reliability_manager.execute_reliable_operation(
            operation=analysis_operation,
            operation_id=operation_id,
            operation_type="comprehensive_analysis",
            use_cache=self.config.enable_caching,
            cache_key=cache_key
        )
        
        if result.is_success():
            return result.result
        else:
            raise result.error or RuntimeError("Analysis operation failed")
    
    def _execute_direct_analysis(self, cipher_samples: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis directly without reliability management."""
        return self.base_framework.comprehensive_analysis(cipher_samples)
    
    def _update_performance_metrics(self, execution_time: float, cache_used: bool):
        """Update performance tracking metrics."""
        self.performance_metrics["operations_count"] += 1
        self.performance_metrics["total_execution_time"] += execution_time
        
        if cache_used:
            self.performance_metrics["cache_hits"] += 1
        else:
            self.performance_metrics["cache_misses"] += 1
    
    def batch_analyze_ciphers(
        self,
        cipher_datasets: List[torch.Tensor],
        analysis_types: Optional[List[str]] = None,
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Analyze multiple cipher datasets in parallel."""
        
        if max_workers is None:
            max_workers = self.config.max_workers
        
        if not self.config.enable_parallel_processing:
            max_workers = 1
        
        self.logger.info(f"Starting batch analysis of {len(cipher_datasets)} datasets")
        
        results = []
        
        if max_workers == 1:
            # Sequential processing
            for i, cipher_data in enumerate(cipher_datasets):
                result = self.analyze_cipher_comprehensive(
                    cipher_data=cipher_data,
                    analysis_types=analysis_types,
                    use_parallel=False
                )
                results.append(result)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(
                        self.analyze_cipher_comprehensive,
                        cipher_data=cipher_data,
                        analysis_types=analysis_types,
                        use_parallel=False  # Already parallelized at batch level
                    ): i
                    for i, cipher_data in enumerate(cipher_datasets)
                }
                
                # Collect results in order
                results = [None] * len(cipher_datasets)
                
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                    except Exception as e:
                        self.logger.error(f"Batch analysis failed for dataset {index}: {e}")
                        results[index] = {
                            "error": {
                                "type": type(e).__name__,
                                "message": str(e),
                                "dataset_index": index
                            }
                        }
        
        self.logger.info(f"Batch analysis completed for {len(cipher_datasets)} datasets")
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and health metrics."""
        status = {
            "timestamp": time.time(),
            "performance_metrics": self.performance_metrics.copy(),
            "config": {
                "security_level": self.config.security_level.value,
                "enable_caching": self.config.enable_caching,
                "enable_parallel_processing": self.config.enable_parallel_processing,
                "max_workers": self.config.max_workers
            }
        }
        
        # Add reliability status if available
        if self.reliability_manager:
            status["reliability"] = self.reliability_manager.get_reliability_status()
        
        # Add computed metrics
        if self.performance_metrics["operations_count"] > 0:
            status["computed_metrics"] = {
                "average_execution_time": (
                    self.performance_metrics["total_execution_time"] / 
                    self.performance_metrics["operations_count"]
                ),
                "cache_hit_ratio": (
                    self.performance_metrics["cache_hits"] / 
                    (self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"])
                ),
                "error_rate": (
                    self.performance_metrics["errors_count"] / 
                    self.performance_metrics["operations_count"]
                )
            }
        
        return status
    
    def clear_cache(self):
        """Clear operation cache."""
        if self.reliability_manager and self.reliability_manager.operation_cache:
            self.reliability_manager.operation_cache.clear()
            self.logger.info("Operation cache cleared")
    
    def shutdown(self):
        """Gracefully shutdown the framework."""
        self.logger.info("Shutting down enhanced cryptanalysis framework")
        
        if self.reliability_manager:
            self.reliability_manager.shutdown()
        
        # Clear performance metrics
        self.performance_metrics.clear()
        
        self.logger.info("Framework shutdown complete")


def create_enhanced_cryptanalysis_framework(
    cipher_type: str = "generic",
    neural_operator_type: str = "fourier",
    security_level: SecurityLevel = SecurityLevel.MEDIUM,
    enable_parallel_processing: bool = True,
    **kwargs
) -> EnhancedCryptanalysisFramework:
    """Create enhanced cryptanalysis framework with security and reliability."""
    
    config = EnhancedCryptanalysisConfig(
        cipher_type=cipher_type,
        neural_operator_type=neural_operator_type,
        security_level=security_level,
        enable_parallel_processing=enable_parallel_processing,
        **kwargs
    )
    
    return EnhancedCryptanalysisFramework(config)


# Convenience function for quick secure analysis
def analyze_cipher_securely(
    cipher_data: torch.Tensor,
    analysis_types: Optional[List[str]] = None,
    security_level: SecurityLevel = SecurityLevel.MEDIUM,
    neural_operator_type: str = "fourier"
) -> Dict[str, Any]:
    """Quick secure cipher analysis with production-ready error handling."""
    
    framework = create_enhanced_cryptanalysis_framework(
        neural_operator_type=neural_operator_type,
        security_level=security_level
    )
    
    try:
        return framework.analyze_cipher_comprehensive(
            cipher_data=cipher_data,
            analysis_types=analysis_types
        )
    finally:
        framework.shutdown()
