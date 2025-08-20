"""Ultra-Robust Neural Operator Cryptanalysis with Advanced Error Handling.

This module implements production-grade neural operator cryptanalysis with comprehensive
error handling, input validation, security measures, and fault tolerance designed for
critical cryptographic analysis applications.
"""

import torch
import torch.nn as nn
import numpy as np
import hashlib
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import queue
import warnings
from loguru import logger
import json

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from .advanced_neural_cryptanalysis import (
        AdvancedCryptanalysisFramework,
        AdvancedResearchConfig,
        ResearchMode
    )
    from .neural_operator_cryptanalysis import CryptanalysisConfig
except ImportError:
    logger.warning("Advanced cryptanalysis modules not available - using fallback")
    AdvancedCryptanalysisFramework = object
    AdvancedResearchConfig = object
    ResearchMode = None
    CryptanalysisConfig = object


class ErrorSeverity(Enum):
    """Error severity levels for graduated response."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"


class ValidationError(Exception):
    """Custom exception for input validation failures."""
    
    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR"):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = time.time()


class SecurityError(Exception):
    """Custom exception for security-related failures."""
    
    def __init__(self, message: str, error_code: str = "SECURITY_ERROR"):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = time.time()


class OperationError(Exception):
    """Custom exception for operation failures."""
    
    def __init__(self, message: str, error_code: str = "OPERATION_ERROR", severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        super().__init__(message)
        self.error_code = error_code
        self.severity = severity
        self.timestamp = time.time()


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationResult:
    """Result of cryptanalysis operation with comprehensive metadata."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    memory_usage: float = 0.0
    validation_result: Optional[ValidationResult] = None
    security_checks: Dict[str, bool] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    operation_id: str = ""


class InputValidator:
    """Comprehensive input validation for cryptanalysis operations."""
    
    def __init__(self, max_data_size: int = 100_000_000, max_memory_mb: int = 8192):
        self.max_data_size = max_data_size
        self.max_memory_mb = max_memory_mb
        self.logger = logger.bind(component="input_validator")
    
    def validate_tensor_input(self, data: torch.Tensor, name: str = "input") -> ValidationResult:
        """Comprehensive tensor validation."""
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Basic type validation
            if not isinstance(data, torch.Tensor):
                errors.append(f"{name} must be a torch.Tensor, got {type(data)}")
                return ValidationResult(False, errors, warnings, metadata)
            
            # Size validation
            numel = data.numel()
            metadata["size"] = numel
            metadata["shape"] = list(data.shape)
            metadata["dtype"] = str(data.dtype)
            
            if numel == 0:
                errors.append(f"{name} cannot be empty")
            elif numel > self.max_data_size:
                errors.append(f"{name} too large: {numel} > {self.max_data_size}")
            
            # Memory estimation
            estimated_memory_mb = (numel * data.element_size()) / (1024 * 1024)
            metadata["estimated_memory_mb"] = estimated_memory_mb
            
            if estimated_memory_mb > self.max_memory_mb:
                errors.append(f"{name} requires too much memory: {estimated_memory_mb:.1f}MB > {self.max_memory_mb}MB")
            
            # Data type validation
            if data.dtype not in [torch.float32, torch.float64, torch.int32, torch.int64, torch.uint8]:
                warnings.append(f"{name} has unusual dtype {data.dtype}")
            
            # NaN/Inf validation
            if data.dtype.is_floating_point:
                if torch.isnan(data).any():
                    errors.append(f"{name} contains NaN values")
                if torch.isinf(data).any():
                    errors.append(f"{name} contains infinite values")
            
            # Range validation for uint8 (common for cipher data)
            if data.dtype == torch.uint8:
                metadata["unique_values"] = len(torch.unique(data))
                metadata["entropy"] = self._estimate_entropy(data)
                
                if metadata["entropy"] < 1.0:
                    warnings.append(f"{name} has very low entropy ({metadata['entropy']:.3f})")
            
            # Shape validation
            if len(data.shape) > 3:
                warnings.append(f"{name} has high dimensionality: {len(data.shape)}D")
            
            # Device validation
            metadata["device"] = str(data.device)
            if data.device.type == "cuda" and not torch.cuda.is_available():
                errors.append(f"{name} is on CUDA device but CUDA not available")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed for {name}: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation exception: {str(e)}"],
                warnings=warnings,
                metadata=metadata
            )
    
    def validate_config(self, config: Any) -> ValidationResult:
        """Validate configuration objects."""
        errors = []
        warnings = []
        metadata = {}
        
        try:
            if config is None:
                errors.append("Configuration cannot be None")
                return ValidationResult(False, errors, warnings, metadata)
            
            # Check for required attributes based on type
            if hasattr(config, 'spectral_resolution'):
                if config.spectral_resolution <= 0:
                    errors.append("spectral_resolution must be positive")
                elif config.spectral_resolution > 2048:
                    warnings.append(f"Large spectral_resolution: {config.spectral_resolution}")
            
            if hasattr(config, 'max_execution_time'):
                if config.max_execution_time <= 0:
                    errors.append("max_execution_time must be positive")
            
            if hasattr(config, 'security_level'):
                metadata["security_level"] = str(config.security_level)
            
            # Validate memory-related settings
            if hasattr(config, 'enable_parallel_processing') and config.enable_parallel_processing:
                if hasattr(config, 'max_workers') and config.max_workers > 16:
                    warnings.append(f"High worker count may cause resource contention: {config.max_workers}")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Config validation exception: {str(e)}"],
                warnings=warnings,
                metadata=metadata
            )
    
    def _estimate_entropy(self, data: torch.Tensor) -> float:
        """Estimate Shannon entropy of data."""
        try:
            # Convert to numpy for histogram
            data_np = data.cpu().numpy().flatten()
            
            # Compute histogram
            hist, _ = np.histogram(data_np, bins=min(256, len(np.unique(data_np))))
            
            # Compute probabilities
            probs = hist / np.sum(hist)
            probs = probs[probs > 0]  # Remove zeros
            
            # Shannon entropy
            entropy = -np.sum(probs * np.log2(probs))
            
            return float(entropy)
            
        except Exception:
            return 0.0


class SecurityManager:
    """Advanced security management for cryptanalysis operations."""
    
    def __init__(self, enable_audit_log: bool = True):
        self.enable_audit_log = enable_audit_log
        self.audit_log = []
        self.failed_operations = []
        self.security_violations = []
        self.logger = logger.bind(component="security_manager")
        self._lock = threading.Lock()
    
    def validate_operation_security(self, operation_type: str, data: torch.Tensor, **kwargs) -> Dict[str, bool]:
        """Comprehensive security validation."""
        checks = {
            "data_size_ok": True,
            "memory_safe": True,
            "no_suspicious_patterns": True,
            "rate_limit_ok": True,
            "device_safe": True
        }
        
        try:
            # Data size check
            if data.numel() > 100_000_000:  # 100M elements
                checks["data_size_ok"] = False
                self._log_security_event("LARGE_DATA_WARNING", f"Operation {operation_type} with {data.numel()} elements")
            
            # Memory safety check
            estimated_memory = (data.numel() * data.element_size()) / (1024 * 1024)  # MB
            if estimated_memory > 4096:  # 4GB
                checks["memory_safe"] = False
                self._log_security_event("MEMORY_WARNING", f"Operation requires {estimated_memory:.1f}MB")
            
            # Pattern analysis for potential attacks
            if self._detect_suspicious_patterns(data):
                checks["no_suspicious_patterns"] = False
                self._log_security_event("SUSPICIOUS_PATTERN", f"Detected in {operation_type}")
            
            # Rate limiting check
            if self._check_rate_limit(operation_type):
                checks["rate_limit_ok"] = False
                self._log_security_event("RATE_LIMIT_EXCEEDED", f"Operation {operation_type}")
            
            # Device security
            if data.device.type == "cuda" and not self._validate_cuda_device():
                checks["device_safe"] = False
                self._log_security_event("CUDA_SECURITY_WARNING", "Potentially unsafe CUDA operation")
            
            return checks
            
        except Exception as e:
            self.logger.error(f"Security validation failed: {e}")
            return {key: False for key in checks.keys()}
    
    def _detect_suspicious_patterns(self, data: torch.Tensor) -> bool:
        """Detect suspicious patterns in data."""
        try:
            # Check for all zeros/ones (potential attack vector)
            if torch.all(data == 0) or torch.all(data == 1):
                return True
            
            # Check for highly repetitive patterns
            if data.numel() > 100:
                sample = data.flatten()[:100]
                unique_ratio = len(torch.unique(sample)) / len(sample)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _check_rate_limit(self, operation_type: str) -> bool:
        """Check if operation exceeds rate limits."""
        try:
            current_time = time.time()
            
            # Count operations in last minute
            recent_ops = [
                entry for entry in self.audit_log
                if entry.get("timestamp", 0) > current_time - 60
                and entry.get("operation") == operation_type
            ]
            
            # Rate limit: 100 operations per minute per type
            return len(recent_ops) > 100
            
        except Exception:
            return False
    
    def _validate_cuda_device(self) -> bool:
        """Validate CUDA device security."""
        try:
            if torch.cuda.is_available():
                # Check available memory
                free_memory = torch.cuda.get_device_properties(0).total_memory
                if free_memory < 1024 * 1024 * 1024:  # Less than 1GB
                    return False
            return True
        except Exception:
            return False
    
    def _log_security_event(self, event_type: str, details: str):
        """Log security event."""
        with self._lock:
            event = {
                "timestamp": time.time(),
                "event_type": event_type,
                "details": details,
                "thread_id": threading.get_ident()
            }
            
            self.security_violations.append(event)
            
            if self.enable_audit_log:
                self.audit_log.append(event)
            
            # Limit log size
            if len(self.audit_log) > 10000:
                self.audit_log = self.audit_log[-5000:]
            
            if len(self.security_violations) > 1000:
                self.security_violations = self.security_violations[-500:]
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report."""
        with self._lock:
            return {
                "total_violations": len(self.security_violations),
                "recent_violations": len([
                    v for v in self.security_violations
                    if v["timestamp"] > time.time() - 3600  # Last hour
                ]),
                "violation_types": list(set([
                    v["event_type"] for v in self.security_violations[-100:]
                ])),
                "audit_log_size": len(self.audit_log)
            }


class ErrorRecoveryManager:
    """Advanced error recovery and fault tolerance."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logger.bind(component="error_recovery")
        self.recovery_strategies = {}
        self.error_history = []
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default error recovery strategies."""
        
        # Memory error recovery
        def memory_error_recovery(operation, *args, **kwargs):
            """Recover from memory errors by reducing batch size."""
            self.logger.info("Attempting memory error recovery")
            
            # Try with reduced data
            if args and isinstance(args[0], torch.Tensor):
                original_data = args[0]
                reduced_data = original_data[:len(original_data)//2]
                new_args = (reduced_data,) + args[1:]
                return operation(*new_args, **kwargs)
            
            raise OperationError("Memory error recovery failed")
        
        # CUDA error recovery
        def cuda_error_recovery(operation, *args, **kwargs):
            """Recover from CUDA errors by moving to CPU."""
            self.logger.info("Attempting CUDA error recovery")
            
            # Move tensors to CPU
            new_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    new_args.append(arg.cpu())
                else:
                    new_args.append(arg)
            
            new_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    new_kwargs[key] = value.cpu()
                else:
                    new_kwargs[key] = value
            
            return operation(*new_args, **new_kwargs)
        
        # Timeout recovery
        def timeout_recovery(operation, *args, **kwargs):
            """Recover from timeout by using simplified operation."""
            self.logger.info("Attempting timeout recovery")
            
            # Simplify operation (implementation-specific)
            kwargs["simplified_mode"] = True
            kwargs["max_iterations"] = kwargs.get("max_iterations", 100) // 2
            
            return operation(*args, **kwargs)
        
        self.recovery_strategies[RuntimeError] = memory_error_recovery
        self.recovery_strategies[torch.cuda.OutOfMemoryError] = cuda_error_recovery
        self.recovery_strategies[TimeoutError] = timeout_recovery
    
    def execute_with_recovery(
        self,
        operation: Callable,
        *args,
        retry_on: Optional[List[type]] = None,
        **kwargs
    ) -> Any:
        """Execute operation with automatic error recovery."""
        
        if retry_on is None:
            retry_on = [RuntimeError, torch.cuda.OutOfMemoryError, TimeoutError]
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    # Apply backoff delay
                    delay = self.backoff_factor ** (attempt - 1)
                    time.sleep(delay)
                    self.logger.info(f"Retry attempt {attempt} after {delay:.1f}s delay")
                
                result = operation(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(f"Operation succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                self._log_error(e, attempt)
                
                # Check if this error type has a recovery strategy
                error_type = type(e)
                if error_type in self.recovery_strategies and attempt < self.max_retries:
                    try:
                        self.logger.info(f"Applying recovery strategy for {error_type.__name__}")
                        return self.recovery_strategies[error_type](operation, *args, **kwargs)
                    except Exception as recovery_error:
                        self.logger.warning(f"Recovery strategy failed: {recovery_error}")
                        last_exception = recovery_error
                
                # Check if we should retry
                if attempt < self.max_retries and any(isinstance(e, retry_type) for retry_type in retry_on):
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    continue
                else:
                    break
        
        # All attempts failed
        raise OperationError(
            f"Operation failed after {self.max_retries + 1} attempts. Last error: {last_exception}",
            "MAX_RETRIES_EXCEEDED",
            ErrorSeverity.HIGH
        )
    
    def _log_error(self, error: Exception, attempt: int):
        """Log error occurrence."""
        error_entry = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "message": str(error),
            "attempt": attempt,
            "thread_id": threading.get_ident()
        }
        
        self.error_history.append(error_entry)
        
        # Limit history size
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]


class PerformanceMonitor:
    """Advanced performance monitoring and optimization."""
    
    def __init__(self):
        self.logger = logger.bind(component="performance_monitor")
        self.metrics_history = []
        self.performance_alerts = []
        self._lock = threading.Lock()
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring operation performance."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        metrics = {
            "operation_name": operation_name,
            "start_time": start_time,
            "start_memory_mb": start_memory,
            "thread_id": threading.get_ident()
        }
        
        try:
            yield metrics
            
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            metrics.update({
                "end_time": end_time,
                "execution_time": end_time - start_time,
                "end_memory_mb": end_memory,
                "memory_delta_mb": end_memory - start_memory,
                "success": True
            })
            
            # Check for performance issues
            self._check_performance_alerts(metrics)
            
            with self._lock:
                self.metrics_history.append(metrics)
                
                # Limit history size
                if len(self.metrics_history) > 5000:
                    self.metrics_history = self.metrics_history[-2500:]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)
            else:
                # Fallback: estimate using torch if available
                if torch.cuda.is_available():
                    return torch.cuda.memory_allocated() / (1024 * 1024)
                return 0.0
        except Exception:
            return 0.0
    
    def _check_performance_alerts(self, metrics: Dict[str, Any]):
        """Check for performance issues and generate alerts."""
        alerts = []
        
        # Execution time alerts
        if metrics["execution_time"] > 60:  # 1 minute
            alerts.append({
                "type": "SLOW_EXECUTION",
                "message": f"Operation {metrics['operation_name']} took {metrics['execution_time']:.1f}s",
                "severity": ErrorSeverity.MEDIUM
            })
        
        # Memory usage alerts
        if metrics["memory_delta_mb"] > 1024:  # 1GB increase
            alerts.append({
                "type": "HIGH_MEMORY_USAGE",
                "message": f"Operation {metrics['operation_name']} used {metrics['memory_delta_mb']:.1f}MB",
                "severity": ErrorSeverity.MEDIUM
            })
        
        # Add alerts with timestamp
        for alert in alerts:
            alert["timestamp"] = time.time()
            alert["operation_metrics"] = metrics
        
        with self._lock:
            self.performance_alerts.extend(alerts)
            
            # Limit alerts history
            if len(self.performance_alerts) > 1000:
                self.performance_alerts = self.performance_alerts[-500:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self._lock:
            if not self.metrics_history:
                return {"message": "No performance data available"}
            
            # Compute statistics
            execution_times = [m["execution_time"] for m in self.metrics_history]
            memory_deltas = [m["memory_delta_mb"] for m in self.metrics_history]
            
            return {
                "total_operations": len(self.metrics_history),
                "execution_time_stats": {
                    "mean": np.mean(execution_times),
                    "median": np.median(execution_times),
                    "max": np.max(execution_times),
                    "min": np.min(execution_times),
                    "std": np.std(execution_times)
                },
                "memory_usage_stats": {
                    "mean_delta": np.mean(memory_deltas),
                    "max_delta": np.max(memory_deltas),
                    "min_delta": np.min(memory_deltas)
                },
                "recent_alerts": len([
                    a for a in self.performance_alerts
                    if a["timestamp"] > time.time() - 3600  # Last hour
                ]),
                "operation_breakdown": self._get_operation_breakdown()
            }
    
    def _get_operation_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get performance breakdown by operation type."""
        breakdown = {}
        
        for metrics in self.metrics_history[-1000:]:  # Last 1000 operations
            op_name = metrics["operation_name"]
            if op_name not in breakdown:
                breakdown[op_name] = {
                    "count": 0,
                    "total_time": 0,
                    "total_memory": 0
                }
            
            breakdown[op_name]["count"] += 1
            breakdown[op_name]["total_time"] += metrics["execution_time"]
            breakdown[op_name]["total_memory"] += metrics["memory_delta_mb"]
        
        # Compute averages
        for op_name, stats in breakdown.items():
            if stats["count"] > 0:
                stats["avg_time"] = stats["total_time"] / stats["count"]
                stats["avg_memory"] = stats["total_memory"] / stats["count"]
        
        return breakdown


class UltraRobustCryptanalysisFramework:
    """Ultra-robust cryptanalysis framework with comprehensive error handling."""
    
    def __init__(self, config: Optional[AdvancedResearchConfig] = None):
        # Initialize configuration
        if config is None:
            config = AdvancedResearchConfig() if AdvancedResearchConfig != object else type('Config', (), {})()
        self.config = config
        
        # Initialize components
        self.logger = logger.bind(component="ultra_robust_cryptanalysis")
        self.validator = InputValidator()
        self.security_manager = SecurityManager()
        self.error_recovery = ErrorRecoveryManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize base framework if available
        try:
            if AdvancedCryptanalysisFramework != object:
                self.base_framework = AdvancedCryptanalysisFramework(config)
            else:
                self.base_framework = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize base framework: {e}")
            self.base_framework = None
        
        # Operation tracking
        self.operation_count = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self._operation_lock = threading.Lock()
        
        # Health status
        self.health_status = {
            "status": "healthy",
            "last_check": time.time(),
            "issues": []
        }
        
        self.logger.info("Ultra-robust cryptanalysis framework initialized")
    
    def analyze_cipher_with_full_protection(
        self,
        cipher_data: torch.Tensor,
        analysis_types: Optional[List[str]] = None,
        max_execution_time: Optional[float] = None,
        enable_recovery: bool = True,
        security_level: str = "high"
    ) -> OperationResult:
        """Perform cipher analysis with full protection and error handling."""
        
        operation_id = self._generate_operation_id()
        
        with self.performance_monitor.monitor_operation("cipher_analysis") as perf_metrics:
            try:
                # Phase 1: Input Validation
                validation_result = self._validate_inputs(cipher_data, analysis_types)
                if not validation_result.is_valid:
                    return OperationResult(
                        success=False,
                        error=ValidationError(f"Input validation failed: {'; '.join(validation_result.errors)}"),
                        validation_result=validation_result,
                        operation_id=operation_id
                    )
                
                # Phase 2: Security Checks
                security_checks = self.security_manager.validate_operation_security(
                    "cipher_analysis", cipher_data
                )
                
                if not all(security_checks.values()):
                    failed_checks = [k for k, v in security_checks.items() if not v]
                    return OperationResult(
                        success=False,
                        error=SecurityError(f"Security checks failed: {failed_checks}"),
                        security_checks=security_checks,
                        operation_id=operation_id
                    )
                
                # Phase 3: Execute Analysis with Recovery
                if enable_recovery:
                    result = self.error_recovery.execute_with_recovery(
                        self._execute_protected_analysis,
                        cipher_data,
                        analysis_types,
                        max_execution_time
                    )
                else:
                    result = self._execute_protected_analysis(
                        cipher_data, analysis_types, max_execution_time
                    )
                
                # Phase 4: Post-processing and validation
                validated_result = self._validate_and_sanitize_result(result)
                
                # Success
                with self._operation_lock:
                    self.successful_operations += 1
                
                return OperationResult(
                    success=True,
                    result=validated_result,
                    execution_time=perf_metrics.get("execution_time", 0),
                    memory_usage=perf_metrics.get("memory_delta_mb", 0),
                    validation_result=validation_result,
                    security_checks=security_checks,
                    performance_metrics=perf_metrics,
                    operation_id=operation_id
                )
                
            except Exception as e:
                # Handle any remaining errors
                with self._operation_lock:
                    self.failed_operations += 1
                
                self.logger.error(f"Analysis failed with error: {e}")
                
                return OperationResult(
                    success=False,
                    error=e,
                    execution_time=perf_metrics.get("execution_time", 0),
                    memory_usage=perf_metrics.get("memory_delta_mb", 0),
                    validation_result=validation_result if 'validation_result' in locals() else None,
                    security_checks=security_checks if 'security_checks' in locals() else {},
                    operation_id=operation_id
                )
    
    def _validate_inputs(self, cipher_data: torch.Tensor, analysis_types: Optional[List[str]]) -> ValidationResult:
        """Comprehensive input validation."""
        # Validate tensor
        tensor_validation = self.validator.validate_tensor_input(cipher_data, "cipher_data")
        
        if not tensor_validation.is_valid:
            return tensor_validation
        
        # Validate analysis types
        if analysis_types is not None:
            valid_types = ["differential", "linear", "frequency", "spectral", "quantum"]
            invalid_types = [t for t in analysis_types if t not in valid_types]
            
            if invalid_types:
                tensor_validation.errors.append(f"Invalid analysis types: {invalid_types}")
                tensor_validation.is_valid = False
        
        return tensor_validation
    
    def _execute_protected_analysis(
        self,
        cipher_data: torch.Tensor,
        analysis_types: Optional[List[str]],
        max_execution_time: Optional[float]
    ) -> Dict[str, Any]:
        """Execute analysis with timeout protection."""
        
        if max_execution_time is None:
            max_execution_time = getattr(self.config, 'max_execution_time', 300.0)
        
        # Use ThreadPoolExecutor for timeout control
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._perform_analysis, cipher_data, analysis_types)
            
            try:
                result = future.result(timeout=max_execution_time)
                return result
            except TimeoutError:
                future.cancel()
                raise OperationError(
                    f"Analysis timed out after {max_execution_time}s",
                    "ANALYSIS_TIMEOUT",
                    ErrorSeverity.HIGH
                )
    
    def _perform_analysis(self, cipher_data: torch.Tensor, analysis_types: Optional[List[str]]) -> Dict[str, Any]:
        """Perform the actual cryptanalysis."""
        
        if self.base_framework:
            # Use advanced framework if available
            return self.base_framework.comprehensive_research_analysis(cipher_data)
        else:
            # Fallback analysis
            return self._fallback_analysis(cipher_data, analysis_types)
    
    def _fallback_analysis(self, cipher_data: torch.Tensor, analysis_types: Optional[List[str]]) -> Dict[str, Any]:
        """Fallback analysis when advanced framework unavailable."""
        
        self.logger.info("Using fallback analysis")
        
        # Basic frequency analysis
        unique_values, counts = torch.unique(cipher_data, return_counts=True)
        frequencies = counts.float() / counts.sum()
        
        # Compute entropy
        entropy = -torch.sum(frequencies * torch.log2(frequencies + 1e-10))
        
        # Basic statistical measures
        data_float = cipher_data.float()
        mean_val = torch.mean(data_float)
        std_val = torch.std(data_float)
        
        # Simple vulnerability assessment
        vulnerability_score = torch.abs(entropy - 8.0) / 8.0  # Distance from maximum entropy
        
        return {
            "fallback_analysis": {
                "entropy": entropy,
                "unique_values": len(unique_values),
                "mean": mean_val,
                "std": std_val,
                "vulnerability_score": vulnerability_score
            },
            "overall": {
                "combined_vulnerability_score": vulnerability_score,
                "overall_vulnerability_level": "LOW" if entropy > 6 else "MEDIUM",
                "recommendation": "Basic fallback analysis completed"
            },
            "analysis_mode": "fallback"
        }
    
    def _validate_and_sanitize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize analysis result."""
        
        sanitized = {}
        
        for key, value in result.items():
            try:
                if isinstance(value, torch.Tensor):
                    # Check for NaN/Inf in tensors
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        self.logger.warning(f"Invalid values detected in {key}, replacing with zeros")
                        sanitized[key] = torch.zeros_like(value)
                    else:
                        sanitized[key] = value
                elif isinstance(value, dict):
                    sanitized[key] = self._validate_and_sanitize_result(value)
                elif isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        self.logger.warning(f"Invalid numeric value in {key}, replacing with 0")
                        sanitized[key] = 0.0
                    else:
                        sanitized[key] = value
                else:
                    sanitized[key] = value
                    
            except Exception as e:
                self.logger.warning(f"Error sanitizing {key}: {e}")
                sanitized[key] = None
        
        return sanitized
    
    def _generate_operation_id(self) -> str:
        """Generate unique operation ID."""
        with self._operation_lock:
            self.operation_count += 1
            timestamp = int(time.time() * 1000)
            return f"robust_crypto_{timestamp}_{self.operation_count}"
    
    def batch_analyze_with_protection(
        self,
        cipher_datasets: List[torch.Tensor],
        max_workers: Optional[int] = None,
        continue_on_error: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> List[OperationResult]:
        """Batch analysis with full protection and error isolation."""
        
        if max_workers is None:
            max_workers = min(4, len(cipher_datasets))
        
        results = [None] * len(cipher_datasets)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(
                    self.analyze_cipher_with_full_protection,
                    cipher_data
                ): i
                for i, cipher_data in enumerate(cipher_datasets)
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    self.logger.error(f"Batch analysis failed for dataset {index}: {e}")
                    results[index] = OperationResult(
                        success=False,
                        error=e,
                        operation_id=f"batch_error_{index}"
                    )
                
                completed += 1
                
                # Progress callback
                if progress_callback:
                    try:
                        progress_callback(completed, len(cipher_datasets))
                    except Exception as e:
                        self.logger.warning(f"Progress callback failed: {e}")
                
                # Stop on error if requested
                if not continue_on_error and not results[index].success:
                    self.logger.warning("Stopping batch analysis due to error")
                    break
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        
        health_report = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "components": {},
            "issues": [],
            "metrics": {}
        }
        
        # Check validator
        try:
            test_tensor = torch.randn(10)
            val_result = self.validator.validate_tensor_input(test_tensor, "test")
            health_report["components"]["validator"] = "healthy" if val_result.is_valid else "issues"
            if not val_result.is_valid:
                health_report["issues"].extend(val_result.errors)
        except Exception as e:
            health_report["components"]["validator"] = "error"
            health_report["issues"].append(f"Validator error: {e}")
        
        # Check security manager
        try:
            security_report = self.security_manager.get_security_report()
            health_report["components"]["security_manager"] = "healthy"
            health_report["metrics"]["security"] = security_report
            
            if security_report["recent_violations"] > 10:
                health_report["issues"].append("High number of recent security violations")
                
        except Exception as e:
            health_report["components"]["security_manager"] = "error"
            health_report["issues"].append(f"Security manager error: {e}")
        
        # Check performance monitor
        try:
            perf_report = self.performance_monitor.get_performance_report()
            health_report["components"]["performance_monitor"] = "healthy"
            health_report["metrics"]["performance"] = perf_report
            
            if perf_report.get("recent_alerts", 0) > 5:
                health_report["issues"].append("High number of performance alerts")
                
        except Exception as e:
            health_report["components"]["performance_monitor"] = "error"
            health_report["issues"].append(f"Performance monitor error: {e}")
        
        # Check base framework
        if self.base_framework:
            health_report["components"]["base_framework"] = "available"
        else:
            health_report["components"]["base_framework"] = "unavailable"
            health_report["issues"].append("Base framework not available - using fallback mode")
        
        # Overall status assessment
        error_components = [k for k, v in health_report["components"].items() if v == "error"]
        if error_components:
            health_report["overall_status"] = "critical"
        elif health_report["issues"]:
            health_report["overall_status"] = "degraded"
        else:
            health_report["overall_status"] = "healthy"
        
        # Operation statistics
        with self._operation_lock:
            health_report["metrics"]["operations"] = {
                "total": self.operation_count,
                "successful": self.successful_operations,
                "failed": self.failed_operations,
                "success_rate": (
                    self.successful_operations / max(self.operation_count, 1)
                )
            }
        
        # Update health status
        self.health_status = health_report
        
        return health_report
    
    def shutdown(self):
        """Graceful shutdown of the framework."""
        self.logger.info("Shutting down ultra-robust cryptanalysis framework")
        
        try:
            # Shutdown base framework if available
            if self.base_framework and hasattr(self.base_framework, 'shutdown'):
                self.base_framework.shutdown()
        except Exception as e:
            self.logger.warning(f"Error shutting down base framework: {e}")
        
        # Clear sensitive data
        self.security_manager.audit_log.clear()
        self.error_recovery.error_history.clear()
        
        self.logger.info("Framework shutdown complete")


def create_ultra_robust_framework(**kwargs) -> UltraRobustCryptanalysisFramework:
    """Create ultra-robust cryptanalysis framework with advanced error handling."""
    
    config = None
    if AdvancedResearchConfig != object:
        config = AdvancedResearchConfig(**kwargs)
    
    return UltraRobustCryptanalysisFramework(config)


# Convenience function for protected analysis
def analyze_cipher_safely(
    cipher_data: torch.Tensor,
    analysis_types: Optional[List[str]] = None,
    max_execution_time: float = 60.0,
    **kwargs
) -> OperationResult:
    """Perform safe cipher analysis with automatic error handling."""
    
    framework = create_ultra_robust_framework(**kwargs)
    
    try:
        return framework.analyze_cipher_with_full_protection(
            cipher_data=cipher_data,
            analysis_types=analysis_types,
            max_execution_time=max_execution_time
        )
    finally:
        framework.shutdown()