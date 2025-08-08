"""Security and Validation Module for Cryptanalysis.

Provides security measures, input validation, and defensive programming
patterns for neural operator cryptanalysis operations.
"""

import hashlib
import secrets
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
from loguru import logger
from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc


class SecurityLevel(Enum):
    """Security levels for cryptanalysis operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class ResourceLimitError(Exception):
    """Custom exception for resource limit violations."""
    pass


@dataclass
class SecurityConfig:
    """Security configuration for cryptanalysis operations."""
    
    max_data_size: int = 10_000_000  # 10MB max
    max_execution_time: float = 300.0  # 5 minutes
    max_memory_usage: int = 2_000_000_000  # 2GB
    max_concurrent_operations: int = 10
    enable_differential_privacy: bool = True
    noise_scale: float = 0.1
    rate_limit_requests: int = 100  # per minute
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    log_security_events: bool = True
    sanitize_outputs: bool = True
    

@dataclass
class ValidationConfig:
    """Validation configuration for inputs and outputs."""
    
    min_data_size: int = 8
    max_tensor_dimensions: int = 4
    allowed_dtypes: List[torch.dtype] = field(default_factory=lambda: [
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        torch.float16, torch.float32, torch.float64
    ])
    max_batch_size: int = 1000
    validate_numerical_stability: bool = True
    check_for_adversarial_inputs: bool = True
    

class CryptanalysisValidator:
    """Validates inputs and outputs for cryptanalysis operations."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logger.bind(component="validator")
    
    def validate_cipher_data(self, data: torch.Tensor, context: str = "unknown") -> bool:
        """Validate cipher data input."""
        try:
            self._validate_tensor_basic(data, context)
            self._validate_tensor_size(data, context)
            self._validate_tensor_content(data, context)
            
            if self.config.check_for_adversarial_inputs:
                self._check_adversarial_patterns(data, context)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation failed for {context}: {e}")
            raise ValidationError(f"Invalid cipher data in {context}: {e}")
    
    def _validate_tensor_basic(self, tensor: torch.Tensor, context: str):
        """Basic tensor validation."""
        if not torch.is_tensor(tensor):
            raise ValidationError(f"Input must be a tensor in {context}")
        
        if tensor.dtype not in self.config.allowed_dtypes:
            raise ValidationError(f"Unsupported dtype {tensor.dtype} in {context}")
        
        if len(tensor.shape) > self.config.max_tensor_dimensions:
            raise ValidationError(f"Too many dimensions ({len(tensor.shape)}) in {context}")
        
        if tensor.numel() == 0:
            raise ValidationError(f"Empty tensor in {context}")
    
    def _validate_tensor_size(self, tensor: torch.Tensor, context: str):
        """Validate tensor size constraints."""
        if tensor.numel() < self.config.min_data_size:
            raise ValidationError(f"Data too small ({tensor.numel()}) in {context}")
        
        if tensor.size(0) > self.config.max_batch_size:
            raise ValidationError(f"Batch size too large ({tensor.size(0)}) in {context}")
    
    def _validate_tensor_content(self, tensor: torch.Tensor, context: str):
        """Validate tensor content for numerical stability."""
        if not self.config.validate_numerical_stability:
            return
        
        if torch.isnan(tensor).any():
            raise ValidationError(f"NaN values detected in {context}")
        
        if torch.isinf(tensor).any():
            raise ValidationError(f"Infinite values detected in {context}")
        
        # Check for extreme values that might cause numerical issues
        if tensor.dtype.is_floating_point:
            abs_max = torch.abs(tensor).max()
            if abs_max > 1e6:
                self.logger.warning(f"Large values detected in {context}: {abs_max}")
    
    def _check_adversarial_patterns(self, tensor: torch.Tensor, context: str):
        """Check for potential adversarial input patterns."""
        # Check for suspiciously uniform distributions
        if tensor.dtype == torch.uint8:
            unique_values = torch.unique(tensor)
            if len(unique_values) == 1:
                raise ValidationError(f"Uniform data detected in {context} - potential adversarial input")
        
        # Check for suspiciously repeating patterns
        if tensor.numel() >= 32:
            flat_tensor = tensor.flatten()
            pattern_length = min(16, len(flat_tensor) // 4)
            pattern = flat_tensor[:pattern_length]
            
            # Check if pattern repeats throughout the data
            repeats = 0
            for i in range(pattern_length, len(flat_tensor) - pattern_length, pattern_length):
                if torch.equal(flat_tensor[i:i+pattern_length], pattern):
                    repeats += 1
            
            repeat_ratio = repeats / ((len(flat_tensor) - pattern_length) // pattern_length)
            if repeat_ratio > 0.8:
                self.logger.warning(f"Highly repetitive pattern detected in {context}")
    
    def validate_analysis_result(self, result: Dict[str, Any], context: str = "unknown") -> bool:
        """Validate analysis result structure and content."""
        try:
            required_keys = ["overall"]
            for key in required_keys:
                if key not in result:
                    raise ValidationError(f"Missing required key '{key}' in result")
            
            # Validate overall result structure
            overall = result["overall"]
            if "combined_vulnerability_score" not in overall:
                raise ValidationError("Missing vulnerability score in overall result")
            
            score = overall["combined_vulnerability_score"]
            if torch.is_tensor(score):
                if torch.isnan(score) or torch.isinf(score):
                    raise ValidationError("Invalid vulnerability score")
                if score < 0 or score > 10:  # Reasonable bounds
                    raise ValidationError(f"Vulnerability score out of bounds: {score}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Result validation failed for {context}: {e}")
            raise ValidationError(f"Invalid analysis result in {context}: {e}")


class SecurityManager:
    """Manages security aspects of cryptanalysis operations."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logger.bind(component="security")
        self.rate_limiter = RateLimiter(config.rate_limit_requests)
        self.resource_monitor = ResourceMonitor(config)
        self.audit_trail = AuditTrail()
        
    def secure_operation(self, operation_type: str, operation_id: str):
        """Context manager for secure operations."""
        return SecureOperationContext(
            self, operation_type, operation_id
        )
    
    def apply_differential_privacy(self, data: torch.Tensor) -> torch.Tensor:
        """Apply differential privacy noise to data."""
        if not self.config.enable_differential_privacy:
            return data
        
        noise_shape = data.shape
        if data.dtype.is_floating_point:
            noise = torch.normal(0, self.config.noise_scale, noise_shape)
            return data + noise
        else:
            # For integer data, add discrete noise
            noise_magnitude = max(1, int(self.config.noise_scale * 255))
            noise = torch.randint(-noise_magnitude, noise_magnitude + 1, noise_shape)
            return torch.clamp(data.long() + noise, 0, 255).to(data.dtype)
    
    def sanitize_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize output to remove sensitive information."""
        if not self.config.sanitize_outputs:
            return result
        
        sanitized = {}
        
        for key, value in result.items():
            if key in ["execution_metadata", "internal_state", "debug_info"]:
                # Remove potentially sensitive metadata
                continue
            
            if isinstance(value, dict):
                sanitized[key] = self.sanitize_output(value)
            elif torch.is_tensor(value):
                # Round tensor values to prevent fingerprinting
                if value.dtype.is_floating_point:
                    sanitized[key] = torch.round(value * 1000) / 1000
                else:
                    sanitized[key] = value
            else:
                sanitized[key] = value
        
        return sanitized
    
    def generate_operation_id(self) -> str:
        """Generate secure operation ID."""
        timestamp = str(int(time.time() * 1000))
        random_part = secrets.token_hex(8)
        return f"op_{timestamp}_{random_part}"
    
    def check_security_constraints(self, operation_type: str, data_size: int):
        """Check if operation meets security constraints."""
        if data_size > self.config.max_data_size:
            raise SecurityError(f"Data size {data_size} exceeds limit {self.config.max_data_size}")
        
        if not self.rate_limiter.allow_request():
            raise SecurityError("Rate limit exceeded")
        
        if not self.resource_monitor.check_resources():
            raise SecurityError("Resource limits exceeded")
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security-related events."""
        if self.config.log_security_events:
            self.audit_trail.record_event(event_type, details)
            self.logger.info(f"Security event: {event_type}", **details)


class SecureOperationContext:
    """Context manager for secure cryptanalysis operations."""
    
    def __init__(self, security_manager: SecurityManager, operation_type: str, operation_id: str):
        self.security_manager = security_manager
        self.operation_type = operation_type
        self.operation_id = operation_id
        self.start_time = None
        self.logger = logger.bind(operation_id=operation_id)
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting secure operation: {self.operation_type}")
        
        # Check resource limits
        if not self.security_manager.resource_monitor.check_resources():
            raise SecurityError("Insufficient resources for operation")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        
        if execution_time > self.security_manager.config.max_execution_time:
            self.logger.warning(f"Operation exceeded time limit: {execution_time:.2f}s")
        
        # Log operation completion
        self.security_manager.log_security_event(
            "operation_completed",
            {
                "operation_type": self.operation_type,
                "operation_id": self.operation_id,
                "execution_time": execution_time,
                "success": exc_type is None
            }
        )
        
        if exc_type is not None:
            self.logger.error(f"Operation failed: {exc_val}")
        else:
            self.logger.info(f"Operation completed successfully in {execution_time:.2f}s")
        
        # Force garbage collection
        gc.collect()


class RateLimiter:
    """Simple rate limiter for API requests."""
    
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()
    
    def allow_request(self) -> bool:
        """Check if request is allowed under rate limit."""
        with self.lock:
            current_time = time.time()
            
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests 
                           if current_time - req_time < 60]
            
            if len(self.requests) >= self.max_requests:
                return False
            
            self.requests.append(current_time)
            return True


class ResourceMonitor:
    """Monitors system resources during operations."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logger.bind(component="resource_monitor")
    
    def check_resources(self) -> bool:
        """Check if system has sufficient resources."""
        try:
            # Check memory usage
            memory_info = psutil.virtual_memory()
            if memory_info.available < self.config.max_memory_usage:
                self.logger.warning(f"Low memory: {memory_info.available} bytes available")
                return False
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 90:
                self.logger.warning(f"High CPU usage: {cpu_percent}%")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Resource check failed: {e}")
            return False
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage statistics."""
        try:
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            return {
                "memory_used_gb": (memory_info.total - memory_info.available) / (1024**3),
                "memory_available_gb": memory_info.available / (1024**3),
                "memory_percent": memory_info.percent,
                "cpu_percent": cpu_percent
            }
        except Exception as e:
            self.logger.error(f"Failed to get resource usage: {e}")
            return {}


class AuditTrail:
    """Maintains audit trail of security events."""
    
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()
        self.max_events = 10000  # Limit memory usage
    
    def record_event(self, event_type: str, details: Dict[str, Any]):
        """Record a security event."""
        with self.lock:
            event = {
                "timestamp": time.time(),
                "event_type": event_type,
                "details": details.copy(),
                "event_hash": self._generate_event_hash(event_type, details)
            }
            
            self.events.append(event)
            
            # Limit number of stored events
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events//2:]
    
    def _generate_event_hash(self, event_type: str, details: Dict[str, Any]) -> str:
        """Generate hash for event integrity."""
        event_str = f"{event_type}_{str(sorted(details.items()))}"
        return hashlib.sha256(event_str.encode()).hexdigest()[:16]
    
    def get_recent_events(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent security events."""
        with self.lock:
            return self.events[-count:]
    
    def clear_old_events(self, max_age_hours: int = 24):
        """Clear events older than specified age."""
        with self.lock:
            cutoff_time = time.time() - (max_age_hours * 3600)
            self.events = [event for event in self.events 
                          if event["timestamp"] > cutoff_time]


class ErrorHandler:
    """Centralized error handling for cryptanalysis operations."""
    
    def __init__(self):
        self.logger = logger.bind(component="error_handler")
    
    @contextmanager
    def handle_operation(self, operation_name: str, operation_id: str):
        """Context manager for operation error handling."""
        try:
            yield
        except ValidationError as e:
            self.logger.error(f"Validation error in {operation_name} ({operation_id}): {e}")
            raise
        except SecurityError as e:
            self.logger.error(f"Security error in {operation_name} ({operation_id}): {e}")
            raise
        except ResourceLimitError as e:
            self.logger.error(f"Resource limit error in {operation_name} ({operation_id}): {e}")
            raise
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"GPU memory error in {operation_name} ({operation_id}): {e}")
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise ResourceLimitError(f"GPU memory exhausted: {e}")
        except Exception as e:
            self.logger.exception(f"Unexpected error in {operation_name} ({operation_id}): {e}")
            raise RuntimeError(f"Operation failed: {e}")
    
    def create_error_report(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized error report."""
        return {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": time.time(),
            "traceback": str(error.__traceback__) if hasattr(error, '__traceback__') else None
        }


def create_secure_cryptanalysis_environment(
    security_level: SecurityLevel = SecurityLevel.MEDIUM,
    max_data_size: int = 10_000_000,
    max_execution_time: float = 300.0
) -> Tuple[SecurityManager, CryptanalysisValidator, ErrorHandler]:
    """Create a complete secure environment for cryptanalysis."""
    
    security_config = SecurityConfig(
        max_data_size=max_data_size,
        max_execution_time=max_execution_time,
        security_level=security_level,
        enable_differential_privacy=(security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]),
        sanitize_outputs=True,
        log_security_events=True
    )
    
    validation_config = ValidationConfig(
        check_for_adversarial_inputs=(security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]),
        validate_numerical_stability=True
    )
    
    security_manager = SecurityManager(security_config)
    validator = CryptanalysisValidator(validation_config)
    error_handler = ErrorHandler()
    
    return security_manager, validator, error_handler


# Decorators for secure operations
def secure_cryptanalysis_operation(security_level: SecurityLevel = SecurityLevel.MEDIUM):
    """Decorator for securing cryptanalysis operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            security_manager, validator, error_handler = create_secure_cryptanalysis_environment(
                security_level=security_level
            )
            
            operation_id = security_manager.generate_operation_id()
            operation_name = func.__name__
            
            with error_handler.handle_operation(operation_name, operation_id):
                with security_manager.secure_operation(operation_name, operation_id):
                    # Validate inputs if they contain tensor data
                    for arg in args:
                        if torch.is_tensor(arg):
                            validator.validate_cipher_data(arg, f"{operation_name}_input")
                    
                    for key, value in kwargs.items():
                        if torch.is_tensor(value):
                            validator.validate_cipher_data(value, f"{operation_name}_{key}")
                    
                    # Execute operation
                    result = func(*args, **kwargs)
                    
                    # Validate and sanitize output
                    if isinstance(result, dict):
                        validator.validate_analysis_result(result, f"{operation_name}_output")
                        result = security_manager.sanitize_output(result)
                    
                    return result
        
        return wrapper
    return decorator
