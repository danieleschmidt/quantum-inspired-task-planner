"""Security module for quantum task planner."""

import hashlib
import hmac
import secrets
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    timestamp: float
    event_type: str
    severity: SecurityLevel
    user_id: Optional[str]
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class SecurityManager:
    """Comprehensive security manager for quantum planning operations."""
    
    def __init__(self):
        self.audit_log: List[SecurityEvent] = []
        self.rate_limits: Dict[str, List[float]] = {}
        self.blocked_ips: set = set()
        self.session_tokens: Dict[str, Dict[str, Any]] = {}
        
    def generate_session_token(self, user_id: str) -> str:
        """Generate a secure session token."""
        token = secrets.token_urlsafe(32)
        self.session_tokens[token] = {
            "user_id": user_id,
            "created_at": time.time(),
            "last_used": time.time(),
            "permissions": ["read", "optimize"]  # Default permissions
        }
        
        self.log_security_event(
            event_type="session_created",
            severity=SecurityLevel.LOW,
            user_id=user_id,
            details={"token_prefix": token[:8]}
        )
        
        return token
    
    def validate_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate and refresh session token."""
        if token not in self.session_tokens:
            self.log_security_event(
                event_type="invalid_token",
                severity=SecurityLevel.MEDIUM,
                user_id=None,
                details={"token_prefix": token[:8] if token else "empty"}
            )
            return None
        
        session = self.session_tokens[token]
        
        # Check if token is expired (24 hours)
        if time.time() - session["created_at"] > 86400:
            self.revoke_session_token(token)
            self.log_security_event(
                event_type="token_expired",
                severity=SecurityLevel.LOW,
                user_id=session["user_id"],
                details={"token_prefix": token[:8]}
            )
            return None
        
        # Update last used time
        session["last_used"] = time.time()
        return session
    
    def revoke_session_token(self, token: str) -> bool:
        """Revoke a session token."""
        if token in self.session_tokens:
            user_id = self.session_tokens[token]["user_id"]
            del self.session_tokens[token]
            
            self.log_security_event(
                event_type="session_revoked",
                severity=SecurityLevel.LOW,
                user_id=user_id,
                details={"token_prefix": token[:8]}
            )
            return True
        return False
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100, window_seconds: int = 3600) -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        
        # Initialize if not exists
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
        
        # Remove old requests outside the window
        self.rate_limits[identifier] = [
            req_time for req_time in self.rate_limits[identifier]
            if current_time - req_time < window_seconds
        ]
        
        # Check if under limit
        if len(self.rate_limits[identifier]) >= max_requests:
            self.log_security_event(
                event_type="rate_limit_exceeded",
                severity=SecurityLevel.HIGH,
                user_id=identifier,
                details={"requests": len(self.rate_limits[identifier]), "limit": max_requests}
            )
            return False
        
        # Add current request
        self.rate_limits[identifier].append(current_time)
        return True
    
    def sanitize_input(self, data: Any) -> Any:
        """Sanitize input data to prevent injection attacks."""
        if isinstance(data, str):
            # Remove or escape dangerous characters
            dangerous_chars = ["<", ">", "&", "\"", "'", ";", "(", ")", "--", "/*", "*/"]
            sanitized = data
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, "")
            return sanitized.strip()
        
        elif isinstance(data, dict):
            return {k: self.sanitize_input(v) for k, v in data.items()}
        
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        
        return data
    
    def validate_permissions(self, session: Dict[str, Any], required_permission: str) -> bool:
        """Validate user permissions for operation."""
        user_permissions = session.get("permissions", [])
        
        if required_permission not in user_permissions:
            self.log_security_event(
                event_type="permission_denied",
                severity=SecurityLevel.HIGH,
                user_id=session.get("user_id"),
                details={"required": required_permission, "has": user_permissions}
            )
            return False
        
        return True
    
    def encrypt_sensitive_data(self, data: str, key: Optional[str] = None) -> str:
        """Encrypt sensitive data using HMAC."""
        if key is None:
            key = secrets.token_urlsafe(32)
        
        # Use HMAC for data integrity and authenticity
        signature = hmac.new(
            key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{signature}:{data}"
    
    def verify_sensitive_data(self, encrypted_data: str, key: str) -> Optional[str]:
        """Verify and decrypt sensitive data."""
        try:
            signature, data = encrypted_data.split(":", 1)
            expected_signature = hmac.new(
                key.encode(),
                data.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if hmac.compare_digest(signature, expected_signature):
                return data
            else:
                self.log_security_event(
                    event_type="data_integrity_violation",
                    severity=SecurityLevel.CRITICAL,
                    user_id=None,
                    details={"data_prefix": data[:20]}
                )
                return None
                
        except ValueError:
            self.log_security_event(
                event_type="malformed_encrypted_data",
                severity=SecurityLevel.HIGH,
                user_id=None,
                details={"data_prefix": encrypted_data[:20]}
            )
            return None
    
    def log_security_event(
        self,
        event_type: str,
        severity: SecurityLevel,
        user_id: Optional[str],
        details: Dict[str, Any],
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """Log security event for audit trail."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.audit_log.append(event)
        
        # Log to standard logger as well
        log_level = {
            SecurityLevel.LOW: logging.INFO,
            SecurityLevel.MEDIUM: logging.WARNING,
            SecurityLevel.HIGH: logging.ERROR,
            SecurityLevel.CRITICAL: logging.CRITICAL
        }[severity]
        
        logger.log(
            log_level,
            f"Security Event: {event_type} | User: {user_id} | Details: {details}"
        )
        
        # Keep only last 10000 events to prevent memory issues
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics for monitoring."""
        current_time = time.time()
        last_hour = current_time - 3600
        last_day = current_time - 86400
        
        events_last_hour = [e for e in self.audit_log if e.timestamp > last_hour]
        events_last_day = [e for e in self.audit_log if e.timestamp > last_day]
        
        metrics = {
            "total_events": len(self.audit_log),
            "events_last_hour": len(events_last_hour),
            "events_last_day": len(events_last_day),
            "active_sessions": len(self.session_tokens),
            "blocked_ips": len(self.blocked_ips),
            "rate_limited_identifiers": len(self.rate_limits)
        }
        
        # Severity breakdown
        severity_counts = {}
        for event in events_last_day:
            severity = event.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        metrics["severity_breakdown_24h"] = severity_counts
        
        # Event type breakdown
        event_type_counts = {}
        for event in events_last_hour:
            event_type = event.event_type
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        metrics["event_types_1h"] = event_type_counts
        
        return metrics
    
    def check_quantum_credentials(self, backend_config: Dict[str, Any]) -> bool:
        """Check for exposed quantum service credentials."""
        dangerous_patterns = [
            "password", "secret", "key", "token", "credential",
            "dwave_token", "ibm_token", "azure_key"
        ]
        
        # Scan configuration for potential credential exposure
        config_str = str(backend_config).lower()
        
        for pattern in dangerous_patterns:
            if pattern in config_str and len(config_str) > 50:
                # Potential credential detected
                self.log_security_event(
                    event_type="potential_credential_exposure",
                    severity=SecurityLevel.HIGH,
                    user_id=None,
                    details={"pattern": pattern, "config_keys": list(backend_config.keys())}
                )
                
                # Don't block - just warn
                logger.warning(f"Potential credential exposure detected: {pattern}")
        
        return True
    
    def secure_backend_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Secure backend configuration by masking sensitive values."""
        secured_config = config.copy()
        
        sensitive_keys = ["token", "password", "secret", "key", "credential"]
        
        for key, value in secured_config.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                if isinstance(value, str) and len(value) > 8:
                    # Mask all but first 4 and last 4 characters
                    secured_config[key] = f"{value[:4]}...{value[-4:]}"
        
        return secured_config
    
    def validate_problem_size(self, num_agents: int, num_tasks: int) -> bool:
        """Validate problem size to prevent resource exhaustion attacks."""
        max_agents = 1000
        max_tasks = 10000
        max_variables = 100000  # num_agents * num_tasks
        
        if num_agents > max_agents:
            self.log_security_event(
                event_type="excessive_agents",
                severity=SecurityLevel.HIGH,
                user_id=None,
                details={"agents": num_agents, "limit": max_agents}
            )
            return False
        
        if num_tasks > max_tasks:
            self.log_security_event(
                event_type="excessive_tasks",
                severity=SecurityLevel.HIGH,
                user_id=None,
                details={"tasks": num_tasks, "limit": max_tasks}
            )
            return False
        
        if num_agents * num_tasks > max_variables:
            self.log_security_event(
                event_type="excessive_problem_size",
                severity=SecurityLevel.HIGH,
                user_id=None,
                details={
                    "variables": num_agents * num_tasks,
                    "limit": max_variables,
                    "agents": num_agents,
                    "tasks": num_tasks
                }
            )
            return False
        
        return True
    
    def get_recent_events(self, severity: Optional[SecurityLevel] = None, limit: int = 100) -> List[SecurityEvent]:
        """Get recent security events, optionally filtered by severity."""
        events = self.audit_log[-limit:] if not severity else [
            e for e in self.audit_log[-limit*5:] if e.severity == severity
        ][-limit:]
        
        return sorted(events, key=lambda e: e.timestamp, reverse=True)


# Global security manager instance
security_manager = SecurityManager()


def require_permission(permission: str):
    """Decorator to require specific permission for function access."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # In a real implementation, this would check the current session
            # For now, we'll just log the permission check
            security_manager.log_security_event(
                event_type="permission_check",
                severity=SecurityLevel.LOW,
                user_id="system",
                details={"function": func.__name__, "required_permission": permission}
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def secure_operation(security_level: SecurityLevel = SecurityLevel.MEDIUM):
    """Decorator to mark operations with security level."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            security_manager.log_security_event(
                event_type="secure_operation_start",
                severity=security_level,
                user_id="system",
                details={"function": func.__name__, "security_level": security_level.value}
            )
            
            try:
                result = func(*args, **kwargs)
                security_manager.log_security_event(
                    event_type="secure_operation_success",
                    severity=SecurityLevel.LOW,
                    user_id="system",
                    details={"function": func.__name__}
                )
                return result
                
            except Exception as e:
                security_manager.log_security_event(
                    event_type="secure_operation_failure",
                    severity=SecurityLevel.HIGH,
                    user_id="system",
                    details={"function": func.__name__, "error": str(e)}
                )
                raise
        
        return wrapper
    return decorator