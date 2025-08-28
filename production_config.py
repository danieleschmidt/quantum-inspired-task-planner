# Quantum Planner Production Configuration
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ProductionConfig:
    """Production configuration settings"""
    
    # Environment
    environment: str = "production"
    debug: bool = False
    testing: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "postgresql://qp_user:password@localhost/quantum_planner")
    database_pool_size: int = 20
    database_max_overflow: int = 30
    
    # Cache
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    cache_ttl: int = 300  # 5 minutes
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # 1 hour
    
    # Quantum backends
    quantum_backends: Dict[str, Any] = None
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance
    max_task_batch_size: int = 1000
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    
    # Monitoring
    metrics_enabled: bool = True
    health_check_endpoint: str = "/health"
    metrics_endpoint: str = "/metrics"
    
    def __post_init__(self):
        """Initialize quantum backends configuration"""
        if self.quantum_backends is None:
            self.quantum_backends = {
                "dwave": {
                    "token": os.getenv("DWAVE_TOKEN"),
                    "solver": "Advantage_system6.1",
                    "enabled": bool(os.getenv("DWAVE_TOKEN"))
                },
                "azure": {
                    "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
                    "resource_group": os.getenv("AZURE_RESOURCE_GROUP"),
                    "workspace": os.getenv("AZURE_QUANTUM_WORKSPACE"),
                    "enabled": bool(os.getenv("AZURE_SUBSCRIPTION_ID"))
                },
                "ibm": {
                    "token": os.getenv("IBM_QUANTUM_TOKEN"),
                    "hub": os.getenv("IBM_QUANTUM_HUB", "ibm-q"),
                    "group": os.getenv("IBM_QUANTUM_GROUP", "open"),
                    "project": os.getenv("IBM_QUANTUM_PROJECT", "main"),
                    "enabled": bool(os.getenv("IBM_QUANTUM_TOKEN"))
                }
            }

# Create production configuration instance
config = ProductionConfig()
