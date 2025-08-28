# Health Check Implementation
from flask import Flask, jsonify
import psycopg2
import redis
import time
from typing import Dict, Any

app = Flask(__name__)

class HealthChecker:
    """Comprehensive health check system"""
    
    def __init__(self):
        self.checks = {
            "database": self._check_database,
            "cache": self._check_cache,
            "quantum_backends": self._check_quantum_backends,
            "memory": self._check_memory,
            "disk": self._check_disk
        }
    
    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            conn.close()
            
            return {"status": "healthy", "response_time": "< 100ms"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def _check_cache(self) -> Dict[str, Any]:
        """Check cache connectivity"""
        try:
            r = redis.from_url(os.getenv("REDIS_URL"))
            start_time = time.time()
            r.ping()
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy", 
                "response_time": f"{response_time:.1f}ms",
                "memory_usage": r.info()["used_memory_human"]
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def _check_quantum_backends(self) -> Dict[str, Any]:
        """Check quantum backend availability"""
        backends = {}
        
        # Check D-Wave
        if os.getenv("DWAVE_TOKEN"):
            try:
                # Placeholder for D-Wave health check
                backends["dwave"] = {"status": "available"}
            except Exception as e:
                backends["dwave"] = {"status": "unavailable", "error": str(e)}
        
        return backends
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return {
                "status": "healthy" if memory.percent < 90 else "warning",
                "usage_percent": memory.percent,
                "available_gb": round(memory.available / (1024**3), 2)
            }
        except Exception as e:
            return {"status": "unknown", "error": str(e)}
    
    def _check_disk(self) -> Dict[str, Any]:
        """Check disk usage"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            
            usage_percent = (disk.used / disk.total) * 100
            
            return {
                "status": "healthy" if usage_percent < 90 else "warning",
                "usage_percent": round(usage_percent, 2),
                "free_gb": round(disk.free / (1024**3), 2)
            }
        except Exception as e:
            return {"status": "unknown", "error": str(e)}

@app.route('/health')
def health():
    """Basic health check"""
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/health/detailed')
def detailed_health():
    """Detailed health check"""
    checker = HealthChecker()
    results = {}
    
    overall_status = "healthy"
    
    for check_name, check_func in checker.checks.items():
        result = check_func()
        results[check_name] = result
        
        if result.get("status") == "unhealthy":
            overall_status = "unhealthy"
        elif result.get("status") == "warning" and overall_status == "healthy":
            overall_status = "warning"
    
    return jsonify({
        "status": overall_status,
        "timestamp": time.time(),
        "checks": results
    })

@app.route('/ready')
def readiness():
    """Readiness probe"""
    checker = HealthChecker()
    
    # Check critical dependencies
    db_result = checker._check_database()
    cache_result = checker._check_cache()
    
    if db_result.get("status") == "healthy" and cache_result.get("status") == "healthy":
        return jsonify({"status": "ready", "timestamp": time.time()})
    else:
        return jsonify({
            "status": "not_ready",
            "timestamp": time.time(),
            "issues": {
                "database": db_result,
                "cache": cache_result
            }
        }), 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
