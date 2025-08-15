"""Advanced caching module for quantum task planner performance optimization."""

import hashlib
import pickle
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() > self.timestamp + self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.timestamp


class LRUCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0
        }
    
    def _calculate_size(self, obj: Any) -> int:
        """Calculate approximate size of object in bytes."""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback estimation
            return len(str(obj))
    
    def _evict_expired(self) -> None:
        """Remove expired entries."""
        expired_keys = []
        for key, entry in self.cache.items():
            if entry.is_expired:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update stats."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.stats["size_bytes"] -= entry.size_bytes
            self.stats["evictions"] += 1
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self.cache:
            key = next(iter(self.cache))
            self._remove_entry(key)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            self._evict_expired()
            
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired:
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    entry.access_count += 1
                    self.stats["hits"] += 1
                    return entry.value
                else:
                    self._remove_entry(key)
            
            self.stats["misses"] += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self.lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Ensure we have space
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                access_count=0,
                size_bytes=size_bytes,
                ttl=ttl or self.default_ttl
            )
            
            self.cache[key] = entry
            self.stats["size_bytes"] += size_bytes
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.stats["size_bytes"] = 0
            self.stats["evictions"] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self.stats["evictions"],
                "size_bytes": self.stats["size_bytes"],
                "avg_entry_size": self.stats["size_bytes"] / len(self.cache) if self.cache else 0
            }


class AdaptiveCache:
    """Multi-level adaptive cache with different strategies."""
    
    def __init__(self):
        # L1: Small, fast cache for frequently accessed items
        self.l1_cache = LRUCache(max_size=100, default_ttl=300)  # 5 minutes
        
        # L2: Larger cache for recent items
        self.l2_cache = LRUCache(max_size=1000, default_ttl=1800)  # 30 minutes
        
        # L3: Persistent cache for problem patterns
        self.l3_cache = LRUCache(max_size=5000, default_ttl=3600)  # 1 hour
        
        self.lock = threading.RLock()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        with self.lock:
            # Try L1 first
            value = self.l1_cache.get(key)
            if value is not None:
                return value
            
            # Try L2
            value = self.l2_cache.get(key)
            if value is not None:
                # Promote to L1
                self.l1_cache.put(key, value)
                return value
            
            # Try L3
            value = self.l3_cache.get(key)
            if value is not None:
                # Promote to L2
                self.l2_cache.put(key, value)
                return value
            
            return None
    
    def put(self, key: str, value: Any, priority: str = "normal") -> None:
        """Put value in appropriate cache level based on priority."""
        with self.lock:
            if priority == "high":
                # High priority goes to L1
                self.l1_cache.put(key, value)
            elif priority == "low":
                # Low priority goes to L3
                self.l3_cache.put(key, value)
            else:
                # Normal priority goes to L2
                self.l2_cache.put(key, value)
    
    def clear_all(self) -> None:
        """Clear all cache levels."""
        with self.lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.l3_cache.clear()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get statistics for all cache levels."""
        with self.lock:
            return {
                "l1_cache": self.l1_cache.get_stats(),
                "l2_cache": self.l2_cache.get_stats(),
                "l3_cache": self.l3_cache.get_stats(),
                "total_size": (
                    self.l1_cache.get_stats()["size"] +
                    self.l2_cache.get_stats()["size"] +
                    self.l3_cache.get_stats()["size"]
                ),
                "total_memory": (
                    self.l1_cache.get_stats()["size_bytes"] +
                    self.l2_cache.get_stats()["size_bytes"] +
                    self.l3_cache.get_stats()["size_bytes"]
                )
            }


class IntelligentCacheManager:
    """Intelligent cache manager with automatic optimization."""
    
    def __init__(self):
        self.caches = {
            "solution": AdaptiveCache(),
            "problem_analysis": LRUCache(max_size=500, default_ttl=3600),
            "qubo_matrices": LRUCache(max_size=200, default_ttl=1800),
            "embeddings": LRUCache(max_size=100, default_ttl=7200),
            "performance_metrics": LRUCache(max_size=1000, default_ttl=600)
        }
        
        self.access_patterns = {}
        self.optimization_stats = {
            "cache_optimizations": 0,
            "total_requests": 0,
            "cache_effectiveness": 0.0
        }
    
    def get_cache(self, cache_name: str) -> Optional[Any]:
        """Get cache by name."""
        return self.caches.get(cache_name)
    
    def record_access_pattern(self, cache_name: str, key: str, hit: bool) -> None:
        """Record access pattern for optimization."""
        if cache_name not in self.access_patterns:
            self.access_patterns[cache_name] = {
                "hot_keys": {},
                "cold_keys": set(),
                "total_accesses": 0
            }
        
        pattern = self.access_patterns[cache_name]
        pattern["total_accesses"] += 1
        
        if hit:
            pattern["hot_keys"][key] = pattern["hot_keys"].get(key, 0) + 1
            pattern["cold_keys"].discard(key)
        else:
            pattern["cold_keys"].add(key)
    
    def optimize_caches(self) -> None:
        """Optimize cache configurations based on access patterns."""
        for cache_name, pattern in self.access_patterns.items():
            if pattern["total_accesses"] < 100:
                continue  # Not enough data
            
            cache = self.caches.get(cache_name)
            if not cache:
                continue
            
            # Analyze hot keys
            hot_keys = sorted(
                pattern["hot_keys"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            # Promote hot keys to higher cache levels for adaptive caches
            if isinstance(cache, AdaptiveCache):
                for key, access_count in hot_keys:
                    value = cache.get(key)
                    if value is not None:
                        cache.put(key, value, priority="high")
            
            self.optimization_stats["cache_optimizations"] += 1
        
        # Reset patterns after optimization
        self.access_patterns.clear()
    
    def get_global_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_hits = 0
        total_misses = 0
        total_size = 0
        total_memory = 0
        
        cache_stats = {}
        
        for name, cache in self.caches.items():
            if hasattr(cache, 'get_comprehensive_stats'):
                stats = cache.get_comprehensive_stats()
                cache_stats[name] = stats
                
                # Aggregate L1 stats for adaptive caches
                total_hits += stats["l1_cache"]["hits"]
                total_misses += stats["l1_cache"]["misses"]
                total_size += stats["total_size"]
                total_memory += stats["total_memory"]
                
            elif hasattr(cache, 'get_stats'):
                stats = cache.get_stats()
                cache_stats[name] = stats
                
                total_hits += stats["hits"]
                total_misses += stats["misses"]
                total_size += stats["size"]
                total_memory += stats["size_bytes"]
        
        total_requests = total_hits + total_misses
        global_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_details": cache_stats,
            "global_stats": {
                "total_hits": total_hits,
                "total_misses": total_misses,
                "global_hit_rate": global_hit_rate,
                "total_size": total_size,
                "total_memory_bytes": total_memory,
                "total_memory_mb": total_memory / (1024 * 1024),
                "optimization_stats": self.optimization_stats
            }
        }
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        for cache in self.caches.values():
            if hasattr(cache, 'clear_all'):
                cache.clear_all()
            elif hasattr(cache, 'clear'):
                cache.clear()


# Global cache manager instance
cache_manager = IntelligentCacheManager()


def cached(cache_name: str, ttl: Optional[float] = None, priority: str = "normal"):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            cache = cache_manager.get_cache(cache_name)
            if not cache:
                # No cache available, execute function
                return func(*args, **kwargs)
            
            # Generate cache key
            key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            key = hashlib.sha256(key_data.encode()).hexdigest()[:16]
            
            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                cache_manager.record_access_pattern(cache_name, key, hit=True)
                logger.debug(f"Cache hit for {func.__name__} with key {key}")
                return cached_result
            
            # Cache miss - execute function
            cache_manager.record_access_pattern(cache_name, key, hit=False)
            logger.debug(f"Cache miss for {func.__name__} with key {key}")
            
            result = func(*args, **kwargs)
            
            # Store in cache
            if hasattr(cache, 'put'):
                if isinstance(cache, AdaptiveCache):
                    cache.put(key, result, priority=priority)
                elif isinstance(cache, LRUCache) and ttl:
                    cache.put(key, result, ttl=ttl)
                else:
                    cache.put(key, result)
            
            return result
        
        return wrapper
    return decorator


def preload_cache(cache_name: str, data: Dict[str, Any]) -> None:
    """Preload cache with data."""
    cache = cache_manager.get_cache(cache_name)
    if not cache:
        logger.warning(f"Cache {cache_name} not found for preloading")
        return
    
    for key, value in data.items():
        if hasattr(cache, 'put'):
            cache.put(key, value)
        
    logger.info(f"Preloaded {len(data)} items into cache {cache_name}")


def warm_up_caches() -> None:
    """Warm up caches with common patterns."""
    # This would be called during system startup
    logger.info("Warming up caches...")
    
    # Example: preload common problem sizes and patterns
    common_patterns = {
        "small_problem": {"agents": 5, "tasks": 10, "complexity": "low"},
        "medium_problem": {"agents": 20, "tasks": 50, "complexity": "medium"},
        "large_problem": {"agents": 100, "tasks": 200, "complexity": "high"}
    }
    
    preload_cache("problem_analysis", common_patterns)
    logger.info("Cache warm-up completed")