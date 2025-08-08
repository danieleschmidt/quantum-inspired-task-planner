"""Performance Optimization Module for Cryptanalysis.

Provides GPU acceleration, distributed computing, advanced caching,
and performance monitoring for large-scale cryptanalysis operations.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import time
import psutil
import gc
from pathlib import Path
import pickle
import hashlib
from loguru import logger
import threading
from contextlib import contextmanager
import warnings

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import numba
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    numba = None
    jit = lambda func: func  # No-op decorator
    cuda = None


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    
    # GPU settings
    enable_gpu: bool = True
    gpu_device_ids: List[int] = field(default_factory=list)
    gpu_memory_fraction: float = 0.8
    enable_mixed_precision: bool = True
    
    # Distributed computing
    enable_distributed: bool = False
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    
    # Parallel processing
    max_workers: int = 4
    chunk_size: int = 1000
    use_process_pool: bool = False
    
    # Memory optimization
    enable_memory_mapping: bool = True
    max_memory_usage: float = 0.8  # 80% of available memory
    enable_gradient_checkpointing: bool = True
    
    # Caching
    enable_result_caching: bool = True
    enable_computation_caching: bool = True
    cache_compression: bool = True
    max_cache_size_gb: float = 2.0
    
    # JIT compilation
    enable_jit_compilation: bool = True
    enable_torch_jit: bool = True
    
    # Profiling
    enable_profiling: bool = False
    profile_memory: bool = False
    profile_gpu: bool = False


class GPUAccelerator:
    """Handles GPU acceleration for cryptanalysis operations."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logger.bind(component="gpu_accelerator")
        self.device = self._initialize_gpu()
        self.memory_pool = None
        
        if CUPY_AVAILABLE and self.device.type == 'cuda':
            self._initialize_cupy()
    
    def _initialize_gpu(self) -> torch.device:
        """Initialize GPU device."""
        if not self.config.enable_gpu or not torch.cuda.is_available():
            self.logger.info("Using CPU device")
            return torch.device('cpu')
        
        if self.config.gpu_device_ids:
            device_id = self.config.gpu_device_ids[0]
        else:
            device_id = 0
        
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device)
        
        # Set memory fraction
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
        
        self.logger.info(f"Initialized GPU device: {device}")
        self.logger.info(f"GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        
        return device
    
    def _initialize_cupy(self):
        """Initialize CuPy for advanced GPU operations."""
        try:
            self.memory_pool = cp.get_default_memory_pool()
            self.logger.info("CuPy memory pool initialized")
        except Exception as e:
            self.logger.warning(f"CuPy initialization failed: {e}")
    
    def optimize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor for GPU processing."""
        if self.device.type == 'cpu':
            return tensor
        
        # Move to GPU
        tensor = tensor.to(self.device, non_blocking=True)
        
        # Enable mixed precision if configured
        if self.config.enable_mixed_precision and tensor.dtype == torch.float32:
            tensor = tensor.half()
        
        return tensor
    
    def batch_process_tensors(
        self,
        tensors: List[torch.Tensor],
        operation: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int = 32
    ) -> List[torch.Tensor]:
        """Process tensors in batches for optimal GPU utilization."""
        results = []
        
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i + batch_size]
            
            # Stack batch for parallel processing
            if batch:
                try:
                    batch_tensor = torch.stack([self.optimize_tensor(t) for t in batch])
                    batch_result = operation(batch_tensor)
                    
                    # Unstack results
                    for j in range(batch_result.size(0)):
                        results.append(batch_result[j])
                        
                except Exception as e:
                    self.logger.warning(f"Batch processing failed, falling back to individual: {e}")
                    # Fallback to individual processing
                    for tensor in batch:
                        results.append(operation(self.optimize_tensor(tensor)))
        
        return results
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
            if self.memory_pool:
                self.memory_pool.free_all_blocks()
                
            self.logger.info("GPU cache cleared")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get GPU memory statistics."""
        if self.device.type == 'cpu':
            return {"device": "cpu"}
        
        return {
            "device": str(self.device),
            "allocated_gb": torch.cuda.memory_allocated(self.device) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated(self.device) / 1e9,
            "total_gb": torch.cuda.get_device_properties(self.device).total_memory / 1e9
        }


class DistributedManager:
    """Manages distributed computing for large-scale cryptanalysis."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logger.bind(component="distributed_manager")
        self.is_initialized = False
        
        if config.enable_distributed:
            self._initialize_distributed()
    
    def _initialize_distributed(self):
        """Initialize distributed processing."""
        try:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.backend,
                    world_size=self.config.world_size,
                    rank=self.config.rank
                )
            
            self.is_initialized = True
            self.logger.info(f"Distributed initialized: rank {self.config.rank}/{self.config.world_size}")
            
        except Exception as e:
            self.logger.error(f"Distributed initialization failed: {e}")
            self.config.enable_distributed = False
    
    def distribute_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for distributed training."""
        if not self.is_initialized:
            return model
        
        return DistributedDataParallel(model)
    
    def all_gather_results(self, local_result: torch.Tensor) -> List[torch.Tensor]:
        """Gather results from all processes."""
        if not self.is_initialized:
            return [local_result]
        
        gathered = [torch.zeros_like(local_result) for _ in range(self.config.world_size)]
        dist.all_gather(gathered, local_result)
        return gathered
    
    def reduce_results(self, local_result: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """Reduce results across all processes."""
        if not self.is_initialized:
            return local_result
        
        dist.all_reduce(local_result, op=op)
        return local_result
    
    def cleanup(self):
        """Cleanup distributed resources."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
            self.logger.info("Distributed cleanup completed")


class AdvancedCache:
    """Advanced caching system with compression and intelligent eviction."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logger.bind(component="advanced_cache")
        
        self.cache_dir = Path("/tmp/cryptanalysis_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.memory_cache = {}
        self.disk_cache_index = {}
        self.access_times = {}
        self.cache_sizes = {}
        self.lock = threading.RLock()
        
        self.max_memory_cache_size = int(self.config.max_cache_size_gb * 0.3 * 1e9)  # 30% in memory
        self.max_disk_cache_size = int(self.config.max_cache_size_gb * 0.7 * 1e9)    # 70% on disk
        
        self._cleanup_old_cache()
    
    def _cleanup_old_cache(self):
        """Clean up old cache files."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                if time.time() - cache_file.stat().st_mtime > 86400:  # 24 hours
                    cache_file.unlink()
        except Exception as e:
            self.logger.warning(f"Cache cleanup warning: {e}")
    
    def _generate_key(self, operation: str, **kwargs) -> str:
        """Generate cache key from operation and parameters."""
        key_data = f"{operation}_{sorted(kwargs.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage."""
        if not self.config.cache_compression:
            return pickle.dumps(data)
        
        try:
            import lz4.frame
            return lz4.frame.compress(pickle.dumps(data))
        except ImportError:
            try:
                import gzip
                return gzip.compress(pickle.dumps(data))
            except Exception:
                return pickle.dumps(data)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data from storage."""
        if not self.config.cache_compression:
            return pickle.loads(compressed_data)
        
        try:
            import lz4.frame
            return pickle.loads(lz4.frame.decompress(compressed_data))
        except ImportError:
            try:
                import gzip
                return pickle.loads(gzip.decompress(compressed_data))
            except Exception:
                return pickle.loads(compressed_data)
    
    def get(self, operation: str, **kwargs) -> Optional[Any]:
        """Get cached result."""
        if not self.config.enable_result_caching:
            return None
        
        key = self._generate_key(operation, **kwargs)
        
        with self.lock:
            # Check memory cache first
            if key in self.memory_cache:
                self.access_times[key] = time.time()
                return self.memory_cache[key]
            
            # Check disk cache
            if key in self.disk_cache_index:
                try:
                    cache_file = self.cache_dir / f"{key}.cache"
                    if cache_file.exists():
                        with open(cache_file, 'rb') as f:
                            data = self._decompress_data(f.read())
                        
                        # Move to memory cache if small enough
                        data_size = len(pickle.dumps(data))
                        if data_size < self.max_memory_cache_size // 100:  # If less than 1% of memory cache
                            self._evict_memory_cache_if_needed(data_size)
                            self.memory_cache[key] = data
                            self.cache_sizes[key] = data_size
                        
                        self.access_times[key] = time.time()
                        return data
                        
                except Exception as e:
                    self.logger.warning(f"Disk cache read failed for {key}: {e}")
                    if key in self.disk_cache_index:
                        del self.disk_cache_index[key]
        
        return None
    
    def put(self, operation: str, result: Any, **kwargs):
        """Cache result."""
        if not self.config.enable_result_caching:
            return
        
        key = self._generate_key(operation, **kwargs)
        
        try:
            data_size = len(pickle.dumps(result))
            
            with self.lock:
                # Decide whether to cache in memory or disk
                if data_size < self.max_memory_cache_size // 10:  # If less than 10% of memory cache
                    self._evict_memory_cache_if_needed(data_size)
                    self.memory_cache[key] = result
                    self.cache_sizes[key] = data_size
                else:
                    # Cache to disk
                    self._evict_disk_cache_if_needed(data_size)
                    cache_file = self.cache_dir / f"{key}.cache"
                    
                    with open(cache_file, 'wb') as f:
                        f.write(self._compress_data(result))
                    
                    self.disk_cache_index[key] = cache_file
                    self.cache_sizes[key] = data_size
                
                self.access_times[key] = time.time()
                
        except Exception as e:
            self.logger.warning(f"Cache put failed for {key}: {e}")
    
    def _evict_memory_cache_if_needed(self, new_size: int):
        """Evict items from memory cache if needed."""
        current_size = sum(self.cache_sizes.get(k, 0) for k in self.memory_cache.keys())
        
        while current_size + new_size > self.max_memory_cache_size and self.memory_cache:
            # Evict least recently used
            lru_key = min(self.memory_cache.keys(), 
                         key=lambda k: self.access_times.get(k, 0))
            
            current_size -= self.cache_sizes.get(lru_key, 0)
            del self.memory_cache[lru_key]
            if lru_key in self.cache_sizes:
                del self.cache_sizes[lru_key]
    
    def _evict_disk_cache_if_needed(self, new_size: int):
        """Evict items from disk cache if needed."""
        current_size = sum(self.cache_sizes.get(k, 0) for k in self.disk_cache_index.keys())
        
        while current_size + new_size > self.max_disk_cache_size and self.disk_cache_index:
            # Evict least recently used
            lru_key = min(self.disk_cache_index.keys(), 
                         key=lambda k: self.access_times.get(k, 0))
            
            cache_file = self.disk_cache_index[lru_key]
            try:
                cache_file.unlink(missing_ok=True)
            except Exception:
                pass
            
            current_size -= self.cache_sizes.get(lru_key, 0)
            del self.disk_cache_index[lru_key]
            if lru_key in self.cache_sizes:
                del self.cache_sizes[lru_key]
    
    def clear(self):
        """Clear all caches."""
        with self.lock:
            self.memory_cache.clear()
            
            for cache_file in self.disk_cache_index.values():
                try:
                    cache_file.unlink(missing_ok=True)
                except Exception:
                    pass
            
            self.disk_cache_index.clear()
            self.cache_sizes.clear()
            self.access_times.clear()
        
        self.logger.info("All caches cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            memory_size = sum(self.cache_sizes.get(k, 0) for k in self.memory_cache.keys())
            disk_size = sum(self.cache_sizes.get(k, 0) for k in self.disk_cache_index.keys())
            
            return {
                "memory_cache_items": len(self.memory_cache),
                "disk_cache_items": len(self.disk_cache_index),
                "memory_size_mb": memory_size / 1e6,
                "disk_size_mb": disk_size / 1e6,
                "total_size_mb": (memory_size + disk_size) / 1e6,
                "max_size_gb": self.config.max_cache_size_gb
            }


class JITCompiler:
    """JIT compilation for performance-critical functions."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logger.bind(component="jit_compiler")
        self.compiled_functions = {}
    
    def compile_function(self, func: Callable, use_gpu: bool = False) -> Callable:
        """Compile function with appropriate JIT compiler."""
        if not self.config.enable_jit_compilation:
            return func
        
        func_key = f"{func.__name__}_{use_gpu}"
        
        if func_key in self.compiled_functions:
            return self.compiled_functions[func_key]
        
        try:
            if use_gpu and NUMBA_AVAILABLE and cuda:
                # CUDA JIT compilation
                compiled_func = cuda.jit(func)
                self.logger.info(f"CUDA JIT compiled: {func.__name__}")
            elif NUMBA_AVAILABLE:
                # CPU JIT compilation
                compiled_func = jit(nopython=True)(func)
                self.logger.info(f"Numba JIT compiled: {func.__name__}")
            else:
                # No JIT available
                compiled_func = func
                self.logger.warning(f"No JIT compiler available for: {func.__name__}")
            
            self.compiled_functions[func_key] = compiled_func
            return compiled_func
            
        except Exception as e:
            self.logger.warning(f"JIT compilation failed for {func.__name__}: {e}")
            return func
    
    def torch_jit_compile(self, model: nn.Module) -> nn.Module:
        """Compile PyTorch model with TorchScript."""
        if not self.config.enable_torch_jit:
            return model
        
        try:
            # Trace the model
            model.eval()
            compiled_model = torch.jit.script(model)
            self.logger.info(f"TorchScript compiled model: {type(model).__name__}")
            return compiled_model
            
        except Exception as e:
            self.logger.warning(f"TorchScript compilation failed: {e}")
            return model


class PerformanceProfiler:
    """Advanced performance profiling and monitoring."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logger.bind(component="performance_profiler")
        self.profiles = {}
        self.active_profiles = {}
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Profile an operation."""
        if not self.config.enable_profiling:
            yield
            return
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        if self.config.profile_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_gpu_memory = torch.cuda.memory_allocated()
        else:
            start_gpu_memory = 0
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            if self.config.profile_gpu and torch.cuda.is_available():
                torch.cuda.synchronize()
                end_gpu_memory = torch.cuda.memory_allocated()
            else:
                end_gpu_memory = start_gpu_memory
            
            profile_data = {
                "execution_time": end_time - start_time,
                "memory_delta_mb": (end_memory - start_memory) / 1e6,
                "gpu_memory_delta_mb": (end_gpu_memory - start_gpu_memory) / 1e6,
                "timestamp": start_time
            }
            
            if operation_name not in self.profiles:
                self.profiles[operation_name] = []
            
            self.profiles[operation_name].append(profile_data)
            
            # Keep only recent profiles
            if len(self.profiles[operation_name]) > 1000:
                self.profiles[operation_name] = self.profiles[operation_name][-500:]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except Exception:
            return 0.0
    
    def get_profile_summary(self, operation_name: str) -> Dict[str, Any]:
        """Get profile summary for operation."""
        if operation_name not in self.profiles:
            return {"error": "No profile data available"}
        
        profiles = self.profiles[operation_name]
        
        if not profiles:
            return {"error": "No profile data available"}
        
        execution_times = [p["execution_time"] for p in profiles]
        memory_deltas = [p["memory_delta_mb"] for p in profiles]
        gpu_memory_deltas = [p["gpu_memory_delta_mb"] for p in profiles]
        
        return {
            "operation": operation_name,
            "sample_count": len(profiles),
            "execution_time": {
                "mean": np.mean(execution_times),
                "median": np.median(execution_times),
                "std": np.std(execution_times),
                "min": np.min(execution_times),
                "max": np.max(execution_times)
            },
            "memory_usage": {
                "mean_delta_mb": np.mean(memory_deltas),
                "max_delta_mb": np.max(memory_deltas)
            },
            "gpu_memory_usage": {
                "mean_delta_mb": np.mean(gpu_memory_deltas),
                "max_delta_mb": np.max(gpu_memory_deltas)
            }
        }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logger.bind(component="performance_optimizer")
        
        # Initialize components
        self.gpu_accelerator = GPUAccelerator(config)
        self.distributed_manager = DistributedManager(config)
        self.cache = AdvancedCache(config)
        self.jit_compiler = JITCompiler(config)
        self.profiler = PerformanceProfiler(config)
        
        # Performance monitoring
        self.performance_stats = {
            "operations_count": 0,
            "total_execution_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "gpu_operations": 0
        }
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply all available optimizations to a model."""
        # Move to GPU
        if self.gpu_accelerator.device.type == 'cuda':
            model = model.to(self.gpu_accelerator.device)
        
        # Distributed wrapper
        model = self.distributed_manager.distribute_model(model)
        
        # JIT compilation
        model = self.jit_compiler.torch_jit_compile(model)
        
        # Mixed precision
        if self.config.enable_mixed_precision and self.gpu_accelerator.device.type == 'cuda':
            model = model.half()
        
        return model
    
    def execute_optimized_operation(
        self,
        operation: Callable,
        operation_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with all optimizations applied."""
        
        with self.profiler.profile_operation(operation_name):
            # Check cache first
            cache_key = f"{operation_name}_{hash(str(args))}_{hash(str(sorted(kwargs.items())))}"
            cached_result = self.cache.get(operation_name, cache_key=cache_key)
            
            if cached_result is not None:
                self.performance_stats["cache_hits"] += 1
                return cached_result
            
            self.performance_stats["cache_misses"] += 1
            
            # Optimize inputs
            optimized_args = []
            for arg in args:
                if torch.is_tensor(arg):
                    optimized_args.append(self.gpu_accelerator.optimize_tensor(arg))
                else:
                    optimized_args.append(arg)
            
            optimized_kwargs = {}
            for key, value in kwargs.items():
                if torch.is_tensor(value):
                    optimized_kwargs[key] = self.gpu_accelerator.optimize_tensor(value)
                else:
                    optimized_kwargs[key] = value
            
            # Execute operation
            start_time = time.time()
            
            try:
                result = operation(*optimized_args, **optimized_kwargs)
                
                # Cache result
                self.cache.put(operation_name, result, cache_key=cache_key)
                
                # Update stats
                execution_time = time.time() - start_time
                self.performance_stats["operations_count"] += 1
                self.performance_stats["total_execution_time"] += execution_time
                
                if self.gpu_accelerator.device.type == 'cuda':
                    self.performance_stats["gpu_operations"] += 1
                
                return result
                
            except Exception as e:
                self.logger.error(f"Optimized operation {operation_name} failed: {e}")
                raise
    
    def parallel_batch_process(
        self,
        data_batches: List[Any],
        process_function: Callable,
        max_workers: Optional[int] = None
    ) -> List[Any]:
        """Process batches in parallel with optimal resource utilization."""
        
        if max_workers is None:
            max_workers = self.config.max_workers
        
        if len(data_batches) <= 1 or max_workers <= 1:
            # Sequential processing for small datasets
            return [process_function(batch) for batch in data_batches]
        
        # Choose executor type based on configuration
        executor_class = ProcessPoolExecutor if self.config.use_process_pool else ThreadPoolExecutor
        
        results = [None] * len(data_batches)
        
        with executor_class(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_function, batch): i
                for i, batch in enumerate(data_batches)
            }
            
            # Collect results in order
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    self.logger.error(f"Batch {index} processing failed: {e}")
                    results[index] = None
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": time.time(),
            "config": {
                "gpu_enabled": self.config.enable_gpu,
                "distributed_enabled": self.config.enable_distributed,
                "caching_enabled": self.config.enable_result_caching,
                "jit_enabled": self.config.enable_jit_compilation
            },
            "runtime_stats": self.performance_stats.copy(),
            "gpu_stats": self.gpu_accelerator.get_memory_stats(),
            "cache_stats": self.cache.get_stats()
        }
        
        # Add computed metrics
        if self.performance_stats["operations_count"] > 0:
            report["computed_metrics"] = {
                "avg_execution_time": (
                    self.performance_stats["total_execution_time"] / 
                    self.performance_stats["operations_count"]
                ),
                "cache_hit_ratio": (
                    self.performance_stats["cache_hits"] / 
                    (self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"])
                ),
                "gpu_operation_ratio": (
                    self.performance_stats["gpu_operations"] / 
                    self.performance_stats["operations_count"]
                )
            }
        
        return report
    
    def cleanup(self):
        """Cleanup performance optimization resources."""
        self.logger.info("Cleaning up performance optimizer")
        
        self.gpu_accelerator.clear_cache()
        self.cache.clear()
        self.distributed_manager.cleanup()
        
        # Clear stats
        self.performance_stats.clear()
        
        self.logger.info("Performance optimizer cleanup completed")


def create_performance_optimizer(
    enable_gpu: bool = True,
    enable_distributed: bool = False,
    enable_caching: bool = True,
    max_workers: int = 4,
    **kwargs
) -> PerformanceOptimizer:
    """Create performance optimizer with specified configuration."""
    
    config = PerformanceConfig(
        enable_gpu=enable_gpu,
        enable_distributed=enable_distributed,
        enable_result_caching=enable_caching,
        max_workers=max_workers,
        **kwargs
    )
    
    return PerformanceOptimizer(config)


# Utility functions for common optimizations
def optimize_tensor_operations():
    """Apply global PyTorch optimizations."""
    # Enable memory efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except AttributeError:
        pass
    
    # Enable cuDNN benchmarking
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Set number of threads for CPU operations
    torch.set_num_threads(min(8, mp.cpu_count()))
    
    logger.info("Global tensor optimizations applied")


def profile_memory_usage(func):
    """Decorator to profile memory usage of a function."""
    def wrapper(*args, **kwargs):
        import tracemalloc
        
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            logger.info(
                f"Memory usage for {func.__name__}: "
                f"Current: {current_memory / 1e6:.1f}MB, "
                f"Peak: {peak_memory / 1e6:.1f}MB, "
                f"Delta: {(current_memory - start_memory) / 1e6:.1f}MB"
            )
    
    return wrapper
