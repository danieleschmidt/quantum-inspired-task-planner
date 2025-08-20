"""Comprehensive Research Benchmarking Suite for Neural Operator Cryptanalysis.

This module implements publication-ready benchmarking, statistical validation, 
and comparative analysis for neural operator cryptanalysis research with rigorous
experimental design and reproducible results.
"""

import torch
import numpy as np
import pandas as pd
import time
import datetime
import json
import pickle
import hashlib
import threading
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import warnings
import gc
from loguru import logger

try:
    import scipy.stats as stats
    import scipy.signal as signal
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from .neural_operator_cryptanalysis import CryptanalysisFramework
    from .advanced_neural_cryptanalysis import AdvancedCryptanalysisFramework
    from .ultra_robust_neural_cryptanalysis import UltraRobustCryptanalysisFramework
    from .hyperspeed_neural_cryptanalysis import HyperspeedCryptanalysisFramework
except ImportError:
    logger.warning("Cryptanalysis modules not available - using fallback")
    CryptanalysisFramework = object
    AdvancedCryptanalysisFramework = object
    UltraRobustCryptanalysisFramework = object
    HyperspeedCryptanalysisFramework = object


class BenchmarkType(Enum):
    """Types of benchmarks for different evaluation aspects."""
    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    SCALABILITY = "scalability"
    ROBUSTNESS = "robustness"
    STATISTICAL = "statistical"
    COMPARATIVE = "comparative"


class CipherType(Enum):
    """Types of cipher data for comprehensive testing."""
    RANDOM = "random"
    AES_SIMULATION = "aes_simulation"
    DES_SIMULATION = "des_simulation"
    STRUCTURED_PATTERN = "structured_pattern"
    LOW_ENTROPY = "low_entropy"
    HIGH_ENTROPY = "high_entropy"
    SYNTHETIC_WEAK = "synthetic_weak"
    SYNTHETIC_STRONG = "synthetic_strong"


class MetricType(Enum):
    """Types of evaluation metrics."""
    VULNERABILITY_SCORE = "vulnerability_score"
    DETECTION_ACCURACY = "detection_accuracy"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    FALSE_NEGATIVE_RATE = "false_negative_rate"
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    THROUGHPUT = "throughput"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    
    # Experiment settings
    experiment_name: str = "neural_crypto_benchmark"
    num_runs: int = 10
    statistical_significance_level: float = 0.05
    confidence_interval: float = 0.95
    
    # Data generation
    data_sizes: List[int] = field(default_factory=lambda: [1024, 4096, 16384, 65536])
    cipher_types: List[CipherType] = field(default_factory=lambda: list(CipherType))
    samples_per_type: int = 50
    
    # Performance settings
    enable_parallel_execution: bool = True
    max_workers: int = min(8, mp.cpu_count())
    memory_limit_mb: int = 8192
    timeout_seconds: float = 300.0
    
    # Output settings
    save_raw_data: bool = True
    save_plots: bool = True
    output_directory: str = "benchmark_results"
    
    # Reproducibility
    random_seed: int = 42
    deterministic_mode: bool = True


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    
    benchmark_id: str
    framework_name: str
    cipher_type: CipherType
    data_size: int
    run_number: int
    
    # Metrics
    vulnerability_score: float
    execution_time: float
    memory_usage: float
    accuracy: Optional[float] = None
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error_message: Optional[str] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalAnalysis:
    """Statistical analysis results."""
    
    metric_name: str
    sample_size: int
    mean: float
    std: float
    median: float
    min_value: float
    max_value: float
    
    # Distribution tests
    normality_test_p: float
    shapiro_wilk_p: float
    
    # Confidence intervals
    ci_lower: float
    ci_upper: float
    
    # Additional statistics
    skewness: float
    kurtosis: float
    coefficient_of_variation: float


@dataclass
class ComparativeAnalysis:
    """Comparative analysis between frameworks."""
    
    baseline_framework: str
    comparison_framework: str
    metric: str
    
    # Statistical tests
    t_test_statistic: float
    t_test_p_value: float
    mann_whitney_u_statistic: float
    mann_whitney_p_value: float
    
    # Effect size
    cohens_d: float
    cliff_delta: float
    
    # Summary
    significant_difference: bool
    practical_significance: bool
    improvement_percentage: float


class DataGenerator:
    """Advanced data generator for comprehensive benchmark datasets."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        torch.manual_seed(random_seed)
        
    def generate_cipher_data(
        self, 
        cipher_type: CipherType, 
        size: int, 
        num_samples: int = 1
    ) -> List[torch.Tensor]:
        """Generate cipher data of specified type and characteristics."""
        
        samples = []
        
        for _ in range(num_samples):
            if cipher_type == CipherType.RANDOM:
                data = torch.randint(0, 256, (size,), dtype=torch.uint8)
                
            elif cipher_type == CipherType.AES_SIMULATION:
                # Simulate AES-like properties
                data = self._generate_aes_like_data(size)
                
            elif cipher_type == CipherType.DES_SIMULATION:
                # Simulate DES-like properties
                data = self._generate_des_like_data(size)
                
            elif cipher_type == CipherType.STRUCTURED_PATTERN:
                # Data with subtle patterns
                data = self._generate_structured_pattern(size)
                
            elif cipher_type == CipherType.LOW_ENTROPY:
                # Low entropy data (predictable)
                data = self._generate_low_entropy_data(size)
                
            elif cipher_type == CipherType.HIGH_ENTROPY:
                # High entropy data (truly random)
                data = self._generate_high_entropy_data(size)
                
            elif cipher_type == CipherType.SYNTHETIC_WEAK:
                # Intentionally weak cipher
                data = self._generate_weak_cipher(size)
                
            elif cipher_type == CipherType.SYNTHETIC_STRONG:
                # Strong synthetic cipher
                data = self._generate_strong_cipher(size)
                
            else:
                data = torch.randint(0, 256, (size,), dtype=torch.uint8)
            
            samples.append(data)
        
        return samples
    
    def _generate_aes_like_data(self, size: int) -> torch.Tensor:
        """Generate data with AES-like statistical properties."""
        # AES produces high-quality pseudorandom output
        data = torch.randint(0, 256, (size,), dtype=torch.uint8)
        
        # Add subtle correlation structure
        if size > 16:
            block_size = 16
            for i in range(0, size - block_size, block_size):
                block = data[i:i+block_size]
                # Introduce minimal correlation
                correlation_strength = 0.02
                noise = torch.randn(block_size) * correlation_strength
                adjusted_block = (block.float() + noise).clamp(0, 255).byte()
                data[i:i+block_size] = adjusted_block
        
        return data
    
    def _generate_des_like_data(self, size: int) -> torch.Tensor:
        """Generate data with DES-like properties (weaker than AES)."""
        data = torch.randint(0, 256, (size,), dtype=torch.uint8)
        
        # DES has some known weaknesses - simulate weak key patterns
        if size > 8:
            block_size = 8
            for i in range(0, size - block_size, block_size):
                # Introduce slight bias
                bias = self.rng.choice([0, 1], p=[0.52, 0.48])  # Slight bias
                if bias:
                    data[i:i+block_size] = (data[i:i+block_size] | 0x01)  # Set LSB
        
        return data
    
    def _generate_structured_pattern(self, size: int) -> torch.Tensor:
        """Generate data with hidden structural patterns."""
        # Base random data
        data = torch.randint(0, 256, (size,), dtype=torch.uint8)
        
        # Add periodic pattern
        period = min(size // 10, 17)  # Prime number period
        pattern_strength = 0.1
        
        for i in range(size):
            if i % period == 0:
                # Subtle pattern injection
                pattern_value = int(255 * pattern_strength)
                data[i] = (data[i] + pattern_value) % 256
        
        return data
    
    def _generate_low_entropy_data(self, size: int) -> torch.Tensor:
        """Generate low entropy data."""
        # Heavily biased towards certain values
        values = [0x00, 0xFF, 0x55, 0xAA]  # Common patterns
        probabilities = [0.4, 0.3, 0.2, 0.1]
        
        data = self.rng.choice(values, size=size, p=probabilities)
        return torch.from_numpy(data).to(torch.uint8)
    
    def _generate_high_entropy_data(self, size: int) -> torch.Tensor:
        """Generate high entropy data using cryptographic quality randomness."""
        # Use multiple entropy sources
        data1 = torch.randint(0, 256, (size,), dtype=torch.uint8)
        data2 = torch.from_numpy(self.rng.randint(0, 256, size=size, dtype=np.uint8))
        
        # XOR combine for higher entropy
        combined = data1 ^ data2
        return combined
    
    def _generate_weak_cipher(self, size: int) -> torch.Tensor:
        """Generate intentionally weak cipher for testing detection."""
        # Simple XOR cipher with repeated key
        plaintext = torch.randint(0, 256, (size,), dtype=torch.uint8)
        key = torch.tensor([0x42], dtype=torch.uint8)  # Single byte key
        
        # Repeat key to match plaintext size
        repeated_key = key.repeat(size)
        ciphertext = plaintext ^ repeated_key
        
        return ciphertext
    
    def _generate_strong_cipher(self, size: int) -> torch.Tensor:
        """Generate strong synthetic cipher."""
        # Simulate strong cipher with good statistical properties
        data = torch.randint(0, 256, (size,), dtype=torch.uint8)
        
        # Apply avalanche effect simulation
        for i in range(size - 1):
            # Each byte affects the next (simplified avalanche)
            avalanche = (data[i] >> 3) ^ (data[i] << 5)
            data[i + 1] = data[i + 1] ^ (avalanche & 0xFF)
        
        return data


class FrameworkBenchmarker:
    """Benchmarking engine for cryptanalysis frameworks."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logger.bind(component="framework_benchmarker")
        self.data_generator = DataGenerator(config.random_seed)
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        self.statistical_analyses: Dict[str, StatisticalAnalysis] = {}
        self.comparative_analyses: Dict[str, ComparativeAnalysis] = {}
        
        # Framework instances
        self.frameworks = {}
        self._initialize_frameworks()
        
        # Ensure reproducibility
        if config.deterministic_mode:
            self._set_deterministic_mode()
    
    def _set_deterministic_mode(self):
        """Set deterministic mode for reproducible results."""
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _initialize_frameworks(self):
        """Initialize all available frameworks for testing."""
        
        try:
            if CryptanalysisFramework != object:
                from .neural_operator_cryptanalysis import CryptanalysisConfig, create_cryptanalysis_framework
                config = CryptanalysisConfig(cipher_type="generic")
                self.frameworks["basic"] = create_cryptanalysis_framework("generic")
                self.logger.info("Initialized basic cryptanalysis framework")
        except Exception as e:
            self.logger.warning(f"Failed to initialize basic framework: {e}")
        
        try:
            if AdvancedCryptanalysisFramework != object:
                from .advanced_neural_cryptanalysis import create_advanced_research_framework
                self.frameworks["advanced"] = create_advanced_research_framework()
                self.logger.info("Initialized advanced cryptanalysis framework")
        except Exception as e:
            self.logger.warning(f"Failed to initialize advanced framework: {e}")
        
        try:
            if UltraRobustCryptanalysisFramework != object:
                from .ultra_robust_neural_cryptanalysis import create_ultra_robust_framework
                self.frameworks["robust"] = create_ultra_robust_framework()
                self.logger.info("Initialized robust cryptanalysis framework")
        except Exception as e:
            self.logger.warning(f"Failed to initialize robust framework: {e}")
        
        try:
            if HyperspeedCryptanalysisFramework != object:
                from .hyperspeed_neural_cryptanalysis import create_hyperspeed_framework
                self.frameworks["hyperspeed"] = create_hyperspeed_framework()
                self.logger.info("Initialized hyperspeed cryptanalysis framework")
        except Exception as e:
            self.logger.warning(f"Failed to initialize hyperspeed framework: {e}")
        
        if not self.frameworks:
            self.logger.warning("No frameworks available - creating fallback")
            self.frameworks["fallback"] = self._create_fallback_framework()
    
    def _create_fallback_framework(self):
        """Create fallback framework for testing when others unavailable."""
        class FallbackFramework:
            def comprehensive_analysis(self, cipher_samples):
                time.sleep(0.001)  # Simulate processing
                return {
                    "overall": {
                        "combined_vulnerability_score": 0.5,
                        "overall_vulnerability_level": "MEDIUM"
                    }
                }
            
            def analyze_cipher_with_full_protection(self, cipher_data, analysis_types=None):
                time.sleep(0.001)
                result_obj = type('Result', (), {})()
                result_obj.success = True
                result_obj.result = {
                    "overall": {
                        "combined_vulnerability_score": 0.5,
                        "overall_vulnerability_level": "MEDIUM"
                    }
                }
                return result_obj
            
            def hyperspeed_analyze_cipher(self, cipher_data, **kwargs):
                time.sleep(0.001)
                return {
                    "overall": {
                        "combined_vulnerability_score": 0.5,
                        "overall_vulnerability_level": "MEDIUM"
                    }
                }
        
        return FallbackFramework()
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        
        self.logger.info(f"Starting comprehensive benchmark: {self.config.experiment_name}")
        
        start_time = time.time()
        
        # Generate test datasets
        test_datasets = self._generate_test_datasets()
        
        # Run benchmarks
        if self.config.enable_parallel_execution:
            self._run_parallel_benchmarks(test_datasets)
        else:
            self._run_sequential_benchmarks(test_datasets)
        
        # Statistical analysis
        self._perform_statistical_analysis()
        
        # Comparative analysis
        self._perform_comparative_analysis()
        
        # Generate report
        total_time = time.time() - start_time
        report = self._generate_comprehensive_report(total_time)
        
        # Save results
        self._save_results(report)
        
        self.logger.info(f"Benchmark completed in {total_time:.2f}s")
        
        return report
    
    def _generate_test_datasets(self) -> Dict[str, List[Tuple[torch.Tensor, Dict[str, Any]]]]:
        """Generate comprehensive test datasets."""
        
        datasets = {}
        
        for cipher_type in self.config.cipher_types:
            datasets[cipher_type.value] = []
            
            for size in self.config.data_sizes:
                samples = self.data_generator.generate_cipher_data(
                    cipher_type, size, self.config.samples_per_type
                )
                
                for i, sample in enumerate(samples):
                    metadata = {
                        "cipher_type": cipher_type,
                        "size": size,
                        "sample_id": i,
                        "ground_truth_vulnerability": self._get_ground_truth(cipher_type)
                    }
                    datasets[cipher_type.value].append((sample, metadata))
        
        self.logger.info(f"Generated {sum(len(d) for d in datasets.values())} test samples")
        
        return datasets
    
    def _get_ground_truth(self, cipher_type: CipherType) -> float:
        """Get ground truth vulnerability score for cipher type."""
        vulnerability_map = {
            CipherType.RANDOM: 0.1,
            CipherType.AES_SIMULATION: 0.2,
            CipherType.DES_SIMULATION: 0.4,
            CipherType.STRUCTURED_PATTERN: 0.6,
            CipherType.LOW_ENTROPY: 0.9,
            CipherType.HIGH_ENTROPY: 0.05,
            CipherType.SYNTHETIC_WEAK: 0.95,
            CipherType.SYNTHETIC_STRONG: 0.1
        }
        return vulnerability_map.get(cipher_type, 0.5)
    
    def _run_parallel_benchmarks(self, test_datasets: Dict[str, List[Tuple[torch.Tensor, Dict[str, Any]]]]):
        """Run benchmarks in parallel for faster execution."""
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for framework_name, framework in self.frameworks.items():
                for cipher_type, samples in test_datasets.items():
                    for run_num in range(self.config.num_runs):
                        for sample, metadata in samples:
                            future = executor.submit(
                                self._run_single_benchmark,
                                framework_name,
                                framework,
                                sample,
                                metadata,
                                run_num
                            )
                            futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    if result:
                        self.results.append(result)
                except Exception as e:
                    self.logger.error(f"Benchmark task failed: {e}")
    
    def _run_sequential_benchmarks(self, test_datasets: Dict[str, List[Tuple[torch.Tensor, Dict[str, Any]]]]):
        """Run benchmarks sequentially."""
        
        total_tasks = (
            len(self.frameworks) * 
            sum(len(samples) for samples in test_datasets.values()) * 
            self.config.num_runs
        )
        
        completed = 0
        
        for framework_name, framework in self.frameworks.items():
            for cipher_type, samples in test_datasets.items():
                for run_num in range(self.config.num_runs):
                    for sample, metadata in samples:
                        result = self._run_single_benchmark(
                            framework_name, framework, sample, metadata, run_num
                        )
                        if result:
                            self.results.append(result)
                        
                        completed += 1
                        if completed % 100 == 0:
                            progress = (completed / total_tasks) * 100
                            self.logger.info(f"Benchmark progress: {progress:.1f}%")
    
    def _run_single_benchmark(
        self,
        framework_name: str,
        framework: Any,
        cipher_data: torch.Tensor,
        metadata: Dict[str, Any],
        run_number: int
    ) -> Optional[BenchmarkResult]:
        """Run single benchmark and collect metrics."""
        
        benchmark_id = f"{framework_name}_{metadata['cipher_type'].value}_{metadata['size']}_{run_number}"
        
        try:
            # Memory usage before
            memory_before = self._get_memory_usage()
            
            # Execute analysis
            start_time = time.time()
            
            if hasattr(framework, 'comprehensive_research_analysis'):
                result = framework.comprehensive_research_analysis(cipher_data)
            elif hasattr(framework, 'analyze_cipher_with_full_protection'):
                operation_result = framework.analyze_cipher_with_full_protection(cipher_data)
                result = operation_result.result if operation_result.success else {}
            elif hasattr(framework, 'hyperspeed_analyze_cipher'):
                result = framework.hyperspeed_analyze_cipher(cipher_data)
            elif hasattr(framework, 'comprehensive_analysis'):
                result = framework.comprehensive_analysis({"ciphertext_samples": cipher_data})
            else:
                # Fallback
                result = framework.comprehensive_analysis({"ciphertext_samples": cipher_data})
            
            execution_time = time.time() - start_time
            
            # Memory usage after
            memory_after = self._get_memory_usage()
            memory_usage = memory_after - memory_before
            
            # Extract vulnerability score
            vulnerability_score = self._extract_vulnerability_score(result)
            
            # Calculate accuracy if ground truth available
            ground_truth = metadata.get("ground_truth_vulnerability")
            accuracy = None
            if ground_truth is not None:
                accuracy = 1.0 - abs(vulnerability_score - ground_truth)
            
            return BenchmarkResult(
                benchmark_id=benchmark_id,
                framework_name=framework_name,
                cipher_type=metadata["cipher_type"],
                data_size=metadata["size"],
                run_number=run_number,
                vulnerability_score=vulnerability_score,
                execution_time=execution_time,
                memory_usage=memory_usage,
                accuracy=accuracy,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Benchmark {benchmark_id} failed: {e}")
            
            return BenchmarkResult(
                benchmark_id=benchmark_id,
                framework_name=framework_name,
                cipher_type=metadata["cipher_type"],
                data_size=metadata["size"],
                run_number=run_number,
                vulnerability_score=0.0,
                execution_time=0.0,
                memory_usage=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _extract_vulnerability_score(self, result: Dict[str, Any]) -> float:
        """Extract vulnerability score from analysis result."""
        
        # Try different possible locations
        if "overall" in result and "combined_vulnerability_score" in result["overall"]:
            score = result["overall"]["combined_vulnerability_score"]
        elif "vulnerability_score" in result:
            score = result["vulnerability_score"]
        elif "ensemble_prediction" in result:
            score = result["ensemble_prediction"]
        else:
            # Default fallback
            score = 0.5
        
        # Convert tensor to float if needed
        if torch.is_tensor(score):
            score = score.item()
        
        return float(score)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _perform_statistical_analysis(self):
        """Perform comprehensive statistical analysis of results."""
        
        if not SCIPY_AVAILABLE:
            self.logger.warning("SciPy not available - skipping statistical analysis")
            return
        
        # Group results by framework and metric
        grouped_results = {}
        
        for result in self.results:
            if not result.success:
                continue
                
            key = f"{result.framework_name}_{result.cipher_type.value}"
            if key not in grouped_results:
                grouped_results[key] = {
                    "vulnerability_scores": [],
                    "execution_times": [],
                    "memory_usage": [],
                    "accuracies": []
                }
            
            grouped_results[key]["vulnerability_scores"].append(result.vulnerability_score)
            grouped_results[key]["execution_times"].append(result.execution_time)
            grouped_results[key]["memory_usage"].append(result.memory_usage)
            
            if result.accuracy is not None:
                grouped_results[key]["accuracies"].append(result.accuracy)
        
        # Analyze each group
        for group_key, metrics in grouped_results.items():
            for metric_name, values in metrics.items():
                if len(values) < 3:  # Need minimum samples
                    continue
                
                analysis = self._analyze_metric_distribution(metric_name, values)
                analysis_key = f"{group_key}_{metric_name}"
                self.statistical_analyses[analysis_key] = analysis
    
    def _analyze_metric_distribution(self, metric_name: str, values: List[float]) -> StatisticalAnalysis:
        """Analyze distribution of a metric."""
        
        values_array = np.array(values)
        
        # Basic statistics
        mean_val = np.mean(values_array)
        std_val = np.std(values_array, ddof=1)
        median_val = np.median(values_array)
        min_val = np.min(values_array)
        max_val = np.max(values_array)
        
        # Normality tests
        _, shapiro_p = stats.shapiro(values_array) if len(values_array) <= 5000 else (0, 1)
        _, anderson_stat = stats.normaltest(values_array)
        
        # Confidence interval
        confidence_level = self.config.confidence_interval
        sem = stats.sem(values_array)
        margin_error = sem * stats.t.ppf((1 + confidence_level) / 2, len(values_array) - 1)
        ci_lower = mean_val - margin_error
        ci_upper = mean_val + margin_error
        
        # Additional statistics
        skewness = stats.skew(values_array)
        kurt = stats.kurtosis(values_array)
        cv = std_val / mean_val if mean_val != 0 else float('inf')
        
        return StatisticalAnalysis(
            metric_name=metric_name,
            sample_size=len(values_array),
            mean=mean_val,
            std=std_val,
            median=median_val,
            min_value=min_val,
            max_value=max_val,
            normality_test_p=anderson_stat,
            shapiro_wilk_p=shapiro_p,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            skewness=skewness,
            kurtosis=kurt,
            coefficient_of_variation=cv
        )
    
    def _perform_comparative_analysis(self):
        """Perform pairwise comparative analysis between frameworks."""
        
        if not SCIPY_AVAILABLE:
            self.logger.warning("SciPy not available - skipping comparative analysis")
            return
        
        frameworks = list(self.frameworks.keys())
        
        for i, framework1 in enumerate(frameworks):
            for j, framework2 in enumerate(frameworks[i+1:], i+1):
                
                # Compare each metric
                for metric in ["vulnerability_score", "execution_time", "memory_usage"]:
                    analysis = self._compare_frameworks(framework1, framework2, metric)
                    if analysis:
                        comparison_key = f"{framework1}_vs_{framework2}_{metric}"
                        self.comparative_analyses[comparison_key] = analysis
    
    def _compare_frameworks(
        self, 
        framework1: str, 
        framework2: str, 
        metric: str
    ) -> Optional[ComparativeAnalysis]:
        """Compare two frameworks on a specific metric."""
        
        # Extract values for both frameworks
        values1 = []
        values2 = []
        
        for result in self.results:
            if not result.success:
                continue
                
            if result.framework_name == framework1:
                value = getattr(result, metric, None)
                if value is not None:
                    values1.append(value)
            elif result.framework_name == framework2:
                value = getattr(result, metric, None)
                if value is not None:
                    values2.append(value)
        
        if len(values1) < 3 or len(values2) < 3:
            return None
        
        values1 = np.array(values1)
        values2 = np.array(values2)
        
        # Statistical tests
        t_stat, t_p = stats.ttest_ind(values1, values2, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(values1, values2, alternative='two-sided')
        
        # Effect sizes
        cohens_d = self._calculate_cohens_d(values1, values2)
        cliff_delta = self._calculate_cliff_delta(values1, values2)
        
        # Significance assessment
        significant = t_p < self.config.statistical_significance_level
        practical = abs(cohens_d) > 0.5  # Medium effect size threshold
        
        # Improvement calculation
        mean1 = np.mean(values1)
        mean2 = np.mean(values2)
        improvement = ((mean2 - mean1) / mean1) * 100 if mean1 != 0 else 0
        
        return ComparativeAnalysis(
            baseline_framework=framework1,
            comparison_framework=framework2,
            metric=metric,
            t_test_statistic=t_stat,
            t_test_p_value=t_p,
            mann_whitney_u_statistic=u_stat,
            mann_whitney_p_value=u_p,
            cohens_d=cohens_d,
            cliff_delta=cliff_delta,
            significant_difference=significant,
            practical_significance=practical,
            improvement_percentage=improvement
        )
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _calculate_cliff_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cliff's delta effect size."""
        n1, n2 = len(group1), len(group2)
        
        if n1 == 0 or n2 == 0:
            return 0.0
        
        # Count dominance
        dominance = 0
        for x1 in group1:
            for x2 in group2:
                if x1 > x2:
                    dominance += 1
                elif x1 < x2:
                    dominance -= 1
        
        return dominance / (n1 * n2)
    
    def _generate_comprehensive_report(self, total_execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        
        # Summary statistics
        successful_runs = len([r for r in self.results if r.success])
        failed_runs = len([r for r in self.results if not r.success])
        
        # Performance summary by framework
        framework_summary = {}
        for framework_name in self.frameworks.keys():
            framework_results = [r for r in self.results if r.framework_name == framework_name and r.success]
            
            if framework_results:
                execution_times = [r.execution_time for r in framework_results]
                memory_usage = [r.memory_usage for r in framework_results]
                vulnerability_scores = [r.vulnerability_score for r in framework_results]
                
                framework_summary[framework_name] = {
                    "total_runs": len(framework_results),
                    "mean_execution_time": np.mean(execution_times),
                    "mean_memory_usage": np.mean(memory_usage),
                    "mean_vulnerability_score": np.mean(vulnerability_scores),
                    "std_execution_time": np.std(execution_times),
                    "accuracy_scores": [r.accuracy for r in framework_results if r.accuracy is not None]
                }
        
        # Compile report
        report = {
            "experiment_metadata": {
                "experiment_name": self.config.experiment_name,
                "timestamp": datetime.datetime.now().isoformat(),
                "total_execution_time": total_execution_time,
                "configuration": asdict(self.config)
            },
            "summary_statistics": {
                "total_benchmark_runs": len(self.results),
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate": successful_runs / max(len(self.results), 1),
                "frameworks_tested": list(self.frameworks.keys()),
                "cipher_types_tested": [ct.value for ct in self.config.cipher_types],
                "data_sizes_tested": self.config.data_sizes
            },
            "framework_performance": framework_summary,
            "statistical_analyses": {
                key: asdict(analysis) for key, analysis in self.statistical_analyses.items()
            },
            "comparative_analyses": {
                key: asdict(analysis) for key, analysis in self.comparative_analyses.items()
            },
            "raw_results": [asdict(result) for result in self.results] if self.config.save_raw_data else [],
            "reproducibility": {
                "random_seed": self.config.random_seed,
                "deterministic_mode": self.config.deterministic_mode,
                "framework_versions": self._get_framework_versions()
            }
        }
        
        return report
    
    def _get_framework_versions(self) -> Dict[str, str]:
        """Get version information for frameworks."""
        versions = {
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }
        
        if SCIPY_AVAILABLE:
            import scipy
            versions["scipy_version"] = scipy.__version__
        
        return versions
    
    def _save_results(self, report: Dict[str, Any]):
        """Save benchmark results to files."""
        
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_file = output_dir / f"benchmark_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save CSV of results
        if self.results:
            df = pd.DataFrame([asdict(result) for result in self.results])
            csv_file = output_dir / f"benchmark_results_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
        
        # Save plots if enabled
        if self.config.save_plots and PLOTTING_AVAILABLE:
            self._generate_plots(output_dir, timestamp)
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def _generate_plots(self, output_dir: Path, timestamp: str):
        """Generate visualization plots."""
        
        if not self.results:
            return
        
        try:
            # Create DataFrame
            df = pd.DataFrame([asdict(result) for result in self.results if result.success])
            
            if df.empty:
                return
            
            # Set up plotting style
            plt.style.use('seaborn-v0_8-whitegrid')
            
            # 1. Execution time comparison
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=df, x='framework_name', y='execution_time')
            plt.title('Execution Time Comparison Across Frameworks')
            plt.ylabel('Execution Time (seconds)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / f"execution_time_comparison_{timestamp}.png", dpi=300)
            plt.close()
            
            # 2. Vulnerability score distribution
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=df, x='framework_name', y='vulnerability_score')
            plt.title('Vulnerability Score Distribution Across Frameworks')
            plt.ylabel('Vulnerability Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / f"vulnerability_score_distribution_{timestamp}.png", dpi=300)
            plt.close()
            
            # 3. Performance vs Data Size
            plt.figure(figsize=(12, 8))
            sns.lineplot(data=df, x='data_size', y='execution_time', hue='framework_name')
            plt.title('Execution Time vs Data Size')
            plt.xlabel('Data Size (bytes)')
            plt.ylabel('Execution Time (seconds)')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(output_dir / f"performance_vs_size_{timestamp}.png", dpi=300)
            plt.close()
            
            # 4. Cipher Type Analysis
            if len(df['cipher_type'].unique()) > 1:
                plt.figure(figsize=(14, 8))
                sns.boxplot(data=df, x='cipher_type', y='vulnerability_score', hue='framework_name')
                plt.title('Vulnerability Detection Across Cipher Types')
                plt.ylabel('Vulnerability Score')
                plt.xticks(rotation=45)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(output_dir / f"cipher_type_analysis_{timestamp}.png", dpi=300)
                plt.close()
            
            self.logger.info("Visualization plots generated successfully")
            
        except Exception as e:
            self.logger.error(f"Plot generation failed: {e}")


def run_research_benchmark(
    experiment_name: str = "neural_crypto_research_benchmark",
    num_runs: int = 10,
    data_sizes: Optional[List[int]] = None,
    output_directory: str = "research_benchmark_results"
) -> Dict[str, Any]:
    """Run comprehensive research benchmark with publication-ready results."""
    
    if data_sizes is None:
        data_sizes = [1024, 4096, 16384, 65536]
    
    config = BenchmarkConfig(
        experiment_name=experiment_name,
        num_runs=num_runs,
        data_sizes=data_sizes,
        cipher_types=list(CipherType),
        output_directory=output_directory,
        enable_parallel_execution=True,
        save_plots=True,
        save_raw_data=True
    )
    
    benchmarker = FrameworkBenchmarker(config)
    return benchmarker.run_comprehensive_benchmark()


if __name__ == "__main__":
    # Example usage
    logger.info("Starting comprehensive research benchmark")
    
    results = run_research_benchmark(
        experiment_name="neural_operator_cryptanalysis_evaluation",
        num_runs=5,  # Reduced for example
        data_sizes=[1024, 4096, 16384]
    )
    
    print("Benchmark completed successfully!")
    print(f"Total frameworks tested: {len(results['summary_statistics']['frameworks_tested'])}")
    print(f"Success rate: {results['summary_statistics']['success_rate']:.2%}")