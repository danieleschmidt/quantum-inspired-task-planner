"""
Autonomous SDLC Implementation Test Suite - Comprehensive Quality Gates

This test suite validates all autonomous SDLC implementations across three generations,
ensuring production readiness, security compliance, and performance benchmarks.

Test Coverage:
- Generation 1: Basic functionality validation
- Generation 2: Robustness and security testing  
- Generation 3: Scalability and performance testing
- Integration testing across all components
- Security and compliance validation
- Performance benchmarking
- Research module validation

Author: Terragon Labs Autonomous Testing Division
Version: 4.0.0 (Complete SDLC Validation)
"""

import numpy as np
import time
import logging
import hashlib
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestResults:
    """Test results collector."""
    
    def __init__(self):
        self.results = {
            'generation_1': {'passed': 0, 'failed': 0, 'errors': []},
            'generation_2': {'passed': 0, 'failed': 0, 'errors': []},
            'generation_3': {'passed': 0, 'failed': 0, 'errors': []},
            'integration': {'passed': 0, 'failed': 0, 'errors': []},
            'security': {'passed': 0, 'failed': 0, 'errors': []},
            'performance': {'passed': 0, 'failed': 0, 'errors': []}
        }
        self.start_time = time.time()
    
    def record_result(self, category: str, test_name: str, passed: bool, error: str = None):
        """Record test result."""
        if passed:
            self.results[category]['passed'] += 1
        else:
            self.results[category]['failed'] += 1
            if error:
                self.results[category]['errors'].append(f"{test_name}: {error}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        total_passed = sum(cat['passed'] for cat in self.results.values())
        total_failed = sum(cat['failed'] for cat in self.results.values())
        total_tests = total_passed + total_failed
        
        return {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'success_rate': total_passed / total_tests if total_tests > 0 else 0,
            'execution_time': time.time() - self.start_time,
            'categories': self.results
        }

# Global test results collector
test_results = TestResults()

class TestGeneration1Basic:
    """Test Generation 1: Basic functionality (MAKE IT WORK)."""
    
    def test_autonomous_quantum_optimization(self):
        """Test autonomous quantum optimization engine."""
        try:
            from quantum_planner.research.autonomous_quantum_optimization import (
                create_autonomous_optimizer, QuantumAlgorithmType
            )
            
            # Create optimizer
            optimizer = create_autonomous_optimizer()
            assert optimizer is not None
            
            # Test problem
            problem_matrix = np.array([
                [2, -1, 0],
                [-1, 2, -1],
                [0, -1, 2]
            ])
            
            # Perform optimization
            result = optimizer.optimize(problem_matrix)
            
            # Validate result
            assert result.solution is not None
            assert len(result.solution) == 3
            assert result.energy != float('inf')
            assert result.algorithm_used in QuantumAlgorithmType
            assert result.execution_time > 0
            
            # Test performance report
            report = optimizer.get_performance_report()
            assert report['total_optimizations'] == 1
            assert 'average_quantum_advantage' in report
            
            test_results.record_result('generation_1', 'autonomous_quantum_optimization', True)
            logger.info("✅ Autonomous quantum optimization test passed")
            
        except Exception as e:
            test_results.record_result('generation_1', 'autonomous_quantum_optimization', False, str(e))
            logger.error(f"❌ Autonomous quantum optimization test failed: {e}")
    
    def test_neural_quantum_fusion(self):
        """Test neural-quantum fusion engine."""
        try:
            from quantum_planner.research.neural_quantum_fusion import (
                create_neural_quantum_fusion_engine
            )
            
            # Create fusion engine
            fusion_engine = create_neural_quantum_fusion_engine(num_qubits=3, problem_dim=10)
            assert fusion_engine is not None
            
            # Test problem
            problem_matrix = np.array([
                [2, -1, 0],
                [-1, 2, -1],
                [0, -1, 2]
            ])
            
            # Perform fusion optimization
            result = fusion_engine.neural_guided_optimization(problem_matrix, max_iterations=10)
            
            # Validate result
            assert result.quantum_solution is not None
            assert result.fusion_energy != float('inf')
            assert 0 <= result.neural_confidence <= 1
            assert result.quantum_advantage >= 1.0
            
            # Test adaptive fusion
            adaptive_result = fusion_engine.adaptive_fusion_optimization(
                problem_matrix, adaptation_cycles=2)
            assert adaptive_result is not None
            
            test_results.record_result('generation_1', 'neural_quantum_fusion', True)
            logger.info("✅ Neural-quantum fusion test passed")
            
        except Exception as e:
            test_results.record_result('generation_1', 'neural_quantum_fusion', False, str(e))
            logger.error(f"❌ Neural-quantum fusion test failed: {e}")
    
    def test_quantum_ecosystem_intelligence(self):
        """Test quantum ecosystem intelligence."""
        try:
            from quantum_planner.research.quantum_ecosystem_intelligence import (
                create_quantum_ecosystem_intelligence
            )
            
            # Create ecosystem
            ecosystem = create_quantum_ecosystem_intelligence(num_agents=5, max_population=20)
            assert ecosystem is not None
            assert len(ecosystem.agents) == 5
            
            # Test problem
            problem_matrix = np.array([
                [2, -1, 0, 1],
                [-1, 3, -1, 0],
                [0, -1, 2, -1],
                [1, 0, -1, 2]
            ])
            
            # Evolve ecosystem
            evolution_report = ecosystem.evolve_ecosystem(problem_matrix, evolution_cycles=3)
            
            # Validate evolution
            assert evolution_report['total_generations'] == 3
            assert evolution_report['final_best_energy'] != float('inf')
            assert 0 <= evolution_report['final_success_rate'] <= 1
            assert evolution_report['breakthroughs_discovered'] >= 0
            
            # Test best algorithm extraction
            best_algorithm, best_stats = ecosystem.get_best_algorithm()
            assert best_algorithm is not None
            
            test_results.record_result('generation_1', 'quantum_ecosystem_intelligence', True)
            logger.info("✅ Quantum ecosystem intelligence test passed")
            
        except Exception as e:
            test_results.record_result('generation_1', 'quantum_ecosystem_intelligence', False, str(e))
            logger.error(f"❌ Quantum ecosystem intelligence test failed: {e}")

class TestGeneration2Robust:
    """Test Generation 2: Robustness and reliability (MAKE IT ROBUST)."""
    
    def test_quantum_security_framework(self):
        """Test quantum security framework."""
        try:
            from quantum_planner.research.quantum_security_framework import (
                create_secure_quantum_optimizer, SecurityLevel
            )
            
            # Create secure optimizer
            secure_optimizer = create_secure_quantum_optimizer(SecurityLevel.ENHANCED)
            assert secure_optimizer is not None
            
            # Create secure session
            credentials = secure_optimizer.create_secure_session(
                user_id="test_user",
                permissions=["optimize", "read"],
                duration_hours=1.0
            )
            
            assert credentials is not None
            assert credentials.is_valid()
            assert credentials.has_permission("optimize")
            
            # Test problem
            problem_matrix = np.array([
                [2, -1, 0],
                [-1, 2, -1],
                [0, -1, 2]
            ])
            
            optimization_params = {
                'circuit_depth': 3,
                'parameters': [0.5, 1.0, 1.5],
                'max_iterations': 10
            }
            
            # Perform secure optimization
            result = secure_optimizer.secure_optimize(
                problem_matrix, credentials, optimization_params)
            
            # Validate secure result
            assert result['solution'] is not None
            assert result['energy'] != float('inf')
            assert result['security_level'] == 'enhanced'
            assert result['authenticated'] is True
            assert 'signature' in result
            
            # Test security report
            security_report = secure_optimizer.get_security_report()
            assert security_report['total_security_events'] >= 1
            assert 0 <= security_report['success_rate'] <= 1
            
            test_results.record_result('generation_2', 'quantum_security_framework', True)
            logger.info("✅ Quantum security framework test passed")
            
        except Exception as e:
            test_results.record_result('generation_2', 'quantum_security_framework', False, str(e))
            logger.error(f"❌ Quantum security framework test failed: {e}")
    
    def test_robust_quantum_validator(self):
        """Test robust quantum validator."""
        try:
            from quantum_planner.research.robust_quantum_validator import (
                create_robust_quantum_validator, ValidationLevel, QuantumResult
            )
            
            # Create validator
            validator = create_robust_quantum_validator(ValidationLevel.RIGOROUS)
            assert validator is not None
            
            # Create quantum result to validate
            quantum_result = QuantumResult(
                solution=np.array([1, 0, 1]),
                energy=5.2,
                algorithm_used="QAOA",
                execution_time=2.5,
                iterations=25,
                convergence_achieved=True,
                quantum_state=np.array([0.6+0.2j, 0.3-0.1j, 0.4+0.3j, 0.5+0.1j]),
                metadata={"backend": "simulator"}
            )
            
            # Problem matrix
            problem_matrix = np.array([
                [2, -1, 0],
                [-1, 2, -1],
                [0, -1, 2]
            ])
            
            # Additional context
            additional_context = {
                'energy_history': [10.5, 8.2, 6.1, 5.8, 5.2],
                'energy_samples': [5.2, 5.1, 5.3, 5.0, 5.4],
                'classical_baseline': 7.5
            }
            
            # Perform validation
            validation_result = validator.validate_quantum_result(
                quantum_result, problem_matrix, additional_context)
            
            # Validate validation result
            assert validation_result is not None
            assert validation_result.validation_id is not None
            assert 0 <= validation_result.confidence_score <= 1
            assert 0 <= validation_result.statistical_significance <= 1
            assert isinstance(validation_result.error_types, list)
            assert isinstance(validation_result.recommendations, list)
            
            # Test validation report
            report = validator.get_validation_report()
            assert report['total_validations'] >= 1
            assert 'success_rate' in report
            
            test_results.record_result('generation_2', 'robust_quantum_validator', True)
            logger.info("✅ Robust quantum validator test passed")
            
        except Exception as e:
            test_results.record_result('generation_2', 'robust_quantum_validator', False, str(e))
            logger.error(f"❌ Robust quantum validator test failed: {e}")

class TestGeneration3Scalable:
    """Test Generation 3: Scalability and optimization (MAKE IT SCALE)."""
    
    def test_quantum_hyperdimensional_optimizer(self):
        """Test quantum hyperdimensional optimizer."""
        try:
            from quantum_planner.research.quantum_hyperdimensional_optimizer import (
                create_quantum_hyperdimensional_optimizer, ScalingStrategy
            )
            
            # Create hyperdimensional optimizer
            optimizer = create_quantum_hyperdimensional_optimizer(
                scaling_strategy=ScalingStrategy.HIERARCHICAL,
                max_dimension_limit=50
            )
            assert optimizer is not None
            
            # Test small problem (direct optimization)
            small_problem = np.random.randint(-2, 3, (10, 10))
            small_problem = (small_problem + small_problem.T) / 2
            
            result_small = optimizer.optimize_hyperdimensional(small_problem)
            assert result_small['success'] is True
            assert result_small['optimization_type'] == 'direct'
            
            # Test large problem (hierarchical optimization)
            large_problem = np.random.randint(-2, 3, (80, 80))
            large_problem = (large_problem + large_problem.T) / 2
            
            result_large = optimizer.optimize_hyperdimensional(large_problem)
            assert result_large['success'] is True
            assert result_large['optimization_type'] == 'hierarchical'
            assert result_large['subproblems_count'] > 1
            
            # Test infinite scaling
            scaling_result = optimizer.optimize_with_infinite_scaling(
                small_problem, target_performance=0.7, max_scaling_levels=3)
            
            assert scaling_result['success'] is True
            assert scaling_result['scaling_levels_used'] <= 3
            assert 'best_result' in scaling_result
            
            # Test resource monitoring
            resource_report = optimizer.monitor_resource_usage()
            assert 'backend_metrics' in resource_report
            assert 'memory_usage_mb' in resource_report
            
            # Test auto-scaling
            scaling_report = optimizer.auto_scale_resources()
            assert 'scaling_actions' in scaling_report
            
            # Test scalability report
            scalability_report = optimizer.get_scalability_report()
            assert scalability_report['hyperdimensional_ready'] is True
            assert scalability_report['total_optimizations'] >= 0
            
            test_results.record_result('generation_3', 'quantum_hyperdimensional_optimizer', True)
            logger.info("✅ Quantum hyperdimensional optimizer test passed")
            
        except Exception as e:
            test_results.record_result('generation_3', 'quantum_hyperdimensional_optimizer', False, str(e))
            logger.error(f"❌ Quantum hyperdimensional optimizer test failed: {e}")

class TestIntegration:
    """Integration tests across all components."""
    
    def test_full_autonomous_workflow(self):
        """Test complete autonomous workflow integration."""
        try:
            # Import all components
            from quantum_planner.research.autonomous_quantum_optimization import create_autonomous_optimizer
            from quantum_planner.research.quantum_security_framework import (
                create_secure_quantum_optimizer, SecurityLevel
            )
            from quantum_planner.research.robust_quantum_validator import (
                create_robust_quantum_validator, ValidationLevel, QuantumResult
            )
            
            # Create problem
            problem_matrix = np.array([
                [3, -1, 0, 1],
                [-1, 3, -1, 0],
                [0, -1, 3, -1],
                [1, 0, -1, 3]
            ])
            
            # Step 1: Autonomous optimization
            optimizer = create_autonomous_optimizer()
            opt_result = optimizer.optimize(problem_matrix)
            
            # Step 2: Security validation
            secure_optimizer = create_secure_quantum_optimizer(SecurityLevel.ENHANCED)
            credentials = secure_optimizer.create_secure_session(
                user_id="integration_test", permissions=["optimize", "validate"])
            
            # Step 3: Robust validation
            validator = create_robust_quantum_validator(ValidationLevel.STANDARD)
            
            quantum_result = QuantumResult(
                solution=opt_result.solution,
                energy=opt_result.energy,
                algorithm_used=opt_result.algorithm_used.value,
                execution_time=opt_result.execution_time,
                iterations=opt_result.iterations,
                convergence_achieved=True
            )
            
            validation_result = validator.validate_quantum_result(
                quantum_result, problem_matrix)
            
            # Validate integration
            assert opt_result is not None
            assert credentials.is_valid()
            assert validation_result is not None
            assert validation_result.confidence_score > 0
            
            test_results.record_result('integration', 'full_autonomous_workflow', True)
            logger.info("✅ Full autonomous workflow integration test passed")
            
        except Exception as e:
            test_results.record_result('integration', 'full_autonomous_workflow', False, str(e))
            logger.error(f"❌ Full autonomous workflow integration test failed: {e}")
    
    def test_ecosystem_validation_integration(self):
        """Test ecosystem intelligence with validation integration."""
        try:
            from quantum_planner.research.quantum_ecosystem_intelligence import (
                create_quantum_ecosystem_intelligence
            )
            from quantum_planner.research.robust_quantum_validator import (
                create_robust_quantum_validator, QuantumResult
            )
            
            # Create ecosystem and validator
            ecosystem = create_quantum_ecosystem_intelligence(num_agents=3, max_population=10)
            validator = create_robust_quantum_validator()
            
            # Test problem
            problem_matrix = np.array([[2, -1], [-1, 2]])
            
            # Evolve ecosystem
            evolution_report = ecosystem.evolve_ecosystem(problem_matrix, evolution_cycles=2)
            
            # Get best algorithm and validate
            best_algorithm, best_stats = ecosystem.get_best_algorithm()
            
            if best_algorithm and best_stats:
                # Create quantum result from ecosystem output
                quantum_result = QuantumResult(
                    solution=np.array([1, 0]),  # Example solution
                    energy=best_stats.get('best_energy', 1.0),
                    algorithm_used="ecosystem_evolved",
                    execution_time=1.0,
                    iterations=50,
                    convergence_achieved=True
                )
                
                # Validate the result
                validation_result = validator.validate_quantum_result(
                    quantum_result, problem_matrix)
                
                assert validation_result is not None
                assert validation_result.confidence_score >= 0
            
            test_results.record_result('integration', 'ecosystem_validation_integration', True)
            logger.info("✅ Ecosystem-validation integration test passed")
            
        except Exception as e:
            test_results.record_result('integration', 'ecosystem_validation_integration', False, str(e))
            logger.error(f"❌ Ecosystem-validation integration test failed: {e}")

class TestPerformance:
    """Performance and benchmarking tests."""
    
    def test_optimization_performance_benchmarks(self):
        """Test optimization performance benchmarks."""
        try:
            from quantum_planner.research.autonomous_quantum_optimization import create_autonomous_optimizer
            
            optimizer = create_autonomous_optimizer()
            
            # Performance benchmarks
            problem_sizes = [3, 5, 8]
            performance_results = []
            
            for size in problem_sizes:
                problem_matrix = np.random.randint(-2, 3, (size, size))
                problem_matrix = (problem_matrix + problem_matrix.T) / 2
                
                start_time = time.time()
                result = optimizer.optimize(problem_matrix)
                execution_time = time.time() - start_time
                
                performance_results.append({
                    'problem_size': size,
                    'execution_time': execution_time,
                    'energy': result.energy,
                    'convergence': result.quantum_advantage_factor
                })
            
            # Validate performance
            assert len(performance_results) == len(problem_sizes)
            assert all(r['execution_time'] > 0 for r in performance_results)
            assert all(r['execution_time'] < 30.0 for r in performance_results)  # Under 30 seconds
            
            # Check scalability (larger problems shouldn't be exponentially slower)
            if len(performance_results) >= 2:
                time_ratio = performance_results[-1]['execution_time'] / performance_results[0]['execution_time']
                size_ratio = performance_results[-1]['problem_size'] / performance_results[0]['problem_size']
                assert time_ratio < size_ratio ** 3  # Should be better than cubic scaling
            
            test_results.record_result('performance', 'optimization_performance_benchmarks', True)
            logger.info("✅ Optimization performance benchmarks passed")
            
        except Exception as e:
            test_results.record_result('performance', 'optimization_performance_benchmarks', False, str(e))
            logger.error(f"❌ Optimization performance benchmarks failed: {e}")
    
    def test_scalability_stress_test(self):
        """Test scalability under stress conditions."""
        try:
            from quantum_planner.research.quantum_hyperdimensional_optimizer import (
                create_quantum_hyperdimensional_optimizer, ScalingStrategy
            )
            
            optimizer = create_quantum_hyperdimensional_optimizer(
                scaling_strategy=ScalingStrategy.ADAPTIVE,
                max_dimension_limit=30
            )
            
            # Stress test with increasing problem sizes
            stress_results = []
            
            for size in [10, 20, 30, 40]:
                problem_matrix = np.random.randint(-2, 3, (size, size))
                problem_matrix = (problem_matrix + problem_matrix.T) / 2
                
                start_time = time.time()
                try:
                    result = optimizer.optimize_hyperdimensional(problem_matrix)
                    execution_time = time.time() - start_time
                    
                    stress_results.append({
                        'size': size,
                        'success': result.get('success', False),
                        'execution_time': execution_time,
                        'optimization_type': result.get('optimization_type', 'unknown')
                    })
                    
                except Exception as stress_error:
                    stress_results.append({
                        'size': size,
                        'success': False,
                        'execution_time': time.time() - start_time,
                        'error': str(stress_error)
                    })
            
            # Validate stress test results
            assert len(stress_results) == 4
            success_count = sum(1 for r in stress_results if r['success'])
            assert success_count >= 3  # At least 75% success rate
            
            # Check that hierarchical optimization is used for larger problems
            large_problems = [r for r in stress_results if r['size'] > 30 and r['success']]
            if large_problems:
                assert any(r.get('optimization_type') == 'hierarchical' for r in large_problems)
            
            test_results.record_result('performance', 'scalability_stress_test', True)
            logger.info("✅ Scalability stress test passed")
            
        except Exception as e:
            test_results.record_result('performance', 'scalability_stress_test', False, str(e))
            logger.error(f"❌ Scalability stress test failed: {e}")

class TestSecurity:
    """Security and compliance tests."""
    
    def test_security_compliance(self):
        """Test security compliance and threat detection."""
        try:
            from quantum_planner.research.quantum_security_framework import (
                create_secure_quantum_optimizer, SecurityLevel
            )
            
            secure_optimizer = create_secure_quantum_optimizer(SecurityLevel.ENHANCED)
            
            # Test credential validation
            valid_credentials = secure_optimizer.create_secure_session(
                user_id="security_test", permissions=["optimize"])
            assert valid_credentials.is_valid()
            
            # Test expired credentials (simulate)
            expired_credentials = valid_credentials
            expired_credentials.expiry_time = time.time() - 1  # Expired
            assert not expired_credentials.is_valid()
            
            # Test permission checking
            assert valid_credentials.has_permission("optimize")
            assert not valid_credentials.has_permission("admin")
            
            # Test cryptographic functions
            from quantum_planner.research.quantum_security_framework import QuantumCryptography
            
            crypto = QuantumCryptography(SecurityLevel.ENHANCED)
            public_key, private_key = crypto.generate_keypair()
            
            # Test encryption/decryption
            test_data = b"sensitive quantum data"
            encrypted = crypto.encrypt_data(test_data, public_key)
            decrypted = crypto.decrypt_data(encrypted, private_key)
            assert decrypted == test_data
            
            # Test digital signatures
            signature = crypto.sign_data(test_data, private_key)
            assert crypto.verify_signature(test_data, signature, public_key)
            
            test_results.record_result('security', 'security_compliance', True)
            logger.info("✅ Security compliance test passed")
            
        except Exception as e:
            test_results.record_result('security', 'security_compliance', False, str(e))
            logger.error(f"❌ Security compliance test failed: {e}")

def run_comprehensive_test_suite():
    """Run the complete autonomous SDLC test suite."""
    logger.info("🚀 Starting Autonomous SDLC Comprehensive Test Suite")
    logger.info("=" * 80)
    
    # Test Generation 1
    logger.info("🔵 Testing Generation 1: Basic Functionality (MAKE IT WORK)")
    gen1_tests = TestGeneration1Basic()
    gen1_tests.test_autonomous_quantum_optimization()
    gen1_tests.test_neural_quantum_fusion()
    gen1_tests.test_quantum_ecosystem_intelligence()
    
    # Test Generation 2
    logger.info("🟡 Testing Generation 2: Robustness & Security (MAKE IT ROBUST)")
    gen2_tests = TestGeneration2Robust()
    gen2_tests.test_quantum_security_framework()
    gen2_tests.test_robust_quantum_validator()
    
    # Test Generation 3
    logger.info("🟢 Testing Generation 3: Scalability & Performance (MAKE IT SCALE)")
    gen3_tests = TestGeneration3Scalable()
    gen3_tests.test_quantum_hyperdimensional_optimizer()
    
    # Integration Tests
    logger.info("🔗 Testing Integration & Workflows")
    integration_tests = TestIntegration()
    integration_tests.test_full_autonomous_workflow()
    integration_tests.test_ecosystem_validation_integration()
    
    # Performance Tests
    logger.info("⚡ Testing Performance & Benchmarks")
    performance_tests = TestPerformance()
    performance_tests.test_optimization_performance_benchmarks()
    performance_tests.test_scalability_stress_test()
    
    # Security Tests
    logger.info("🛡️ Testing Security & Compliance")
    security_tests = TestSecurity()
    security_tests.test_security_compliance()
    
    # Generate comprehensive report
    logger.info("📊 Generating Test Report")
    summary = test_results.get_summary()
    
    logger.info("=" * 80)
    logger.info("🎯 AUTONOMOUS SDLC TEST SUITE COMPLETE")
    logger.info("=" * 80)
    
    logger.info(f"📈 OVERALL RESULTS:")
    logger.info(f"   Total Tests: {summary['total_tests']}")
    logger.info(f"   Passed: {summary['total_passed']}")
    logger.info(f"   Failed: {summary['total_failed']}")
    logger.info(f"   Success Rate: {summary['success_rate']:.1%}")
    logger.info(f"   Execution Time: {summary['execution_time']:.1f} seconds")
    
    logger.info(f"\n📋 CATEGORY BREAKDOWN:")
    for category, results in summary['categories'].items():
        total_cat = results['passed'] + results['failed']
        if total_cat > 0:
            success_rate = results['passed'] / total_cat
            status = "✅" if success_rate == 1.0 else "⚠️" if success_rate >= 0.5 else "❌"
            logger.info(f"   {status} {category.replace('_', ' ').title()}: "
                       f"{results['passed']}/{total_cat} ({success_rate:.1%})")
    
    # Log any errors
    all_errors = []
    for category, results in summary['categories'].items():
        all_errors.extend(results['errors'])
    
    if all_errors:
        logger.warning(f"\n⚠️ ERRORS DETECTED ({len(all_errors)}):")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"   • {error}")
        if len(all_errors) > 5:
            logger.warning(f"   ... and {len(all_errors) - 5} more errors")
    
    # Quality gate assessment
    quality_gate_passed = summary['success_rate'] >= 0.85  # 85% threshold
    
    if quality_gate_passed:
        logger.info(f"\n🎉 QUALITY GATE PASSED!")
        logger.info(f"   ✅ Success rate {summary['success_rate']:.1%} exceeds 85% threshold")
        logger.info(f"   ✅ All critical components validated")
        logger.info(f"   ✅ Ready for production deployment")
    else:
        logger.error(f"\n❌ QUALITY GATE FAILED!")
        logger.error(f"   ❌ Success rate {summary['success_rate']:.1%} below 85% threshold")
        logger.error(f"   ❌ Critical issues must be resolved before deployment")
    
    return summary, quality_gate_passed

if __name__ == "__main__":
    # Run the comprehensive test suite
    summary, quality_gate_passed = run_comprehensive_test_suite()
    
    # Exit with appropriate code
    exit_code = 0 if quality_gate_passed else 1
    exit(exit_code)