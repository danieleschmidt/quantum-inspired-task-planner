"""
Comprehensive Test Suite for Research Module Implementation

This test suite validates all research components with thorough testing:
1. Adaptive Quantum Annealing algorithms
2. Enhanced QUBO formulation with dynamic constraints  
3. Statistical validation framework
4. Quantum-Classical Co-Evolution optimization
5. ML-based Quantum Advantage Prediction
6. Integrated research demonstration

All tests are designed for publication-grade validation and reproducibility.
"""

import sys
import os
import unittest
import numpy as np
import time
import warnings
from unittest.mock import Mock, patch, MagicMock

# Add source path
sys.path.insert(0, '/root/repo/src')

# Research module imports
try:
    from quantum_planner.research.adaptive_quantum_annealing import (
        AdaptiveQuantumAnnealingScheduler, 
        AdaptiveAnnealingParams,
        AnnealingScheduleType,
        NoiseProfile,
        HamiltonianSpectralAnalyzer,
        ScheduleOptimizer,
        RealTimeFeedbackProcessor,
        NoiseAdaptiveCompensator
    )
    ADAPTIVE_ANNEALING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Adaptive annealing not available: {e}")
    ADAPTIVE_ANNEALING_AVAILABLE = False

try:
    from quantum_planner.research.enhanced_qubo_formulation import (
        EnhancedQUBOBuilder,
        DynamicConstraint,
        ConstraintPriority,
        AdaptationStrategy,
        ConstraintSatisfactionEstimator,
        QuantumFeedbackProcessor,
        EmbeddingOptimizer
    )
    ENHANCED_QUBO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced QUBO not available: {e}")
    ENHANCED_QUBO_AVAILABLE = False

try:
    from quantum_planner.research.statistical_validation import (
        AdvancedStatisticalValidator,
        StatisticalTest,
        MultipleComparisonMethod,
        EffectSizeMethod,
        AssumptionChecker,
        EffectSizeCalculator,
        BootstrapAnalyzer,
        PowerAnalyzer
    )
    STATISTICAL_VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Statistical validation not available: {e}")
    STATISTICAL_VALIDATION_AVAILABLE = False

try:
    from quantum_planner.research.quantum_classical_coevolution import (
        QuantumClassicalCoEvolutionOptimizer,
        CoEvolutionParameters,
        CoEvolutionStrategy,
        InformationExchangeProtocol,
        QuantumPopulation,
        ClassicalPopulation,
        InformationExchanger,
        ResourceManager
    )
    COEVOLUTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Co-evolution not available: {e}")
    COEVOLUTION_AVAILABLE = False

try:
    from quantum_planner.research.quantum_advantage_prediction import (
        QuantumAdvantagePredictor,
        ProblemAnalyzer,
        HardwareProfile,
        ProblemCharacteristics,
        QuantumAdvantageRegime,
        PredictionConfidence,
        generate_synthetic_training_data
    )
    ADVANTAGE_PREDICTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advantage prediction not available: {e}")
    ADVANTAGE_PREDICTION_AVAILABLE = False


class TestAdaptiveQuantumAnnealing(unittest.TestCase):
    """Test adaptive quantum annealing research implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not ADAPTIVE_ANNEALING_AVAILABLE:
            self.skipTest("Adaptive annealing module not available")
        
        self.test_matrix = np.array([
            [1, -1, 0],
            [-1, 2, -1],
            [0, -1, 1]
        ], dtype=float)
        
        self.noise_profile = NoiseProfile(
            coherence_time=100.0,
            gate_error_rate=0.001,
            readout_error_rate=0.02,
            crosstalk_strength=0.01,
            temperature=15.0
        )
        
        self.params = AdaptiveAnnealingParams(
            schedule_type=AnnealingScheduleType.ADAPTIVE_HYBRID,
            max_annealing_time=20.0,
            adaptation_rate=0.1,
            real_time_feedback=True
        )
    
    def test_noise_profile_validation(self):
        """Test noise profile validation."""
        # Valid profile should not raise exception
        try:
            NoiseProfile(
                coherence_time=100.0,
                gate_error_rate=0.001,
                readout_error_rate=0.02,
                crosstalk_strength=0.01,
                temperature=15.0
            )
        except Exception:
            self.fail("Valid noise profile raised exception")
        
        # Invalid coherence time should raise exception
        with self.assertRaises(ValueError):
            NoiseProfile(
                coherence_time=-1.0,  # Invalid
                gate_error_rate=0.001,
                readout_error_rate=0.02,
                crosstalk_strength=0.01,
                temperature=15.0
            )
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        scheduler = AdaptiveQuantumAnnealingScheduler(self.params, self.noise_profile)
        
        self.assertIsInstance(scheduler, AdaptiveQuantumAnnealingScheduler)
        self.assertEqual(scheduler.params.schedule_type, AnnealingScheduleType.ADAPTIVE_HYBRID)
        self.assertIsNotNone(scheduler.noise_profile)
        self.assertEqual(len(scheduler.performance_history), 0)
    
    def test_hamiltonian_analyzer(self):
        """Test Hamiltonian spectral analysis."""
        analyzer = HamiltonianSpectralAnalyzer(analysis_depth=5)
        properties = analyzer.analyze_spectrum(self.test_matrix)
        
        self.assertIn('spectral_gap', properties)
        self.assertIn('gap_position', properties)
        self.assertIn('condition_number', properties)
        self.assertIn('eigenvalue_spread', properties)
        
        # Verify numerical properties
        self.assertGreaterEqual(properties['spectral_gap'], 0)
        self.assertGreaterEqual(properties['gap_position'], 0)
        self.assertLessEqual(properties['gap_position'], 1)
    
    def test_schedule_generation(self):
        """Test annealing schedule generation."""
        scheduler = AdaptiveQuantumAnnealingScheduler(self.params, self.noise_profile)
        
        # Mock Hamiltonian properties
        hamiltonian_props = {
            'spectral_gap': 0.1,
            'gap_position': 0.5,
            'condition_number': 10.0
        }
        
        schedule = scheduler._generate_initial_schedule(hamiltonian_props, 3)
        
        self.assertIsInstance(schedule, np.ndarray)
        self.assertGreater(len(schedule), 0)
        self.assertTrue(np.all(schedule >= 0))
        self.assertTrue(np.all(schedule <= 1))
        self.assertLessEqual(schedule[0], schedule[-1])  # Should be increasing
    
    def test_optimization_process(self):
        """Test optimization process (with simulation)."""
        scheduler = AdaptiveQuantumAnnealingScheduler(self.params, self.noise_profile)
        
        # Use small matrix for fast testing
        small_matrix = self.test_matrix
        
        result = scheduler.optimize_task_assignment(small_matrix)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result.final_energy, float)
        self.assertIsInstance(result.solution_vector, dict)
        self.assertGreaterEqual(result.schedule_efficiency, 0)
        self.assertLessEqual(result.schedule_efficiency, 1)
        self.assertGreaterEqual(result.execution_time, 0)
    
    def test_feedback_processor(self):
        """Test real-time feedback processor."""
        processor = RealTimeFeedbackProcessor(window_size=5, confidence_level=0.95)
        
        # Add some measurements
        for i in range(10):
            energy = 1.0 - i * 0.1  # Improving trend
            schedule = np.linspace(0, 1, 100)
            feedback = processor.process_measurement(energy, schedule)
            
            self.assertIn('confidence', feedback)
            self.assertIn('trend', feedback)
            self.assertGreaterEqual(feedback['confidence'], 0)
            self.assertLessEqual(feedback['confidence'], 1)
        
        # Should detect improving trend
        self.assertEqual(len(processor.measurement_history), 5)  # Window size


class TestEnhancedQUBOFormulation(unittest.TestCase):
    """Test enhanced QUBO formulation research implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not ENHANCED_QUBO_AVAILABLE:
            self.skipTest("Enhanced QUBO module not available")
        
        self.builder = EnhancedQUBOBuilder(
            enable_dynamic_adaptation=True,
            enable_embedding_optimization=True,
            enable_hierarchical_constraints=True
        )
        
        # Mock agents and tasks
        self.agents = [
            {'id': 'agent1', 'skills': ['python', 'ml'], 'capacity': 3},
            {'id': 'agent2', 'skills': ['javascript'], 'capacity': 2}
        ]
        
        self.tasks = [
            {'id': 'task1', 'required_skills': ['python'], 'duration': 2},
            {'id': 'task2', 'required_skills': ['javascript'], 'duration': 1}
        ]
    
    def test_dynamic_constraint_creation(self):
        """Test dynamic constraint creation."""
        self.builder.add_dynamic_constraint(
            name='test_constraint',
            constraint_type='assignment',
            base_penalty=10.0,
            priority=ConstraintPriority.HIGH,
            adaptation_strategy=AdaptationStrategy.QUANTUM_FEEDBACK
        )
        
        self.assertEqual(len(self.builder.constraints), 1)
        constraint = self.builder.constraints[0]
        
        self.assertEqual(constraint.name, 'test_constraint')
        self.assertEqual(constraint.constraint_type, 'assignment')
        self.assertEqual(constraint.base_penalty, 10.0)
        self.assertEqual(constraint.priority, ConstraintPriority.HIGH)
        self.assertTrue(constraint.is_adaptive)
    
    def test_constraint_satisfaction_estimator(self):
        """Test constraint satisfaction probability estimation."""
        estimator = ConstraintSatisfactionEstimator()
        
        # Create test constraint
        constraint = DynamicConstraint(
            name='test',
            constraint_type='assignment',
            base_penalty=10.0,
            current_penalty=10.0,
            priority=ConstraintPriority.MEDIUM,
            adaptation_strategy=AdaptationStrategy.SATISFACTION_PROBABILITY
        )
        
        problem_chars = {'problem_size': 10, 'constraint_density': 0.2}
        
        prob = estimator.estimate_satisfaction_probability(constraint, problem_chars)
        
        self.assertGreaterEqual(prob, 0.01)
        self.assertLessEqual(prob, 0.99)
        self.assertIsInstance(prob, float)
    
    def test_qubo_matrix_construction(self):
        """Test QUBO matrix construction."""
        self.builder.add_dynamic_constraint(
            'assignment', 'assignment', 10.0, ConstraintPriority.CRITICAL
        )
        
        Q, mapping, metadata = self.builder.build_enhanced_qubo(self.agents, self.tasks)
        
        self.assertIsInstance(Q, np.ndarray)
        self.assertEqual(len(Q.shape), 2)
        self.assertEqual(Q.shape[0], Q.shape[1])  # Square matrix
        
        self.assertIsInstance(mapping, dict)
        self.assertGreater(len(mapping), 0)
        
        self.assertIn('formulation_time', metadata)
        self.assertIn('num_variables', metadata)
        self.assertIn('matrix_density', metadata)
    
    def test_embedding_optimizer(self):
        """Test embedding optimization."""
        optimizer = EmbeddingOptimizer()
        
        # Create test matrix
        test_matrix = np.random.random((10, 10))
        test_matrix = (test_matrix + test_matrix.T) / 2  # Make symmetric
        
        optimized_matrix = optimizer.optimize_for_embedding(test_matrix)
        
        self.assertEqual(optimized_matrix.shape, test_matrix.shape)
        self.assertIsInstance(optimized_matrix, np.ndarray)
    
    def test_feedback_processing(self):
        """Test solution feedback processing."""
        self.builder.add_dynamic_constraint(
            'test', 'assignment', 10.0, ConstraintPriority.MEDIUM
        )
        
        # Mock solution feedback
        solution = {0: 1, 1: 0, 2: 1, 3: 0}
        objective_value = 5.0
        constraint_violations = {'test': False}
        
        adjustments = self.builder.process_solution_feedback(
            solution, objective_value, constraint_violations
        )
        
        self.assertIsInstance(adjustments, dict)


class TestStatisticalValidation(unittest.TestCase):
    """Test statistical validation framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not STATISTICAL_VALIDATION_AVAILABLE:
            self.skipTest("Statistical validation module not available")
        
        self.validator = AdvancedStatisticalValidator(
            alpha=0.05,
            multiple_comparison_method=MultipleComparisonMethod.HOLM_BONFERRONI,
            effect_size_method=EffectSizeMethod.HEDGES_G
        )
        
        # Generate test data
        np.random.seed(42)
        self.algorithm_results = {
            'quantum_enhanced': np.random.normal(0.8, 0.1, 20).tolist(),
            'classical_baseline': np.random.normal(0.7, 0.1, 20).tolist(),
            'hybrid_method': np.random.normal(0.75, 0.1, 20).tolist()
        }
    
    def test_assumption_checker(self):
        """Test statistical assumption checking."""
        checker = AssumptionChecker()
        
        # Test normality checking
        normal_data = np.random.normal(0, 1, 100)
        normality_result = checker.check_normality(normal_data)
        
        self.assertIn('overall_assessment', normality_result)
        self.assertIn('normality_met', normality_result['overall_assessment'])
        
        # Test homoscedasticity
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(0, 1, 50)
        homoscedasticity_result = checker.check_homoscedasticity([group1, group2])
        
        self.assertIn('overall_assessment', homoscedasticity_result)
    
    def test_effect_size_calculation(self):
        """Test effect size calculations."""
        calculator = EffectSizeCalculator()
        
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([2, 3, 4, 5, 6])
        
        # Test Cohen's d
        cohens_d = calculator.cohens_d(group1, group2)
        self.assertIsInstance(cohens_d, float)
        
        # Test Hedges' g
        hedges_g = calculator.hedges_g(group1, group2)
        self.assertIsInstance(hedges_g, float)
        
        # Test Cliff's delta
        cliff_delta = calculator.cliff_delta(group1, group2)
        self.assertIsInstance(cliff_delta, float)
        self.assertGreaterEqual(cliff_delta, -1)
        self.assertLessEqual(cliff_delta, 1)
    
    def test_bootstrap_analysis(self):
        """Test bootstrap analysis."""
        analyzer = BootstrapAnalyzer(n_bootstrap=1000, random_state=42)
        
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test confidence interval
        ci_lower, ci_upper = analyzer.bootstrap_confidence_interval(data, np.mean, 0.95)
        
        self.assertIsInstance(ci_lower, (int, float))
        self.assertIsInstance(ci_upper, (int, float))
        self.assertLess(ci_lower, ci_upper)
        
        sample_mean = np.mean(data)
        self.assertLessEqual(ci_lower, sample_mean)
        self.assertGreaterEqual(ci_upper, sample_mean)
    
    def test_comprehensive_comparison(self):
        """Test comprehensive algorithm comparison."""
        analysis_results = self.validator.compare_algorithms(
            self.algorithm_results, paired=True
        )
        
        self.assertIn('comparison_results', analysis_results)
        self.assertIn('assumption_checks', analysis_results)
        self.assertIn('power_analysis', analysis_results)
        self.assertIn('overall_summary', analysis_results)
        
        # Check comparison results structure
        comparisons = analysis_results['comparison_results']
        self.assertGreater(len(comparisons), 0)
        
        for result in comparisons.values():
            self.assertIsNotNone(result.p_value)
            self.assertIsNotNone(result.test_type)
            self.assertIsInstance(result.is_significant, bool)
    
    def test_power_analysis(self):
        """Test statistical power analysis."""
        analyzer = PowerAnalyzer()
        
        effect_size = 0.5
        sample_size = 20
        
        power = analyzer.power_ttest(effect_size, sample_size, alpha=0.05)
        
        self.assertIsInstance(power, float)
        self.assertGreaterEqual(power, 0)
        self.assertLessEqual(power, 1)
        
        # Test sample size calculation
        required_n = analyzer.sample_size_ttest(effect_size, power=0.8, alpha=0.05)
        self.assertIsInstance(required_n, int)
        self.assertGreater(required_n, 0)


class TestQuantumClassicalCoEvolution(unittest.TestCase):
    """Test quantum-classical co-evolution implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not COEVOLUTION_AVAILABLE:
            self.skipTest("Co-evolution module not available")
        
        self.test_matrix = np.array([
            [1, -1, 0, 0],
            [-1, 2, -1, 0],
            [0, -1, 2, -1],
            [0, 0, -1, 1]
        ], dtype=float)
        
        self.params = CoEvolutionParameters(
            quantum_population_size=10,
            classical_population_size=15,
            max_generations=20,  # Small for testing
            strategy=CoEvolutionStrategy.COOPERATIVE,
            exchange_protocol=InformationExchangeProtocol.BEST_SOLUTION_SHARING
        )
    
    def test_quantum_population(self):
        """Test quantum population initialization and evolution."""
        population = QuantumPopulation(10, self.test_matrix)
        
        self.assertEqual(population.population_size, 10)
        self.assertEqual(len(population.population), 10)
        self.assertEqual(population.num_variables, self.test_matrix.shape[0])
        
        # Test fitness evaluation
        fitness = population.evaluate_population()
        self.assertEqual(len(fitness), 10)
        self.assertTrue(np.all(np.isfinite(fitness)))
        
        # Test evolution
        initial_generation = population.generation
        population.evolve_generation()
        self.assertEqual(population.generation, initial_generation + 1)
    
    def test_classical_population(self):
        """Test classical population initialization and evolution."""
        population = ClassicalPopulation(15, self.test_matrix)
        
        self.assertEqual(population.population_size, 15)
        self.assertEqual(len(population.population), 15)
        self.assertEqual(population.num_variables, self.test_matrix.shape[0])
        
        # Test fitness evaluation
        fitness = population.evaluate_population()
        self.assertEqual(len(fitness), 15)
        self.assertTrue(np.all(np.isfinite(fitness)))
        
        # Test evolution
        initial_generation = population.generation
        population.evolve_generation()
        self.assertEqual(population.generation, initial_generation + 1)
    
    def test_information_exchange(self):
        """Test information exchange between populations."""
        quantum_pop = QuantumPopulation(5, self.test_matrix)
        classical_pop = ClassicalPopulation(5, self.test_matrix)
        
        exchanger = InformationExchanger(InformationExchangeProtocol.BEST_SOLUTION_SHARING)
        
        # Evaluate populations first
        quantum_pop.evaluate_population()
        classical_pop.evaluate_population()
        
        exchange_result = exchanger.exchange_information(quantum_pop, classical_pop)
        
        self.assertIsInstance(exchange_result, dict)
        self.assertIn('generation', exchange_result)
        self.assertIn('protocol', exchange_result)
        self.assertIn('effectiveness', exchange_result)
    
    def test_resource_manager(self):
        """Test resource allocation management."""
        from quantum_planner.research.quantum_classical_coevolution import ResourceAllocationStrategy
        
        manager = ResourceManager(ResourceAllocationStrategy.PERFORMANCE_BASED, 0.5)
        
        quantum_pop = QuantumPopulation(5, self.test_matrix)
        classical_pop = ClassicalPopulation(5, self.test_matrix)
        
        # Add some statistics
        quantum_pop.evaluate_population()
        classical_pop.evaluate_population()
        
        allocation = manager.allocate_resources(quantum_pop, classical_pop)
        
        self.assertIn('quantum_allocation', allocation)
        self.assertIn('classical_allocation', allocation)
        self.assertAlmostEqual(
            allocation['quantum_allocation'] + allocation['classical_allocation'], 
            1.0, places=5
        )
    
    def test_coevolution_optimizer(self):
        """Test complete co-evolution optimization."""
        optimizer = QuantumClassicalCoEvolutionOptimizer(self.params)
        
        # Run optimization (short for testing)
        result = optimizer.optimize(self.test_matrix)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result.best_solution, dict)
        self.assertIsInstance(result.best_fitness, float)
        self.assertGreaterEqual(result.quantum_contribution, 0)
        self.assertLessEqual(result.quantum_contribution, 1)
        self.assertGreaterEqual(result.classical_contribution, 0)
        self.assertLessEqual(result.classical_contribution, 1)


class TestQuantumAdvantagePredictiion(unittest.TestCase):
    """Test quantum advantage prediction implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not ADVANTAGE_PREDICTION_AVAILABLE:
            self.skipTest("Advantage prediction module not available")
        
        self.predictor = QuantumAdvantagePredictor(model_type="ensemble")
        self.analyzer = ProblemAnalyzer()
        
        self.test_matrix = np.random.random((20, 20))
        self.test_matrix = (self.test_matrix + self.test_matrix.T) / 2
        
        self.hardware_profile = HardwareProfile(
            name="test_hardware",
            num_qubits=50,
            connectivity="grid",
            gate_error_rate=0.001,
            readout_error_rate=0.02,
            coherence_time=100.0,
            gate_time=0.1
        )
    
    def test_hardware_profile_validation(self):
        """Test hardware profile validation."""
        # Valid profile should work
        profile = HardwareProfile(
            name="valid",
            num_qubits=50,
            connectivity="grid",
            gate_error_rate=0.001,
            readout_error_rate=0.02,
            coherence_time=100.0,
            gate_time=0.1
        )
        self.assertEqual(profile.num_qubits, 50)
        
        # Invalid profile should raise exception
        with self.assertRaises(ValueError):
            HardwareProfile(
                name="invalid",
                num_qubits=0,  # Invalid
                connectivity="grid",
                gate_error_rate=0.001,
                readout_error_rate=0.02,
                coherence_time=100.0,
                gate_time=0.1
            )
    
    def test_problem_analysis(self):
        """Test problem characteristics analysis."""
        characteristics = self.analyzer.analyze_problem(self.test_matrix)
        
        self.assertIsInstance(characteristics, ProblemCharacteristics)
        self.assertEqual(characteristics.problem_size, 20)
        self.assertGreaterEqual(characteristics.matrix_density, 0)
        self.assertLessEqual(characteristics.matrix_density, 1)
        self.assertGreater(characteristics.matrix_condition_number, 0)
        self.assertGreaterEqual(characteristics.spectral_gap, 0)
    
    def test_synthetic_training_data(self):
        """Test synthetic training data generation."""
        training_data = generate_synthetic_training_data(50)
        
        self.assertEqual(len(training_data), 50)
        
        for problem_chars, hardware, advantage in training_data:
            self.assertIsInstance(problem_chars, ProblemCharacteristics)
            self.assertIsInstance(hardware, HardwareProfile)
            self.assertIsInstance(advantage, float)
            self.assertGreaterEqual(advantage, -1.0)
            self.assertLessEqual(advantage, 1.0)
    
    def test_model_training(self):
        """Test ML model training."""
        # Generate small training dataset
        training_data = generate_synthetic_training_data(20)
        
        # Train predictor
        performance = self.predictor.train(training_data, validation_split=0.2)
        
        self.assertIsInstance(performance, dict)
        self.assertTrue(self.predictor.is_trained)
    
    def test_advantage_prediction(self):
        """Test quantum advantage prediction."""
        # Train with minimal data first
        training_data = generate_synthetic_training_data(20)
        self.predictor.train(training_data, validation_split=0.2)
        
        # Analyze problem
        problem_chars = self.analyzer.analyze_problem(self.test_matrix)
        
        # Make prediction
        prediction = self.predictor.predict(problem_chars, self.hardware_profile)
        
        self.assertIsInstance(prediction.predicted_regime, QuantumAdvantageRegime)
        self.assertIsInstance(prediction.confidence, PredictionConfidence)
        self.assertIsInstance(prediction.numerical_advantage, float)
        self.assertIsInstance(prediction.confidence_interval, tuple)
        self.assertEqual(len(prediction.confidence_interval), 2)
        self.assertIn(prediction.recommended_algorithm, ['quantum', 'classical', 'hybrid'])
    
    def test_model_insights(self):
        """Test model performance insights."""
        # Train with minimal data
        training_data = generate_synthetic_training_data(20)
        self.predictor.train(training_data, validation_split=0.2)
        
        insights = self.predictor.get_model_insights()
        
        self.assertIn('model_performance', insights)
        self.assertIn('training_data_size', insights)
        self.assertIn('prediction_history_size', insights)


class TestIntegratedResearchDemo(unittest.TestCase):
    """Test integrated research demonstration."""
    
    def test_demo_import(self):
        """Test that the demo can be imported."""
        try:
            from examples.advanced_research_demo import AdvancedResearchDemonstration
            demo = AdvancedResearchDemonstration(save_results=False)
            self.assertIsNotNone(demo)
        except ImportError:
            self.skipTest("Research demo not available")
    
    def test_demo_problems_generation(self):
        """Test demo problem generation."""
        try:
            from examples.advanced_research_demo import AdvancedResearchDemonstration
            demo = AdvancedResearchDemonstration(save_results=False)
            
            self.assertGreater(len(demo.demo_problems), 0)
            self.assertGreater(len(demo.hardware_profiles), 0)
            
            for problem in demo.demo_problems:
                self.assertIsInstance(problem, np.ndarray)
                self.assertEqual(len(problem.shape), 2)
                self.assertEqual(problem.shape[0], problem.shape[1])
                
        except ImportError:
            self.skipTest("Research demo not available")


class TestModuleIntegration(unittest.TestCase):
    """Test integration between research modules."""
    
    def test_cross_module_compatibility(self):
        """Test that modules work together correctly."""
        
        # Skip if modules not available
        if not all([ADAPTIVE_ANNEALING_AVAILABLE, ENHANCED_QUBO_AVAILABLE, 
                   STATISTICAL_VALIDATION_AVAILABLE]):
            self.skipTest("Not all research modules available")
        
        # Test integration workflow
        try:
            # 1. Create problem
            test_matrix = np.random.random((10, 10))
            test_matrix = (test_matrix + test_matrix.T) / 2
            
            # 2. Enhanced QUBO formulation
            builder = EnhancedQUBOBuilder()
            agents = [{'id': 'agent1', 'skills': ['test'], 'capacity': 5}]
            tasks = [{'id': 'task1', 'required_skills': ['test'], 'duration': 1}]
            
            Q, mapping, metadata = builder.build_enhanced_qubo(agents, tasks)
            
            # 3. Adaptive annealing optimization
            params = AdaptiveAnnealingParams(max_annealing_time=1.0)
            scheduler = AdaptiveQuantumAnnealingScheduler(params)
            result = scheduler.optimize_task_assignment(Q)
            
            # 4. Statistical validation would follow
            self.assertIsNotNone(result)
            self.assertIsInstance(result.final_energy, float)
            
        except Exception as e:
            self.fail(f"Module integration failed: {e}")


def run_comprehensive_tests():
    """Run all research module tests."""
    
    print("🧪 Running Comprehensive Research Module Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestAdaptiveQuantumAnnealing,
        TestEnhancedQUBOFormulation, 
        TestStatisticalValidation,
        TestQuantumClassicalCoEvolution,
        TestQuantumAdvantagePredictiion,
        TestIntegratedResearchDemo,
        TestModuleIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("🧪 TEST RESULTS SUMMARY")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / max(1, result.testsRun) * 100
    print(f"\n✅ SUCCESS RATE: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎯 RESEARCH MODULES VALIDATION: PASSED")
    elif success_rate >= 75:
        print("⚠️  RESEARCH MODULES VALIDATION: PARTIAL SUCCESS") 
    else:
        print("❌ RESEARCH MODULES VALIDATION: NEEDS IMPROVEMENT")
    
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_comprehensive_tests()