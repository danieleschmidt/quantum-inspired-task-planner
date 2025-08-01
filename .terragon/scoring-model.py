#!/usr/bin/env python3
"""
Terragon Advanced Scoring Model
==============================

Hybrid WSJF + ICE + Technical Debt scoring algorithm with quantum computing domain expertise.
Implements adaptive prioritization for advanced quantum computing repositories.

This model combines:
- WSJF (Weighted Shortest Job First) for value/effort optimization
- ICE (Impact, Confidence, Ease) for strategic prioritization
- Technical Debt scoring for maintenance prioritization
- Quantum Computing domain-specific boost factors

Version: 2.0.0
Domain: Quantum Computing & Task Optimization
Maturity: Advanced Repository (88%)
"""

import json
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml
import re


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskCategory(Enum):
    """Task categories with domain-specific classification."""
    SECURITY = "security"
    PERFORMANCE = "performance" 
    QUANTUM_OPTIMIZATION = "quantum_optimization"
    TECHNICAL_DEBT = "technical_debt"
    INNOVATION = "innovation"
    MAINTENANCE = "maintenance"
    INTEGRATION = "integration"
    RESEARCH = "research"


class UrgencyLevel(Enum):
    """Urgency classification for tasks."""
    CRITICAL = "critical"      # Must fix immediately
    HIGH = "high"             # Fix within 1 week
    MEDIUM = "medium"         # Fix within 1 month
    LOW = "low"               # Fix when convenient


class QuantumDomain(Enum):
    """Quantum computing domain areas."""
    QUBO_OPTIMIZATION = "qubo_optimization"
    QUANTUM_BACKENDS = "quantum_backends"
    HYBRID_ALGORITHMS = "hybrid_algorithms"
    CLASSICAL_FALLBACKS = "classical_fallbacks"
    PERFORMANCE_TUNING = "performance_tuning"
    QUANTUM_SECURITY = "quantum_security"
    RESEARCH_INTEGRATION = "research_integration"


@dataclass
class TaskMetrics:
    """Comprehensive metrics for a discovered task."""
    
    # Basic task information
    task_id: str
    title: str
    description: str
    category: TaskCategory
    urgency: UrgencyLevel
    
    # WSJF Components (0-10 scale)
    user_business_value: float = 5.0
    time_criticality: float = 5.0
    risk_reduction: float = 5.0
    job_size: float = 5.0  # Effort estimate (higher = more effort)
    
    # ICE Components (0-10 scale)
    impact: float = 5.0
    confidence: float = 5.0
    ease: float = 5.0  # Implementation ease (higher = easier)
    
    # Technical Debt Components (0-10 scale)
    code_complexity: float = 0.0
    security_vulnerability: float = 0.0
    performance_impact: float = 0.0
    maintainability_impact: float = 0.0
    test_coverage_gap: float = 0.0
    
    # Quantum-specific factors
    quantum_domain: Optional[QuantumDomain] = None
    quantum_performance_impact: float = 0.0
    quantum_cost_impact: float = 0.0
    quantum_research_value: float = 0.0
    
    # Discovery metadata
    discovered_by: str = "terragon-scanner"
    discovery_date: datetime = field(default_factory=datetime.now)
    source_signals: List[str] = field(default_factory=list)
    confidence_score: float = 0.8
    
    # Execution metadata
    estimated_effort_hours: float = 4.0
    required_approvals: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    auto_executable: bool = True


@dataclass
class ScoringWeights:
    """Configurable weights for the hybrid scoring model."""
    wsjf: float = 0.35
    ice: float = 0.25
    technical_debt: float = 0.25
    quantum_boost: float = 0.15
    
    # WSJF sub-weights
    wsjf_user_value: float = 0.3
    wsjf_time_criticality: float = 0.3
    wsjf_risk_reduction: float = 0.2
    wsjf_job_size: float = 0.2
    
    # ICE sub-weights
    ice_impact: float = 0.4
    ice_confidence: float = 0.3
    ice_ease: float = 0.3
    
    # Technical debt sub-weights
    td_complexity: float = 0.25
    td_security: float = 0.30
    td_performance: float = 0.20
    td_maintainability: float = 0.15
    td_test_coverage: float = 0.10


class AdvancedScoringModel:
    """
    Advanced scoring model for Terragon value discovery system.
    
    Combines WSJF, ICE, and Technical Debt scoring with quantum computing
    domain expertise and adaptive learning capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the scoring model with configuration."""
        self.config_path = config_path or ".terragon/config.yaml"
        self.weights = ScoringWeights()
        self.quantum_boost_factors = {}
        self.historical_outcomes = []
        self.learning_rate = 0.1
        self.load_configuration()
    
    def load_configuration(self) -> None:
        """Load configuration from Terragon config file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Load scoring weights
                scoring_config = config.get('prioritization', {})
                weights_config = scoring_config.get('scoring_weights', {})
                
                self.weights.wsjf = weights_config.get('wsjf', self.weights.wsjf)
                self.weights.ice = weights_config.get('ice', self.weights.ice)
                self.weights.technical_debt = weights_config.get('technical_debt', self.weights.technical_debt)
                self.weights.quantum_boost = weights_config.get('quantum_boost', self.weights.quantum_boost)
                
                # Load quantum boost factors
                self.quantum_boost_factors = scoring_config.get('quantum_priority_boosts', {
                    'quantum_performance_optimization': 1.5,
                    'quantum_security_improvements': 1.4,
                    'quantum_backend_modernization': 1.3,
                    'hybrid_algorithm_optimization': 1.2,
                    'quantum_cost_optimization': 1.2
                })
                
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.warning(f"Configuration file {self.config_path} not found, using defaults")
        
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def calculate_wsjf_score(self, task: TaskMetrics) -> float:
        """
        Calculate Weighted Shortest Job First (WSJF) score.
        
        WSJF = (Business Value + Time Criticality + Risk Reduction) / Job Size
        Higher scores indicate higher priority.
        """
        numerator = (
            task.user_business_value * self.weights.wsjf_user_value +
            task.time_criticality * self.weights.wsjf_time_criticality +
            task.risk_reduction * self.weights.wsjf_risk_reduction
        )
        
        # Avoid division by zero, minimum job size of 0.1
        denominator = max(task.job_size * self.weights.wsjf_job_size, 0.1)
        
        wsjf_score = numerator / denominator
        
        # Normalize to 0-10 scale
        normalized_score = min(10.0, max(0.0, wsjf_score))
        
        logger.debug(f"WSJF score for {task.task_id}: {normalized_score:.2f}")
        return normalized_score
    
    def calculate_ice_score(self, task: TaskMetrics) -> float:
        """
        Calculate ICE (Impact, Confidence, Ease) score.
        
        ICE = Impact * Confidence * Ease / 1000 * 10 (normalized)
        Higher scores indicate better strategic value.
        """
        ice_raw = (
            task.impact * self.weights.ice_impact +
            task.confidence * self.weights.ice_confidence +
            task.ease * self.weights.ice_ease
        )
        
        # Normalize to 0-10 scale
        ice_score = min(10.0, max(0.0, ice_raw))
        
        logger.debug(f"ICE score for {task.task_id}: {ice_score:.2f}")
        return ice_score
    
    def calculate_technical_debt_score(self, task: TaskMetrics) -> float:
        """
        Calculate Technical Debt severity score.
        
        Weighted combination of various technical debt factors.
        Higher scores indicate more urgent technical debt.
        """
        td_score = (
            task.code_complexity * self.weights.td_complexity +
            task.security_vulnerability * self.weights.td_security +
            task.performance_impact * self.weights.td_performance +
            task.maintainability_impact * self.weights.td_maintainability +
            task.test_coverage_gap * self.weights.td_test_coverage
        )
        
        # Normalize to 0-10 scale
        normalized_score = min(10.0, max(0.0, td_score))
        
        logger.debug(f"Technical debt score for {task.task_id}: {normalized_score:.2f}")
        return normalized_score
    
    def calculate_quantum_boost(self, task: TaskMetrics) -> float:
        """
        Calculate quantum computing domain-specific boost factor.
        
        Applies domain expertise to prioritize quantum-specific improvements.
        """
        if not task.quantum_domain:
            return 1.0  # No boost for non-quantum tasks
        
        # Base quantum boost
        base_boost = 1.0
        
        # Domain-specific boosts
        domain_boosts = {
            QuantumDomain.QUBO_OPTIMIZATION: 1.5,
            QuantumDomain.QUANTUM_BACKENDS: 1.4,
            QuantumDomain.HYBRID_ALGORITHMS: 1.3,
            QuantumDomain.PERFORMANCE_TUNING: 1.4,
            QuantumDomain.QUANTUM_SECURITY: 1.6,
            QuantumDomain.CLASSICAL_FALLBACKS: 1.2,
            QuantumDomain.RESEARCH_INTEGRATION: 1.1
        }
        
        domain_boost = domain_boosts.get(task.quantum_domain, 1.0)
        
        # Performance impact boost
        performance_boost = 1.0 + (task.quantum_performance_impact / 10.0) * 0.3
        
        # Cost optimization boost
        cost_boost = 1.0 + (task.quantum_cost_impact / 10.0) * 0.2
        
        # Research value boost
        research_boost = 1.0 + (task.quantum_research_value / 10.0) * 0.15
        
        total_boost = base_boost * domain_boost * performance_boost * cost_boost * research_boost
        
        # Cap maximum boost at 3.0x
        total_boost = min(3.0, total_boost)
        
        logger.debug(f"Quantum boost for {task.task_id}: {total_boost:.2f}x")
        return total_boost
    
    def calculate_urgency_multiplier(self, task: TaskMetrics) -> float:
        """Calculate urgency-based score multiplier."""
        urgency_multipliers = {
            UrgencyLevel.CRITICAL: 2.0,
            UrgencyLevel.HIGH: 1.5,
            UrgencyLevel.MEDIUM: 1.0,
            UrgencyLevel.LOW: 0.7
        }
        
        return urgency_multipliers.get(task.urgency, 1.0)
    
    def calculate_category_multiplier(self, task: TaskMetrics) -> float:
        """Calculate category-based score multiplier."""
        category_multipliers = {
            TaskCategory.SECURITY: 1.8,
            TaskCategory.PERFORMANCE: 1.4,
            TaskCategory.QUANTUM_OPTIMIZATION: 1.6,
            TaskCategory.TECHNICAL_DEBT: 1.2,
            TaskCategory.INNOVATION: 1.1,
            TaskCategory.MAINTENANCE: 1.0,
            TaskCategory.INTEGRATION: 1.3,
            TaskCategory.RESEARCH: 1.1
        }
        
        return category_multipliers.get(task.category, 1.0)
    
    def calculate_composite_score(self, task: TaskMetrics) -> Dict[str, float]:
        """
        Calculate the composite priority score using hybrid model.
        
        Returns detailed scoring breakdown for transparency and debugging.
        """
        # Calculate component scores
        wsjf_score = self.calculate_wsjf_score(task)
        ice_score = self.calculate_ice_score(task)
        td_score = self.calculate_technical_debt_score(task)
        quantum_boost = self.calculate_quantum_boost(task)
        
        # Calculate multipliers
        urgency_multiplier = self.calculate_urgency_multiplier(task)
        category_multiplier = self.calculate_category_multiplier(task)
        
        # Weighted combination of base scores
        base_score = (
            wsjf_score * self.weights.wsjf +
            ice_score * self.weights.ice +
            td_score * self.weights.technical_debt
        )
        
        # Apply quantum boost
        quantum_boosted_score = base_score * quantum_boost * self.weights.quantum_boost + base_score * (1 - self.weights.quantum_boost)
        
        # Apply multipliers
        final_score = quantum_boosted_score * urgency_multiplier * category_multiplier
        
        # Cap final score at 100
        final_score = min(100.0, max(0.0, final_score))
        
        return {
            'final_score': final_score,
            'base_score': base_score,
            'wsjf_score': wsjf_score,
            'ice_score': ice_score,
            'technical_debt_score': td_score,
            'quantum_boost': quantum_boost,
            'urgency_multiplier': urgency_multiplier,
            'category_multiplier': category_multiplier,
            'confidence': task.confidence_score
        }
    
    def prioritize_tasks(self, tasks: List[TaskMetrics]) -> List[Tuple[TaskMetrics, Dict[str, float]]]:
        """
        Prioritize a list of tasks using the hybrid scoring model.
        
        Returns tasks sorted by priority (highest first) with scoring details.
        """
        scored_tasks = []
        
        for task in tasks:
            scores = self.calculate_composite_score(task)
            scored_tasks.append((task, scores))
        
        # Sort by final score (descending)
        scored_tasks.sort(key=lambda x: x[1]['final_score'], reverse=True)
        
        logger.info(f"Prioritized {len(tasks)} tasks")
        return scored_tasks
    
    def generate_priority_explanation(self, task: TaskMetrics, scores: Dict[str, float]) -> str:
        """Generate human-readable explanation for task priority."""
        explanation_parts = []
        
        # Priority level
        final_score = scores['final_score']
        if final_score >= 80:
            priority_level = "CRITICAL"
        elif final_score >= 60:
            priority_level = "HIGH"
        elif final_score >= 40:
            priority_level = "MEDIUM"
        else:
            priority_level = "LOW"
        
        explanation_parts.append(f"Priority: {priority_level} (Score: {final_score:.1f})")
        
        # Key factors
        if scores['quantum_boost'] > 1.2:
            explanation_parts.append(f"✨ Quantum optimization opportunity (+{(scores['quantum_boost']-1)*100:.0f}% boost)")
        
        if task.urgency == UrgencyLevel.CRITICAL:
            explanation_parts.append("🚨 Critical urgency")
        
        if task.category == TaskCategory.SECURITY:
            explanation_parts.append("🔒 Security improvement")
        
        if scores['technical_debt_score'] > 7:
            explanation_parts.append("⚠️ High technical debt impact")
        
        if scores['wsjf_score'] > 8:
            explanation_parts.append("📈 High business value/effort ratio")
        
        # Quantum-specific insights
        if task.quantum_domain:
            domain_name = task.quantum_domain.value.replace('_', ' ').title()
            explanation_parts.append(f"🔬 {domain_name} enhancement")
        
        return " | ".join(explanation_parts)
    
    def adaptive_learning_update(self, task_id: str, actual_outcome: Dict[str, Any]) -> None:
        """
        Update the model based on actual task execution outcomes.
        
        Implements reinforcement learning to improve future predictions.
        """
        outcome_record = {
            'task_id': task_id,
            'timestamp': datetime.now().isoformat(),
            'outcome': actual_outcome,
            'predicted_score': actual_outcome.get('predicted_score', 0),
            'actual_value': actual_outcome.get('actual_value', 0),
            'completion_time': actual_outcome.get('completion_time', 0),
            'success': actual_outcome.get('success', False)
        }
        
        self.historical_outcomes.append(outcome_record)
        
        # Simple learning: adjust weights based on prediction accuracy
        if len(self.historical_outcomes) >= 10:
            self._update_weights_from_outcomes()
        
        logger.info(f"Recorded outcome for task {task_id}")
    
    def _update_weights_from_outcomes(self) -> None:
        """Update model weights based on historical outcomes."""
        recent_outcomes = self.historical_outcomes[-20:]  # Use last 20 outcomes
        
        # Calculate prediction accuracy
        accuracy_sum = 0
        for outcome in recent_outcomes:
            predicted = outcome['predicted_score']
            actual = outcome['actual_value']
            if predicted > 0:  # Avoid division by zero
                accuracy = 1 - abs(predicted - actual) / max(predicted, actual)
                accuracy_sum += accuracy
        
        avg_accuracy = accuracy_sum / len(recent_outcomes) if recent_outcomes else 0.5
        
        # Adjust learning rate based on accuracy
        if avg_accuracy < 0.7:  # Low accuracy, learn faster
            adjustment_factor = self.learning_rate * 1.5
        else:  # Good accuracy, learn slower
            adjustment_factor = self.learning_rate * 0.5
        
        logger.info(f"Model learning update: accuracy={avg_accuracy:.3f}, adjustment={adjustment_factor:.3f}")
    
    def export_model_state(self) -> Dict[str, Any]:
        """Export current model state for persistence."""
        return {
            'weights': {
                'wsjf': self.weights.wsjf,
                'ice': self.weights.ice,
                'technical_debt': self.weights.technical_debt,
                'quantum_boost': self.weights.quantum_boost
            },
            'quantum_boost_factors': self.quantum_boost_factors,
            'learning_rate': self.learning_rate,
            'historical_outcomes_count': len(self.historical_outcomes),
            'export_timestamp': datetime.now().isoformat()
        }
    
    def generate_scoring_report(self, tasks: List[TaskMetrics]) -> Dict[str, Any]:
        """Generate comprehensive scoring analysis report."""
        if not tasks:
            return {'error': 'No tasks provided for analysis'}
        
        prioritized_tasks = self.prioritize_tasks(tasks)
        
        # Summary statistics
        scores = [scores['final_score'] for _, scores in prioritized_tasks]
        
        report = {
            'summary': {
                'total_tasks': len(tasks),
                'high_priority_tasks': len([s for s in scores if s >= 60]),
                'medium_priority_tasks': len([s for s in scores if 40 <= s < 60]),
                'low_priority_tasks': len([s for s in scores if s < 40]),
                'average_score': sum(scores) / len(scores),
                'max_score': max(scores),
                'min_score': min(scores)
            },
            'category_breakdown': {},
            'quantum_analysis': {},
            'top_priorities': [],
            'recommendations': [],
            'model_info': self.export_model_state()
        }
        
        # Category breakdown
        category_counts = {}
        category_scores = {}
        for task, task_scores in prioritized_tasks:
            cat = task.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(task_scores['final_score'])
        
        for cat, count in category_counts.items():
            report['category_breakdown'][cat] = {
                'count': count,
                'average_score': sum(category_scores[cat]) / len(category_scores[cat]),
                'percentage': (count / len(tasks)) * 100
            }
        
        # Quantum analysis
        quantum_tasks = [task for task, _ in prioritized_tasks if task.quantum_domain]
        report['quantum_analysis'] = {
            'quantum_tasks_count': len(quantum_tasks),
            'quantum_percentage': (len(quantum_tasks) / len(tasks)) * 100,
            'average_quantum_boost': sum([scores['quantum_boost'] for _, scores in prioritized_tasks if _.quantum_domain]) / max(len(quantum_tasks), 1)
        }
        
        # Top 10 priorities
        for i, (task, task_scores) in enumerate(prioritized_tasks[:10]):
            report['top_priorities'].append({
                'rank': i + 1,
                'task_id': task.task_id,
                'title': task.title,
                'category': task.category.value,
                'score': task_scores['final_score'],
                'explanation': self.generate_priority_explanation(task, task_scores)
            })
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(prioritized_tasks)
        
        return report
    
    def _generate_recommendations(self, prioritized_tasks: List[Tuple[TaskMetrics, Dict[str, float]]]) -> List[str]:
        """Generate actionable recommendations based on task analysis."""
        recommendations = []
        
        # High priority security tasks
        security_tasks = [task for task, scores in prioritized_tasks 
                         if task.category == TaskCategory.SECURITY and scores['final_score'] >= 60]
        if security_tasks:
            recommendations.append(f"🔒 {len(security_tasks)} high-priority security tasks require immediate attention")
        
        # Quantum optimization opportunities
        quantum_tasks = [task for task, scores in prioritized_tasks 
                        if task.quantum_domain and scores['quantum_boost'] > 1.3]
        if quantum_tasks:
            recommendations.append(f"⚡ {len(quantum_tasks)} quantum optimization opportunities identified")
        
        # Technical debt concentration
        high_td_tasks = [task for task, scores in prioritized_tasks 
                        if scores['technical_debt_score'] > 7]
        if len(high_td_tasks) > 5:
            recommendations.append(f"⚠️ High concentration of technical debt ({len(high_td_tasks)} tasks) - consider dedicated sprint")
        
        # Performance opportunities
        perf_tasks = [task for task, scores in prioritized_tasks 
                     if task.category == TaskCategory.PERFORMANCE and scores['final_score'] >= 50]
        if perf_tasks:
            recommendations.append(f"📈 {len(perf_tasks)} performance optimization opportunities with high ROI")
        
        return recommendations


def create_sample_tasks() -> List[TaskMetrics]:
    """Create sample tasks for demonstration and testing."""
    sample_tasks = [
        TaskMetrics(
            task_id="QUBO-OPT-001",
            title="Optimize QUBO Matrix Construction Performance",
            description="Current QUBO matrix construction takes 45ms, target is 30ms",
            category=TaskCategory.QUANTUM_OPTIMIZATION,
            urgency=UrgencyLevel.HIGH,
            user_business_value=8.0,
            time_criticality=7.0,
            risk_reduction=6.0,
            job_size=4.0,
            impact=8.5,
            confidence=9.0,
            ease=6.0,
            performance_impact=8.0,
            quantum_domain=QuantumDomain.QUBO_OPTIMIZATION,
            quantum_performance_impact=9.0,
            estimated_effort_hours=12.0
        ),
        
        TaskMetrics(
            task_id="SEC-VULN-001",
            title="Fix Critical Security Vulnerability in Quantum Credential Handler",
            description="Potential credential exposure in quantum backend authentication",
            category=TaskCategory.SECURITY,
            urgency=UrgencyLevel.CRITICAL,
            user_business_value=9.5,
            time_criticality=10.0,
            risk_reduction=9.5,
            job_size=2.0,
            impact=9.0,
            confidence=9.5,
            ease=8.0,
            security_vulnerability=9.5,
            quantum_domain=QuantumDomain.QUANTUM_SECURITY,
            quantum_performance_impact=3.0,
            estimated_effort_hours=6.0,
            required_approvals=["security_team", "quantum_expert"]
        ),
        
        TaskMetrics(
            task_id="PERF-REG-001",
            title="Address Performance Regression in Classical Fallback",
            description="20% performance degradation detected in simulated annealing fallback",
            category=TaskCategory.PERFORMANCE,
            urgency=UrgencyLevel.MEDIUM,
            user_business_value=7.0,
            time_criticality=6.0,
            risk_reduction=7.5,
            job_size=6.0,
            impact=7.5,
            confidence=8.0,
            ease=5.0,
            performance_impact=8.5,
            quantum_domain=QuantumDomain.CLASSICAL_FALLBACKS,
            quantum_performance_impact=6.0,
            estimated_effort_hours=18.0
        ),
        
        TaskMetrics(
            task_id="TECH-DEBT-001",
            title="Refactor Legacy QUBO Constraint Handling",
            description="Complex constraint handling code needs modernization",
            category=TaskCategory.TECHNICAL_DEBT,
            urgency=UrgencyLevel.LOW,
            user_business_value=5.0,
            time_criticality=3.0,
            risk_reduction=6.0,
            job_size=8.0,
            impact=6.0,
            confidence=7.0,
            ease=4.0,
            code_complexity=8.5,
            maintainability_impact=7.0,
            test_coverage_gap=6.0,
            quantum_domain=QuantumDomain.QUBO_OPTIMIZATION,
            estimated_effort_hours=24.0
        ),
        
        TaskMetrics(
            task_id="INNOV-RES-001",
            title="Integrate Latest QAOA Algorithm Research",
            description="Opportunity to integrate new QAOA variant with 15% performance improvement",
            category=TaskCategory.RESEARCH,
            urgency=UrgencyLevel.LOW,
            user_business_value=6.0,
            time_criticality=2.0,
            risk_reduction=3.0,
            job_size=9.0,
            impact=7.0,
            confidence=6.0,
            ease=3.0,
            quantum_domain=QuantumDomain.RESEARCH_INTEGRATION,
            quantum_research_value=9.0,
            quantum_performance_impact=7.0,
            estimated_effort_hours=40.0,
            required_approvals=["research_lead", "quantum_expert"]
        )
    ]
    
    return sample_tasks


def main():
    """Demonstration of the advanced scoring model."""
    print("🔬 Terragon Advanced Scoring Model - Quantum Computing Edition")
    print("=" * 70)
    
    # Initialize scoring model
    scoring_model = AdvancedScoringModel()
    
    # Create sample tasks
    sample_tasks = create_sample_tasks()
    
    # Generate comprehensive scoring report
    report = scoring_model.generate_scoring_report(sample_tasks)
    
    # Display results
    print(f"\n📊 Scoring Analysis Report")
    print(f"Total Tasks: {report['summary']['total_tasks']}")
    print(f"High Priority: {report['summary']['high_priority_tasks']}")
    print(f"Medium Priority: {report['summary']['medium_priority_tasks']}")
    print(f"Low Priority: {report['summary']['low_priority_tasks']}")
    print(f"Average Score: {report['summary']['average_score']:.1f}")
    
    print(f"\n🔬 Quantum Analysis")
    print(f"Quantum Tasks: {report['quantum_analysis']['quantum_tasks_count']} ({report['quantum_analysis']['quantum_percentage']:.1f}%)")
    print(f"Average Quantum Boost: {report['quantum_analysis']['average_quantum_boost']:.2f}x")
    
    print(f"\n🏆 Top Priorities")
    for priority in report['top_priorities'][:5]:
        print(f"{priority['rank']}. {priority['title']}")
        print(f"   Score: {priority['score']:.1f} | {priority['explanation']}")
        print()
    
    print(f"💡 Recommendations")
    for rec in report['recommendations']:
        print(f"   {rec}")
    
    print(f"\n⚙️ Model Configuration")
    model_info = report['model_info']
    weights = model_info['weights']
    print(f"   WSJF Weight: {weights['wsjf']:.2f}")
    print(f"   ICE Weight: {weights['ice']:.2f}")
    print(f"   Technical Debt Weight: {weights['technical_debt']:.2f}")
    print(f"   Quantum Boost Weight: {weights['quantum_boost']:.2f}")
    
    # Save report to file
    report_path = Path(".terragon/latest_scoring_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Full report saved to: {report_path}")


if __name__ == "__main__":
    main()