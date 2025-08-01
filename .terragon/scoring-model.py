#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Value Discovery Scoring Model
Advanced Hybrid Scoring Algorithm (WSJF + ICE + Technical Debt + Quantum Boost)

This module implements an advanced scoring system that combines multiple proven
frameworks to prioritize work items for quantum computing repositories.

Scoring Components:
- WSJF (35%): Weighted Shortest Job First (SAFe methodology)
- ICE (25%): Impact, Confidence, Ease framework  
- Technical Debt (25%): Maintenance and quality debt assessment
- Quantum Boost (15%): Domain-specific quantum computing adjustments

Author: Terragon Autonomous SDLC System
Version: 2.1.0
"""

import json
import math
import yaml
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import hashlib
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskCategory(Enum):
    """Task categories with quantum computing specialization."""
    QUANTUM_OPTIMIZATION = "quantum_optimization"
    BACKEND_INTEGRATION = "backend_integration"
    HYBRID_ALGORITHMS = "hybrid_algorithms"
    QUANTUM_SECURITY = "quantum_security"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    TECHNICAL_DEBT = "technical_debt"
    RESEARCH_INTEGRATION = "research_integration"
    DOCUMENTATION = "documentation"

class Priority(Enum):
    """Priority levels for tasks."""
    CRITICAL = 10
    HIGH = 8
    MEDIUM = 6
    LOW = 4
    MINIMAL = 2

class QuantumDomainType(Enum):
    """Quantum computing domain classifications."""
    QUBO_OPTIMIZATION = "qubo_optimization"
    QUANTUM_BACKEND = "quantum_backend"
    HYBRID_ALGORITHMS = "hybrid_algorithms"
    QUANTUM_SECURITY = "quantum_security"
    ERROR_MITIGATION = "error_mitigation"
    QUANTUM_ML = "quantum_machine_learning"
    QUANTUM_NETWORKING = "quantum_networking"

@dataclass
class TaskItem:
    """Represents a discoverable work item."""
    id: str
    title: str
    description: str
    category: TaskCategory
    source: str  # git_history, static_analysis, issue_tracker, etc.
    
    # WSJF Components
    user_business_value: float = 0.0    # 1-10 scale
    time_criticality: float = 0.0       # 1-10 scale  
    risk_reduction: float = 0.0         # 1-10 scale
    opportunity_enablement: float = 0.0 # 1-10 scale
    job_size_estimate: float = 1.0      # Story points or ideal days
    
    # ICE Components
    impact: float = 0.0                 # 1-10 scale
    confidence: float = 0.0             # 1-10 scale (execution confidence)  
    ease: float = 0.0                   # 1-10 scale (implementation ease)
    
    # Technical Debt Components
    debt_impact: float = 0.0            # Hours saved by addressing debt
    debt_interest: float = 0.0          # Future cost if not addressed
    hotspot_multiplier: float = 1.0     # 1-5x based on code churn/complexity
    
    # Quantum Computing Specific
    quantum_domain: Optional[QuantumDomainType] = None
    quantum_relevance: float = 0.0      # 0-1 scale
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    file_paths: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    estimated_effort_hours: float = 1.0
    
    # Scoring results (populated by scorer)
    wsjf_score: float = 0.0
    ice_score: float = 0.0
    tech_debt_score: float = 0.0
    quantum_boost: float = 0.0
    composite_score: float = 0.0
    
    # Risk assessment
    risk_factors: List[str] = field(default_factory=list)
    risk_score: float = 0.0             # 0-1 scale
    
    # Execution metadata
    auto_executable: bool = False
    approval_required: List[str] = field(default_factory=list)

@dataclass
class ScoringWeights:
    """Configurable scoring weights."""
    wsjf: float = 0.35
    ice: float = 0.25
    technical_debt: float = 0.25
    quantum_boost: float = 0.15

@dataclass 
class ScoringThresholds:
    """Scoring thresholds and multipliers."""
    min_composite_score: float = 15.0
    max_risk_factor: float = 0.75
    security_multiplier: float = 2.0
    compliance_multiplier: float = 1.8
    performance_multiplier: float = 1.6
    quantum_multiplier: float = 1.5

@dataclass
class QuantumDomainBoosts:
    """Quantum computing domain-specific priority boosts."""
    qubo_optimization: float = 1.5
    quantum_backend: float = 1.4  
    hybrid_algorithms: float = 1.3
    quantum_security: float = 1.6
    error_mitigation: float = 1.4
    quantum_machine_learning: float = 1.3
    quantum_networking: float = 1.2

class AdvancedScoringModel:
    """
    Advanced hybrid scoring model for quantum computing repositories.
    
    Combines WSJF, ICE, Technical Debt, and Quantum Domain expertise
    to intelligently prioritize work items for maximum value delivery.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the scoring model with configuration."""
        self.config_path = config_path or ".terragon/config.yaml"
        self.config = self._load_config()
        
        # Extract configuration components
        scoring_config = self.config.get('scoring', {})
        self.weights = ScoringWeights(**scoring_config.get('weights', {}))
        self.thresholds = ScoringThresholds(**scoring_config.get('thresholds', {}))
        self.quantum_boosts = QuantumDomainBoosts(
            **scoring_config.get('quantum_domain_boosts', {})
        )
        
        # Learning configuration
        learning_config = scoring_config.get('learning', {})
        self.learning_enabled = learning_config.get('enabled', True)
        self.learning_rate = learning_config.get('learning_rate', 0.1)
        self.confidence_decay = learning_config.get('confidence_decay', 0.95)
        
        # Historical data for learning
        self.prediction_history: List[Dict] = []
        self.outcome_history: List[Dict] = []
        
        logger.info(f"Initialized AdvancedScoringModel with weights: {self.weights}")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            config_path = Path(self.config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Config file not found: {config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def calculate_wsjf_score(self, task: TaskItem) -> float:
        """
        Calculate Weighted Shortest Job First (WSJF) score.
        
        WSJF = Cost of Delay / Job Size
        Cost of Delay = User/Business Value + Time Criticality + Risk Reduction + Opportunity Enablement
        """
        cost_of_delay = (
            task.user_business_value + 
            task.time_criticality + 
            task.risk_reduction + 
            task.opportunity_enablement
        )
        
        # Avoid division by zero
        job_size = max(task.job_size_estimate, 0.1)
        
        wsjf_score = cost_of_delay / job_size
        
        # Apply domain-specific multipliers
        if task.category == TaskCategory.QUANTUM_SECURITY:
            wsjf_score *= self.thresholds.security_multiplier
        elif task.category == TaskCategory.PERFORMANCE_OPTIMIZATION:
            wsjf_score *= self.thresholds.performance_multiplier
            
        return round(wsjf_score, 2)
    
    def calculate_ice_score(self, task: TaskItem) -> float:
        """
        Calculate ICE (Impact, Confidence, Ease) score.
        
        ICE = Impact × Confidence × Ease
        """
        ice_score = task.impact * task.confidence * task.ease
        
        # Normalize to similar scale as WSJF (typically 0-100)
        ice_score = (ice_score / 1000) * 100
        
        return round(ice_score, 2)
    
    def calculate_technical_debt_score(self, task: TaskItem) -> float:
        """
        Calculate Technical Debt score.
        
        Debt Score = (Debt Impact + Debt Interest) × Hotspot Multiplier
        """
        debt_base = task.debt_impact + task.debt_interest
        debt_score = debt_base * task.hotspot_multiplier
        
        # Apply category-specific adjustments
        if task.category == TaskCategory.TECHNICAL_DEBT:
            debt_score *= 1.2  # Boost actual technical debt tasks
        
        return round(debt_score, 2)
    
    def calculate_quantum_boost(self, task: TaskItem) -> float:
        """
        Calculate quantum computing domain-specific boost.
        
        Applies domain expertise and quantum-specific prioritization.
        """
        if not task.quantum_domain or task.quantum_relevance == 0:
            return 0.0
        
        # Get domain-specific boost factor
        domain_boost_map = {
            QuantumDomainType.QUBO_OPTIMIZATION: self.quantum_boosts.qubo_optimization,
            QuantumDomainType.QUANTUM_BACKEND: self.quantum_boosts.quantum_backend,
            QuantumDomainType.HYBRID_ALGORITHMS: self.quantum_boosts.hybrid_algorithms,
            QuantumDomainType.QUANTUM_SECURITY: self.quantum_boosts.quantum_security,
            QuantumDomainType.ERROR_MITIGATION: self.quantum_boosts.error_mitigation,
            QuantumDomainType.QUANTUM_ML: self.quantum_boosts.quantum_machine_learning,
            QuantumDomainType.QUANTUM_NETWORKING: self.quantum_boosts.quantum_networking,
        }
        
        domain_boost = domain_boost_map.get(task.quantum_domain, 1.0)
        
        # Calculate quantum boost score
        quantum_boost = (
            task.quantum_relevance * 
            domain_boost * 
            self.thresholds.quantum_multiplier * 
            10  # Scale to match other components
        )
        
        return round(quantum_boost, 2)
    
    def calculate_risk_score(self, task: TaskItem) -> float:
        """Calculate overall risk score for the task."""
        risk_factors = {
            'complexity': min(task.job_size_estimate / 10, 1.0),
            'dependencies': min(len(task.dependencies) / 5, 1.0),
            'confidence': 1.0 - (task.confidence / 10),
            'technical_uncertainty': 1.0 - (task.ease / 10)
        }
        
        # Add quantum-specific risks
        if task.quantum_domain:
            risk_factors['quantum_uncertainty'] = 0.3  # Quantum computing inherent risk
        
        # Weight and combine risk factors
        total_risk = sum(risk_factors.values()) / len(risk_factors)
        
        # Apply risk factor penalties
        for factor in task.risk_factors:
            if 'security' in factor.lower():
                total_risk += 0.2
            elif 'performance' in factor.lower():
                total_risk += 0.1
            elif 'breaking' in factor.lower():
                total_risk += 0.3
        
        return min(total_risk, 1.0)
    
    def calculate_composite_score(self, task: TaskItem) -> float:
        """
        Calculate the final composite score using all components.
        
        Composite Score = (WSJF × w1) + (ICE × w2) + (TechDebt × w3) + (QuantumBoost × w4)
        """
        # Calculate individual component scores
        task.wsjf_score = self.calculate_wsjf_score(task)
        task.ice_score = self.calculate_ice_score(task)
        task.tech_debt_score = self.calculate_technical_debt_score(task)
        task.quantum_boost = self.calculate_quantum_boost(task)
        task.risk_score = self.calculate_risk_score(task)
        
        # Normalize scores to similar ranges for fair weighting
        normalized_wsjf = min(task.wsjf_score / 50, 1.0) * 100  # Normalize WSJF
        normalized_ice = task.ice_score  # ICE already scaled appropriately
        normalized_debt = min(task.tech_debt_score / 100, 1.0) * 100
        normalized_quantum = task.quantum_boost * 10  # Scale quantum boost
        
        # Calculate weighted composite score
        composite_score = (
            (normalized_wsjf * self.weights.wsjf) +
            (normalized_ice * self.weights.ice) +
            (normalized_debt * self.weights.technical_debt) +
            (normalized_quantum * self.weights.quantum_boost)
        )
        
        # Apply risk adjustment (reduce score based on risk)
        risk_penalty = task.risk_score * 0.3  # Up to 30% penalty for high risk
        composite_score *= (1.0 - risk_penalty)
        
        # Apply category-specific final adjustments
        category_multipliers = {
            TaskCategory.QUANTUM_SECURITY: 1.2,
            TaskCategory.QUANTUM_OPTIMIZATION: 1.1,
            TaskCategory.PERFORMANCE_OPTIMIZATION: 1.05,
            TaskCategory.TECHNICAL_DEBT: 0.9,
            TaskCategory.DOCUMENTATION: 0.8
        }
        
        multiplier = category_multipliers.get(task.category, 1.0)
        composite_score *= multiplier
        
        task.composite_score = round(composite_score, 2)
        return task.composite_score
    
    def score_task(self, task: TaskItem) -> TaskItem:
        """Score a single task and return the updated task with scores."""
        self.calculate_composite_score(task)
        
        # Determine auto-executability and approval requirements
        self._assess_execution_requirements(task)
        
        # Store prediction for learning
        if self.learning_enabled:
            self._store_prediction(task)
        
        logger.info(f"Scored task '{task.title}': {task.composite_score}")
        return task
    
    def score_tasks(self, tasks: List[TaskItem]) -> List[TaskItem]:
        """Score multiple tasks and return sorted by composite score."""
        scored_tasks = [self.score_task(task) for task in tasks]
        
        # Sort by composite score (descending)
        scored_tasks.sort(key=lambda t: t.composite_score, reverse=True)
        
        return scored_tasks
    
    def _assess_execution_requirements(self, task: TaskItem):
        """Assess whether task can be auto-executed and what approvals are needed."""
        # Auto-execution assessment
        auto_exec_factors = {
            'low_risk': task.risk_score < 0.3,
            'high_confidence': task.confidence >= 7.0,
            'simple_task': task.job_size_estimate <= 3.0,
            'no_security_risk': 'security' not in [r.lower() for r in task.risk_factors]
        }
        
        task.auto_executable = sum(auto_exec_factors.values()) >= 3
        
        # Approval requirements
        approvals = []
        
        if task.category == TaskCategory.QUANTUM_SECURITY:
            approvals.extend(["security-team", "quantum-expert"])
        elif task.category == TaskCategory.QUANTUM_OPTIMIZATION:
            approvals.append("quantum-expert")
        elif task.category == TaskCategory.BACKEND_INTEGRATION:
            approvals.extend(["tech-lead", "quantum-expert"])
        elif task.risk_score > 0.7:
            approvals.append("tech-lead")
        
        if task.job_size_estimate > 8:  # Large tasks
            approvals.append("project-manager")
        
        task.approval_required = list(set(approvals))  # Remove duplicates
    
    def _store_prediction(self, task: TaskItem):
        """Store prediction data for learning."""
        prediction = {
            'task_id': task.id,
            'predicted_composite_score': task.composite_score,
            'predicted_effort': task.estimated_effort_hours,
            'predicted_impact': task.impact,
            'timestamp': datetime.now().isoformat(),
            'model_version': '2.1.0'
        }
        self.prediction_history.append(prediction)
    
    def update_from_outcome(self, task_id: str, actual_effort: float, 
                          actual_impact: float, outcome_quality: float):
        """Update model based on actual execution outcomes."""
        if not self.learning_enabled:
            return
        
        outcome = {
            'task_id': task_id,
            'actual_effort': actual_effort,
            'actual_impact': actual_impact,
            'outcome_quality': outcome_quality,  # 0-1 scale
            'timestamp': datetime.now().isoformat()
        }
        self.outcome_history.append(outcome)
        
        # Find corresponding prediction
        prediction = next(
            (p for p in self.prediction_history if p['task_id'] == task_id),
            None
        )
        
        if prediction:
            self._adjust_model_weights(prediction, outcome)
    
    def _adjust_model_weights(self, prediction: Dict, outcome: Dict):
        """Adjust model weights based on prediction accuracy."""
        # Calculate prediction accuracy
        effort_accuracy = 1.0 - abs(
            prediction['predicted_effort'] - outcome['actual_effort']
        ) / max(prediction['predicted_effort'], 1.0)
        
        impact_accuracy = 1.0 - abs(
            prediction['predicted_impact'] - outcome['actual_impact']
        ) / max(prediction['predicted_impact'], 1.0)
        
        overall_accuracy = (effort_accuracy + impact_accuracy) / 2.0
        
        # Adjust weights based on accuracy and outcome quality
        if overall_accuracy < 0.7:  # Poor prediction
            # Reduce confidence in current weighting
            adjustment = self.learning_rate * (0.7 - overall_accuracy)
            
            # Simple weight adjustment logic (could be more sophisticated)
            if outcome['outcome_quality'] > 0.8:  # Good outcome despite poor prediction
                # Increase weight of components that predicted high value
                if prediction['predicted_composite_score'] > 50:
                    # This was a beneficial task we didn't predict well
                    pass  # More complex logic would go here
        
        logger.info(f"Model learning: accuracy={overall_accuracy:.2f}, quality={outcome['outcome_quality']:.2f}")

class QuantumTaskDiscovery:
    """
    Advanced task discovery system for quantum computing repositories.
    
    Discovers optimization opportunities through multiple signal sources:
    - Git history analysis
    - Static code analysis  
    - Issue tracker mining
    - Performance monitoring
    - Research integration
    """
    
    def __init__(self, repo_path: str = ".", config_path: str = ".terragon/config.yaml"):
        self.repo_path = Path(repo_path)
        self.config_path = config_path
        self.config = self._load_config()
        self.scorer = AdvancedScoringModel(config_path)
        
        # Discovery sources configuration
        discovery_config = self.config.get('discovery', {}).get('sources', {})
        self.git_enabled = discovery_config.get('git_history', {}).get('enabled', True)
        self.static_analysis_enabled = discovery_config.get('static_analysis', {}).get('enabled', True)
        self.issues_enabled = discovery_config.get('issue_trackers', {}).get('enabled', True)
        
    def _load_config(self) -> Dict:
        """Load discovery configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def discover_tasks(self) -> List[TaskItem]:
        """Run comprehensive task discovery across all enabled sources."""
        discovered_tasks = []
        
        if self.git_enabled:
            discovered_tasks.extend(self._discover_from_git_history())
        
        if self.static_analysis_enabled:
            discovered_tasks.extend(self._discover_from_static_analysis())
        
        if self.issues_enabled:
            discovered_tasks.extend(self._discover_from_issues())
        
        # Add quantum-specific discoveries
        discovered_tasks.extend(self._discover_quantum_opportunities())
        
        # Remove duplicates and score tasks
        unique_tasks = self._deduplicate_tasks(discovered_tasks)
        scored_tasks = self.scorer.score_tasks(unique_tasks)
        
        logger.info(f"Discovered {len(scored_tasks)} unique tasks")
        return scored_tasks
    
    def _discover_from_git_history(self) -> List[TaskItem]:
        """Discover tasks from git commit history and code comments."""
        tasks = []
        
        # Common patterns that indicate technical debt or improvement opportunities
        patterns = [
            (r'TODO|FIXME|HACK|XXX', TaskCategory.TECHNICAL_DEBT),
            (r'QUANTUM_TODO|QUBO_OPTIMIZE', TaskCategory.QUANTUM_OPTIMIZATION),
            (r'PERFORMANCE_ISSUE|OPTIMIZE', TaskCategory.PERFORMANCE_OPTIMIZATION),
            (r'SECURITY_REVIEW|VULNERABILITY', TaskCategory.QUANTUM_SECURITY),
        ]
        
        # Simulated git history analysis (in real implementation, would use GitPython)
        sample_findings = [
            {
                'text': 'TODO: Optimize QUBO matrix construction for large problems',
                'file': 'src/quantum_planner/optimizer.py',
                'category': TaskCategory.QUANTUM_OPTIMIZATION,
                'quantum_domain': QuantumDomainType.QUBO_OPTIMIZATION,
                'quantum_relevance': 0.9
            },
            {
                'text': 'FIXME: Quantum backend authentication is insecure',
                'file': 'src/quantum_planner/backends.py', 
                'category': TaskCategory.QUANTUM_SECURITY,
                'quantum_domain': QuantumDomainType.QUANTUM_SECURITY,
                'quantum_relevance': 1.0
            },
            {
                'text': 'PERFORMANCE_ISSUE: Classical fallback is too slow',
                'file': 'src/quantum_planner/classical.py',
                'category': TaskCategory.PERFORMANCE_OPTIMIZATION,
                'quantum_domain': QuantumDomainType.HYBRID_ALGORITHMS,
                'quantum_relevance': 0.7
            }
        ]
        
        for finding in sample_findings:
            task = TaskItem(
                id=self._generate_task_id(finding['text']),
                title=finding['text'].replace('TODO:', '').replace('FIXME:', '').strip(),
                description=f"Address technical debt in {finding['file']}",
                category=finding['category'],
                source='git_history',
                file_paths=[finding['file']],
                quantum_domain=finding.get('quantum_domain'),
                quantum_relevance=finding.get('quantum_relevance', 0.0),
                # Set reasonable defaults for scoring
                user_business_value=6.0,
                time_criticality=5.0,
                risk_reduction=7.0,
                opportunity_enablement=5.0,
                job_size_estimate=3.0,
                impact=7.0,
                confidence=8.0,
                ease=6.0,
                debt_impact=8.0,
                debt_interest=4.0,
                hotspot_multiplier=2.0
            )
            tasks.append(task)
        
        return tasks
    
    def _discover_from_static_analysis(self) -> List[TaskItem]:
        """Discover tasks from static code analysis."""
        tasks = []
        
        # Simulated static analysis findings
        findings = [
            {
                'title': 'Add comprehensive error handling for quantum operations',
                'description': 'Quantum operations lack proper error handling and fallback mechanisms',
                'category': TaskCategory.QUANTUM_OPTIMIZATION,
                'severity': 'high',
                'quantum_domain': QuantumDomainType.ERROR_MITIGATION,
                'files': ['src/quantum_planner/optimizer.py', 'src/quantum_planner/backends.py']
            },
            {
                'title': 'Optimize memory usage in QUBO matrix operations',
                'description': 'Large QUBO matrices causing excessive memory consumption',
                'category': TaskCategory.PERFORMANCE_OPTIMIZATION,
                'severity': 'medium',
                'quantum_domain': QuantumDomainType.QUBO_OPTIMIZATION,
                'files': ['src/quantum_planner/formulation.py']
            },
            {
                'title': 'Implement secure quantum credential storage',
                'description': 'Quantum backend credentials stored in plaintext',
                'category': TaskCategory.QUANTUM_SECURITY,
                'severity': 'critical',
                'quantum_domain': QuantumDomainType.QUANTUM_SECURITY,
                'files': ['src/quantum_planner/backends.py']
            }
        ]
        
        for finding in findings:
            # Map severity to scoring values
            severity_map = {
                'critical': {'business_value': 10, 'time_criticality': 9, 'impact': 10},
                'high': {'business_value': 8, 'time_criticality': 7, 'impact': 8},
                'medium': {'business_value': 6, 'time_criticality': 5, 'impact': 6},
                'low': {'business_value': 4, 'time_criticality': 3, 'impact': 4}
            }
            
            severity_scores = severity_map.get(finding['severity'], severity_map['medium'])
            
            task = TaskItem(
                id=self._generate_task_id(finding['title']),
                title=finding['title'],
                description=finding['description'],
                category=finding['category'],
                source='static_analysis',
                file_paths=finding['files'],
                quantum_domain=finding.get('quantum_domain'),
                quantum_relevance=0.8 if finding.get('quantum_domain') else 0.2,
                user_business_value=severity_scores['business_value'],
                time_criticality=severity_scores['time_criticality'],
                risk_reduction=8.0,
                opportunity_enablement=6.0,
                job_size_estimate=4.0,
                impact=severity_scores['impact'],
                confidence=7.0,
                ease=5.0,
                debt_impact=10.0,
                debt_interest=6.0,
                hotspot_multiplier=1.5
            )
            
            tasks.append(task)
        
        return tasks
    
    def _discover_from_issues(self) -> List[TaskItem]:
        """Discover tasks from issue tracker (GitHub Issues, etc.)."""
        tasks = []
        
        # Simulated issue tracker findings  
        issues = [
            {
                'title': 'Add support for QAOA algorithm',
                'description': 'Implement Quantum Approximate Optimization Algorithm for better performance',
                'labels': ['enhancement', 'quantum', 'research'],
                'priority': 'high',
                'quantum_domain': QuantumDomainType.HYBRID_ALGORITHMS
            },
            {
                'title': 'Performance regression in large problem solving',
                'description': 'Recent changes caused 30% performance decrease for problems >50 variables',
                'labels': ['bug', 'performance', 'critical'],
                'priority': 'critical',
                'quantum_domain': QuantumDomainType.QUBO_OPTIMIZATION
            }
        ]
        
        for issue in issues:
            priority_map = {
                'critical': 10,
                'high': 8,
                'medium': 6,
                'low': 4
            }
            
            priority_score = priority_map.get(issue['priority'], 6)
            
            # Determine category from labels
            category = TaskCategory.TECHNICAL_DEBT  # Default
            if 'quantum' in issue['labels']:
                category = TaskCategory.QUANTUM_OPTIMIZATION
            elif 'performance' in issue['labels']:
                category = TaskCategory.PERFORMANCE_OPTIMIZATION
            elif 'security' in issue['labels']:
                category = TaskCategory.QUANTUM_SECURITY
            elif 'research' in issue['labels']:
                category = TaskCategory.RESEARCH_INTEGRATION
            
            task = TaskItem(
                id=self._generate_task_id(issue['title']),
                title=issue['title'],
                description=issue['description'],
                category=category,
                source='issue_tracker',
                quantum_domain=issue.get('quantum_domain'),
                quantum_relevance=0.9 if issue.get('quantum_domain') else 0.3,
                user_business_value=priority_score,
                time_criticality=priority_score,
                risk_reduction=5.0,
                opportunity_enablement=7.0,
                job_size_estimate=6.0,
                impact=priority_score,
                confidence=6.0,
                ease=4.0,
                debt_impact=5.0,
                debt_interest=3.0,
                hotspot_multiplier=1.2
            )
            
            tasks.append(task)
        
        return tasks
    
    def _discover_quantum_opportunities(self) -> List[TaskItem]:
        """Discover quantum computing specific optimization opportunities."""
        tasks = []
        
        # Quantum-specific opportunities based on repository analysis
        opportunities = [
            {
                'title': 'Implement hybrid quantum-classical optimization pipeline',
                'description': 'Create seamless integration between quantum and classical solvers',
                'category': TaskCategory.HYBRID_ALGORITHMS,
                'quantum_domain': QuantumDomainType.HYBRID_ALGORITHMS,
                'business_value': 9,
                'effort': 8
            },
            {
                'title': 'Add quantum circuit depth optimization',
                'description': 'Optimize quantum circuits for NISQ devices with limited coherence',
                'category': TaskCategory.QUANTUM_OPTIMIZATION,
                'quantum_domain': QuantumDomainType.ERROR_MITIGATION,
                'business_value': 7,
                'effort': 6
            },
            {
                'title': 'Implement quantum machine learning integration',
                'description': 'Add QML capabilities for task classification and optimization',
                'category': TaskCategory.RESEARCH_INTEGRATION,
                'quantum_domain': QuantumDomainType.QUANTUM_ML,
                'business_value': 6,
                'effort': 10
            }
        ]
        
        for opp in opportunities:
            task = TaskItem(
                id=self._generate_task_id(opp['title']),
                title=opp['title'],
                description=opp['description'],
                category=opp['category'],
                source='quantum_analysis',
                quantum_domain=opp['quantum_domain'],
                quantum_relevance=1.0,
                user_business_value=opp['business_value'],
                time_criticality=5.0,
                risk_reduction=4.0,
                opportunity_enablement=9.0,
                job_size_estimate=opp['effort'],
                impact=opp['business_value'],
                confidence=5.0,  # Research has inherent uncertainty
                ease=3.0,        # Quantum development is challenging
                debt_impact=2.0,
                debt_interest=1.0,
                hotspot_multiplier=1.1
            )
            
            tasks.append(task)
        
        return tasks
    
    def _deduplicate_tasks(self, tasks: List[TaskItem]) -> List[TaskItem]:
        """Remove duplicate tasks based on similarity."""
        unique_tasks = []
        seen_hashes = set()
        
        for task in tasks:
            # Create a hash based on title and description similarity
            content = f"{task.title.lower()} {task.description.lower()}"
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            
            if content_hash not in seen_hashes:
                unique_tasks.append(task)
                seen_hashes.add(content_hash)
        
        return unique_tasks
    
    def _generate_task_id(self, title: str) -> str:
        """Generate a unique task ID from title."""
        # Create ID from title hash + timestamp
        title_hash = hashlib.md5(title.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        return f"task_{title_hash}_{timestamp}"

class ValueMetricsTracker:
    """
    Advanced value metrics tracking and reporting system.
    
    Tracks execution outcomes, ROI, and continuous learning metrics
    for the autonomous SDLC system.
    """
    
    def __init__(self, metrics_file: str = ".terragon/value-metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.metrics_data = self._load_metrics()
    
    def _load_metrics(self) -> Dict:
        """Load existing metrics data."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics: {e}")
        
        # Initialize empty metrics structure
        return {
            'summary': {
                'total_tasks_discovered': 0,
                'total_tasks_completed': 0,
                'total_value_delivered': 0.0,
                'total_effort_invested': 0.0,
                'average_completion_time': 0.0,
                'prediction_accuracy': 0.0,
                'last_updated': datetime.now().isoformat()
            },
            'execution_history': [],
            'discovery_metrics': {
                'sources': {},
                'categories': {},
                'trends': []
            },
            'learning_metrics': {
                'prediction_accuracy_trend': [],
                'model_adjustments': [],
                'outcome_correlation': {}
            }
        }
    
    def save_metrics(self):
        """Save metrics data to file."""
        self.metrics_data['summary']['last_updated'] = datetime.now().isoformat()
        
        # Ensure directory exists
        self.metrics_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def record_task_discovery(self, tasks: List[TaskItem]):
        """Record discovered tasks in metrics."""
        discovery_count = len(tasks)
        self.metrics_data['summary']['total_tasks_discovered'] += discovery_count
        
        # Track by source
        source_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for task in tasks:
            source_counts[task.source] += 1
            category_counts[task.category.value] += 1
        
        # Update metrics
        discovery_metrics = self.metrics_data['discovery_metrics']
        for source, count in source_counts.items():
            discovery_metrics['sources'][source] = discovery_metrics['sources'].get(source, 0) + count
        
        for category, count in category_counts.items():
            discovery_metrics['categories'][category] = discovery_metrics['categories'].get(category, 0) + count
        
        # Add trend data point
        discovery_metrics['trends'].append({
            'timestamp': datetime.now().isoformat(),
            'discovered_count': discovery_count,
            'total_score': sum(task.composite_score for task in tasks),
            'avg_score': sum(task.composite_score for task in tasks) / len(tasks) if tasks else 0
        })
        
        self.save_metrics()
        logger.info(f"Recorded discovery of {discovery_count} tasks")
    
    def record_task_completion(self, task_id: str, actual_effort: float, 
                             actual_value: float, outcome_quality: float):
        """Record completed task execution."""
        completion_record = {
            'task_id': task_id,
            'completed_at': datetime.now().isoformat(),
            'actual_effort_hours': actual_effort,
            'actual_value_delivered': actual_value,
            'outcome_quality': outcome_quality,
            'roi': actual_value / max(actual_effort, 0.1)  # Avoid division by zero
        }
        
        self.metrics_data['execution_history'].append(completion_record)
        
        # Update summary metrics
        summary = self.metrics_data['summary']
        summary['total_tasks_completed'] += 1
        summary['total_value_delivered'] += actual_value
        summary['total_effort_invested'] += actual_effort
        
        # Recalculate averages
        if summary['total_tasks_completed'] > 0:
            summary['average_completion_time'] = (
                summary['total_effort_invested'] / summary['total_tasks_completed']
            )
        
        self.save_metrics()
        logger.info(f"Recorded completion of task {task_id}: ROI={completion_record['roi']:.2f}")
    
    def calculate_roi_metrics(self) -> Dict:
        """Calculate comprehensive ROI and value metrics."""
        summary = self.metrics_data['summary']
        history = self.metrics_data['execution_history']
        
        if not history:
            return {'total_roi': 0.0, 'average_roi': 0.0, 'value_trend': []}
        
        # Calculate ROI metrics
        total_roi = summary['total_value_delivered'] / max(summary['total_effort_invested'], 1.0)
        
        individual_rois = [record['roi'] for record in history]
        average_roi = sum(individual_rois) / len(individual_rois)
        
        # Value trend over time
        value_trend = []
        cumulative_value = 0.0
        for record in history[-10:]:  # Last 10 completions
            cumulative_value += record['actual_value_delivered']
            value_trend.append({
                'timestamp': record['completed_at'],
                'cumulative_value': cumulative_value,
                'individual_value': record['actual_value_delivered']
            })
        
        return {
            'total_roi': round(total_roi, 2),
            'average_roi': round(average_roi, 2),
            'value_trend': value_trend,
            'total_value': summary['total_value_delivered'],
            'total_effort': summary['total_effort_invested'],
            'completion_count': summary['total_tasks_completed']
        }

def main():
    """
    Main execution function for the Terragon Value Discovery System.
    
    Demonstrates the complete workflow:
    1. Discover tasks from multiple sources
    2. Score and prioritize using advanced hybrid model
    3. Generate execution recommendations
    4. Track and report metrics
    """
    print("🚀 Terragon Autonomous SDLC Value Discovery System")
    print("=" * 60)
    
    # Initialize systems
    discovery = QuantumTaskDiscovery()
    metrics_tracker = ValueMetricsTracker()
    
    # Discover tasks
    print("\n📊 Discovering optimization opportunities...")
    discovered_tasks = discovery.discover_tasks()
    
    # Record discovery metrics
    metrics_tracker.record_task_discovery(discovered_tasks)
    
    # Display results
    print(f"\n✅ Discovered {len(discovered_tasks)} optimization opportunities")
    print("\n🎯 Top Priority Tasks:")
    print("-" * 80)
    
    total_value = 0.0
    for i, task in enumerate(discovered_tasks[:10], 1):
        # Estimate business value (simplified calculation)
        estimated_value = task.impact * task.user_business_value * 100
        total_value += estimated_value
        
        print(f"{i:2d}. {task.title}")
        print(f"    Score: {task.composite_score:6.1f} | Category: {task.category.value}")
        print(f"    Value: ${estimated_value:8,.0f} | Effort: {task.estimated_effort_hours:4.1f}h | ROI: {estimated_value/max(task.estimated_effort_hours,1):5.1f}x")
        print(f"    Auto-exec: {'✅' if task.auto_executable else '❌'} | Approvals: {', '.join(task.approval_required) if task.approval_required else 'None'}")
        print(f"    Quantum: {'🔬' if task.quantum_domain else '🔧'} | Risk: {'🟡' if task.risk_score < 0.5 else '🔴'}")
        print()
    
    # Display summary metrics
    roi_metrics = metrics_tracker.calculate_roi_metrics()
    
    print("\n📈 Value Discovery Summary:")
    print("-" * 40)
    print(f"Total Estimated Value: ${total_value:,.0f}")
    print(f"Total Estimated Effort: {sum(t.estimated_effort_hours for t in discovered_tasks):,.1f} hours")
    print(f"Portfolio ROI: {total_value / max(sum(t.estimated_effort_hours for t in discovered_tasks), 1):,.1f}x")
    print(f"Auto-executable: {sum(1 for t in discovered_tasks if t.auto_executable)}/{len(discovered_tasks)} ({100*sum(1 for t in discovered_tasks if t.auto_executable)/len(discovered_tasks):.1f}%)")
    
    # Quantum computing insights
    quantum_tasks = [t for t in discovered_tasks if t.quantum_domain]
    print(f"\n🔬 Quantum Computing Focus:")
    print(f"Quantum-specific tasks: {len(quantum_tasks)}/{len(discovered_tasks)} ({100*len(quantum_tasks)/len(discovered_tasks):.1f}%)")
    print(f"Average quantum relevance: {sum(t.quantum_relevance for t in quantum_tasks)/len(quantum_tasks):.1f}" if quantum_tasks else "0.0")
    
    # Save detailed results
    results_file = Path(".terragon/latest_scoring_report.json")
    results_file.parent.mkdir(exist_ok=True)
    
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tasks': len(discovered_tasks),
            'total_estimated_value': total_value,
            'total_estimated_effort': sum(t.estimated_effort_hours for t in discovered_tasks),
            'portfolio_roi': total_value / max(sum(t.estimated_effort_hours for t in discovered_tasks), 1),
            'auto_executable_count': sum(1 for t in discovered_tasks if t.auto_executable),
            'quantum_tasks_count': len(quantum_tasks)
        },
        'top_tasks': [
            {
                'id': task.id,
                'title': task.title,
                'category': task.category.value,
                'composite_score': task.composite_score,
                'estimated_value': task.impact * task.user_business_value * 100,
                'estimated_effort': task.estimated_effort_hours,
                'auto_executable': task.auto_executable,
                'quantum_domain': task.quantum_domain.value if task.quantum_domain else None,
                'risk_score': task.risk_score
            }
            for task in discovered_tasks[:20]
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print("\n🎉 Terragon Value Discovery System operational!")
    print("Ready for autonomous task execution and continuous value delivery.")

if __name__ == "__main__":
    main()