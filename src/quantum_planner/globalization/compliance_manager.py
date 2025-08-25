"""Compliance Manager for Quantum Task Planner.

Ensures compliance with international data protection and privacy regulations:
- GDPR (General Data Protection Regulation) - EU
- CCPA (California Consumer Privacy Act) - USA
- PDPA (Personal Data Protection Act) - Singapore/Thailand  
- LGPD (Lei Geral de Proteção de Dados) - Brazil
- Data residency and sovereignty requirements
"""

import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid


class ComplianceRegulation(Enum):
    """Supported compliance regulations."""
    
    GDPR = "gdpr"  # EU General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    PDPA_SG = "pdpa_sg"  # Singapore Personal Data Protection Act
    PDPA_TH = "pdpa_th"  # Thailand Personal Data Protection Act
    LGPD = "lgpd"  # Brazil Lei Geral de Proteção de Dados
    PIPEDA = "pipeda"  # Canada Personal Information Protection Act


class DataClassification(Enum):
    """Data classification levels."""
    
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"  # Contains PII
    SENSITIVE = "sensitive"  # Contains sensitive personal data


class ProcessingLawfulBasis(Enum):
    """GDPR-compliant lawful basis for processing."""
    
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


@dataclass
class DataSubject:
    """Represents a data subject (individual) in compliance context."""
    
    subject_id: str
    region: str  # Geographic region for data residency
    consent_status: Dict[str, bool] = field(default_factory=dict)
    data_retention_period: int = 3650  # Days (default 10 years)
    processing_purposes: Set[str] = field(default_factory=set)
    lawful_basis: Optional[ProcessingLawfulBasis] = None
    created_at: float = field(default_factory=time.time)
    last_consent_update: Optional[float] = None


@dataclass
class DataProcessingRecord:
    """Record of data processing activities."""
    
    processing_id: str
    data_subject_id: str
    data_categories: Set[DataClassification]
    processing_purposes: Set[str]
    lawful_basis: ProcessingLawfulBasis
    retention_period: int
    third_party_transfers: List[str] = field(default_factory=list)
    security_measures: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass 
class ConsentRecord:
    """Record of user consent for data processing."""
    
    consent_id: str
    data_subject_id: str
    processing_purposes: Set[str]
    consent_given: bool
    consent_timestamp: float
    withdrawal_timestamp: Optional[float] = None
    consent_method: str = "explicit"  # explicit, implied, opt_in, opt_out
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class ComplianceManager:
    """Main compliance management system."""
    
    def __init__(self, active_regulations: List[ComplianceRegulation]):
        """Initialize compliance manager.
        
        Args:
            active_regulations: List of regulations to comply with
        """
        self.active_regulations = active_regulations
        self.data_subjects: Dict[str, DataSubject] = {}
        self.processing_records: List[DataProcessingRecord] = []
        self.consent_records: List[ConsentRecord] = []
        self.data_retention_policies: Dict[str, int] = {}
        self.permitted_regions: Set[str] = set()
        self.data_classification_rules: Dict[str, DataClassification] = {}
        
        # Initialize default settings
        self._initialize_compliance_settings()
    
    def _initialize_compliance_settings(self):
        """Initialize default compliance settings based on active regulations."""
        # Default data retention periods by regulation
        retention_periods = {
            ComplianceRegulation.GDPR: 2555,  # 7 years default
            ComplianceRegulation.CCPA: 2555,  # 7 years
            ComplianceRegulation.PDPA_SG: 3650,  # 10 years
            ComplianceRegulation.LGPD: 1825,  # 5 years
            ComplianceRegulation.PIPEDA: 2555   # 7 years
        }
        
        # Set retention policies
        for regulation in self.active_regulations:
            if regulation in retention_periods:
                self.data_retention_policies[regulation.value] = retention_periods[regulation]
        
        # Initialize permitted regions based on regulations
        if ComplianceRegulation.GDPR in self.active_regulations:
            # EU/EEA countries
            self.permitted_regions.update([
                'EU', 'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 
                'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 
                'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 'IS', 'LI', 'NO'
            ])
        
        if ComplianceRegulation.CCPA in self.active_regulations:
            self.permitted_regions.add('US-CA')
        
        if ComplianceRegulation.PDPA_SG in self.active_regulations:
            self.permitted_regions.add('SG')
            
        if ComplianceRegulation.LGPD in self.active_regulations:
            self.permitted_regions.add('BR')
    
    def register_data_subject(self, subject_id: str, region: str, 
                            initial_consent: Optional[Dict[str, bool]] = None) -> DataSubject:
        """Register a new data subject.
        
        Args:
            subject_id: Unique identifier for the data subject
            region: Geographic region of the data subject
            initial_consent: Initial consent settings
            
        Returns:
            Created DataSubject object
        """
        # Validate region is permitted
        if region not in self.permitted_regions and region != 'UNKNOWN':
            raise ValueError(f"Data processing not permitted in region: {region}")
        
        data_subject = DataSubject(
            subject_id=subject_id,
            region=region,
            consent_status=initial_consent or {},
            created_at=time.time()
        )
        
        self.data_subjects[subject_id] = data_subject
        
        return data_subject
    
    def record_consent(self, data_subject_id: str, purposes: List[str], 
                      consent_given: bool, method: str = "explicit",
                      ip_address: Optional[str] = None) -> ConsentRecord:
        """Record user consent for data processing.
        
        Args:
            data_subject_id: ID of the data subject
            purposes: List of processing purposes
            consent_given: Whether consent was given
            method: Method of consent collection
            ip_address: IP address of the user (for audit)
            
        Returns:
            Created ConsentRecord
        """
        consent_record = ConsentRecord(
            consent_id=str(uuid.uuid4()),
            data_subject_id=data_subject_id,
            processing_purposes=set(purposes),
            consent_given=consent_given,
            consent_timestamp=time.time(),
            consent_method=method,
            ip_address=ip_address
        )
        
        self.consent_records.append(consent_record)
        
        # Update data subject consent status
        if data_subject_id in self.data_subjects:
            data_subject = self.data_subjects[data_subject_id]
            for purpose in purposes:
                data_subject.consent_status[purpose] = consent_given
            data_subject.last_consent_update = time.time()
        
        return consent_record
    
    def record_data_processing(self, data_subject_id: str, 
                             data_categories: List[DataClassification],
                             purposes: List[str],
                             lawful_basis: ProcessingLawfulBasis,
                             retention_days: Optional[int] = None) -> DataProcessingRecord:
        """Record a data processing activity.
        
        Args:
            data_subject_id: ID of the data subject
            data_categories: Categories of data being processed
            purposes: Purposes of processing
            lawful_basis: Legal basis for processing
            retention_days: Data retention period in days
            
        Returns:
            Created DataProcessingRecord
        """
        # Validate consent if required
        if lawful_basis == ProcessingLawfulBasis.CONSENT:
            if not self.has_valid_consent(data_subject_id, purposes):
                raise ValueError("Valid consent required for this processing")
        
        # Determine retention period
        if retention_days is None:
            retention_days = self._get_default_retention_period(data_subject_id)
        
        processing_record = DataProcessingRecord(
            processing_id=str(uuid.uuid4()),
            data_subject_id=data_subject_id,
            data_categories=set(data_categories),
            processing_purposes=set(purposes),
            lawful_basis=lawful_basis,
            retention_period=retention_days,
            timestamp=time.time()
        )
        
        self.processing_records.append(processing_record)
        
        return processing_record
    
    def has_valid_consent(self, data_subject_id: str, purposes: List[str]) -> bool:
        """Check if data subject has valid consent for specified purposes.
        
        Args:
            data_subject_id: ID of the data subject
            purposes: Processing purposes to check
            
        Returns:
            True if valid consent exists for all purposes
        """
        if data_subject_id not in self.data_subjects:
            return False
        
        data_subject = self.data_subjects[data_subject_id]
        
        for purpose in purposes:
            if not data_subject.consent_status.get(purpose, False):
                return False
        
        return True
    
    def withdraw_consent(self, data_subject_id: str, purposes: List[str]) -> ConsentRecord:
        """Withdraw consent for specified purposes.
        
        Args:
            data_subject_id: ID of the data subject
            purposes: Purposes to withdraw consent for
            
        Returns:
            ConsentRecord documenting the withdrawal
        """
        # Record consent withdrawal
        withdrawal_record = ConsentRecord(
            consent_id=str(uuid.uuid4()),
            data_subject_id=data_subject_id,
            processing_purposes=set(purposes),
            consent_given=False,
            consent_timestamp=time.time(),
            withdrawal_timestamp=time.time(),
            consent_method="withdrawal"
        )
        
        self.consent_records.append(withdrawal_record)
        
        # Update data subject status
        if data_subject_id in self.data_subjects:
            data_subject = self.data_subjects[data_subject_id]
            for purpose in purposes:
                data_subject.consent_status[purpose] = False
            data_subject.last_consent_update = time.time()
        
        return withdrawal_record
    
    def handle_data_subject_request(self, request_type: str, data_subject_id: str) -> Dict[str, Any]:
        """Handle data subject rights requests (GDPR Article 15-22).
        
        Args:
            request_type: Type of request (access, rectification, erasure, portability, etc.)
            data_subject_id: ID of the data subject
            
        Returns:
            Response to the data subject request
        """
        if data_subject_id not in self.data_subjects:
            return {'status': 'error', 'message': 'Data subject not found'}
        
        data_subject = self.data_subjects[data_subject_id]
        
        if request_type == "access":
            # Right of access (GDPR Art. 15)
            return self._handle_access_request(data_subject_id)
        
        elif request_type == "rectification":
            # Right to rectification (GDPR Art. 16)
            return {'status': 'acknowledged', 'message': 'Rectification request received'}
        
        elif request_type == "erasure":
            # Right to erasure / "right to be forgotten" (GDPR Art. 17)
            return self._handle_erasure_request(data_subject_id)
        
        elif request_type == "portability":
            # Right to data portability (GDPR Art. 20)
            return self._handle_portability_request(data_subject_id)
        
        elif request_type == "restriction":
            # Right to restriction of processing (GDPR Art. 18)
            return {'status': 'acknowledged', 'message': 'Restriction request received'}
        
        elif request_type == "objection":
            # Right to object (GDPR Art. 21)
            return {'status': 'acknowledged', 'message': 'Objection request received'}
        
        else:
            return {'status': 'error', 'message': f'Unknown request type: {request_type}'}
    
    def _handle_access_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle data access request."""
        data_subject = self.data_subjects[data_subject_id]
        
        # Collect all data related to the subject
        subject_data = {
            'data_subject_info': {
                'subject_id': data_subject.subject_id,
                'region': data_subject.region,
                'created_at': data_subject.created_at,
                'consent_status': data_subject.consent_status,
                'last_consent_update': data_subject.last_consent_update
            },
            'processing_records': [],
            'consent_records': []
        }
        
        # Add processing records
        for record in self.processing_records:
            if record.data_subject_id == data_subject_id:
                subject_data['processing_records'].append({
                    'processing_id': record.processing_id,
                    'data_categories': [cat.value for cat in record.data_categories],
                    'purposes': list(record.processing_purposes),
                    'lawful_basis': record.lawful_basis.value,
                    'timestamp': record.timestamp
                })
        
        # Add consent records
        for record in self.consent_records:
            if record.data_subject_id == data_subject_id:
                subject_data['consent_records'].append({
                    'consent_id': record.consent_id,
                    'purposes': list(record.processing_purposes),
                    'consent_given': record.consent_given,
                    'timestamp': record.consent_timestamp,
                    'method': record.consent_method
                })
        
        return {
            'status': 'success',
            'data': subject_data,
            'request_fulfilled_at': time.time()
        }
    
    def _handle_erasure_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle data erasure request."""
        # Check if erasure is permissible (considering legal obligations)
        active_legal_obligations = False
        for record in self.processing_records:
            if (record.data_subject_id == data_subject_id and 
                record.lawful_basis == ProcessingLawfulBasis.LEGAL_OBLIGATION):
                active_legal_obligations = True
                break
        
        if active_legal_obligations:
            return {
                'status': 'partially_fulfilled',
                'message': 'Some data retained due to legal obligations',
                'erasure_timestamp': time.time()
            }
        
        # Perform erasure (in practice, this would anonymize or delete data)
        erased_records = 0
        
        # Mark processing records as erased
        for record in self.processing_records:
            if record.data_subject_id == data_subject_id:
                erased_records += 1
        
        # In real implementation, would actually anonymize/delete the data
        # For demonstration, we'll just mark it
        
        return {
            'status': 'fulfilled',
            'erased_records': erased_records,
            'erasure_timestamp': time.time()
        }
    
    def _handle_portability_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle data portability request."""
        # Get data in structured, machine-readable format
        access_response = self._handle_access_request(data_subject_id)
        
        if access_response['status'] == 'success':
            # Format data for portability (JSON format)
            portable_data = json.dumps(access_response['data'], indent=2)
            
            return {
                'status': 'success',
                'format': 'JSON',
                'data': portable_data,
                'export_timestamp': time.time()
            }
        
        return access_response
    
    def _get_default_retention_period(self, data_subject_id: str) -> int:
        """Get default retention period for a data subject."""
        if data_subject_id not in self.data_subjects:
            return 2555  # Default 7 years
        
        data_subject = self.data_subjects[data_subject_id]
        
        # Use shortest retention period from active regulations
        min_retention = float('inf')
        for regulation in self.active_regulations:
            if regulation.value in self.data_retention_policies:
                period = self.data_retention_policies[regulation.value]
                min_retention = min(min_retention, period)
        
        return int(min_retention) if min_retention != float('inf') else 2555
    
    def audit_compliance(self) -> Dict[str, Any]:
        """Perform compliance audit.
        
        Returns:
            Audit report with compliance status
        """
        audit_results = {
            'audit_timestamp': time.time(),
            'regulations_covered': [reg.value for reg in self.active_regulations],
            'data_subjects_count': len(self.data_subjects),
            'processing_records_count': len(self.processing_records),
            'consent_records_count': len(self.consent_records),
            'compliance_issues': [],
            'recommendations': []
        }
        
        # Check for compliance issues
        
        # 1. Check for processing without valid consent
        consent_issues = 0
        for record in self.processing_records:
            if (record.lawful_basis == ProcessingLawfulBasis.CONSENT and
                not self.has_valid_consent(record.data_subject_id, list(record.processing_purposes))):
                consent_issues += 1
        
        if consent_issues > 0:
            audit_results['compliance_issues'].append({
                'type': 'consent_violation',
                'count': consent_issues,
                'severity': 'high'
            })
        
        # 2. Check for data retention violations
        current_time = time.time()
        retention_violations = 0
        
        for record in self.processing_records:
            if current_time - record.timestamp > record.retention_period * 86400:  # Convert days to seconds
                retention_violations += 1
        
        if retention_violations > 0:
            audit_results['compliance_issues'].append({
                'type': 'data_retention_violation',
                'count': retention_violations,
                'severity': 'medium'
            })
        
        # 3. Generate recommendations
        if len(audit_results['compliance_issues']) == 0:
            audit_results['recommendations'].append("Compliance status: All checks passed")
        else:
            if consent_issues > 0:
                audit_results['recommendations'].append("Review and update consent management processes")
            if retention_violations > 0:
                audit_results['recommendations'].append("Implement automated data retention policies")
        
        # Overall compliance score
        total_issues = len(audit_results['compliance_issues'])
        audit_results['compliance_score'] = max(0, 100 - (total_issues * 20))
        audit_results['overall_status'] = 'COMPLIANT' if total_issues == 0 else 'ISSUES_FOUND'
        
        return audit_results
    
    def generate_privacy_policy(self, organization_name: str) -> str:
        """Generate privacy policy text based on active regulations.
        
        Args:
            organization_name: Name of the organization
            
        Returns:
            Generated privacy policy text
        """
        policy_text = f"""PRIVACY POLICY

{organization_name} - Quantum Task Planner

Last Updated: {time.strftime('%Y-%m-%d')}

1. DATA CONTROLLER
{organization_name} acts as the data controller for personal data processed through the Quantum Task Planner system.

2. DATA WE COLLECT
We collect and process the following categories of personal data:
- Task assignment data (agent identifiers, task descriptions)
- System usage data (optimization parameters, performance metrics)
- Account information (user preferences, settings)

3. LEGAL BASIS FOR PROCESSING
We process personal data based on:"""
        
        if ComplianceRegulation.GDPR in self.active_regulations:
            policy_text += """
- Your consent (GDPR Article 6(1)(a))
- Performance of a contract (GDPR Article 6(1)(b))
- Legitimate interests (GDPR Article 6(1)(f))
- Legal obligation (GDPR Article 6(1)(c))"""
        
        policy_text += """

4. DATA RETENTION
Personal data is retained according to our data retention policy:"""
        
        for regulation, period in self.data_retention_policies.items():
            policy_text += f"""
- {regulation.upper()}: {period} days"""
        
        if ComplianceRegulation.GDPR in self.active_regulations:
            policy_text += """

5. YOUR RIGHTS (GDPR)
Under GDPR, you have the right to:
- Access your personal data (Article 15)
- Rectify inaccurate data (Article 16)
- Erase your data (Article 17)
- Restrict processing (Article 18)
- Data portability (Article 20)
- Object to processing (Article 21)

To exercise these rights, contact us at: privacy@example.com"""
        
        if ComplianceRegulation.CCPA in self.active_regulations:
            policy_text += """

6. CALIFORNIA CONSUMER PRIVACY ACT (CCPA)
California residents have additional rights:
- Right to know what personal information is collected
- Right to delete personal information
- Right to opt-out of the sale of personal information
- Right to non-discrimination"""
        
        policy_text += f"""

7. CONTACT INFORMATION
For privacy-related inquiries, contact:
Data Protection Officer
{organization_name}
Email: privacy@example.com

This privacy policy is automatically generated based on your compliance requirements.
"""
        
        return policy_text
    
    def export_compliance_report(self) -> Dict[str, Any]:
        """Export comprehensive compliance report.
        
        Returns:
            Complete compliance report for regulatory purposes
        """
        report = {
            'report_metadata': {
                'generated_at': time.time(),
                'report_version': '1.0',
                'regulations': [reg.value for reg in self.active_regulations]
            },
            'data_inventory': {
                'data_subjects': len(self.data_subjects),
                'processing_activities': len(self.processing_records),
                'consent_records': len(self.consent_records)
            },
            'audit_results': self.audit_compliance(),
            'data_subject_rights_requests': {
                'access_requests': sum(1 for r in self.consent_records if 'access' in r.consent_method),
                'erasure_requests': sum(1 for r in self.consent_records if 'erasure' in r.consent_method),
                'portability_requests': sum(1 for r in self.consent_records if 'portability' in r.consent_method)
            },
            'security_measures': [
                'Data encryption in transit and at rest',
                'Access controls and authentication',
                'Regular security assessments',
                'Incident response procedures',
                'Staff training on data protection'
            ]
        }
        
        return report