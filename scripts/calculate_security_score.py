#!/usr/bin/env python3
"""Calculate overall security score from various security reports."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any


def calculate_dependency_score(safety_report: Dict[str, Any]) -> float:
    """Calculate dependency security score from Safety report."""
    try:
        vulnerabilities = safety_report.get('vulnerabilities', [])
        if not vulnerabilities:
            return 100.0
        
        high_severity = sum(1 for v in vulnerabilities if v.get('severity') == 'high')
        medium_severity = sum(1 for v in vulnerabilities if v.get('severity') == 'medium') 
        low_severity = sum(1 for v in vulnerabilities if v.get('severity') == 'low')
        
        # Weight vulnerabilities by severity
        penalty = (high_severity * 20) + (medium_severity * 10) + (low_severity * 5)
        score = max(0, 100 - penalty)
        
        return score
    except Exception:
        return 50.0  # Default score if parsing fails


def calculate_code_security_score(bandit_report: Dict[str, Any]) -> float:
    """Calculate code security score from Bandit report."""
    try:
        results = bandit_report.get('results', [])
        if not results:
            return 100.0
        
        high_severity = sum(1 for r in results if r.get('issue_severity') == 'HIGH')
        medium_severity = sum(1 for r in results if r.get('issue_severity') == 'MEDIUM')
        low_severity = sum(1 for r in results if r.get('issue_severity') == 'LOW')
        
        # Weight issues by severity
        penalty = (high_severity * 15) + (medium_severity * 8) + (low_severity * 3)
        score = max(0, 100 - penalty)
        
        return score
    except Exception:
        return 50.0


def calculate_container_security_score(trivy_report: Dict[str, Any]) -> float:
    """Calculate container security score from Trivy report."""
    try:
        vulnerabilities = trivy_report.get('Results', [])
        if not vulnerabilities:
            return 100.0
        
        total_vulns = 0
        critical_vulns = 0
        high_vulns = 0
        
        for result in vulnerabilities:
            vulns = result.get('Vulnerabilities', [])
            for vuln in vulns:
                total_vulns += 1
                severity = vuln.get('Severity', '').upper()
                if severity == 'CRITICAL':
                    critical_vulns += 1
                elif severity == 'HIGH':
                    high_vulns += 1
        
        if total_vulns == 0:
            return 100.0
        
        # Calculate penalty based on vulnerability distribution
        penalty = (critical_vulns * 25) + (high_vulns * 15)
        score = max(0, 100 - penalty)
        
        return score
    except Exception:
        return 50.0


def calculate_license_compliance_score(licenses_report: list) -> float:
    """Calculate license compliance score."""
    try:
        prohibited_licenses = {
            'GPL-3.0', 'AGPL-3.0', 'LGPL-3.0', 'SSPL-1.0'
        }
        
        violations = 0
        total_packages = len(licenses_report)
        
        for package in licenses_report:
            license_name = package.get('License', '')
            if license_name in prohibited_licenses:
                violations += 1
        
        if total_packages == 0:
            return 100.0
        
        # Each violation reduces score by 10%
        penalty = (violations / total_packages) * 100
        score = max(0, 100 - penalty)
        
        return score
    except Exception:
        return 90.0  # High default for license compliance


def main():
    parser = argparse.ArgumentParser(description='Calculate security score')
    parser.add_argument('--dependency-report', type=Path, help='Safety report JSON file')
    parser.add_argument('--code-report', type=Path, help='Bandit report JSON file')
    parser.add_argument('--container-report', type=Path, help='Trivy report JSON file')
    parser.add_argument('--license-report', type=Path, help='License report JSON file')
    parser.add_argument('--output', type=Path, help='Output JSON file')
    
    args = parser.parse_args()
    
    scores = {}
    
    # Calculate dependency security score
    if args.dependency_report and args.dependency_report.exists():
        try:
            with open(args.dependency_report) as f:
                dependency_data = json.load(f)
            scores['dependencies'] = calculate_dependency_score(dependency_data)
        except Exception as e:
            print(f"Error processing dependency report: {e}", file=sys.stderr)
            scores['dependencies'] = 50.0
    else:
        scores['dependencies'] = 100.0
    
    # Calculate code security score
    if args.code_report and args.code_report.exists():
        try:
            with open(args.code_report) as f:
                code_data = json.load(f)
            scores['code_security'] = calculate_code_security_score(code_data)
        except Exception as e:
            print(f"Error processing code report: {e}", file=sys.stderr)
            scores['code_security'] = 50.0
    else:
        scores['code_security'] = 100.0
    
    # Calculate container security score
    if args.container_report and args.container_report.exists():
        try:
            with open(args.container_report) as f:
                container_data = json.load(f)
            scores['container_security'] = calculate_container_security_score(container_data)
        except Exception as e:
            print(f"Error processing container report: {e}", file=sys.stderr)
            scores['container_security'] = 50.0
    else:
        scores['container_security'] = 100.0
    
    # Calculate license compliance score
    if args.license_report and args.license_report.exists():
        try:
            with open(args.license_report) as f:
                license_data = json.load(f)
            scores['license_compliance'] = calculate_license_compliance_score(license_data)
        except Exception as e:
            print(f"Error processing license report: {e}", file=sys.stderr)
            scores['license_compliance'] = 90.0
    else:
        scores['license_compliance'] = 100.0
    
    # Calculate overall score (weighted average)
    weights = {
        'dependencies': 0.3,
        'code_security': 0.3,
        'container_security': 0.25,
        'license_compliance': 0.15
    }
    
    overall_score = sum(scores[key] * weights[key] for key in scores.keys())
    scores['overall_score'] = round(overall_score, 1)
    
    # Add score interpretation
    if overall_score >= 90:
        scores['grade'] = 'A'
        scores['status'] = 'Excellent'
    elif overall_score >= 80:
        scores['grade'] = 'B'
        scores['status'] = 'Good'
    elif overall_score >= 70:
        scores['grade'] = 'C'
        scores['status'] = 'Fair'
    elif overall_score >= 60:
        scores['grade'] = 'D'
        scores['status'] = 'Poor'
    else:
        scores['grade'] = 'F'
        scores['status'] = 'Critical'
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(scores, f, indent=2)
    else:
        print(json.dumps(scores, indent=2))
    
    # Print summary
    print(f"\n🔒 Security Score Summary", file=sys.stderr)
    print(f"Overall Score: {scores['overall_score']}% ({scores['grade']} - {scores['status']})", file=sys.stderr)
    print(f"Dependencies: {scores['dependencies']}%", file=sys.stderr)
    print(f"Code Security: {scores['code_security']}%", file=sys.stderr)
    print(f"Container Security: {scores['container_security']}%", file=sys.stderr)
    print(f"License Compliance: {scores['license_compliance']}%", file=sys.stderr)
    
    # Exit with non-zero code if score is too low
    if overall_score < 70:
        sys.exit(1)


if __name__ == '__main__':
    main()