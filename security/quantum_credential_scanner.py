#!/usr/bin/env python3
"""
Quantum Credential Scanner
Specialized security scanner for quantum computing API keys and sensitive credentials.
"""

import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class CredentialMatch:
    """Represents a found credential match."""
    file_path: str
    line_number: int
    match_type: str
    severity: str
    context: str
    confidence: float


class QuantumCredentialScanner:
    """Scans for quantum computing credentials and API keys."""
    
    # Quantum backend patterns
    QUANTUM_PATTERNS = {
        # D-Wave API tokens
        'dwave_token': {
            'pattern': r'(?i)(?:dwave[_-]?token|DEV[_-]?TOKEN)["\'\s]*[:=]["\'\s]*([A-Za-z0-9+/]{40,})',
            'severity': 'HIGH',
            'confidence': 0.9
        },
        
        # IBM Quantum tokens
        'ibm_quantum_token': {
            'pattern': r'(?i)(?:ibm[_-]?quantum[_-]?token|IBMQ[_-]?TOKEN)["\'\s]*[:=]["\'\s]*([A-Za-z0-9]{40,})',
            'severity': 'HIGH',
            'confidence': 0.9
        },
        
        # Azure Quantum subscription keys
        'azure_quantum_key': {
            'pattern': r'(?i)(?:azure[_-]?quantum[_-]?key|subscription[_-]?key)["\'\s]*[:=]["\'\s]*([A-Za-z0-9]{32,})',
            'severity': 'HIGH',
            'confidence': 0.8
        },
        
        # Rigetti API keys
        'rigetti_api_key': {
            'pattern': r'(?i)(?:rigetti[_-]?api[_-]?key|PYQUIL[_-]?CONFIG)["\'\s]*[:=]["\'\s]*([A-Za-z0-9-]{20,})',
            'severity': 'HIGH',
            'confidence': 0.8
        },
        
        # IonQ API keys
        'ionq_api_key': {
            'pattern': r'(?i)(?:ionq[_-]?api[_-]?key)["\'\s]*[:=]["\'\s]*([A-Za-z0-9.]{30,})',
            'severity': 'HIGH',
            'confidence': 0.8
        },
        
        # Generic quantum service URLs with credentials
        'quantum_url_with_creds': {
            'pattern': r'https?://[^/\s]*:[^@/\s]*@[^/\s]*(?:quantum|dwave|ibm|azure|rigetti|ionq)',
            'severity': 'MEDIUM',
            'confidence': 0.7
        },
        
        # Quantum configuration files
        'quantum_config_file': {
            'pattern': r'(?i)(?:\.dwave[_-]?config|\.qiskit|quantum[_-]?credentials?\.json)',
            'severity': 'MEDIUM',
            'confidence': 0.6
        }
    }
    
    # General high-entropy patterns (potential secrets)
    ENTROPY_PATTERNS = {
        'high_entropy_string': {
            'pattern': r'["\']([A-Za-z0-9+/]{32,}={0,2})["\']',
            'severity': 'LOW',
            'confidence': 0.3
        }
    }
    
    # File extensions to scan
    SCAN_EXTENSIONS = {
        '.py', '.yaml', '.yml', '.json', '.toml', '.env', '.cfg', '.ini',
        '.sh', '.bash', '.zsh', '.config', '.conf', '.txt', '.md'
    }
    
    # Files to exclude from scanning
    EXCLUDE_PATTERNS = {
        r'\.git/',
        r'__pycache__/',
        r'\.pytest_cache/',
        r'\.mypy_cache/',
        r'node_modules/',
        r'\.venv/',
        r'venv/',
        r'dist/',
        r'build/',
        r'htmlcov/',
        r'\.coverage',
        r'test_.*\.py$',
        r'.*_test\.py$'
    }
    
    def __init__(self):
        self.matches: List[CredentialMatch] = []
        
    def scan_file(self, file_path: Path) -> List[CredentialMatch]:
        """Scan a single file for credentials."""
        if not self._should_scan_file(file_path):
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            return self._scan_content(str(file_path), content)
            
        except (IOError, UnicodeDecodeError) as e:
            print(f"Warning: Could not scan {file_path}: {e}", file=sys.stderr)
            return []
    
    def scan_directory(self, directory: Path) -> List[CredentialMatch]:
        """Recursively scan a directory for credentials."""
        all_matches = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                matches = self.scan_file(file_path)
                all_matches.extend(matches)
                
        return all_matches
    
    def _should_scan_file(self, file_path: Path) -> bool:
        """Determine if a file should be scanned."""
        # Check extension
        if file_path.suffix not in self.SCAN_EXTENSIONS:
            return False
            
        # Check exclude patterns
        file_str = str(file_path)
        for pattern in self.EXCLUDE_PATTERNS:
            if re.search(pattern, file_str):
                return False
                
        return True
    
    def _scan_content(self, file_path: str, content: str) -> List[CredentialMatch]:
        """Scan file content for credential patterns."""
        matches = []
        lines = content.split('\n')
        
        # Scan for quantum-specific patterns
        for match_type, pattern_info in self.QUANTUM_PATTERNS.items():
            pattern = pattern_info['pattern']
            severity = pattern_info['severity']
            confidence = pattern_info['confidence']
            
            for line_num, line in enumerate(lines, 1):
                for match in re.finditer(pattern, line):
                    matches.append(CredentialMatch(
                        file_path=file_path,
                        line_number=line_num,
                        match_type=match_type,
                        severity=severity,
                        context=line.strip()[:100],
                        confidence=confidence
                    ))
        
        # Scan for high-entropy strings (lower priority)
        for match_type, pattern_info in self.ENTROPY_PATTERNS.items():
            pattern = pattern_info['pattern']
            severity = pattern_info['severity']
            base_confidence = pattern_info['confidence']
            
            for line_num, line in enumerate(lines, 1):
                for match in re.finditer(pattern, line):
                    matched_string = match.group(1)
                    entropy = self._calculate_entropy(matched_string)
                    
                    # Only report high-entropy strings
                    if entropy > 4.5:
                        confidence = min(base_confidence + (entropy - 4.5) * 0.1, 0.8)
                        matches.append(CredentialMatch(
                            file_path=file_path,
                            line_number=line_num,
                            match_type=f'{match_type}_entropy_{entropy:.1f}',
                            severity=severity,
                            context=line.strip()[:100],
                            confidence=confidence
                        ))
        
        return matches
    
    def _calculate_entropy(self, string: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not string:
            return 0
            
        # Count character frequencies
        char_counts = {}
        for char in string:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        length = len(string)
        entropy = 0
        for count in char_counts.values():
            probability = count / length
            entropy -= probability * (probability.bit_length() - 1)
            
        return entropy
    
    def generate_report(self, matches: List[CredentialMatch], format_type: str = 'json') -> str:
        """Generate a security report in the specified format."""
        if format_type == 'json':
            return json.dumps([asdict(match) for match in matches], indent=2)
        elif format_type == 'sarif':
            return self._generate_sarif_report(matches)
        else:
            return self._generate_text_report(matches)
    
    def _generate_sarif_report(self, matches: List[CredentialMatch]) -> str:
        """Generate SARIF format report for security tools integration."""
        sarif = {
            "version": "2.1.0",
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "Quantum Credential Scanner",
                        "version": "1.0.0",
                        "informationUri": "https://github.com/your-org/quantum-inspired-task-planner"
                    }
                },
                "results": []
            }]
        }
        
        for match in matches:
            result = {
                "ruleId": match.match_type,
                "level": self._severity_to_sarif_level(match.severity),
                "message": {
                    "text": f"Potential {match.match_type} credential detected"
                },
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": match.file_path
                        },
                        "region": {
                            "startLine": match.line_number
                        }
                    }
                }],
                "properties": {
                    "confidence": match.confidence,
                    "context": match.context
                }
            }
            sarif["runs"][0]["results"].append(result)
        
        return json.dumps(sarif, indent=2)
    
    def _generate_text_report(self, matches: List[CredentialMatch]) -> str:
        """Generate human-readable text report."""
        if not matches:
            return "✅ No quantum credentials or suspicious patterns detected.\n"
        
        report = f"🚨 Quantum Credential Scanner Report - {len(matches)} issues found\n"
        report += "=" * 70 + "\n\n"
        
        # Group by severity
        by_severity = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
        for match in matches:
            by_severity[match.severity].append(match)
        
        for severity in ['HIGH', 'MEDIUM', 'LOW']:
            if by_severity[severity]:
                report += f"{severity} SEVERITY ({len(by_severity[severity])} issues):\n"
                report += "-" * 40 + "\n"
                
                for match in by_severity[severity]:
                    report += f"  📁 {match.file_path}:{match.line_number}\n"
                    report += f"  🔍 Type: {match.match_type}\n"
                    report += f"  📊 Confidence: {match.confidence:.1%}\n"
                    report += f"  📝 Context: {match.context}\n\n"
        
        return report
    
    def _severity_to_sarif_level(self, severity: str) -> str:
        """Convert severity to SARIF level."""
        mapping = {
            'HIGH': 'error',
            'MEDIUM': 'warning',
            'LOW': 'note'
        }
        return mapping.get(severity, 'note')


def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantum Credential Scanner')
    parser.add_argument('path', help='File or directory to scan')
    parser.add_argument('--format', choices=['text', 'json', 'sarif'], 
                       default='text', help='Output format')
    parser.add_argument('--output', help='Output file (default: stdout)')
    parser.add_argument('--fail-on-match', action='store_true',
                       help='Exit with non-zero code if matches found')
    
    args = parser.parse_args()
    
    scanner = QuantumCredentialScanner()
    path = Path(args.path)
    
    if path.is_file():
        matches = scanner.scan_file(path)
    else:
        matches = scanner.scan_directory(path)
    
    report = scanner.generate_report(matches, args.format)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
    else:
        print(report)
    
    if args.fail_on_match and matches:
        sys.exit(1)


if __name__ == '__main__':
    main()