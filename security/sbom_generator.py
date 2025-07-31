#!/usr/bin/env python3
"""
Software Bill of Materials (SBOM) Generator for Quantum Task Planner
Generates comprehensive SBOM including quantum dependencies and security metadata.
"""

import json
import sys
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import toml


@dataclass
class Component:
    """Represents a software component in the SBOM."""
    name: str
    version: str
    type: str
    supplier: Optional[str] = None
    download_location: Optional[str] = None
    files_analyzed: bool = False
    license_concluded: Optional[str] = None
    license_declared: Optional[str] = None
    copyright_text: Optional[str] = None
    checksums: Optional[Dict[str, str]] = None
    external_refs: Optional[List[Dict[str, str]]] = None


class QuantumSBOMGenerator:
    """Generates SBOM with quantum computing specific metadata."""
    
    # Quantum computing specific packages
    QUANTUM_PACKAGES = {
        'dwave-ocean-sdk': 'D-Wave Systems Inc.',
        'qiskit': 'IBM',
        'qiskit-optimization': 'IBM',
        'azure-quantum': 'Microsoft',
        'cirq': 'Google',
        'pennylane': 'Xanadu',
        'pyquil': 'Rigetti Computing',
        'braket': 'Amazon Web Services',
        'strawberry-fields': 'Xanadu',
        'quantum-planner': 'Terragon Labs'
    }
    
    # Security-sensitive packages that need extra attention
    SECURITY_SENSITIVE = {
        'cryptography', 'pycryptodome', 'paramiko', 'requests',
        'urllib3', 'certifi', 'jwt', 'oauth', 'azure-identity'
    }
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.project_metadata = self._load_project_metadata()
        
    def generate_sbom(self, format_type: str = 'spdx-json') -> str:
        """Generate SBOM in the specified format."""
        if format_type == 'spdx-json':
            return self._generate_spdx_json()
        elif format_type == 'cyclonedx':
            return self._generate_cyclonedx()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _load_project_metadata(self) -> Dict[str, Any]:
        """Load project metadata from pyproject.toml."""
        pyproject_path = self.project_root / 'pyproject.toml'
        if pyproject_path.exists():
            with open(pyproject_path, 'r') as f:
                return toml.load(f)
        return {}
    
    def _get_installed_packages(self) -> List[Component]:
        """Get list of installed packages with metadata."""
        try:
            # Use pip list to get installed packages
            result = subprocess.run(
                ['pip', 'list', '--format=json'],
                capture_output=True,
                text=True,
                check=True
            )
            pip_packages = json.loads(result.stdout)
            
            components = []
            for package in pip_packages:
                name = package['name']
                version = package['version']
                
                # Get additional metadata
                component = Component(
                    name=name,
                    version=version,
                    type='python-package',
                    supplier=self._get_package_supplier(name),
                    license_declared=self._get_package_license(name),
                    external_refs=self._get_external_refs(name)
                )
                
                components.append(component)
            
            return components
            
        except subprocess.CalledProcessError as e:
            print(f"Error getting installed packages: {e}", file=sys.stderr)
            return []
    
    def _get_package_supplier(self, package_name: str) -> Optional[str]:
        """Get package supplier/maintainer."""
        if package_name in self.QUANTUM_PACKAGES:
            return self.QUANTUM_PACKAGES[package_name]
        
        # Try to get from pip show
        try:
            result = subprocess.run(
                ['pip', 'show', package_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.split('\n'):
                if line.startswith('Author:'):
                    return line.replace('Author:', '').strip()
            
        except subprocess.CalledProcessError:
            pass
        
        return None
    
    def _get_package_license(self, package_name: str) -> Optional[str]:
        """Get package license information."""
        try:
            result = subprocess.run(
                ['pip', 'show', package_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.split('\n'):
                if line.startswith('License:'):
                    license_text = line.replace('License:', '').strip()
                    return license_text if license_text != 'UNKNOWN' else None
            
        except subprocess.CalledProcessError:
            pass
        
        return None
    
    def _get_external_refs(self, package_name: str) -> List[Dict[str, str]]:
        """Get external references for a package."""
        refs = []
        
        # PyPI reference
        refs.append({
            'type': 'distribution',
            'locator': f'https://pypi.org/project/{package_name}/'
        })
        
        # Security advisory references for sensitive packages
        if package_name in self.SECURITY_SENSITIVE:
            refs.append({
                'type': 'security-advisory',
                'locator': f'https://github.com/advisories?query={package_name}'
            })
        
        # Quantum-specific references
        if package_name in self.QUANTUM_PACKAGES:
            refs.append({
                'type': 'quantum-backend',
                'locator': f'quantum-computing-package-{package_name}'
            })
        
        return refs
    
    def _generate_spdx_json(self) -> str:
        """Generate SBOM in SPDX JSON format."""
        components = self._get_installed_packages()
        
        spdx_doc = {
            'spdxVersion': 'SPDX-2.3',
            'dataLicense': 'CC0-1.0',
            'SPDXID': 'SPDXRef-DOCUMENT',
            'name': 'Quantum Task Planner SBOM',
            'documentNamespace': f'https://github.com/your-org/quantum-inspired-task-planner/sbom/{datetime.now(timezone.utc).isoformat()}',
            'creationInfo': {
                'created': datetime.now(timezone.utc).isoformat(),
                'creators': ['Tool: Quantum SBOM Generator'],
                'licenseListVersion': '3.19'
            },
            'packages': [],
            'relationships': []
        }
        
        # Add main package
        main_package = {
            'SPDXID': 'SPDXRef-Package-quantum-planner',
            'name': 'quantum-inspired-task-planner',
            'downloadLocation': 'https://github.com/your-org/quantum-inspired-task-planner',
            'filesAnalyzed': False,
            'supplier': 'Organization: Terragon Labs',
            'versionInfo': self.project_metadata.get('tool', {}).get('poetry', {}).get('version', '1.0.0'),
            'licenseConcluded': 'MIT',
            'licenseDeclared': 'MIT',
            'copyrightText': 'Copyright 2024 Terragon Labs',
            'externalRefs': [
                {
                    'referenceCategory': 'PACKAGE-MANAGER',
                    'referenceType': 'purl',
                    'referenceLocator': 'pkg:pypi/quantum-inspired-task-planner'
                }
            ]
        }
        spdx_doc['packages'].append(main_package)
        
        # Add dependencies
        for i, component in enumerate(components):
            package_id = f'SPDXRef-Package-{component.name}-{i}'
            
            package = {
                'SPDXID': package_id,
                'name': component.name,
                'versionInfo': component.version,
                'downloadLocation': f'https://pypi.org/project/{component.name}/',
                'filesAnalyzed': False,
                'supplier': f'Organization: {component.supplier}' if component.supplier else 'NOASSERTION',
                'licenseConcluded': component.license_declared or 'NOASSERTION',
                'licenseDeclared': component.license_declared or 'NOASSERTION',
                'copyrightText': 'NOASSERTION',
                'externalRefs': [
                    {
                        'referenceCategory': 'PACKAGE-MANAGER',
                        'referenceType': 'purl',
                        'referenceLocator': f'pkg:pypi/{component.name}@{component.version}'
                    }
                ]
            }
            
            # Add external references
            if component.external_refs:
                for ref in component.external_refs:
                    package['externalRefs'].append({
                        'referenceCategory': 'OTHER',
                        'referenceType': ref['type'],
                        'referenceLocator': ref['locator']
                    })
            
            spdx_doc['packages'].append(package)
            
            # Add dependency relationship
            spdx_doc['relationships'].append({
                'spdxElementId': 'SPDXRef-Package-quantum-planner',
                'relationshipType': 'DEPENDS_ON',
                'relatedSpdxElement': package_id
            })
        
        return json.dumps(spdx_doc, indent=2)
    
    def _generate_cyclonedx(self) -> str:
        """Generate SBOM in CycloneDX format."""
        components = self._get_installed_packages()
        
        cyclone_doc = {
            'bomFormat': 'CycloneDX',
            'specVersion': '1.4',
            'serialNumber': f'urn:uuid:{self._generate_uuid()}',
            'version': 1,
            'metadata': {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'tools': [
                    {
                        'vendor': 'Terragon Labs',
                        'name': 'Quantum SBOM Generator',
                        'version': '1.0.0'
                    }
                ],
                'component': {
                    'type': 'application',
                    'bom-ref': 'quantum-planner',
                    'name': 'quantum-inspired-task-planner',
                    'version': self.project_metadata.get('tool', {}).get('poetry', {}).get('version', '1.0.0'),
                    'description': 'QUBO-based task scheduler for agent pools using quantum annealing',
                    'licenses': [{'license': {'id': 'MIT'}}],
                    'supplier': {
                        'name': 'Terragon Labs',
                        'url': ['https://terragon.ai']
                    }
                }
            },
            'components': []
        }
        
        # Add components
        for component in components:
            cyclone_component = {
                'type': 'library',
                'bom-ref': f'{component.name}@{component.version}',
                'name': component.name,
                'version': component.version,
                'scope': 'required',
                'hashes': [],
                'licenses': [],
                'purl': f'pkg:pypi/{component.name}@{component.version}',
                'externalReferences': []
            }
            
            # Add license if available
            if component.license_declared:
                cyclone_component['licenses'].append({
                    'license': {'name': component.license_declared}
                })
            
            # Add external references
            if component.external_refs:
                for ref in component.external_refs:
                    cyclone_component['externalReferences'].append({
                        'type': ref['type'],
                        'url': ref['locator']
                    })
            
            # Add quantum-specific metadata
            if component.name in self.QUANTUM_PACKAGES:
                cyclone_component['properties'] = [
                    {
                        'name': 'quantum:backend-type',
                        'value': 'quantum-computing'
                    },
                    {
                        'name': 'quantum:supplier',
                        'value': self.QUANTUM_PACKAGES[component.name]
                    }
                ]
            
            # Add security metadata for sensitive packages
            if component.name in self.SECURITY_SENSITIVE:
                if 'properties' not in cyclone_component:
                    cyclone_component['properties'] = []
                cyclone_component['properties'].append({
                    'name': 'security:sensitive',
                    'value': 'true'
                })
            
            cyclone_doc['components'].append(cyclone_component)
        
        return json.dumps(cyclone_doc, indent=2)
    
    def _generate_uuid(self) -> str:
        """Generate a UUID for the SBOM."""
        import uuid
        return str(uuid.uuid4())
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate security-focused component analysis."""
        components = self._get_installed_packages()
        
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_components': len(components),
            'quantum_components': [],
            'security_sensitive_components': [],
            'license_summary': {},
            'supplier_summary': {},
            'recommendations': []
        }
        
        # Analyze components
        for component in components:
            if component.name in self.QUANTUM_PACKAGES:
                report['quantum_components'].append({
                    'name': component.name,
                    'version': component.version,
                    'supplier': component.supplier
                })
            
            if component.name in self.SECURITY_SENSITIVE:
                report['security_sensitive_components'].append({
                    'name': component.name,
                    'version': component.version,
                    'license': component.license_declared
                })
            
            # License summary
            license_key = component.license_declared or 'Unknown'
            report['license_summary'][license_key] = report['license_summary'].get(license_key, 0) + 1
            
            # Supplier summary
            supplier_key = component.supplier or 'Unknown'
            report['supplier_summary'][supplier_key] = report['supplier_summary'].get(supplier_key, 0) + 1
        
        # Generate recommendations
        if report['quantum_components']:
            report['recommendations'].append(
                'Monitor quantum backend dependencies for security updates and API changes'
            )
        
        if report['security_sensitive_components']:
            report['recommendations'].append(
                'Regularly audit security-sensitive packages for vulnerabilities'
            )
        
        if 'Unknown' in report['license_summary']:
            report['recommendations'].append(
                'Review packages with unknown licenses for compliance'
            )
        
        return report


def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantum SBOM Generator')
    parser.add_argument('--format', choices=['spdx-json', 'cyclonedx'], 
                       default='spdx-json', help='SBOM format')
    parser.add_argument('--output', help='Output file (default: stdout)')
    parser.add_argument('--security-report', action='store_true',
                       help='Generate security analysis report')
    parser.add_argument('--project-root', type=Path, default=Path('.'),
                       help='Project root directory')
    
    args = parser.parse_args()
    
    generator = QuantumSBOMGenerator(args.project_root)
    
    if args.security_report:
        report = generator.generate_security_report()
        output = json.dumps(report, indent=2)
    else:
        output = generator.generate_sbom(args.format)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"SBOM generated: {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    main()