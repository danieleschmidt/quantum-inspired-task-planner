#!/usr/bin/env python3
"""Generate Software Bill of Materials (SBOM) for the project."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import hashlib

def get_git_info():
    """Get git repository information."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            text=True
        ).strip()
        
        commit_date = subprocess.check_output(
            ["git", "log", "-1", "--format=%ci"], 
            text=True
        ).strip()
        
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
            text=True
        ).strip()
        
        remote_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], 
            text=True
        ).strip()
        
        return {
            "commit": commit_hash,
            "commit_date": commit_date,
            "branch": branch,
            "repository": remote_url
        }
    except subprocess.CalledProcessError:
        return None

def get_poetry_dependencies():
    """Get dependencies from poetry."""
    try:
        result = subprocess.check_output(
            ["poetry", "show", "--no-ansi"], 
            text=True
        )
        
        dependencies = []
        for line in result.strip().split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    version = parts[1]
                    description = ' '.join(parts[2:]) if len(parts) > 2 else ""
                    
                    dependencies.append({
                        "name": name,
                        "version": version,
                        "description": description,
                        "type": "python"
                    })
        
        return dependencies
    except subprocess.CalledProcessError:
        return []

def get_system_info():
    """Get system information."""
    import platform
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation()
    }

def calculate_file_hashes(dist_dir):
    """Calculate hashes for distribution files."""
    hashes = {}
    
    if not dist_dir.exists():
        return hashes
    
    for file_path in dist_dir.glob("*"):
        if file_path.is_file():
            with open(file_path, 'rb') as f:
                content = f.read()
                sha256_hash = hashlib.sha256(content).hexdigest()
                sha1_hash = hashlib.sha1(content).hexdigest()
                
                hashes[file_path.name] = {
                    "sha256": sha256_hash,
                    "sha1": sha1_hash,
                    "size": len(content)
                }
    
    return hashes

def generate_sbom():
    """Generate the complete SBOM."""
    
    # Basic project information
    sbom = {
        "sbom_version": "1.0",
        "format": "terragon-sbom",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "generator": {
            "name": "generate-sbom.py",
            "version": "1.0.0"
        },
        "project": {
            "name": "quantum-inspired-task-planner",
            "version": "1.0.0",
            "description": "QUBO-based task scheduler for agent pools using quantum annealing and classical optimization",
            "homepage": "https://github.com/your-org/quantum-inspired-task-planner",
            "license": "MIT"
        }
    }
    
    # Add git information
    git_info = get_git_info()
    if git_info:
        sbom["source"] = git_info
    
    # Add system information
    sbom["build_environment"] = get_system_info()
    
    # Add dependencies
    dependencies = get_poetry_dependencies()
    sbom["dependencies"] = {
        "count": len(dependencies),
        "packages": dependencies
    }
    
    # Add build artifacts
    dist_dir = Path("dist")
    file_hashes = calculate_file_hashes(dist_dir)
    sbom["artifacts"] = {
        "count": len(file_hashes),
        "files": file_hashes
    }
    
    # Security considerations
    sbom["security"] = {
        "supply_chain": {
            "verified_dependencies": False,  # Would need additional tooling
            "signed_artifacts": False,       # Would need signing infrastructure
            "vulnerability_scan": False     # Would need integration with vuln scanners
        },
        "build_reproducibility": {
            "reproducible": False,  # Would need hermetic builds
            "build_hash": None      # Would need build environment fingerprinting
        }
    }
    
    # Compliance information
    sbom["compliance"] = {
        "standards": [
            "SPDX-2.3",
            "CycloneDX-1.4"
        ],
        "export_control": {
            "eccn": "EAR99",  # Standard software classification
            "restrictions": "None"
        }
    }
    
    return sbom

def main():
    """Main function."""
    print("🔍 Generating Software Bill of Materials (SBOM)...")
    
    try:
        sbom = generate_sbom()
        
        # Write SBOM to file
        sbom_file = Path("build") / "sbom.json"
        sbom_file.parent.mkdir(exist_ok=True)
        
        with open(sbom_file, 'w') as f:
            json.dump(sbom, f, indent=2, sort_keys=True)
        
        print(f"✅ SBOM generated: {sbom_file}")
        print(f"   Dependencies: {sbom['dependencies']['count']}")
        print(f"   Artifacts: {sbom['artifacts']['count']}")
        
        # Also create a human-readable version
        readme_file = sbom_file.with_suffix('.txt')
        with open(readme_file, 'w') as f:
            f.write("Software Bill of Materials (SBOM)\n")
            f.write("=" * 35 + "\n\n")
            
            f.write(f"Project: {sbom['project']['name']}\n")
            f.write(f"Version: {sbom['project']['version']}\n")
            f.write(f"Generated: {sbom['generated_at']}\n\n")
            
            if 'source' in sbom:
                f.write("Source Information:\n")
                f.write(f"  Repository: {sbom['source']['repository']}\n")
                f.write(f"  Commit: {sbom['source']['commit']}\n")
                f.write(f"  Branch: {sbom['source']['branch']}\n\n")
            
            f.write("Dependencies:\n")
            for dep in sbom['dependencies']['packages']:
                f.write(f"  {dep['name']} {dep['version']}\n")
            
            f.write(f"\nBuild Artifacts:\n")
            for filename, info in sbom['artifacts']['files'].items():
                f.write(f"  {filename}\n")
                f.write(f"    SHA256: {info['sha256']}\n")
                f.write(f"    Size: {info['size']:,} bytes\n")
        
        print(f"✅ Human-readable SBOM: {readme_file}")
        
    except Exception as e:
        print(f"❌ Error generating SBOM: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()