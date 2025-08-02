#!/usr/bin/env python3
"""
Automated dependency update script for the Quantum-Inspired Task Planner.

This script automates the process of checking, updating, and validating
dependencies while ensuring compatibility and security.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import shutil


class DependencyUpdater:
    """Automated dependency management and updates."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.requirements_file = repo_root / "requirements.txt"
        self.dev_requirements_file = repo_root / "dev-requirements.txt"
        self.pyproject_file = repo_root / "pyproject.toml"
        self.backup_dir = repo_root / ".dependency_backups"
        
    def _run_command(self, command: str, cwd: Optional[Path] = None) -> Tuple[bool, str]:
        """Run shell command and return success status and output."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd or self.repo_root
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)
    
    def create_backup(self) -> str:
        """Create backup of current dependency files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        files_to_backup = [
            self.requirements_file,
            self.dev_requirements_file,
            self.pyproject_file
        ]
        
        for file_path in files_to_backup:
            if file_path.exists():
                shutil.copy2(file_path, backup_path)
        
        print(f"📦 Created backup at {backup_path}")
        return str(backup_path)
    
    def restore_backup(self, backup_path: str):
        """Restore dependencies from backup."""
        backup_dir = Path(backup_path)
        if not backup_dir.exists():
            print(f"❌ Backup directory {backup_path} does not exist")
            return False
        
        for file_name in ["requirements.txt", "dev-requirements.txt", "pyproject.toml"]:
            backup_file = backup_dir / file_name
            target_file = self.repo_root / file_name
            
            if backup_file.exists():
                shutil.copy2(backup_file, target_file)
                print(f"📄 Restored {file_name}")
        
        print(f"✅ Restored dependencies from {backup_path}")
        return True
    
    def check_outdated_packages(self) -> List[Dict[str, str]]:
        """Check for outdated packages."""
        print("🔍 Checking for outdated packages...")
        
        success, output = self._run_command("pip list --outdated --format=json")
        if not success:
            print(f"❌ Failed to check outdated packages: {output}")
            return []
        
        try:
            outdated = json.loads(output)
            print(f"📊 Found {len(outdated)} outdated packages")
            return outdated
        except json.JSONDecodeError:
            print("❌ Failed to parse outdated packages output")
            return []
    
    def check_security_vulnerabilities(self) -> List[Dict[str, str]]:
        """Check for security vulnerabilities in dependencies."""
        print("🔒 Checking for security vulnerabilities...")
        
        success, output = self._run_command("pip-audit --format=json")
        if not success:
            print(f"⚠️ pip-audit check completed with warnings: {output}")
            # pip-audit returns non-zero exit code when vulnerabilities found
            # Try to parse output anyway
        
        try:
            audit_data = json.loads(output)
            vulnerabilities = audit_data.get("vulnerabilities", [])
            print(f"🛡️ Found {len(vulnerabilities)} security vulnerabilities")
            return vulnerabilities
        except json.JSONDecodeError:
            print("❌ Failed to parse security audit output")
            return []
    
    def update_package(self, package_name: str, target_version: Optional[str] = None) -> bool:
        """Update a specific package."""
        version_spec = f"=={target_version}" if target_version else ""
        command = f"pip install --upgrade {package_name}{version_spec}"
        
        print(f"📦 Updating {package_name}{' to ' + target_version if target_version else ''}...")
        
        success, output = self._run_command(command)
        if success:
            print(f"✅ Successfully updated {package_name}")
            return True
        else:
            print(f"❌ Failed to update {package_name}: {output}")
            return False
    
    def update_requirements_file(self, file_path: Path, updates: Dict[str, str]):
        """Update requirements file with new versions."""
        if not file_path.exists():
            print(f"⚠️ Requirements file {file_path} does not exist")
            return
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                updated_lines.append(line)
                continue
            
            # Parse package name from requirement line
            package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('!=')[0]
            
            if package_name in updates:
                new_version = updates[package_name]
                updated_lines.append(f"{package_name}=={new_version}")
                print(f"📝 Updated {package_name} to {new_version} in {file_path.name}")
            else:
                updated_lines.append(line)
        
        with open(file_path, 'w') as f:
            f.write('\n'.join(updated_lines) + '\n')
    
    def run_tests(self) -> bool:
        """Run test suite to validate dependency updates."""
        print("🧪 Running test suite to validate updates...")
        
        # Run unit tests
        success, output = self._run_command("python -m pytest tests/unit/ -v")
        if not success:
            print(f"❌ Unit tests failed: {output}")
            return False
        
        # Run integration tests
        success, output = self._run_command("python -m pytest tests/integration/ -v")
        if not success:
            print(f"❌ Integration tests failed: {output}")
            return False
        
        # Run security checks
        success, output = self._run_command("python security/quantum_credential_scanner.py --scan-dir=src/")
        if not success:
            print(f"❌ Security checks failed: {output}")
            return False
        
        print("✅ All tests passed!")
        return True
    
    def check_compatibility(self) -> bool:
        """Check compatibility of updated dependencies."""
        print("🔧 Checking dependency compatibility...")
        
        # Check for dependency conflicts
        success, output = self._run_command("pip check")
        if not success:
            print(f"❌ Dependency conflicts detected: {output}")
            return False
        
        # Try importing main package
        success, output = self._run_command("python -c 'import quantum_planner; print(\"Import successful\")'")
        if not success:
            print(f"❌ Failed to import main package: {output}")
            return False
        
        print("✅ Compatibility check passed!")
        return True
    
    def update_security_vulnerabilities(self) -> Tuple[bool, List[str]]:
        """Update packages with security vulnerabilities."""
        vulnerabilities = self.check_security_vulnerabilities()
        if not vulnerabilities:
            print("✅ No security vulnerabilities found!")
            return True, []
        
        updated_packages = []
        failed_updates = []
        
        for vuln in vulnerabilities:
            package_name = vuln.get("package", "")
            current_version = vuln.get("installed_version", "")
            fixed_versions = vuln.get("fixed_versions", [])
            
            if not package_name or not fixed_versions:
                continue
            
            # Use the latest fixed version
            target_version = max(fixed_versions) if fixed_versions else None
            
            print(f"🛡️ Updating {package_name} from {current_version} to {target_version} (security fix)")
            
            if self.update_package(package_name, target_version):
                updated_packages.append(f"{package_name}: {current_version} -> {target_version}")
            else:
                failed_updates.append(package_name)
        
        if failed_updates:
            print(f"❌ Failed to update security vulnerabilities in: {', '.join(failed_updates)}")
            return False, failed_updates
        
        print(f"✅ Successfully updated {len(updated_packages)} packages for security fixes")
        return True, updated_packages
    
    def update_outdated_packages(self, auto_update_minor: bool = True, auto_update_patch: bool = True) -> Tuple[bool, List[str]]:
        """Update outdated packages based on update policy."""
        outdated = self.check_outdated_packages()
        if not outdated:
            print("✅ All packages are up to date!")
            return True, []
        
        updated_packages = []
        failed_updates = []
        
        for package in outdated:
            package_name = package["name"]
            current_version = package["version"]
            latest_version = package["latest_version"]
            
            # Parse version numbers
            current_parts = current_version.split(".")
            latest_parts = latest_version.split(".")
            
            # Determine update type
            if len(current_parts) >= 3 and len(latest_parts) >= 3:
                is_major = current_parts[0] != latest_parts[0]
                is_minor = current_parts[1] != latest_parts[1]
                is_patch = current_parts[2] != latest_parts[2]
                
                # Apply update policy
                should_update = False
                if is_patch and auto_update_patch:
                    should_update = True
                elif is_minor and auto_update_minor and not is_major:
                    should_update = True
                elif not is_major and not is_minor and is_patch:
                    should_update = True
                
                if should_update:
                    print(f"📦 Updating {package_name} from {current_version} to {latest_version}")
                    if self.update_package(package_name, latest_version):
                        updated_packages.append(f"{package_name}: {current_version} -> {latest_version}")
                    else:
                        failed_updates.append(package_name)
                else:
                    print(f"⏭️ Skipping {package_name} (major version update requires manual review)")
            else:
                print(f"⚠️ Skipping {package_name} (unable to parse version)")
        
        if failed_updates:
            print(f"❌ Failed to update: {', '.join(failed_updates)}")
            return False, failed_updates
        
        print(f"✅ Successfully updated {len(updated_packages)} packages")
        return True, updated_packages
    
    def generate_update_report(self, security_updates: List[str], package_updates: List[str]) -> str:
        """Generate dependency update report."""
        report_path = self.repo_root / f"dependency_update_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write("# Dependency Update Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
            
            f.write("## Security Updates\n\n")
            if security_updates:
                for update in security_updates:
                    f.write(f"- {update}\n")
            else:
                f.write("No security updates required.\n")
            
            f.write("\n## Package Updates\n\n")
            if package_updates:
                for update in package_updates:
                    f.write(f"- {update}\n")
            else:
                f.write("No package updates performed.\n")
            
            f.write("\n## Validation\n\n")
            f.write("- [x] Compatibility check passed\n")
            f.write("- [x] Test suite passed\n")
            f.write("- [x] Security scan clean\n")
            
            f.write("\n## Next Steps\n\n")
            f.write("1. Review changes and commit if appropriate\n")
            f.write("2. Monitor application performance\n")
            f.write("3. Update deployment if necessary\n")
        
        print(f"📄 Generated update report: {report_path}")
        return str(report_path)
    
    def perform_full_update(self, dry_run: bool = False) -> bool:
        """Perform full dependency update workflow."""
        if dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")
        
        # Create backup
        backup_path = self.create_backup()
        
        try:
            # Update security vulnerabilities first
            if not dry_run:
                security_success, security_updates = self.update_security_vulnerabilities()
                if not security_success:
                    print("❌ Security updates failed, restoring backup")
                    self.restore_backup(backup_path)
                    return False
            else:
                security_updates = []
            
            # Update outdated packages
            if not dry_run:
                package_success, package_updates = self.update_outdated_packages()
                if not package_success:
                    print("❌ Package updates failed, restoring backup")
                    self.restore_backup(backup_path)
                    return False
            else:
                package_updates = []
            
            # Validate updates
            if not dry_run:
                if not self.check_compatibility():
                    print("❌ Compatibility check failed, restoring backup")
                    self.restore_backup(backup_path)
                    return False
                
                if not self.run_tests():
                    print("❌ Tests failed, restoring backup")
                    self.restore_backup(backup_path)
                    return False
            
            # Generate report
            if not dry_run:
                report_path = self.generate_update_report(security_updates, package_updates)
            
            print("✅ Dependency update completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Unexpected error during update: {e}")
            if not dry_run:
                self.restore_backup(backup_path)
            return False


def main():
    """Main function for dependency updates."""
    repo_root = Path(__file__).parent.parent.parent
    updater = DependencyUpdater(repo_root)
    
    import argparse
    parser = argparse.ArgumentParser(description="Automated dependency updater")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without making changes")
    parser.add_argument("--security-only", action="store_true", help="Only update packages with security vulnerabilities")
    parser.add_argument("--check-only", action="store_true", help="Only check for outdated packages and vulnerabilities")
    
    args = parser.parse_args()
    
    try:
        if args.check_only:
            print("🔍 Checking dependencies...")
            outdated = updater.check_outdated_packages()
            vulnerabilities = updater.check_security_vulnerabilities()
            
            if outdated:
                print(f"\n📊 Outdated packages ({len(outdated)}):")
                for pkg in outdated[:10]:  # Show first 10
                    print(f"  - {pkg['name']}: {pkg['version']} -> {pkg['latest_version']}")
                if len(outdated) > 10:
                    print(f"  ... and {len(outdated) - 10} more")
            
            if vulnerabilities:
                print(f"\n🛡️ Security vulnerabilities ({len(vulnerabilities)}):")
                for vuln in vulnerabilities[:5]:  # Show first 5
                    pkg = vuln.get("package", "unknown")
                    severity = vuln.get("severity", "unknown")
                    print(f"  - {pkg}: {severity} severity")
                if len(vulnerabilities) > 5:
                    print(f"  ... and {len(vulnerabilities) - 5} more")
            
        elif args.security_only:
            print("🛡️ Updating security vulnerabilities only...")
            backup_path = updater.create_backup()
            success, updates = updater.update_security_vulnerabilities()
            
            if success and not args.dry_run:
                if updater.check_compatibility() and updater.run_tests():
                    print("✅ Security updates completed successfully!")
                else:
                    updater.restore_backup(backup_path)
                    sys.exit(1)
            elif not success:
                sys.exit(1)
        else:
            success = updater.perform_full_update(dry_run=args.dry_run)
            if not success:
                sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Update interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during dependency update: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()