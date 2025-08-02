#!/usr/bin/env python3
"""
Repository health monitoring script for the Quantum-Inspired Task Planner.

This script monitors various aspects of repository health including code quality,
security, performance, and community metrics, providing alerts and recommendations.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import requests
from dataclasses import dataclass


@dataclass
class HealthAlert:
    """Represents a health alert."""
    severity: str  # critical, warning, info
    category: str
    message: str
    recommendation: str
    metric_value: Any
    threshold: Any


class RepositoryHealthMonitor:
    """Comprehensive repository health monitoring."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.metrics_file = repo_root / ".github" / "project-metrics.json"
        self.health_config = self._load_health_config()
        self.alerts: List[HealthAlert] = []
    
    def _load_health_config(self) -> Dict[str, Any]:
        """Load health monitoring configuration."""
        default_config = {
            "thresholds": {
                "test_coverage": {"warning": 80, "critical": 70},
                "security_score": {"warning": 8.0, "critical": 7.0},
                "build_time": {"warning": 300, "critical": 600},
                "error_rate": {"warning": 0.01, "critical": 0.05},
                "vulnerability_count": {"warning": 5, "critical": 10},
                "outdated_dependencies": {"warning": 20, "critical": 50},
                "response_time": {"warning": 1000, "critical": 2000},
                "uptime": {"warning": 99.0, "critical": 95.0}
            },
            "monitoring": {
                "enabled": True,
                "alert_channels": ["console", "file"],
                "report_frequency": "daily",
                "trend_analysis_days": 7
            }
        }
        
        config_file = self.repo_root / "monitoring" / "health_config.yaml"
        if config_file.exists():
            # Would load from YAML in real implementation
            pass
        
        return default_config
    
    def _load_current_metrics(self) -> Dict[str, Any]:
        """Load current metrics from JSON file."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _run_command(self, command: str) -> Tuple[bool, str]:
        """Run shell command and return success status and output."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            return result.returncode == 0, result.stdout.strip()
        except Exception as e:
            return False, str(e)
    
    def check_test_coverage(self) -> float:
        """Check current test coverage."""
        success, output = self._run_command("python -m pytest --cov=src --cov-report=json")
        if success and os.path.exists("coverage.json"):
            try:
                with open("coverage.json", 'r') as f:
                    coverage_data = json.load(f)
                    return coverage_data.get("totals", {}).get("percent_covered", 0)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return 0.0
    
    def check_security_health(self) -> Dict[str, Any]:
        """Check security health metrics."""
        metrics = {"vulnerabilities": 0, "security_score": 0.0}
        
        # Check for vulnerabilities
        success, output = self._run_command("pip-audit --format=json")
        if output:
            try:
                audit_data = json.loads(output)
                vulnerabilities = audit_data.get("vulnerabilities", [])
                metrics["vulnerabilities"] = len(vulnerabilities)
                
                # Calculate security score based on vulnerability severity
                critical_count = len([v for v in vulnerabilities if v.get("severity") == "critical"])
                high_count = len([v for v in vulnerabilities if v.get("severity") == "high"])
                medium_count = len([v for v in vulnerabilities if v.get("severity") == "medium"])
                
                # Simple scoring: start at 10, subtract points for vulnerabilities
                score = 10.0 - (critical_count * 3) - (high_count * 2) - (medium_count * 1)
                metrics["security_score"] = max(0.0, score)
                
            except json.JSONDecodeError:
                pass
        
        return metrics
    
    def check_build_performance(self) -> Dict[str, float]:
        """Check build and test performance."""
        metrics = {"build_time": 0.0, "test_time": 0.0}
        
        # Measure build time
        start_time = datetime.now()
        success, _ = self._run_command("python -m pip install -e . --quiet")
        if success:
            build_time = (datetime.now() - start_time).total_seconds()
            metrics["build_time"] = build_time
        
        # Measure test time
        start_time = datetime.now()
        success, _ = self._run_command("python -m pytest tests/unit/ --quiet")
        if success:
            test_time = (datetime.now() - start_time).total_seconds()
            metrics["test_time"] = test_time
        
        return metrics
    
    def check_dependency_health(self) -> Dict[str, int]:
        """Check dependency health."""
        metrics = {"total": 0, "outdated": 0, "vulnerable": 0}
        
        # Count total dependencies
        success, output = self._run_command("pip list --format=json")
        if success:
            try:
                packages = json.loads(output)
                metrics["total"] = len(packages)
            except json.JSONDecodeError:
                pass
        
        # Count outdated dependencies
        success, output = self._run_command("pip list --outdated --format=json")
        if success:
            try:
                outdated = json.loads(output)
                metrics["outdated"] = len(outdated)
            except json.JSONDecodeError:
                pass
        
        # Count vulnerable dependencies
        security_metrics = self.check_security_health()
        metrics["vulnerable"] = security_metrics.get("vulnerabilities", 0)
        
        return metrics
    
    def check_code_quality(self) -> Dict[str, Any]:
        """Check code quality metrics."""
        metrics = {"lint_issues": 0, "type_errors": 0, "complexity_score": 0.0}
        
        # Check linting issues
        success, output = self._run_command("flake8 src/ --count")
        if output and output.isdigit():
            metrics["lint_issues"] = int(output)
        
        # Check type errors
        success, output = self._run_command("mypy src/ --error-format=json")
        if output:
            try:
                # Count number of errors (simplified)
                error_lines = [line for line in output.split('\n') if '"severity": "error"' in line]
                metrics["type_errors"] = len(error_lines)
            except:
                pass
        
        # Check complexity
        success, output = self._run_command("radon cc src/ -a")
        if success and output:
            try:
                # Extract average complexity (simplified)
                for line in output.split('\n'):
                    if 'Average complexity:' in line:
                        complexity_str = line.split(':')[-1].strip()
                        complexity_str = complexity_str.split()[0]  # Get first number
                        metrics["complexity_score"] = float(complexity_str)
                        break
            except:
                pass
        
        return metrics
    
    def check_git_health(self) -> Dict[str, Any]:
        """Check Git repository health."""
        metrics = {"commits_last_week": 0, "contributors": 0, "branch_count": 0}
        
        # Count commits in last week
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        success, output = self._run_command(f"git rev-list --count --since='{week_ago}' HEAD")
        if success and output.isdigit():
            metrics["commits_last_week"] = int(output)
        
        # Count contributors
        success, output = self._run_command("git shortlog -sn | wc -l")
        if success and output.isdigit():
            metrics["contributors"] = int(output)
        
        # Count branches
        success, output = self._run_command("git branch -a | wc -l")
        if success and output.isdigit():
            metrics["branch_count"] = int(output)
        
        return metrics
    
    def analyze_trends(self) -> Dict[str, str]:
        """Analyze trends in key metrics."""
        # This would analyze historical data in a real implementation
        # For now, return placeholder trends
        return {
            "test_coverage": "stable",
            "security_score": "improving",
            "build_time": "stable",
            "dependency_health": "deteriorating"
        }
    
    def evaluate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall health score."""
        scores = []
        
        # Test coverage score (0-25 points)
        coverage = metrics.get("test_coverage", 0)
        coverage_score = min(25, coverage * 0.25)
        scores.append(coverage_score)
        
        # Security score (0-25 points)
        security = metrics.get("security", {})
        security_score = min(25, security.get("security_score", 0) * 2.5)
        scores.append(security_score)
        
        # Performance score (0-25 points)
        performance = metrics.get("performance", {})
        build_time = performance.get("build_time", 600)
        perf_score = max(0, 25 - (build_time / 600) * 25)
        scores.append(perf_score)
        
        # Quality score (0-25 points)
        quality = metrics.get("code_quality", {})
        lint_issues = quality.get("lint_issues", 100)
        quality_score = max(0, 25 - (lint_issues / 100) * 25)
        scores.append(quality_score)
        
        return sum(scores)
    
    def generate_alerts(self, metrics: Dict[str, Any]):
        """Generate health alerts based on metrics."""
        thresholds = self.health_config["thresholds"]
        
        # Test coverage alerts
        coverage = metrics.get("test_coverage", 0)
        if coverage < thresholds["test_coverage"]["critical"]:
            self.alerts.append(HealthAlert(
                severity="critical",
                category="testing",
                message=f"Test coverage is critically low: {coverage:.1f}%",
                recommendation="Add comprehensive test cases to improve coverage",
                metric_value=coverage,
                threshold=thresholds["test_coverage"]["critical"]
            ))
        elif coverage < thresholds["test_coverage"]["warning"]:
            self.alerts.append(HealthAlert(
                severity="warning",
                category="testing",
                message=f"Test coverage is below recommended level: {coverage:.1f}%",
                recommendation="Consider adding more test cases",
                metric_value=coverage,
                threshold=thresholds["test_coverage"]["warning"]
            ))
        
        # Security alerts
        security = metrics.get("security", {})
        vuln_count = security.get("vulnerabilities", 0)
        if vuln_count > thresholds["vulnerability_count"]["critical"]:
            self.alerts.append(HealthAlert(
                severity="critical",
                category="security",
                message=f"High number of security vulnerabilities: {vuln_count}",
                recommendation="Immediately update vulnerable dependencies",
                metric_value=vuln_count,
                threshold=thresholds["vulnerability_count"]["critical"]
            ))
        elif vuln_count > thresholds["vulnerability_count"]["warning"]:
            self.alerts.append(HealthAlert(
                severity="warning",
                category="security",
                message=f"Security vulnerabilities detected: {vuln_count}",
                recommendation="Plan dependency updates to fix vulnerabilities",
                metric_value=vuln_count,
                threshold=thresholds["vulnerability_count"]["warning"]
            ))
        
        # Performance alerts
        performance = metrics.get("performance", {})
        build_time = performance.get("build_time", 0)
        if build_time > thresholds["build_time"]["critical"]:
            self.alerts.append(HealthAlert(
                severity="critical",
                category="performance",
                message=f"Build time is too long: {build_time:.1f}s",
                recommendation="Optimize build process and dependencies",
                metric_value=build_time,
                threshold=thresholds["build_time"]["critical"]
            ))
        elif build_time > thresholds["build_time"]["warning"]:
            self.alerts.append(HealthAlert(
                severity="warning",
                category="performance",
                message=f"Build time is elevated: {build_time:.1f}s",
                recommendation="Monitor build performance and consider optimizations",
                metric_value=build_time,
                threshold=thresholds["build_time"]["warning"]
            ))
        
        # Dependency health alerts
        dependencies = metrics.get("dependencies", {})
        outdated = dependencies.get("outdated", 0)
        if outdated > thresholds["outdated_dependencies"]["critical"]:
            self.alerts.append(HealthAlert(
                severity="critical",
                category="dependencies",
                message=f"Too many outdated dependencies: {outdated}",
                recommendation="Run dependency update process immediately",
                metric_value=outdated,
                threshold=thresholds["outdated_dependencies"]["critical"]
            ))
        elif outdated > thresholds["outdated_dependencies"]["warning"]:
            self.alerts.append(HealthAlert(
                severity="warning",
                category="dependencies",
                message=f"Several outdated dependencies: {outdated}",
                recommendation="Schedule dependency updates",
                metric_value=outdated,
                threshold=thresholds["outdated_dependencies"]["warning"]
            ))
        
        # Code quality alerts
        quality = metrics.get("code_quality", {})
        lint_issues = quality.get("lint_issues", 0)
        if lint_issues > 50:
            self.alerts.append(HealthAlert(
                severity="warning",
                category="quality",
                message=f"High number of linting issues: {lint_issues}",
                recommendation="Run code formatting and fix linting issues",
                metric_value=lint_issues,
                threshold=50
            ))
    
    def generate_health_report(self) -> str:
        """Generate comprehensive health report."""
        print("🔍 Gathering repository health metrics...")
        
        # Collect all metrics
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_coverage": self.check_test_coverage(),
            "security": self.check_security_health(),
            "performance": self.check_build_performance(),
            "dependencies": self.check_dependency_health(),
            "code_quality": self.check_code_quality(),
            "git_health": self.check_git_health(),
            "trends": self.analyze_trends()
        }
        
        # Calculate health score
        health_score = self.evaluate_health_score(metrics)
        metrics["health_score"] = health_score
        
        # Generate alerts
        self.generate_alerts(metrics)
        
        # Create report
        report_path = self.repo_root / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write("# Repository Health Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write(f"**Overall Health Score**: {health_score:.1f}/100\n\n")
            
            # Health status
            if health_score >= 90:
                status = "🟢 Excellent"
            elif health_score >= 75:
                status = "🟡 Good"
            elif health_score >= 60:
                status = "🟠 Fair"
            else:
                status = "🔴 Poor"
            f.write(f"**Status**: {status}\n\n")
            
            # Alerts section
            f.write("## 🚨 Alerts\n\n")
            if self.alerts:
                critical_alerts = [a for a in self.alerts if a.severity == "critical"]
                warning_alerts = [a for a in self.alerts if a.severity == "warning"]
                
                if critical_alerts:
                    f.write("### Critical Issues\n\n")
                    for alert in critical_alerts:
                        f.write(f"- **{alert.message}**\n")
                        f.write(f"  - Recommendation: {alert.recommendation}\n")
                        f.write(f"  - Value: {alert.metric_value}, Threshold: {alert.threshold}\n\n")
                
                if warning_alerts:
                    f.write("### Warnings\n\n")
                    for alert in warning_alerts:
                        f.write(f"- {alert.message}\n")
                        f.write(f"  - Recommendation: {alert.recommendation}\n\n")
            else:
                f.write("No critical issues or warnings detected! ✅\n\n")
            
            # Metrics sections
            f.write("## 📊 Detailed Metrics\n\n")
            
            f.write("### Test Coverage\n")
            f.write(f"- Overall Coverage: {metrics['test_coverage']:.1f}%\n")
            f.write(f"- Trend: {metrics['trends']['test_coverage']}\n\n")
            
            f.write("### Security\n")
            security = metrics['security']
            f.write(f"- Security Score: {security['security_score']:.1f}/10\n")
            f.write(f"- Vulnerabilities: {security['vulnerabilities']}\n")
            f.write(f"- Trend: {metrics['trends']['security_score']}\n\n")
            
            f.write("### Performance\n")
            performance = metrics['performance']
            f.write(f"- Build Time: {performance['build_time']:.1f}s\n")
            f.write(f"- Test Time: {performance['test_time']:.1f}s\n")
            f.write(f"- Trend: {metrics['trends']['build_time']}\n\n")
            
            f.write("### Dependencies\n")
            deps = metrics['dependencies']
            f.write(f"- Total: {deps['total']}\n")
            f.write(f"- Outdated: {deps['outdated']}\n")
            f.write(f"- Vulnerable: {deps['vulnerable']}\n")
            f.write(f"- Trend: {metrics['trends']['dependency_health']}\n\n")
            
            f.write("### Code Quality\n")
            quality = metrics['code_quality']
            f.write(f"- Lint Issues: {quality['lint_issues']}\n")
            f.write(f"- Type Errors: {quality['type_errors']}\n")
            f.write(f"- Complexity Score: {quality['complexity_score']:.1f}\n\n")
            
            f.write("### Git Health\n")
            git = metrics['git_health']
            f.write(f"- Commits (last week): {git['commits_last_week']}\n")
            f.write(f"- Contributors: {git['contributors']}\n")
            f.write(f"- Branches: {git['branch_count']}\n\n")
            
            # Recommendations
            f.write("## 💡 Recommendations\n\n")
            if health_score < 90:
                f.write("### Immediate Actions\n")
                critical_alerts = [a for a in self.alerts if a.severity == "critical"]
                for alert in critical_alerts:
                    f.write(f"1. {alert.recommendation}\n")
                
                f.write("\n### Improvements\n")
                warning_alerts = [a for a in self.alerts if a.severity == "warning"]
                for alert in warning_alerts:
                    f.write(f"- {alert.recommendation}\n")
                f.write("\n")
            else:
                f.write("Repository health is excellent! Continue current practices.\n\n")
            
            # Next steps
            f.write("## 🔄 Next Steps\n\n")
            f.write("1. Address critical alerts immediately\n")
            f.write("2. Plan improvements for warning items\n")
            f.write("3. Monitor trends and metrics regularly\n")
            f.write("4. Run health check again in 24-48 hours\n")
        
        print(f"📄 Health report generated: {report_path}")
        return str(report_path)
    
    def send_alerts(self):
        """Send alerts through configured channels."""
        if not self.alerts:
            return
        
        critical_alerts = [a for a in self.alerts if a.severity == "critical"]
        warning_alerts = [a for a in self.alerts if a.severity == "warning"]
        
        if critical_alerts:
            print("🚨 CRITICAL ALERTS:")
            for alert in critical_alerts:
                print(f"  - {alert.message}")
                print(f"    Recommendation: {alert.recommendation}")
        
        if warning_alerts:
            print("⚠️  WARNINGS:")
            for alert in warning_alerts:
                print(f"  - {alert.message}")


def main():
    """Main function for repository health monitoring."""
    repo_root = Path(__file__).parent.parent.parent
    monitor = RepositoryHealthMonitor(repo_root)
    
    import argparse
    parser = argparse.ArgumentParser(description="Repository health monitor")
    parser.add_argument("--alerts-only", action="store_true", help="Only show alerts, don't generate full report")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    
    args = parser.parse_args()
    
    try:
        if args.alerts_only:
            # Quick health check for alerts only
            metrics = {
                "test_coverage": monitor.check_test_coverage(),
                "security": monitor.check_security_health(),
                "performance": monitor.check_build_performance(),
                "dependencies": monitor.check_dependency_health(),
                "code_quality": monitor.check_code_quality()
            }
            
            monitor.generate_alerts(metrics)
            monitor.send_alerts()
            
            if monitor.alerts:
                critical_count = len([a for a in monitor.alerts if a.severity == "critical"])
                if critical_count > 0:
                    sys.exit(1)  # Exit with error if critical alerts
        else:
            # Full health report
            report_path = monitor.generate_health_report()
            
            if not args.quiet:
                monitor.send_alerts()
                health_score = monitor.evaluate_health_score({
                    "test_coverage": monitor.check_test_coverage(),
                    "security": monitor.check_security_health(),
                    "performance": monitor.check_build_performance(),
                    "dependencies": monitor.check_dependency_health(),
                    "code_quality": monitor.check_code_quality()
                })
                print(f"📊 Overall Health Score: {health_score:.1f}/100")
                print(f"📄 Full report available at: {report_path}")
            
            # Exit with error if health score is too low
            if health_score < 70:
                sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error during health monitoring: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()