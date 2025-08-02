#!/usr/bin/env python3
"""
Automation scheduler for the Quantum-Inspired Task Planner repository.

This script coordinates and schedules various automation tasks including
metrics collection, dependency updates, health monitoring, and reporting.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import schedule
import logging
from dataclasses import dataclass, asdict


@dataclass
class AutomationTask:
    """Represents an automation task."""
    name: str
    script_path: str
    schedule_type: str  # daily, weekly, monthly, hourly
    schedule_time: str  # e.g., "09:00", "monday", "1st"
    enabled: bool = True
    last_run: Optional[str] = None
    last_status: Optional[str] = None
    last_duration: Optional[float] = None
    max_duration: Optional[int] = None  # seconds
    retry_count: int = 3
    dependencies: List[str] = None
    notification_channels: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.notification_channels is None:
            self.notification_channels = ["console"]


class AutomationScheduler:
    """Coordinates and schedules repository automation tasks."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.scripts_dir = repo_root / "scripts" / "automation"
        self.config_file = repo_root / ".github" / "automation-config.json"
        self.log_file = repo_root / "logs" / "automation.log"
        self.tasks: Dict[str, AutomationTask] = {}
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self._load_configuration()
        
        # Initialize scheduler
        self._setup_scheduler()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.repo_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_configuration(self):
        """Load automation configuration."""
        default_tasks = [
            AutomationTask(
                name="metrics_collection",
                script_path="scripts/automation/metrics_collector.py",
                schedule_type="daily",
                schedule_time="09:00",
                max_duration=300,  # 5 minutes
                notification_channels=["console", "slack"]
            ),
            AutomationTask(
                name="dependency_security_check",
                script_path="scripts/automation/dependency_updater.py",
                schedule_type="daily",
                schedule_time="10:00",
                max_duration=600,  # 10 minutes
                notification_channels=["console", "slack"]
            ),
            AutomationTask(
                name="health_monitoring",
                script_path="scripts/automation/repository_health_monitor.py",
                schedule_type="daily",
                schedule_time="11:00",
                max_duration=180,  # 3 minutes
                dependencies=["metrics_collection"],
                notification_channels=["console", "slack"]
            ),
            AutomationTask(
                name="weekly_dependency_update",
                script_path="scripts/automation/dependency_updater.py",
                schedule_type="weekly",
                schedule_time="monday",
                max_duration=1800,  # 30 minutes
                notification_channels=["console", "slack", "email"]
            ),
            AutomationTask(
                name="weekly_security_audit",
                script_path="scripts/automation/security_audit.py",
                schedule_type="weekly",
                schedule_time="sunday",
                max_duration=900,  # 15 minutes
                notification_channels=["console", "slack", "email"]
            ),
            AutomationTask(
                name="monthly_cleanup",
                script_path="scripts/automation/repository_cleanup.py",
                schedule_type="monthly",
                schedule_time="1st",
                max_duration=3600,  # 1 hour
                notification_channels=["console", "email"]
            ),
            AutomationTask(
                name="performance_benchmark",
                script_path="scripts/automation/performance_monitor.py",
                schedule_type="daily",
                schedule_time="14:00",
                max_duration=1200,  # 20 minutes
                notification_channels=["console"]
            )
        ]
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    
                # Load tasks from config
                for task_data in config_data.get("tasks", []):
                    task = AutomationTask(**task_data)
                    self.tasks[task.name] = task
                    
            except (json.JSONDecodeError, TypeError) as e:
                self.logger.warning(f"Failed to load config file: {e}, using defaults")
                
        # Use defaults if no config loaded
        if not self.tasks:
            for task in default_tasks:
                self.tasks[task.name] = task
            self._save_configuration()
    
    def _save_configuration(self):
        """Save current configuration to file."""
        config_data = {
            "tasks": [asdict(task) for task in self.tasks.values()],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def _setup_scheduler(self):
        """Setup task scheduler."""
        for task_name, task in self.tasks.items():
            if not task.enabled:
                continue
                
            if task.schedule_type == "daily":
                schedule.every().day.at(task.schedule_time).do(
                    self._execute_task, task_name
                )
            elif task.schedule_type == "weekly":
                getattr(schedule.every(), task.schedule_time.lower()).do(
                    self._execute_task, task_name
                )
            elif task.schedule_type == "monthly":
                # Simplified monthly scheduling (run on first day of month)
                schedule.every().day.at("00:00").do(
                    self._check_monthly_task, task_name
                )
            elif task.schedule_type == "hourly":
                schedule.every().hour.do(self._execute_task, task_name)
    
    def _check_monthly_task(self, task_name: str):
        """Check if monthly task should run today."""
        today = datetime.now()
        if today.day == 1:  # First day of month
            self._execute_task(task_name)
    
    def _execute_task(self, task_name: str) -> bool:
        """Execute a specific automation task."""
        task = self.tasks.get(task_name)
        if not task or not task.enabled:
            return False
        
        self.logger.info(f"Starting task: {task_name}")
        start_time = datetime.now()
        
        # Check dependencies
        if not self._check_dependencies(task):
            self.logger.error(f"Dependencies not met for task: {task_name}")
            return False
        
        # Execute task with retry logic
        success = False
        attempt = 0
        
        while attempt < task.retry_count and not success:
            attempt += 1
            try:
                success = self._run_task_script(task, attempt)
            except Exception as e:
                self.logger.error(f"Task {task_name} failed on attempt {attempt}: {e}")
        
        # Update task status
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        task.last_run = start_time.isoformat()
        task.last_status = "success" if success else "failed"
        task.last_duration = duration
        
        # Check for timeout
        if task.max_duration and duration > task.max_duration:
            self.logger.warning(f"Task {task_name} exceeded max duration: {duration}s > {task.max_duration}s")
        
        # Send notifications
        self._send_notifications(task, success, duration)
        
        # Save updated configuration
        self._save_configuration()
        
        if success:
            self.logger.info(f"Task {task_name} completed successfully in {duration:.1f}s")
        else:
            self.logger.error(f"Task {task_name} failed after {attempt} attempts")
        
        return success
    
    def _check_dependencies(self, task: AutomationTask) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_name in task.dependencies:
            dep_task = self.tasks.get(dep_name)
            if not dep_task:
                continue
                
            # Check if dependency ran successfully recently (within 24 hours)
            if dep_task.last_run:
                last_run = datetime.fromisoformat(dep_task.last_run.replace('Z', '+00:00'))
                if (datetime.now(timezone.utc) - last_run).total_seconds() > 86400:  # 24 hours
                    return False
                    
                if dep_task.last_status != "success":
                    return False
            else:
                return False
        
        return True
    
    def _run_task_script(self, task: AutomationTask, attempt: int) -> bool:
        """Run the task's script."""
        script_path = self.repo_root / task.script_path
        
        if not script_path.exists():
            self.logger.error(f"Script not found: {script_path}")
            return False
        
        # Prepare command
        if task.name == "dependency_security_check":
            command = f"python {script_path} --security-only"
        elif task.name == "weekly_dependency_update":
            command = f"python {script_path}"
        elif task.name == "health_monitoring":
            command = f"python {script_path}"
        else:
            command = f"python {script_path}"
        
        # Add attempt info for logging
        env = os.environ.copy()
        env["AUTOMATION_ATTEMPT"] = str(attempt)
        env["AUTOMATION_TASK"] = task.name
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                env=env,
                timeout=task.max_duration
            )
            
            if result.returncode == 0:
                self.logger.debug(f"Task {task.name} output: {result.stdout}")
                return True
            else:
                self.logger.error(f"Task {task.name} failed with exit code {result.returncode}")
                self.logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Task {task.name} timed out after {task.max_duration}s")
            return False
        except Exception as e:
            self.logger.error(f"Failed to execute task {task.name}: {e}")
            return False
    
    def _send_notifications(self, task: AutomationTask, success: bool, duration: float):
        """Send notifications about task completion."""
        status = "✅ Success" if success else "❌ Failed"
        message = f"Task '{task.name}' {status.lower()} in {duration:.1f}s"
        
        for channel in task.notification_channels:
            if channel == "console":
                print(f"{status}: {message}")
            elif channel == "slack":
                self._send_slack_notification(message, success)
            elif channel == "email":
                self._send_email_notification(task, success, duration)
    
    def _send_slack_notification(self, message: str, success: bool):
        """Send Slack notification."""
        # Placeholder for Slack integration
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if webhook_url:
            try:
                import requests
                payload = {
                    "text": f"🤖 Automation: {message}",
                    "color": "good" if success else "danger"
                }
                requests.post(webhook_url, json=payload, timeout=10)
            except Exception as e:
                self.logger.warning(f"Failed to send Slack notification: {e}")
    
    def _send_email_notification(self, task: AutomationTask, success: bool, duration: float):
        """Send email notification."""
        # Placeholder for email integration
        if not success or (task.max_duration and duration > task.max_duration):
            self.logger.info(f"Would send email notification for task: {task.name}")
    
    def run_task_now(self, task_name: str) -> bool:
        """Run a specific task immediately."""
        if task_name not in self.tasks:
            self.logger.error(f"Unknown task: {task_name}")
            return False
        
        return self._execute_task(task_name)
    
    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all configured tasks."""
        task_list = []
        for task in self.tasks.values():
            task_info = {
                "name": task.name,
                "enabled": task.enabled,
                "schedule": f"{task.schedule_type} at {task.schedule_time}",
                "last_run": task.last_run,
                "last_status": task.last_status,
                "last_duration": f"{task.last_duration:.1f}s" if task.last_duration else None
            }
            task_list.append(task_info)
        return task_list
    
    def enable_task(self, task_name: str):
        """Enable a task."""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = True
            self._save_configuration()
            self.logger.info(f"Enabled task: {task_name}")
        else:
            self.logger.error(f"Unknown task: {task_name}")
    
    def disable_task(self, task_name: str):
        """Disable a task."""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = False
            self._save_configuration()
            self.logger.info(f"Disabled task: {task_name}")
        else:
            self.logger.error(f"Unknown task: {task_name}")
    
    def generate_status_report(self) -> str:
        """Generate automation status report."""
        report_path = self.repo_root / f"automation_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write("# Automation Status Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
            
            # Task status summary
            total_tasks = len(self.tasks)
            enabled_tasks = len([t for t in self.tasks.values() if t.enabled])
            recent_failures = len([t for t in self.tasks.values() 
                                 if t.last_status == "failed" and t.last_run and 
                                 (datetime.now() - datetime.fromisoformat(t.last_run.replace('Z', '+00:00'))).days < 7])
            
            f.write(f"## Summary\n\n")
            f.write(f"- **Total Tasks**: {total_tasks}\n")
            f.write(f"- **Enabled Tasks**: {enabled_tasks}\n")
            f.write(f"- **Recent Failures**: {recent_failures}\n\n")
            
            # Individual task status
            f.write("## Task Status\n\n")
            for task in self.tasks.values():
                status_icon = "✅" if task.last_status == "success" else "❌" if task.last_status == "failed" else "⏸️"
                enabled_icon = "🟢" if task.enabled else "🔴"
                
                f.write(f"### {enabled_icon} {task.name}\n")
                f.write(f"- **Status**: {status_icon} {task.last_status or 'never run'}\n")
                f.write(f"- **Schedule**: {task.schedule_type} at {task.schedule_time}\n")
                f.write(f"- **Last Run**: {task.last_run or 'never'}\n")
                f.write(f"- **Duration**: {task.last_duration:.1f}s" if task.last_duration else "N/A")
                f.write("\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if recent_failures > 0:
                f.write("- ⚠️ Review recent task failures and resolve issues\n")
            if enabled_tasks < total_tasks:
                f.write("- 🔧 Consider enabling disabled tasks if appropriate\n")
            f.write("- 📊 Monitor task execution times and adjust schedules if needed\n")
            f.write("- 🔄 Review automation logs regularly for issues\n")
        
        self.logger.info(f"Status report generated: {report_path}")
        return str(report_path)
    
    def start_scheduler(self):
        """Start the scheduler daemon."""
        self.logger.info("Starting automation scheduler...")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"Scheduler error: {e}")


def main():
    """Main function for automation scheduler."""
    repo_root = Path(__file__).parent.parent.parent
    scheduler = AutomationScheduler(repo_root)
    
    import argparse
    parser = argparse.ArgumentParser(description="Repository automation scheduler")
    parser.add_argument("--run-task", help="Run a specific task immediately")
    parser.add_argument("--list-tasks", action="store_true", help="List all configured tasks")
    parser.add_argument("--status-report", action="store_true", help="Generate status report")
    parser.add_argument("--enable-task", help="Enable a specific task")
    parser.add_argument("--disable-task", help="Disable a specific task")
    parser.add_argument("--start-daemon", action="store_true", help="Start scheduler daemon")
    
    args = parser.parse_args()
    
    try:
        if args.run_task:
            success = scheduler.run_task_now(args.run_task)
            sys.exit(0 if success else 1)
        
        elif args.list_tasks:
            tasks = scheduler.list_tasks()
            print("\n📋 Configured Tasks:")
            for task in tasks:
                status_icon = "✅" if task["last_status"] == "success" else "❌" if task["last_status"] == "failed" else "⏸️"
                enabled_icon = "🟢" if task["enabled"] else "🔴"
                print(f"  {enabled_icon} {status_icon} {task['name']}")
                print(f"    Schedule: {task['schedule']}")
                print(f"    Last run: {task['last_run'] or 'never'}")
                if task['last_duration']:
                    print(f"    Duration: {task['last_duration']}")
                print()
        
        elif args.status_report:
            report_path = scheduler.generate_status_report()
            print(f"📄 Status report generated: {report_path}")
        
        elif args.enable_task:
            scheduler.enable_task(args.enable_task)
        
        elif args.disable_task:
            scheduler.disable_task(args.disable_task)
        
        elif args.start_daemon:
            scheduler.start_scheduler()
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"❌ Error in automation scheduler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()