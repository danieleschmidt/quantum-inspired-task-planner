"""Pytest configuration — add src to Python path."""

import sys
import os

# Add src/ so tests can import quantum_dag_scheduler without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
