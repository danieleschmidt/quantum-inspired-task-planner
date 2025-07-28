#!/usr/bin/env python3
"""Check that quantum imports are properly guarded."""

import ast
import sys
from pathlib import Path
from typing import List, Set


class QuantumImportChecker(ast.NodeVisitor):
    """AST visitor to check for unguarded quantum imports."""

    def __init__(self):
        self.imports: List[ast.Import | ast.ImportFrom] = []
        self.try_blocks: Set[int] = set()
        self.errors: List[str] = []

    def visit_Try(self, node: ast.Try) -> None:
        """Track try blocks that might guard imports."""
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                self.try_blocks.add(child.lineno)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Check import statements."""
        for alias in node.names:
            if self._is_quantum_module(alias.name):
                if node.lineno not in self.try_blocks:
                    self.errors.append(
                        f"Line {node.lineno}: Unguarded quantum import '{alias.name}'"
                    )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check from...import statements."""
        if node.module and self._is_quantum_module(node.module):
            if node.lineno not in self.try_blocks:
                self.errors.append(
                    f"Line {node.lineno}: Unguarded quantum import 'from {node.module}'"
                )
        self.generic_visit(node)

    def _is_quantum_module(self, module_name: str) -> bool:
        """Check if module is a quantum computing library."""
        quantum_modules = {
            "dwave",
            "azure.quantum",
            "qiskit",
            "cirq",
            "pennylane",
            "pyquil",
            "braket",
        }
        return any(module_name.startswith(qm) for qm in quantum_modules)


def check_file(file_path: Path) -> List[str]:
    """Check a single Python file for quantum import issues."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))
        checker = QuantumImportChecker()
        checker.visit(tree)
        
        # Add file path to errors
        return [f"{file_path}:{error}" for error in checker.errors]
        
    except SyntaxError as e:
        return [f"{file_path}: Syntax error - {e}"]
    except Exception as e:
        return [f"{file_path}: Error reading file - {e}"]


def main():
    """Main function to check all Python files."""
    if len(sys.argv) > 1:
        # Check specific files passed as arguments
        files = [Path(arg) for arg in sys.argv[1:]]
    else:
        # Check all Python files in src/
        src_dir = Path("src")
        if not src_dir.exists():
            print("No src/ directory found")
            return 0
        
        files = list(src_dir.rglob("*.py"))

    all_errors = []
    for file_path in files:
        if file_path.is_file():
            errors = check_file(file_path)
            all_errors.extend(errors)

    if all_errors:
        print("❌ Quantum import issues found:")
        for error in all_errors:
            print(f"  {error}")
        print()
        print("💡 Tip: Wrap quantum imports in try/except blocks:")
        print("   try:")
        print("       from dwave import ...")
        print("   except ImportError:")
        print("       # Handle missing quantum dependency")
        return 1
    else:
        print("✅ All quantum imports are properly guarded")
        return 0


if __name__ == "__main__":
    sys.exit(main())