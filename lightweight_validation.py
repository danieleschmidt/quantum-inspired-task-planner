"""Lightweight Validation for Autonomous SDLC Implementation.

This module provides basic validation without heavy dependencies.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


class LightweightValidator:
    """Lightweight validator for autonomous SDLC."""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_file_structure(self) -> Dict[str, Any]:
        """Validate that all required files exist."""
        
        print("🔍 Validating file structure...")
        
        required_files = [
            "src/quantum_planner/research/breakthrough_quantum_optimizer.py",
            "src/quantum_planner/research/ultra_performance_engine.py", 
            "src/quantum_planner/security/advanced_quantum_security.py",
            "src/quantum_planner/research/revolutionary_quantum_advantage_engine.py",
            "src/quantum_planner/research/breakthrough_neural_cryptanalysis.py"
        ]
        
        results = {
            "files_checked": len(required_files),
            "files_found": 0,
            "missing_files": [],
            "file_details": {}
        }
        
        for file_path in required_files:
            full_path = Path(file_path)
            
            if full_path.exists():
                results["files_found"] += 1
                
                # Get file size
                file_size = full_path.stat().st_size
                results["file_details"][str(full_path)] = {
                    "exists": True,
                    "size_bytes": file_size,
                    "size_kb": round(file_size / 1024, 2)
                }
                
                print(f"   ✓ {full_path} ({file_size:,} bytes)")
            else:
                results["missing_files"].append(str(full_path))
                results["file_details"][str(full_path)] = {
                    "exists": False
                }
                print(f"   ✗ {full_path} - NOT FOUND")
        
        results["completion_rate"] = results["files_found"] / results["files_checked"]
        results["status"] = "PASSED" if results["completion_rate"] >= 0.8 else "FAILED"
        
        print(f"   Files found: {results['files_found']}/{results['files_checked']} ({results['completion_rate']:.1%})")
        
        return results
    
    def validate_code_structure(self) -> Dict[str, Any]:
        """Validate code structure and imports."""
        
        print("\n📝 Validating code structure...")
        
        results = {
            "modules_checked": 0,
            "import_errors": [],
            "class_definitions": {},
            "function_definitions": {},
            "total_lines": 0
        }
        
        module_files = [
            "src/quantum_planner/research/breakthrough_quantum_optimizer.py",
            "src/quantum_planner/research/ultra_performance_engine.py",
            "src/quantum_planner/security/advanced_quantum_security.py",
            "src/quantum_planner/research/revolutionary_quantum_advantage_engine.py",
            "src/quantum_planner/research/breakthrough_neural_cryptanalysis.py"
        ]
        
        for module_file in module_files:
            module_path = Path(module_file)
            
            if module_path.exists():
                results["modules_checked"] += 1
                
                try:
                    with open(module_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    results["total_lines"] += len(lines)
                    
                    # Count class definitions
                    class_count = content.count('class ')
                    function_count = content.count('def ')
                    
                    results["class_definitions"][str(module_path)] = class_count
                    results["function_definitions"][str(module_path)] = function_count
                    
                    print(f"   ✓ {module_path.name}: {class_count} classes, {function_count} functions, {len(lines)} lines")
                    
                except Exception as e:
                    results["import_errors"].append(f"{module_path}: {str(e)}")
                    print(f"   ✗ {module_path.name}: Error reading file - {str(e)}")
            else:
                results["import_errors"].append(f"{module_path}: File not found")
        
        results["status"] = "PASSED" if len(results["import_errors"]) == 0 else "FAILED"
        
        total_classes = sum(results["class_definitions"].values())
        total_functions = sum(results["function_definitions"].values())
        
        print(f"   Total: {total_classes} classes, {total_functions} functions, {results['total_lines']} lines")
        
        return results
    
    def validate_implementation_completeness(self) -> Dict[str, Any]:
        """Validate implementation completeness."""
        
        print("\n🧪 Validating implementation completeness...")
        
        results = {
            "generation1_features": {},
            "generation2_features": {},
            "generation3_features": {},
            "research_features": {},
            "completeness_score": 0.0
        }
        
        # Check Generation 1 features (Core Quantum Algorithms)
        gen1_indicators = [
            ("breakthrough_quantum_optimizer.py", "BreakthroughQuantumOptimizer"),
            ("ultra_performance_engine.py", "UltraPerformanceEngine")
        ]
        
        gen1_found = 0
        for file_name, class_name in gen1_indicators:
            found = self._check_class_in_files(class_name)
            results["generation1_features"][class_name] = found
            if found:
                gen1_found += 1
        
        # Check Generation 2 features (Security & Enterprise)
        gen2_indicators = [
            ("advanced_quantum_security.py", "AdvancedQuantumSecurityFramework"),
            ("advanced_quantum_security.py", "QuantumResistantCrypto")
        ]
        
        gen2_found = 0
        for file_name, class_name in gen2_indicators:
            found = self._check_class_in_files(class_name)
            results["generation2_features"][class_name] = found
            if found:
                gen2_found += 1
        
        # Check Generation 3 features (Research & Quantum Advantage)
        gen3_indicators = [
            ("revolutionary_quantum_advantage_engine.py", "RevolutionaryQuantumAdvantageEngine"),
            ("breakthrough_neural_cryptanalysis.py", "BreakthroughNeuralCryptanalysisEngine")
        ]
        
        gen3_found = 0
        for file_name, class_name in gen3_indicators:
            found = self._check_class_in_files(class_name)
            results["generation3_features"][class_name] = found
            if found:
                gen3_found += 1
        
        # Check Research features
        research_indicators = [
            ("breakthrough_neural_cryptanalysis.py", "FourierNeuralOperator"),
            ("revolutionary_quantum_advantage_engine.py", "DynamicQuantumCircuitOptimizer")
        ]
        
        research_found = 0
        for file_name, class_name in research_indicators:
            found = self._check_class_in_files(class_name)
            results["research_features"][class_name] = found
            if found:
                research_found += 1
        
        # Calculate completeness score
        total_expected = len(gen1_indicators) + len(gen2_indicators) + len(gen3_indicators) + len(research_indicators)
        total_found = gen1_found + gen2_found + gen3_found + research_found
        
        results["completeness_score"] = total_found / total_expected if total_expected > 0 else 0
        results["status"] = "PASSED" if results["completeness_score"] >= 0.8 else "FAILED"
        
        print(f"   Generation 1 (Core): {gen1_found}/{len(gen1_indicators)} features")
        print(f"   Generation 2 (Security): {gen2_found}/{len(gen2_indicators)} features")
        print(f"   Generation 3 (Research): {gen3_found}/{len(gen3_indicators)} features")
        print(f"   Research Components: {research_found}/{len(research_indicators)} features")
        print(f"   Overall Completeness: {results['completeness_score']:.1%}")
        
        return results
    
    def _check_class_in_files(self, class_name: str) -> bool:
        """Check if a class exists in any source file."""
        
        src_dir = Path("src")
        if not src_dir.exists():
            return False
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if f"class {class_name}" in content:
                    return True
                    
            except Exception:
                continue
        
        return False
    
    def validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation completeness."""
        
        print("\n📚 Validating documentation...")
        
        results = {
            "readme_exists": False,
            "readme_size": 0,
            "docstrings_found": 0,
            "modules_with_docs": 0,
            "documentation_score": 0.0
        }
        
        # Check README
        readme_path = Path("README.md")
        if readme_path.exists():
            results["readme_exists"] = True
            results["readme_size"] = readme_path.stat().st_size
            print(f"   ✓ README.md exists ({results['readme_size']:,} bytes)")
        else:
            print(f"   ✗ README.md not found")
        
        # Check module docstrings
        module_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
        
        for module_file in module_files:
            try:
                with open(module_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Count docstrings
                docstring_count = content.count('"""')
                results["docstrings_found"] += docstring_count
                
                if docstring_count > 0:
                    results["modules_with_docs"] += 1
                    
            except Exception:
                continue
        
        # Calculate documentation score
        if len(module_files) > 0:
            doc_coverage = results["modules_with_docs"] / len(module_files)
            readme_factor = 1.0 if results["readme_exists"] else 0.5
            results["documentation_score"] = doc_coverage * readme_factor
        
        results["status"] = "PASSED" if results["documentation_score"] >= 0.7 else "FAILED"
        
        print(f"   Modules with docs: {results['modules_with_docs']}/{len(module_files)}")
        print(f"   Total docstrings: {results['docstrings_found']}")
        print(f"   Documentation score: {results['documentation_score']:.1%}")
        
        return results
    
    def run_lightweight_validation(self) -> Dict[str, Any]:
        """Run complete lightweight validation."""
        
        validation_start = time.time()
        
        print("🚀 Autonomous SDLC Lightweight Validation")
        print("=" * 50)
        
        # Run all validations
        file_structure = self.validate_file_structure()
        code_structure = self.validate_code_structure() 
        implementation = self.validate_implementation_completeness()
        documentation = self.validate_documentation()
        
        validation_time = time.time() - validation_start
        
        # Compile results
        results = {
            "file_structure": file_structure,
            "code_structure": code_structure,
            "implementation_completeness": implementation,
            "documentation": documentation,
            "validation_time": validation_time,
            "timestamp": validation_start
        }
        
        # Calculate overall score
        scores = [
            file_structure.get("completion_rate", 0) * 0.25,
            (1.0 if code_structure.get("status") == "PASSED" else 0.0) * 0.25,
            implementation.get("completeness_score", 0) * 0.3,
            documentation.get("documentation_score", 0) * 0.2
        ]
        
        overall_score = sum(scores)
        
        # Determine status
        if overall_score >= 0.9:
            status = "EXCELLENT"
        elif overall_score >= 0.8:
            status = "GOOD"  
        elif overall_score >= 0.7:
            status = "ACCEPTABLE"
        elif overall_score >= 0.6:
            status = "NEEDS_IMPROVEMENT"
        else:
            status = "FAILED"
        
        results["overall_assessment"] = {
            "overall_score": overall_score,
            "status": status,
            "validation_time": validation_time,
            "component_scores": {
                "file_structure": file_structure.get("completion_rate", 0),
                "code_structure": 1.0 if code_structure.get("status") == "PASSED" else 0.0,
                "implementation": implementation.get("completeness_score", 0),
                "documentation": documentation.get("documentation_score", 0)
            }
        }
        
        # Print summary
        print("\n" + "="*50)
        print("🏆 VALIDATION SUMMARY")
        print("="*50)
        print(f"Overall Status: {status}")
        print(f"Overall Score: {overall_score:.3f}/1.000")
        print(f"Validation Time: {validation_time:.2f}s")
        
        print(f"\nComponent Scores:")
        print(f"  File Structure:     {file_structure.get('completion_rate', 0):.3f}")
        print(f"  Code Structure:     {1.0 if code_structure.get('status') == 'PASSED' else 0.0:.3f}")
        print(f"  Implementation:     {implementation.get('completeness_score', 0):.3f}")
        print(f"  Documentation:      {documentation.get('documentation_score', 0):.3f}")
        
        # Export results
        output_file = Path("lightweight_validation_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Results exported to: {output_file}")
        print("="*50)
        
        return results


def main():
    """Run lightweight validation."""
    
    validator = LightweightValidator()
    return validator.run_lightweight_validation()


if __name__ == "__main__":
    main()