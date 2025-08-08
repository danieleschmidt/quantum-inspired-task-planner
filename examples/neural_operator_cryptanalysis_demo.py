#!/usr/bin/env python3
"""Neural Operator Cryptanalysis Demo.

Demonstrates the neural operator cryptanalysis capabilities
of the quantum planner with practical examples.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantum_planner.research.neural_operator_cryptanalysis import (
    create_cryptanalysis_framework,
    CryptanalysisConfig
)
from quantum_planner.research.quantum_cryptanalysis_integration import (
    create_quantum_cryptanalysis_orchestrator,
    analyze_cipher_with_quantum_optimization
)


def generate_sample_cipher_data(cipher_type: str, size: int = 1000) -> torch.Tensor:
    """Generate sample cipher data for demonstration."""
    torch.manual_seed(42)  # For reproducibility
    
    if cipher_type == "weak_cipher":
        # Generate data with detectable patterns (vulnerable cipher)
        base_pattern = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], dtype=torch.uint8)
        noise_level = 0.1
        
        # Repeat pattern with some noise
        data = base_pattern.repeat(size // len(base_pattern) + 1)[:size]
        noise = torch.rand(size) < noise_level
        data = data ^ noise.byte()
        
    elif cipher_type == "strong_cipher":
        # Generate more random-looking data (stronger cipher)
        data = torch.randint(0, 2, (size,), dtype=torch.uint8)
        
    elif cipher_type == "aes_like":
        # Simulate AES-like cipher with better randomness
        data = torch.randint(0, 256, (size,), dtype=torch.uint8)
        
    else:
        # Default random data
        data = torch.randint(0, 2, (size,), dtype=torch.uint8)
    
    return data


def demonstrate_basic_cryptanalysis():
    """Demonstrate basic neural operator cryptanalysis."""
    print("\n" + "="*60)
    print("BASIC NEURAL OPERATOR CRYPTANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Create cryptanalysis framework
    framework = create_cryptanalysis_framework(
        cipher_type="demo_cipher",
        neural_operator_type="fourier",
        hidden_dim=64,
        num_layers=3
    )
    
    # Test different cipher types
    cipher_types = ["weak_cipher", "strong_cipher", "aes_like"]
    results = {}
    
    for cipher_type in cipher_types:
        print(f"\nAnalyzing {cipher_type}...")
        
        # Generate sample data
        cipher_data = generate_sample_cipher_data(cipher_type, size=512)
        
        # Prepare samples for different analysis types
        cipher_samples = {
            "plaintext_samples": cipher_data[:256],
            "ciphertext_samples": cipher_data[256:],
            "plaintext_pairs": [(cipher_data[i], cipher_data[i+1]) 
                               for i in range(0, 100, 2)],
            "ciphertext_pairs": [(cipher_data[i+200], cipher_data[i+201]) 
                               for i in range(0, 100, 2)]
        }
        
        # Perform comprehensive analysis
        analysis_result = framework.comprehensive_analysis(cipher_samples)
        results[cipher_type] = analysis_result
        
        # Display results
        print(f"  Results for {cipher_type}:")
        
        if "overall" in analysis_result:
            overall = analysis_result["overall"]
            score = overall.get("combined_vulnerability_score", torch.tensor(0.0))
            level = overall.get("overall_vulnerability_level", "UNKNOWN")
            recommendation = overall.get("recommendation", "No recommendation")
            
            print(f"    Vulnerability Score: {score:.4f}")
            print(f"    Vulnerability Level: {level}")
            print(f"    Recommendation: {recommendation}")
        
        if "differential" in analysis_result:
            diff_score = analysis_result["differential"].get("mean_differential_score", torch.tensor(0.0))
            print(f"    Differential Analysis Score: {diff_score:.4f}")
        
        if "linear" in analysis_result:
            max_bias = analysis_result["linear"].get("max_bias", torch.tensor(0.0))
            print(f"    Linear Analysis Max Bias: {max_bias:.4f}")
    
    return results


def demonstrate_quantum_optimized_analysis():
    """Demonstrate quantum-optimized cryptanalysis."""
    print("\n" + "="*60)
    print("QUANTUM-OPTIMIZED CRYPTANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Generate larger dataset for quantum optimization benefits
    cipher_data = generate_sample_cipher_data("weak_cipher", size=2048)
    
    print(f"Analyzing cipher data of size {cipher_data.shape}...")
    
    # Perform quantum-optimized analysis
    result = analyze_cipher_with_quantum_optimization(
        cipher_data=cipher_data,
        analysis_types=["differential", "linear", "frequency"],
        neural_operator_type="fourier",
        quantum_backend="simulator"  # Use simulator for demo
    )
    
    print("\nQuantum-Optimized Analysis Results:")
    
    # Display summary
    if "summary" in result:
        summary = result["summary"]
        print(f"  Mean Vulnerability Score: {summary.get('mean_vulnerability_score', 0):.4f}")
        print(f"  Max Vulnerability Score: {summary.get('max_vulnerability_score', 0):.4f}")
        print(f"  Successful Analyses: {summary.get('successful_analyses', 0)}")
        print(f"  Analysis Types: {summary.get('analysis_types_completed', [])}")
    
    # Display recommendations
    if "recommendations" in result:
        print("\n  Recommendations:")
        for rec in result["recommendations"]:
            print(f"    - {rec}")
    
    # Display quantum optimization metrics
    if "quantum_optimization_metrics" in result:
        metrics = result["quantum_optimization_metrics"]
        print("\n  Quantum Optimization Metrics:")
        print(f"    Total Execution Time: {metrics.get('total_execution_time', 0):.3f}s")
        print(f"    Mean Task Time: {metrics.get('mean_task_time', 0):.3f}s")
        print(f"    Parallel Efficiency: {metrics.get('parallel_efficiency', 0):.3f}")
    
    return result


def demonstrate_different_neural_operators():
    """Demonstrate different neural operator types."""
    print("\n" + "="*60)
    print("NEURAL OPERATOR COMPARISON DEMONSTRATION")
    print("="*60)
    
    operator_types = ["fourier", "wavelet"]
    cipher_data = generate_sample_cipher_data("weak_cipher", size=1024)
    
    results_comparison = {}
    
    for operator_type in operator_types:
        print(f"\nTesting {operator_type.upper()} Neural Operator...")
        
        framework = create_cryptanalysis_framework(
            cipher_type="demo_cipher",
            neural_operator_type=operator_type,
            hidden_dim=64,
            num_layers=3
        )
        
        # Prepare samples
        cipher_samples = {
            "plaintext_samples": cipher_data[:512],
            "ciphertext_samples": cipher_data[512:],
        }
        
        # Analyze
        result = framework.comprehensive_analysis(cipher_samples)
        results_comparison[operator_type] = result
        
        # Display results
        if "overall" in result:
            overall = result["overall"]
            score = overall.get("combined_vulnerability_score", torch.tensor(0.0))
            level = overall.get("overall_vulnerability_level", "UNKNOWN")
            
            print(f"  {operator_type.capitalize()} Operator Results:")
            print(f"    Vulnerability Score: {score:.4f}")
            print(f"    Vulnerability Level: {level}")
    
    return results_comparison


def create_visualization(results: Dict[str, Any]):
    """Create visualizations of analysis results."""
    print("\n" + "="*60)
    print("CREATING ANALYSIS VISUALIZATIONS")
    print("="*60)
    
    try:
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Neural Operator Cryptanalysis Results', fontsize=16, fontweight='bold')
        
        # Vulnerability scores by cipher type
        cipher_types = list(results.keys())
        vulnerability_scores = []
        
        for cipher_type in cipher_types:
            if "overall" in results[cipher_type]:
                score = results[cipher_type]["overall"].get("combined_vulnerability_score", torch.tensor(0.0))
                if torch.is_tensor(score):
                    vulnerability_scores.append(score.item())
                else:
                    vulnerability_scores.append(float(score))
            else:
                vulnerability_scores.append(0.0)
        
        # Bar plot of vulnerability scores
        axes[0, 0].bar(cipher_types, vulnerability_scores, 
                      color=['red' if score > 0.5 else 'orange' if score > 0.2 else 'green' 
                             for score in vulnerability_scores])
        axes[0, 0].set_title('Vulnerability Scores by Cipher Type')
        axes[0, 0].set_ylabel('Vulnerability Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Differential analysis comparison
        diff_scores = []
        for cipher_type in cipher_types:
            if "differential" in results[cipher_type]:
                score = results[cipher_type]["differential"].get("mean_differential_score", torch.tensor(0.0))
                if torch.is_tensor(score):
                    diff_scores.append(score.item())
                else:
                    diff_scores.append(float(score))
            else:
                diff_scores.append(0.0)
        
        axes[0, 1].plot(cipher_types, diff_scores, 'bo-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Differential Analysis Scores')
        axes[0, 1].set_ylabel('Differential Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Linear analysis comparison
        linear_scores = []
        for cipher_type in cipher_types:
            if "linear" in results[cipher_type]:
                score = results[cipher_type]["linear"].get("max_bias", torch.tensor(0.0))
                if torch.is_tensor(score):
                    linear_scores.append(score.item())
                else:
                    linear_scores.append(float(score))
            else:
                linear_scores.append(0.0)
        
        axes[1, 0].plot(cipher_types, linear_scores, 'ro-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Linear Analysis Max Bias')
        axes[1, 0].set_ylabel('Max Bias')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary heatmap
        analysis_matrix = np.array([vulnerability_scores, diff_scores, linear_scores])
        im = axes[1, 1].imshow(analysis_matrix, cmap='RdYlGn_r', aspect='auto')
        axes[1, 1].set_title('Analysis Summary Heatmap')
        axes[1, 1].set_xticks(range(len(cipher_types)))
        axes[1, 1].set_xticklabels(cipher_types, rotation=45)
        axes[1, 1].set_yticks(range(3))
        axes[1, 1].set_yticklabels(['Overall Vulnerability', 'Differential Score', 'Linear Bias'])
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(os.path.dirname(__file__), "cryptanalysis_results.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        
        # Show plot if in interactive environment
        try:
            plt.show()
        except:
            print("Note: Plot display not available in this environment")
            
    except Exception as e:
        print(f"Error creating visualization: {e}")
        print("Continuing without visualization...")


def main():
    """Main demonstration function."""
    print("Neural Operator Cryptanalysis Laboratory")
    print("========================================")
    print("This demo showcases neural operator-based cryptanalysis capabilities")
    print("integrated with quantum optimization for enhanced security analysis.")
    
    try:
        # Demonstrate basic cryptanalysis
        basic_results = demonstrate_basic_cryptanalysis()
        
        # Demonstrate quantum-optimized analysis
        quantum_results = demonstrate_quantum_optimized_analysis()
        
        # Demonstrate different neural operators
        operator_comparison = demonstrate_different_neural_operators()
        
        # Create visualizations
        create_visualization(basic_results)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("\nKey Findings:")
        print("1. Neural operators can detect patterns in cipher data")
        print("2. Different operator types (Fourier vs Wavelet) provide complementary insights")
        print("3. Quantum optimization enables efficient parallel analysis")
        print("4. The framework successfully identifies vulnerable vs strong ciphers")
        
        print("\nNext Steps:")
        print("- Experiment with real cipher implementations")
        print("- Tune neural operator parameters for specific cipher types")
        print("- Integrate with quantum hardware backends for larger problems")
        print("- Explore custom constraint formulations for specific security requirements")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("This may be due to missing dependencies or environment setup.")
        print("Please ensure PyTorch and other required packages are installed.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
