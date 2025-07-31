#!/usr/bin/env python3
"""
Benchmark comparison script for performance regression detection.
Compares performance metrics between main branch and PR branches.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_benchmark_results(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Benchmark file {filepath} not found")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filepath}")
        return {}


def extract_benchmarks(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Extract benchmark metrics from results data."""
    benchmarks = {}
    
    if 'benchmarks' not in data:
        return benchmarks
    
    for benchmark in data['benchmarks']:
        name = benchmark.get('name', 'unknown')
        stats = benchmark.get('stats', {})
        
        benchmarks[name] = {
            'mean': stats.get('mean', 0),
            'stddev': stats.get('stddev', 0),
            'min': stats.get('min', 0),
            'max': stats.get('max', 0),
            'rounds': stats.get('rounds', 0)
        }
    
    return benchmarks


def calculate_performance_change(main_value: float, pr_value: float) -> tuple:
    """Calculate percentage change and determine if it's a regression."""
    if main_value == 0:
        return 0.0, False
    
    change_percent = ((pr_value - main_value) / main_value) * 100
    is_regression = change_percent > 10.0  # 10% slower is considered regression
    
    return change_percent, is_regression


def format_performance_change(change: float, is_regression: bool) -> str:
    """Format performance change for display."""
    if change > 0:
        emoji = "🔴" if is_regression else "🟡"
        return f"{emoji} +{change:.1f}% slower"
    elif change < 0:
        return f"🟢 {abs(change):.1f}% faster"
    else:
        return "⚪ No change"


def generate_comparison_report(main_results: Dict, pr_results: Dict) -> str:
    """Generate markdown comparison report."""
    main_benchmarks = extract_benchmarks(main_results)
    pr_benchmarks = extract_benchmarks(pr_results)
    
    if not main_benchmarks and not pr_benchmarks:
        return "No benchmark data available for comparison."
    
    report = []
    report.append("### Performance Benchmark Comparison")
    report.append("")
    
    # Header
    report.append("| Benchmark | Main Branch | PR Branch | Change | Status |")
    report.append("|-----------|-------------|-----------|---------|--------|")
    
    # Combine all benchmark names
    all_benchmarks = set(main_benchmarks.keys()) | set(pr_benchmarks.keys())
    
    regressions = []
    improvements = []
    
    for benchmark_name in sorted(all_benchmarks):
        main_data = main_benchmarks.get(benchmark_name, {})
        pr_data = pr_benchmarks.get(benchmark_name, {})
        
        if not main_data and not pr_data:
            continue
        
        main_time = main_data.get('mean', 0)
        pr_time = pr_data.get('mean', 0)
        
        if main_time == 0 and pr_time == 0:
            status = "⚪ No data"
            change_text = "N/A"
        elif main_time == 0:
            status = "🆕 New benchmark"
            change_text = f"{pr_time:.4f}s"
        elif pr_time == 0:
            status = "❌ Removed"
            change_text = "N/A"
        else:
            change_percent, is_regression = calculate_performance_change(main_time, pr_time)
            status = format_performance_change(change_percent, is_regression)
            change_text = f"{change_percent:+.1f}%"
            
            if is_regression:
                regressions.append(benchmark_name)
            elif change_percent < -5:  # 5% improvement
                improvements.append(benchmark_name)
        
        # Format times
        main_time_str = f"{main_time:.4f}s" if main_time > 0 else "N/A"
        pr_time_str = f"{pr_time:.4f}s" if pr_time > 0 else "N/A"
        
        report.append(f"| `{benchmark_name}` | {main_time_str} | {pr_time_str} | {change_text} | {status} |")
    
    # Summary
    report.append("")
    report.append("### Summary")
    report.append("")
    
    if regressions:
        report.append(f"🔴 **Performance Regressions Detected**: {len(regressions)} benchmark(s)")
        for regression in regressions:
            report.append(f"- `{regression}`")
        report.append("")
    
    if improvements:
        report.append(f"🟢 **Performance Improvements**: {len(improvements)} benchmark(s)")
        for improvement in improvements:
            report.append(f"- `{improvement}`")
        report.append("")
    
    if not regressions and not improvements:
        report.append("⚪ **No significant performance changes detected**")
        report.append("")
    
    # Guidelines
    report.append("### Guidelines")
    report.append("")
    report.append("- 🟢 **Improvement**: >5% faster")
    report.append("- 🔴 **Regression**: >10% slower")
    report.append("- 🟡 **Minor slowdown**: 0-10% slower")
    report.append("- ⚪ **No change**: <5% difference")
    
    return "\n".join(report)


def main():
    """Main function to compare benchmark results."""
    if len(sys.argv) != 3:
        print("Usage: python compare_benchmarks.py <main_results.json> <pr_results.json>")
        sys.exit(1)
    
    main_file = sys.argv[1]
    pr_file = sys.argv[2]
    
    main_results = load_benchmark_results(main_file)
    pr_results = load_benchmark_results(pr_file)
    
    if not main_results and not pr_results:
        print("Error: No valid benchmark data found in either file")
        sys.exit(1)
    
    report = generate_comparison_report(main_results, pr_results)
    print(report)


if __name__ == "__main__":
    main()