#!/usr/bin/env python3
"""
Example 01: Quick Start Guide

This example demonstrates the basic usage of the Wiener Index Analysis framework.
Perfect for getting started and understanding the core concepts.

Author: Generated by AI Assistant
Date: July 2, 2025
"""

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hamiltonian_wiener import WienerAnalysisOrchestrator


def main():
    """Quick start example demonstrating basic framework usage."""
    
    print("=" * 60)
    print("WIENER INDEX ANALYSIS - QUICK START GUIDE")
    print("=" * 60)
    
    # Step 1: Initialize the orchestrator
    print("Step 1: Initializing the analysis framework...")
    orchestrator = WienerAnalysisOrchestrator(
        output_dir="example_output",
        log_dir="example_logs"
    )
    print("✓ Framework initialized successfully")
    
    # Step 2: Run a quick test to verify everything works
    print("\nStep 2: Running quick verification test...")
    test_results = orchestrator.run_quick_test(n_points=5)
    
    print("✓ Quick test completed!")
    print(f"  - Tested {test_results['n_points']} points")
    print(f"  - Algorithms: {test_results['algorithms_tested']}")
    
    for alg, results in test_results['results'].items():
        print(f"  - {alg}: Wiener={results['wiener_index']:.3f}, "
              f"Time={results['execution_time']:.3f}s")
    
    if 'approximation_ratio' in test_results:
        print(f"  - Approximation ratio: {test_results['approximation_ratio']:.3f}")
    
    # Step 3: Run a small comparison study
    print("\nStep 3: Running comparative analysis...")
    study_results = orchestrator.run_comparison_study(
        point_sizes=[4, 5, 6],       # Small sizes for quick demo
        trials_per_size=2,           # 2 trials per size
        point_type="convex",         # Use convex hull points
        algorithms=['brute_force', 'divide_conquer'],
        save_results=True,           # Save results to disk
        generate_plots=True          # Generate visualization plots
    )
    
    print("✓ Comparative study completed!")
    print(f"  - Total experiments: {len(study_results.experiments)}")
    print(f"  - Algorithms compared: {list(study_results.statistics.keys())}")
    
    # Step 4: Display results summary
    print("\nStep 4: Results Summary")
    print("-" * 30)
    
    for algorithm, stats in study_results.statistics.items():
        print(f"{algorithm.replace('_', ' ').title()}:")
        print(f"  Average Wiener Index: {stats.wiener_index_mean:.3f} ± {stats.wiener_index_std:.3f}")
        print(f"  Average Execution Time: {stats.execution_time_mean:.3f}s ± {stats.execution_time_std:.3f}s")
        print(f"  Experiments: {stats.num_experiments}")
    
    print("\n" + "=" * 60)
    print("QUICK START COMPLETE!")
    print("=" * 60)
    print("Next steps:")
    print("- Check the generated plots in wiener_comparison_plots_convex/")
    print("- View saved data in example_output/")
    print("- Explore more advanced examples (02-18)")
    print("- Read the documentation in README.md")


if __name__ == "__main__":
    main()
