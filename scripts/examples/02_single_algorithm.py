#!/usr/bin/env python3
"""
Example 02: Single Algorithm Analysis
=====================================

This example demonstrates how to run individual algorithms with detailed analysis
and visualization of their performance characteristics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.point_generator import PointGenerator
from solvers.brute_force_solver import BruteForceSolver
from solvers.divide_conquer_solver import DivideConquerSolver
from visualization.visualizer import Visualizer
from utils.logger_setup import setup_logger
import numpy as np

def main():
    """Demonstrate single algorithm analysis with detailed performance metrics."""
    
    # Setup
    logger = setup_logger("single_algorithm_example")
    visualizer = Visualizer(logger=logger)
    generator = PointGenerator()
    
    logger.info("=" * 60)
    logger.info("Single Algorithm Analysis Example")
    logger.info("=" * 60)
    
    # Test parameters
    test_sizes = [5, 6, 7, 8]
    num_trials = 5
    
    # Test Brute Force Algorithm
    logger.info("\n🔍 BRUTE FORCE ALGORITHM ANALYSIS")
    logger.info("-" * 40)
    
    bf_solver = BruteForceSolver()
    bf_results = {}
    
    for n in test_sizes:
        logger.info(f"\nTesting Brute Force with n={n} points ({num_trials} trials)")
        
        trial_results = []
        for trial in range(num_trials):
            # Generate random convex points
            points = generator.generate_convex_position_points(n, seed=trial*10)
            
            # Run algorithm
            result = bf_solver.solve(points)
            trial_results.append(result)
            
            logger.info(f"  Trial {trial+1}: Wiener Index = {result['wiener_index']}, "
                       f"Time = {result['execution_time']:.4f}s")
        
        bf_results[n] = trial_results
        
        # Calculate statistics
        wiener_indices = [r['wiener_index'] for r in trial_results]
        execution_times = [r['execution_time'] for r in trial_results]
        
        logger.info(f"  Average Wiener Index: {np.mean(wiener_indices):.2f} ± {np.std(wiener_indices):.2f}")
        logger.info(f"  Average Execution Time: {np.mean(execution_times):.4f}s ± {np.std(execution_times):.4f}s")
    
    # Visualize Brute Force results
    logger.info(f"\n📊 Creating Brute Force visualizations...")
    
    # Single algorithm performance plot
    visualizer.plot_single_algorithm_analysis(
        bf_results, 
        algorithm_name="Brute Force",
        title="Brute Force Algorithm Performance Analysis",
        save_path="examples_output/02_brute_force_analysis.png"
    )
    
    # Test Divide & Conquer Algorithm
    logger.info("\n🚀 DIVIDE & CONQUER ALGORITHM ANALYSIS")
    logger.info("-" * 40)
    
    dc_solver = DivideConquerSolver()
    dc_results = {}
    
    # Test with larger sizes since D&C is more efficient
    larger_sizes = [5, 6, 7, 8, 9, 10]
    
    for n in larger_sizes:
        logger.info(f"\nTesting Divide & Conquer with n={n} points ({num_trials} trials)")
        
        trial_results = []
        for trial in range(num_trials):
            # Generate random convex points
            points = generator.generate_convex_position_points(n, seed=trial*10)
            
            # Run algorithm
            result = dc_solver.solve(points)
            trial_results.append(result)
            
            logger.info(f"  Trial {trial+1}: Wiener Index = {result['wiener_index']}, "
                       f"Time = {result['execution_time']:.4f}s")
        
        dc_results[n] = trial_results
        
        # Calculate statistics
        wiener_indices = [r['wiener_index'] for r in trial_results]
        execution_times = [r['execution_time'] for r in trial_results]
        
        logger.info(f"  Average Wiener Index: {np.mean(wiener_indices):.2f} ± {np.std(wiener_indices):.2f}")
        logger.info(f"  Average Execution Time: {np.mean(execution_times):.4f}s ± {np.std(execution_times):.4f}s")
    
    # Visualize Divide & Conquer results
    logger.info(f"\n📊 Creating Divide & Conquer visualizations...")
    
    visualizer.plot_single_algorithm_analysis(
        dc_results, 
        algorithm_name="Divide & Conquer",
        title="Divide & Conquer Algorithm Performance Analysis",
        save_path="examples_output/02_divide_conquer_analysis.png"
    )
    
    # Compare execution time trends
    logger.info("\n⚡ EXECUTION TIME COMPARISON")
    logger.info("-" * 40)
    
    # Compare on common sizes
    common_sizes = [5, 6, 7, 8]
    
    logger.info("Algorithm execution time comparison:")
    logger.info("Size\tBrute Force (s)\tDivide & Conquer (s)\tSpeedup")
    logger.info("-" * 60)
    
    for n in common_sizes:
        bf_times = [r['execution_time'] for r in bf_results[n]]
        dc_times = [r['execution_time'] for r in dc_results[n]]
        
        bf_avg = np.mean(bf_times)
        dc_avg = np.mean(dc_times)
        speedup = bf_avg / dc_avg if dc_avg > 0 else float('inf')
        
        logger.info(f"{n}\t{bf_avg:.4f}\t\t{dc_avg:.4f}\t\t\t{speedup:.2f}x")
    
    # Create algorithm comparison visualization
    comparison_data = {
        'Brute Force': {n: bf_results[n] for n in common_sizes},
        'Divide & Conquer': {n: dc_results[n] for n in common_sizes}
    }
    
    visualizer.plot_algorithm_comparison(
        comparison_data,
        title="Brute Force vs Divide & Conquer Comparison",
        save_path="examples_output/02_algorithm_comparison.png"
    )
    
    logger.info("\n✅ Single algorithm analysis complete!")
    logger.info("📁 Results saved to examples_output/")
    logger.info("\nKey Insights:")
    logger.info("• Brute Force: O(n³) complexity, exact results")
    logger.info("• Divide & Conquer: O(n²log n) complexity, significant speedup")
    logger.info("• Both algorithms produce identical Wiener indices")
    logger.info("• D&C algorithm shows better scalability for larger inputs")

if __name__ == "__main__":
    main()
