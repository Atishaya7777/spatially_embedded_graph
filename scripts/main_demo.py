"""
Main entry point for Wiener Index Analysis using the refactored modular architecture.

This demonstrates the proper usage of the decoupled components.
"""

import logging
from hamiltonian_wiener import WienerAnalysisOrchestrator
from core.point import Point
from generators.point_generator import PointGenerator
from utils.logger_setup import setup_logging


def demo_basic_analysis():
    """Demonstrate basic algorithm comparison."""
    print("=== Basic Algorithm Comparison Demo ===")
    
    # Set up orchestrator
    orchestrator = WienerAnalysisOrchestrator()
    
    # Generate a small test case
    generator = PointGenerator()
    points = generator.generate_convex_hull_points(7, seed=42)
    
    print(f"Generated {len(points)} points for comparison")
    print("Points:", [f"({p.x:.2f}, {p.y:.2f})" for p in points])
    
    # Compare algorithms using run_single_experiment
    experiment = orchestrator.run_single_experiment(
        n_points=7, 
        point_type="convex", 
        seed=42,
        algorithms=['brute_force', 'divide_conquer']
    )
    
    print("\nComparison completed!")
    for alg_name, result in experiment.results.items():
        print(f"{alg_name}: Wiener={result.wiener_index:.4f}, Time={result.execution_time:.4f}s")


def demo_targeted_study():
    """Demonstrate targeted study functionality."""
    print("\n=== Targeted Study Demo ===")
    
    orchestrator = WienerAnalysisOrchestrator()
    
    # Run a small targeted study using run_comparison_study
    results = orchestrator.run_comparison_study(
        point_sizes=[5, 6, 7],    # Small sizes for demo
        trials_per_size=3,        # Few trials for demo
        point_type="convex",      # Convex points
        algorithms=['brute_force', 'divide_conquer'],
        save_results=False,       # Skip saving for demo
        generate_plots=False      # Skip plots for demo
    )
    
    print("Targeted study completed!")
    print(f"Analyzed {len(results.experiments)} experiments")
    print("Algorithm performance:")
    for algorithm, stats in results.statistics.items():
        print(f"  {algorithm}: Avg Wiener={stats.wiener_index_mean:.4f}")


def demo_comprehensive_study():
    """Demonstrate comprehensive study functionality."""
    print("\n=== Comprehensive Study Demo ===")
    
    orchestrator = WienerAnalysisOrchestrator()
    
    # Run a small comprehensive study using run_comparison_study
    results = orchestrator.run_comparison_study(
        point_sizes=[5, 6],       # Very small sizes for demo
        trials_per_size=2,        # Very few trials for demo
        point_type="general",     # General (random) points
        algorithms=['brute_force', 'divide_conquer'],
        save_results=False,       # Skip saving for demo
        generate_plots=False      # Skip plots for demo
    )
    
    print("Comprehensive study completed!")
    print(f"Analyzed {len(results.experiments)} experiments")
    print("Algorithm performance:")
    for algorithm, stats in results.statistics.items():
        print(f"  {algorithm}: Avg Wiener={stats.wiener_index_mean:.4f}")


def demo_individual_components():
    """Demonstrate usage of individual components."""
    print("\n=== Individual Components Demo ===")
    
    # Point generation
    generator = PointGenerator()
    convex_points = generator.generate_convex_hull_points(6, seed=123)
    random_points = generator.generate_random_points(6, seed=123)
    
    print(f"Generated {len(convex_points)} convex points and {len(random_points)} random points")
    
    # Solver usage
    from solvers.divide_conquer_solver import DivideConquerSolver
    from solvers.brute_force_solver import BruteForceSolver
    from core.wiener_index_calculator import WienerIndexCalculator
    
    dc_solver = DivideConquerSolver()
    bf_solver = BruteForceSolver()
    calculator = WienerIndexCalculator()
    
    # Solve with divide and conquer
    dc_path = dc_solver.solve(convex_points)
    dc_wiener = calculator.calculate_wiener_index(dc_path)
    
    # Solve with brute force
    bf_path = bf_solver.solve_simple(convex_points)
    bf_wiener = calculator.calculate_wiener_index(bf_path)
    
    print(f"D&C Wiener: {dc_wiener:.4f}")
    print(f"BF Wiener: {bf_wiener:.4f}")
    print(f"Approximation ratio: {dc_wiener/bf_wiener:.4f}")
    
    # Visualization
    from visualization.visualizer import Visualizer
    visualizer = Visualizer()
    
    print("Visualizing results...")
    visualizer.visualize_comparison(
        convex_points, dc_path, bf_path,
        dc_wiener, bf_wiener, dc_wiener/bf_wiener
    )


def main():
    """Main function demonstrating all capabilities."""
    print("Wiener Index Analysis - Modular Architecture Demo")
    print("=" * 60)
    
    # Set up logging
    logger = setup_logging("demo_logs")
    logger.info("Starting modular architecture demo")
    
    try:
        # Run demonstrations
        demo_basic_analysis()
        demo_individual_components()
        demo_targeted_study()
        demo_comprehensive_study()
        
        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("The modular architecture is working correctly.")
        print("\nFor full analysis, run: python hamiltonian_wiener.py")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\nDemo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
