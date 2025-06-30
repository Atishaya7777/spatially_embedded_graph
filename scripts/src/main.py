"""
Complete example of how to use the ComparisonAnalyzer for Wiener index optimization.

This script demonstrates:
1. Setting up the analyzer with your existing components
2. Registering algorithms
3. Running comparison studies
4. Analyzing and visualizing results
"""

import logging
from typing import List

# Import your existing modules
from core.point import Point
from generators.point_generator import PointGenerator
from utils.data_manager import DataManager
from utils.logger_setup import setup_logger
from visualization.visualizer import Visualizer
from solvers.brute_force_solver import BruteForceSolver
from solvers.divide_conquer_solver import DivideConquerSolver
from analysis.comparison_analyzer import (
    ComparisonAnalyzer,
    create_wiener_comparison_analyzer,
    register_wiener_algorithms,
    WienerIndexAlgorithmAdapter
)


def setup_comparison_analyzer():
    """Set up the comparison analyzer with all components."""

    # Initialize logger
    logger = setup_logger("comparison_study", level=logging.INFO)

    # Initialize components
    point_generator = PointGenerator()
    data_manager = DataManager()
    visualizer = Visualizer()

    # Create the analyzer
    analyzer = create_wiener_comparison_analyzer(
        point_generator=point_generator,
        data_manager=data_manager,
        visualizer=visualizer,
        logger=logger
    )

    return analyzer, logger


def register_your_algorithms(analyzer: ComparisonAnalyzer):
    """Register your specific algorithms with the analyzer."""

    # Initialize your solvers
    brute_force_solver = BruteForceSolver()
    divide_conquer_solver = DivideConquerSolver()

    # Create wrapper functions that match the expected interface
    def brute_force_wrapper(points: List[Point]) -> List[Point]:
        return brute_force_solver.solve(points)

    def divide_conquer_wrapper(points: List[Point]) -> List[Point]:
        return divide_conquer_solver.solve(points)

    # If you have a parallel brute force implementation
    def parallel_brute_force_wrapper(points: List[Point]) -> List[Point]:
        # Replace this with your actual parallel implementation
        return brute_force_solver.solve(points)  # Placeholder

    # Register algorithms using the helper function
    register_wiener_algorithms(
        analyzer=analyzer,
        divide_conquer_func=divide_conquer_wrapper,
        brute_force_func=brute_force_wrapper,
        parallel_brute_force_func=parallel_brute_force_wrapper
    )

    # Or register them individually for more control
    # Example of registering a custom algorithm:
    """
    custom_algorithm = WienerIndexAlgorithmAdapter(
        algorithm_func=your_custom_function,
        name="custom_algorithm",
        is_exact=False,  # Set to True if it's an exact algorithm
        max_size=None    # Set maximum feasible problem size
    )
    analyzer.register_algorithm(custom_algorithm)
    """


def run_basic_comparison():
    """Run a basic comparison study."""

    analyzer, logger = setup_comparison_analyzer()
    register_your_algorithms(analyzer)

    logger.info("Starting basic comparison study...")

    # Run comparison on small problem sizes
    point_counts = [4, 5, 6, 7, 8]  # Start small for testing
    num_seeds = 10  # Reduce for testing

    cases = analyzer.run_comparison_study(
        point_counts=point_counts,
        num_seeds=num_seeds,
        # Optional: specify which algorithms to run
        algorithms=['divide_conquer', 'brute_force'],
        # Generator parameters (adjust based on your PointGenerator)
        shape='convex_hull'  # or whatever parameters your generator accepts
    )

    # Analyze results
    analysis = analyzer.analyze_results(cases)

    # Save results
    analyzer.save_results("basic_comparison_results.json", cases)

    # Find and visualize interesting cases
    interesting_cases = analyzer.find_interesting_cases(cases)
    analyzer.visualize_results(interesting_cases, max_cases=5)

    # Plot performance statistics
    analyzer.visualize_performance_analysis(analysis)

    return analyzer, cases, analysis


def run_comprehensive_study():
    """Run a comprehensive comparison study."""

    analyzer, logger = setup_comparison_analyzer()
    register_your_algorithms(analyzer)

    logger.info("Starting comprehensive comparison study...")

    # Define test parameters
    small_sizes = [4, 5, 6, 7, 8]  # For all algorithms
    medium_sizes = [10, 12, 15, 20]  # Only heuristics
    large_sizes = [25, 30, 40, 50]  # Only fast heuristics

    all_cases = []

    # Test small sizes with all algorithms
    logger.info("Testing small sizes with all algorithms...")
    small_cases = analyzer.run_comparison_study(
        point_counts=small_sizes,
        num_seeds=50,
        algorithms=['divide_conquer', 'brute_force'],
        shape='convex_hull'
    )
    all_cases.extend(small_cases)

    # Test medium sizes with heuristics only
    logger.info("Testing medium sizes with heuristics...")
    medium_cases = analyzer.run_comparison_study(
        point_counts=medium_sizes,
        num_seeds=30,
        algorithms=['divide_conquer'],  # Only fast algorithms
        shape='convex_hull'
    )
    all_cases.extend(medium_cases)

    # Test large sizes with fastest algorithms
    logger.info("Testing large sizes with fastest algorithms...")
    large_cases = analyzer.run_comparison_study(
        point_counts=large_sizes,
        num_seeds=20,
        algorithms=['divide_conquer'],
        shape='convex_hull'
    )
    all_cases.extend(large_cases)

    # Comprehensive analysis
    analysis = analyzer.analyze_results(all_cases)

    # Save comprehensive results
    analyzer.save_results("comprehensive_comparison_results.json", all_cases)

    # Generate visualizations
    interesting_cases = analyzer.find_interesting_cases(all_cases)
    analyzer.visualize_results(interesting_cases, max_cases=15)
    analyzer.visualize_performance_analysis(analysis)

    return analyzer, all_cases, analysis


def run_single_case_analysis():
    """Run analysis on a single test case for debugging."""

    analyzer, logger = setup_comparison_analyzer()
    register_your_algorithms(analyzer)

    # Generate a single test case
    points = analyzer.point_generator.generate_points(
        n=6,
        seed=42,
        point_type='general'
    )

    # Run comparison on this single case
    case = analyzer.run_single_comparison(
        points=points,
        case_id="debug_case",
        algorithms=['divide_conquer', 'brute_force']
    )

    # Print results
    logger.info("Single case analysis results:")
    for alg_name, result in case.results.items():
        logger.info(f"{alg_name}:")
        logger.info(f"  Objective: {result.objective_value}")
        logger.info(f"  Time: {result.execution_time:.4f}s")
        logger.info(f"  Success: {result.metadata.get('success', False)}")

    # Calculate approximation ratio
    if 'divide_conquer' in case.results and 'brute_force' in case.results:
        ratio = case.get_approximation_ratio('divide_conquer', 'brute_force')
        logger.info(f"Approximation ratio: {ratio}")

    return case


def custom_analysis_example():
    """Example of custom analysis using the analyzer."""

    analyzer, logger = setup_comparison_analyzer()
    register_your_algorithms(analyzer)

    # Run comparison study
    cases = analyzer.run_comparison_study(
        point_counts=[5, 6, 7],
        num_seeds=20,
        shape='convex_hull'
    )

    # Custom analysis: Find cases where divide_conquer performs poorly
    poor_performance_cases = []
    for case in cases:
        if 'divide_conquer' in case.results and 'brute_force' in case.results:
            ratio = case.get_approximation_ratio(
                'divide_conquer', 'brute_force')
            if ratio and ratio > 1.2:  # More than 20% worse than optimal
                poor_performance_cases.append(case)

    logger.info(f"Found {len(poor_performance_cases)
                         } cases with poor performance")

    # Visualize these cases
    if poor_performance_cases:
        analyzer.visualize_results(poor_performance_cases, max_cases=5)

    # Custom metric analysis
    execution_time_analysis = {}
    for case in cases:
        n_points = case.metadata.get('n_points', 'unknown')
        if n_points not in execution_time_analysis:
            execution_time_analysis[n_points] = {}

        for alg_name, result in case.results.items():
            if result.metadata.get('success', False):
                if alg_name not in execution_time_analysis[n_points]:
                    execution_time_analysis[n_points][alg_name] = []
                execution_time_analysis[n_points][alg_name].append(
                    result.execution_time)

    # Print custom analysis
    for n_points, alg_data in execution_time_analysis.items():
        logger.info(f"\nExecution time analysis for {n_points} points:")
        for alg_name, times in alg_data.items():
            avg_time = sum(times) / len(times)
            logger.info(f"  {alg_name}: {avg_time:.4f}s average")


if __name__ == "__main__":
    # Choose which analysis to run

    print("Choose analysis type:")
    print("1. Single case analysis (debugging)")
    print("2. Basic comparison study")
    print("3. Comprehensive study")
    print("4. Custom analysis example")

    choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        case = run_single_case_analysis()
        print(f"Single case completed. Results: {
              len(case.results)} algorithms")

    elif choice == "2":
        analyzer, cases, analysis = run_basic_comparison()
        print(f"Basic study completed. {len(cases)} cases analyzed")

    elif choice == "3":
        analyzer, cases, analysis = run_comprehensive_study()
        print(f"Comprehensive study completed. {len(cases)} cases analyzed")

    elif choice == "4":
        custom_analysis_example()
        print("Custom analysis completed")

    else:
        print("Invalid choice. Running basic comparison by default.")
        analyzer, cases, analysis = run_basic_comparison()
        print(f"Basic study completed. {len(cases)} cases analyzed")
