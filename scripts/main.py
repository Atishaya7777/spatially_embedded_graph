"""
Usage Example: Wiener Index Algorithm Performance Analysis

This example demonstrates how to use the WienerIndexComparisonAnalyzer to compare
different algorithms for computing Wiener indices with comprehensive performance analysis.

This example uses the actual solvers from your project structure.
"""

from analysis.comparison_analyzer import (
    WienerIndexComparisonAnalyzer,
    create_wiener_index_analyzer,
    register_wiener_algorithms
)
from solvers.divide_conquer_solver import DivideConquerSolver
from solvers.brute_force_solver import BruteForceSolver
from visualization.visualizer import Visualizer
from utils.data_manager import DataManager
from generators.point_generator import PointGenerator
from core.point import Point
import logging
import numpy as np
from typing import List, Dict, Tuple
import os
import sys
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def setup_logging() -> logging.Logger:
    """Set up logging for the analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('wiener_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def print_performance_table(all_study_cases: List, logger: logging.Logger) -> None:
    """
    Print a comprehensive performance table to terminal and log.
    
    Args:
        all_study_cases: List of all study cases
        logger: Logger instance
    """
    # Group cases by point count and distribution
    grouped_cases = {}
    
    for case in all_study_cases:
        n_points = case.metadata.get('n_points', 'unknown')
        distribution = case.metadata.get('distribution', 'unknown')
        
        key = (n_points, distribution)
        if key not in grouped_cases:
            grouped_cases[key] = []
        grouped_cases[key].append(case)
    
    # Create performance table
    table_header = """
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                          WIENER INDEX ALGORITHM PERFORMANCE COMPARISON                                                              ║
╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║ Points │ Distribution │  Cases  │              D&C (BBox Bisection)               │              D&C (Median Bisection)             │    Comparison vs Brute Force    ║
║        │              │         │ Optimal │ Avg Ratio │  Std   │ Success │ Avg Time│ Optimal │ Avg Ratio │  Std   │ Success │ Avg Time│  BBox Better │ Median Better║
╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣"""
    
    print(table_header)
    logger.info(table_header)
    
    # Sort by point count then distribution
    sorted_keys = sorted(grouped_cases.keys(), key=lambda x: (x[0] if isinstance(x[0], int) else 999, x[1]))

    # Fix: Iterate directly over the sorted keys and get cases from grouped_cases
    for (n_points, distribution) in sorted_keys:
        cases = grouped_cases[(n_points, distribution)]
        
        if not cases:
            continue
            
        # Calculate statistics for each algorithm
        bbox_stats = calculate_algorithm_stats(cases, 'D&C (BBox Bisection)')
        median_stats = calculate_algorithm_stats(cases, 'D&C (Median Bisection)')
        
        # Compare algorithms
        bbox_better, median_better = compare_algorithms(cases)
        
        # Format row
        row = f"║   {str(n_points):2s}   │  {distribution:10s}  │  {len(cases):3d}    │"
        row += f"  {bbox_stats['optimal_rate']:5.1%}  │  {bbox_stats['avg_ratio']:6.4f}  │ {bbox_stats['std_ratio']:6.4f} │  {bbox_stats['success_rate']:5.1%}  │ {bbox_stats['avg_time']:7.4f}s│"
        row += f"  {median_stats['optimal_rate']:5.1%}  │  {median_stats['avg_ratio']:6.4f}  │ {median_stats['std_ratio']:6.4f} │  {median_stats['success_rate']:5.1%}  │ {median_stats['avg_time']:7.4f}s│"
        row += f"    {bbox_better:3d} ({bbox_better/len(cases)*100:4.1f}%)  │    {median_better:3d} ({median_better/len(cases)*100:4.1f}%)  ║"
        
        print(row)
        logger.info(row)
    
    table_footer = """╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"""
    print(table_footer)
    logger.info(table_footer)
    
    # Print summary statistics
    print_summary_statistics(all_study_cases, logger)

def calculate_algorithm_stats(cases: List, algorithm_name: str) -> Dict:
    """Calculate statistics for a specific algorithm."""
    ratios = []
    times = []
    optimal_count = 0
    success_count = 0
    
    for case in cases:
        if algorithm_name in case.results:
            result = case.results[algorithm_name]
            
            if result.metadata.get('success', False):
                success_count += 1
                ratio = case.get_approximation_ratio(algorithm_name)
                if ratio is not None:
                    ratios.append(ratio)
                    if ratio <= 1.001:  # Consider optimal if within 0.1%
                        optimal_count += 1
                
                # Get timing if available
                time_taken = result.metadata.get('time_taken', 0)
                times.append(time_taken)
    
    return {
        'optimal_rate': optimal_count / len(cases) if cases else 0,
        'avg_ratio': np.mean(ratios) if ratios else 0,
        'std_ratio': np.std(ratios) if ratios else 0,
        'success_rate': success_count / len(cases) if cases else 0,
        'avg_time': np.mean(times) if times else 0
    }


def compare_algorithms(cases: List) -> Tuple[int, int]:
    """Compare which algorithm performs better in each case."""
    bbox_better = 0
    median_better = 0
    
    for case in cases:
        bbox_ratio = case.get_approximation_ratio('D&C (BBox Bisection)')
        median_ratio = case.get_approximation_ratio('D&C (Median Bisection)')
        
        if bbox_ratio is not None and median_ratio is not None:
            if bbox_ratio < median_ratio:
                bbox_better += 1
            elif median_ratio < bbox_ratio:
                median_better += 1
    
    return bbox_better, median_better


def print_summary_statistics(all_study_cases: List, logger: logging.Logger) -> None:
    """Print overall summary statistics."""
    summary = """
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                    OVERALL SUMMARY                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"""
    
    print(summary)
    logger.info(summary)
    
    # Calculate overall statistics
    total_cases = len(all_study_cases)
    bbox_optimal = 0
    median_optimal = 0
    bbox_ratios = []
    median_ratios = []
    
    for case in all_study_cases:
        bbox_ratio = case.get_approximation_ratio('D&C (BBox Bisection)')
        median_ratio = case.get_approximation_ratio('D&C (Median Bisection)')
        
        if bbox_ratio is not None:
            bbox_ratios.append(bbox_ratio)
            if bbox_ratio <= 1.001:
                bbox_optimal += 1
        
        if median_ratio is not None:
            median_ratios.append(median_ratio)
            if median_ratio <= 1.001:
                median_optimal += 1
    
    print(f"Total test cases: {total_cases}")
    print(f"D&C (BBox Bisection)  - Optimal solutions: {bbox_optimal}/{len(bbox_ratios)} ({bbox_optimal/len(bbox_ratios)*100:.1f}%), Avg ratio: {np.mean(bbox_ratios):.4f} ± {np.std(bbox_ratios):.4f}")
    print(f"D&C (Median Bisection) - Optimal solutions: {median_optimal}/{len(median_ratios)} ({median_optimal/len(median_ratios)*100:.1f}%), Avg ratio: {np.mean(median_ratios):.4f} ± {np.std(median_ratios):.4f}")
    
    logger.info(f"Total test cases: {total_cases}")
    logger.info(f"D&C (BBox Bisection)  - Optimal solutions: {bbox_optimal}/{len(bbox_ratios)} ({bbox_optimal/len(bbox_ratios)*100:.1f}%), Avg ratio: {np.mean(bbox_ratios):.4f} ± {np.std(bbox_ratios):.4f}")
    logger.info(f"D&C (Median Bisection) - Optimal solutions: {median_optimal}/{len(median_ratios)} ({median_optimal/len(median_ratios)*100:.1f}%), Avg ratio: {np.mean(median_ratios):.4f} ± {np.std(median_ratios):.4f}")


def find_best_and_worst_cases(all_study_cases: List) -> Tuple[List, List]:
    """Find the best 3 and worst 5 cases based on approximation ratios."""
    cases_with_ratios = []
    
    for case in all_study_cases:
        # Use BBox Bisection as primary metric
        bbox_ratio = case.get_approximation_ratio('D&C (BBox Bisection)')
        median_ratio = case.get_approximation_ratio('D&C (Median Bisection)')
        
        if bbox_ratio is not None:
            cases_with_ratios.append((case, bbox_ratio, median_ratio))
    
    # Sort by approximation ratio
    cases_with_ratios.sort(key=lambda x: x[1])  # Sort by bbox ratio
    
    # Get best 3 and worst 5
    best_cases = cases_with_ratios[:3]
    worst_cases = cases_with_ratios[-5:]
    
    return best_cases, worst_cases


def visualize_best_and_worst_cases(best_cases: List, worst_cases: List, logger: logging.Logger) -> None:
    """
    Visualize the best 3 and worst 5 cases in a single comprehensive figure.
    """
    logger.info("Creating comprehensive visualization of best and worst cases...")
    
    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Title for the entire figure
    fig.suptitle('Wiener Index Algorithm Comparison: Best 3 and Worst 5 Cases', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Create grid layout
    # Top 3 rows for best cases (3 cases × 3 algorithms = 9 subplots)
    # Bottom 5 rows for worst cases (5 cases × 3 algorithms = 15 subplots)
    
    subplot_idx = 1
    
    # Plot best cases
    for i, (case, bbox_ratio, median_ratio) in enumerate(best_cases):
        plot_case_comparison(fig, subplot_idx, case, bbox_ratio, median_ratio, 
                           f"BEST #{i+1}", logger, is_best=True)
        subplot_idx += 3
    
    # Plot worst cases  
    for i, (case, bbox_ratio, median_ratio) in enumerate(worst_cases):
        plot_case_comparison(fig, subplot_idx, case, bbox_ratio, median_ratio, 
                           f"WORST #{i+1}", logger, is_best=False)
        subplot_idx += 3
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.2)
    
    # Save the figure
    plt.savefig('wiener_comparison_best_worst.png', dpi=300, bbox_inches='tight')
    logger.info("Saved comprehensive visualization as 'wiener_comparison_best_worst.png'")
    
    plt.show()


def plot_case_comparison(fig, start_idx: int, case, bbox_ratio: float, median_ratio: float, 
                        case_label: str, logger: logging.Logger, is_best: bool = True) -> None:
    """Plot a single case comparison across three algorithms."""
    
    points = case.points if hasattr(case, 'points') else []
    n_points = case.metadata.get('n_points', len(points))
    distribution = case.metadata.get('distribution', 'unknown')
    seed = case.metadata.get('seed', 'unknown')
    
    # Get results
    brute_force_result = case.results.get('Brute Force (Exact)')
    bbox_result = case.results.get('D&C (BBox Bisection)')
    median_result = case.results.get('D&C (Median Bisection)')
    
    # Extract paths and Wiener indices
    bf_path = brute_force_result.path if brute_force_result else []
    bf_wiener = brute_force_result.wiener_index if brute_force_result else None
    
    bbox_path = bbox_result.path if bbox_result else []
    bbox_wiener = bbox_result.wiener_index if bbox_result else None
    
    median_path = median_result.path if median_result else []
    median_wiener = median_result.wiener_index if median_result else None
    
    # Color scheme
    color_scheme = 'green' if is_best else 'red'
    
    # Extract coordinates
    if points:
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
    else:
        x_coords, y_coords = [], []
    
    # Plot each algorithm
    algorithms = [
        ('Brute Force\n(Optimal)', bf_path, bf_wiener, 'blue'),
        ('D&C (BBox)\nRatio: {:.4f}'.format(bbox_ratio), bbox_path, bbox_wiener, color_scheme),
        ('D&C (Median)\nRatio: {:.4f}'.format(median_ratio if median_ratio else 0), median_path, median_wiener, color_scheme)
    ]
    
    for i, (alg_name, path, wiener, path_color) in enumerate(algorithms):
        ax = fig.add_subplot(8, 3, start_idx + i)  # 8 rows, 3 columns
        
        # Plot points
        if x_coords and y_coords:
            ax.scatter(x_coords, y_coords, c='black', s=60, alpha=0.7, zorder=3)
            
            # Draw convex hull for reference
            if len(points) >= 3:
                hull_points = np.array([(p.x, p.y) for p in points])
                try:
                    hull = ConvexHull(hull_points)
                    for simplex in hull.simplices:
                        ax.plot(hull_points[simplex, 0], hull_points[simplex, 1], 
                               'k--', alpha=0.3, linewidth=1)
                except:
                    pass  # Skip if convex hull fails
        
        # Draw path
        if path:
            path_x = [p.x for p in path]
            path_y = [p.y for p in path]
            ax.plot(path_x, path_y, color=path_color, linewidth=2, alpha=0.8, zorder=2)
            
            # Mark start and end
            ax.scatter(path[0].x, path[0].y, c='lime', s=100, marker='s', 
                      zorder=4, edgecolors='black', linewidth=1)
            ax.scatter(path[-1].x, path[-1].y, c='red', s=100, marker='^', 
                      zorder=4, edgecolors='black', linewidth=1)
        
        # Title with algorithm name and Wiener index
        title = alg_name
        if wiener is not None:
            title += f'\nWiener: {wiener:.2f}'
        ax.set_title(title, fontsize=10, fontweight='bold')
        
        # Add case information on the first subplot of each row
        if i == 0:
            ax.text(0.02, 0.98, f'{case_label}\n{n_points} pts, {distribution}\nSeed: {seed}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Remove axis labels to save space, but keep ticks
        if start_idx <= 18:  # Not bottom row
            ax.set_xticklabels([])
        if (start_idx - 1) % 3 != 0:  # Not leftmost column
            ax.set_yticklabels([])


# ============================================================================
# Algorithm Wrapper Functions
# ============================================================================

def brute_force_algorithm(points: List[Point]) -> List[Point]:
    """
    Wrapper for the brute force solver.
    """
    solver = BruteForceSolver()
    optimal_path, _ = solver.solve(points, use_parallel=True)
    return optimal_path


def divide_conquer_bbox_algorithm(points: List[Point]) -> List[Point]:
    """
    Wrapper for divide and conquer with bounding box bisection.
    """
    solver = DivideConquerSolver(max_depth=10, base_case_size=4)
    return solver.solve(points, use_median_bisection=False)


def divide_conquer_median_algorithm(points: List[Point]) -> List[Point]:
    """
    Wrapper for divide and conquer with median bisection.
    """
    solver = DivideConquerSolver(max_depth=10, base_case_size=4)
    return solver.solve(points, use_median_bisection=True)



# ============================================================================
# Main Usage Example
# ============================================================================

def main():
    """Main function demonstrating the usage of WienerIndexComparisonAnalyzer."""

    # Set up logging
    logger = setup_logging()
    logger.info("Starting Wiener Index Algorithm Comparison Example")

    # ========================================================================
    # Step 1: Initialize Components
    # ========================================================================

    # Create point generator
    point_generator = PointGenerator()

    # Create data manager (optional)
    data_manager = DataManager()

    # Create visualizer (optional)
    visualizer = Visualizer()

    # Create the comparison analyzer
    analyzer = create_wiener_index_analyzer(
        point_generator=point_generator,
        data_manager=data_manager,
        visualizer=visualizer,
        logger=logger
    )

    # ========================================================================
    # Step 2: Register Algorithms
    # ========================================================================

    logger.info("Registering algorithms...")

    # Register algorithms with custom names
    register_wiener_algorithms(
        analyzer=analyzer,
        brute_force_func=brute_force_algorithm,
        divide_conquer_func=divide_conquer_bbox_algorithm,
        divide_conquer_alt_func=divide_conquer_median_algorithm,
        algorithm_names={
            'brute_force': 'Brute Force (Exact)',
            'divide_conquer': 'D&C (BBox Bisection)',
            'divide_conquer_alt': 'D&C (Median Bisection)'
        }
    )

    # ========================================================================
    # Step 3: Run Comparison Study
    # ========================================================================

    logger.info("Running comparison study...")

    # Define test parameters
    point_counts = [4, 5, 6, 7, 8]  # Small sizes for brute force feasibility
    num_seeds = 1  # Number of random instances per point count

    # Run the study with different point distributions
    test_distributions = [
        {'distribution': 'uniform', 'bounds': (0, 100)},
        {'distribution': 'clustered', 'bounds': (0, 100), 'clusters': 2},
        {'distribution': 'circular', 'bounds': (0, 100)}
    ]

    all_study_cases = []

    for dist_config in test_distributions:
        logger.info(f"Testing distribution: {dist_config['distribution']}")

        # Run comparison study for this distribution
        cases = analyzer.run_comparison_study(
            point_counts=point_counts,
            num_seeds=num_seeds,
            algorithms=None,  # Use all registered algorithms
            **dist_config
        )

        # Add distribution info to metadata
        for case in cases:
            case.metadata['distribution'] = dist_config['distribution']

        all_study_cases.extend(cases)

    # ========================================================================
    # Step 4: Print Performance Table
    # ========================================================================

    print_performance_table(all_study_cases, logger)

    # ========================================================================
    # Step 5: Find and Visualize Best/Worst Cases
    # ========================================================================

    logger.info("Finding best and worst cases for visualization...")
    best_cases, worst_cases = find_best_and_worst_cases(all_study_cases)
    
    logger.info(f"Best 3 cases (lowest approximation ratios):")
    for i, (case, bbox_ratio, median_ratio) in enumerate(best_cases):
        logger.info(f"  #{i+1}: {case.metadata.get('n_points')} points, "
                   f"{case.metadata.get('distribution')}, seed {case.metadata.get('seed')}, "
                   f"BBox ratio: {bbox_ratio:.4f}, Median ratio: {median_ratio:.4f}")
    
    logger.info(f"Worst 5 cases (highest approximation ratios):")
    for i, (case, bbox_ratio, median_ratio) in enumerate(worst_cases):
        logger.info(f"  #{i+1}: {case.metadata.get('n_points')} points, "
                   f"{case.metadata.get('distribution')}, seed {case.metadata.get('seed')}, "
                   f"BBox ratio: {bbox_ratio:.4f}, Median ratio: {median_ratio:.4f}")

    # Create comprehensive visualization
    visualize_best_and_worst_cases(best_cases, worst_cases, logger)

    # ========================================================================
    # Step 6: Analyze Results (Original Analysis)
    # ========================================================================

    logger.info("Analyzing approximation ratios...")

    # Analyze approximation ratios
    approximation_analysis = analyzer.analyze_approximation_ratios(
        all_study_cases)

    # Compare divide and conquer algorithms
    logger.info("Comparing divide and conquer algorithms...")
    dc_comparison = analyzer.compare_divide_conquer_algorithms(all_study_cases)

    # ========================================================================
    # Step 7: Generate Performance Summary
    # ========================================================================

    logger.info("Generating performance summary...")

    # Generate comprehensive summary report
    summary_report = analyzer.generate_performance_summary(
        approximation_analysis, dc_comparison
    )

    # Print the summary
    print("\n" + summary_report)

    # ========================================================================
    # Step 8: Additional Analysis Examples
    # ========================================================================

    # Example: Analyze performance by distribution type
    logger.info("Analyzing performance by distribution type...")

    distribution_analysis = {}
    for dist_name in ['uniform', 'clustered', 'circular']:
        dist_cases = [case for case in all_study_cases
                      if case.metadata.get('distribution') == dist_name]

        if dist_cases:
            dist_analysis = analyzer.analyze_approximation_ratios(dist_cases)
            distribution_analysis[dist_name] = dist_analysis

            logger.info(f"\n=== {dist_name.upper()} DISTRIBUTION RESULTS ===")
            for n_points, analysis in dist_analysis.items():
                if n_points == 'unknown':
                    continue

                logger.info(f"\n{n_points} points:")
                for alg_name, stats in analysis.items():
                    if alg_name in ['n_points', 'total_cases']:
                        continue
                    logger.info(f"  {alg_name}: {stats['optimal_rate']:.1%} optimal, "
                                f"avg ratio {stats['ratio_mean']:.4f}")

    # ========================================================================
    # Step 9: Save Results
    # ========================================================================

    logger.info("Saving results...")

    # Save detailed results to JSON file
    analyzer.save_results('wiener_comparison_results.json', all_study_cases)

    # Save summary report to text file
    with open('wiener_performance_summary.txt', 'w') as f:
        f.write(summary_report)
        f.write("\n\n" + "="*80)
        f.write("\nDETAILED DISTRIBUTION ANALYSIS")
        f.write("\n" + "="*80)

        for dist_name, dist_analysis in distribution_analysis.items():
            f.write(f"\n\n{dist_name.upper()} DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")

            for n_points, analysis in dist_analysis.items():
                if n_points == 'unknown':
                    continue

                f.write(f"\n{n_points} points ({
                        analysis['total_cases']} cases):\n")
                for alg_name, stats in analysis.items():
                    if alg_name in ['n_points', 'total_cases']:
                        continue
                    f.write(f"  {alg_name}:\n")
                    f.write(f"    Optimal rate: {stats['optimal_rate']:.1%}\n")
                    f.write(f"    Avg ratio: {stats['ratio_mean']:.4f} ± {
                            stats['ratio_std']:.4f}\n")
                    f.write(f"    Success rate: {stats['success_rate']:.1%}\n")

    # ========================================================================
    # Step 10: Example of Custom Analysis
    # ========================================================================

    # logger.info("Running custom analysis example...")

    # # Example: Find cases where divide and conquer algorithms disagree significantly
    # disagreement_cases = []
    #
    # for case in all_study_cases:
    #     if ('D&C (BBox Bisection)' in case.results and
    #             'D&C (Median Bisection)' in case.results):
    #
    #         bbox_result = case.results['D&C (BBox Bisection)']
    #         median_result = case.results['D&C (Median Bisection)']
    #         bbox_wiener = bbox_result.wiener_index if bbox_result else None
    #         median_wiener = median_result.wiener_index if median_result else None
    #         if bbox_wiener is not None and median_wiener is not None:
    #             ratio_diff = abs(bbox_wiener - median_wiener)
    #             if ratio_diff > 0.1:  # Arbitrary threshold for significant disagreement
    #                 disagreement_cases.append(case)
            

if __name__ == "__main__":
    main()
