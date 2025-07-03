#!/usr/bin/env python3
"""
Refactored Wiener Analysis Orchestrator

This module provides a clean, modular orchestrator for running Wiener index
analysis experiments comparing different Hamiltonian path algorithms.

The original monolithic code has been refactored to use the modular components:
- generators.point_generator: Point generation
- solvers.brute_force_solver: Brute force algorithm
- solvers.divide_conquer_solver: Divide and conquer algorithm
- analysis.statistical_analyzer: Statistical analysis and result containers
- utils.data_manager: Data persistence
- visualization.visualizer: Result visualization
"""

import time
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

from generators.point_generator import PointGenerator
from solvers.brute_force_solver import BruteForceSolver
from solvers.divide_conquer_solver import DivideConquerSolver
from analysis.statistical_analyzer import (
    StatisticalAnalyzer,
    AlgorithmResult,
    MultiAlgorithmExperiment,
    StudyResults
)
from utils.data_manager import DataManager
from visualization.visualizer import Visualizer
from utils.logger_setup import setup_logging


class WienerAnalysisOrchestrator:
    """
    Orchestrates Wiener index analysis experiments comparing different algorithms.

    This class coordinates all the modular components to run comprehensive
    experiments and generate analysis results.
    """

    def __init__(self,
                 output_dir: str = "wiener_analysis_data",
                 log_dir: str = "wiener_analysis_logs",
                 enable_visualization: bool = True):
        """
        Initialize the orchestrator with all necessary components.

        Args:
            output_dir: Directory for saving experiment data
            log_dir: Directory for saving logs
            enable_visualization: Whether to enable visualization components
        """
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.enable_visualization = enable_visualization

        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        # Setup logging
        self.logger = setup_logging(log_dir=str(self.log_dir))

        # Initialize components
        self.point_generator = PointGenerator()
        self.brute_force_solver = BruteForceSolver()
        self.divide_conquer_solver = DivideConquerSolver()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.data_manager = DataManager(output_dir=str(self.output_dir))

        if self.enable_visualization:
            self.visualizer = Visualizer()

        self.logger.info("WienerAnalysisOrchestrator initialized")

    def run_single_experiment(self,
                              n_points: int,
                              point_type: str = "convex",
                              seed: Optional[int] = None,
                              algorithms: Optional[List[str]] = None) -> MultiAlgorithmExperiment:
        """
        Run a single experiment comparing algorithms on one point set.

        Args:
            n_points: Number of points to generate
            point_type: Type of points ('convex', 'general', etc.)
            seed: Random seed for reproducibility
            algorithms: List of algorithms to run ('brute_force', 'divide_conquer')

        Returns:
            MultiAlgorithmExperiment containing all results
        """
        if algorithms is None:
            algorithms = ['brute_force', 'divide_conquer']

        self.logger.info(f"Starting experiment: n={
                         n_points}, type={point_type}, seed={seed}")

        # Generate points
        points = self.point_generator.generate_points(
            n_points, point_type, seed)
        experiment = MultiAlgorithmExperiment(
            n_points=n_points, seed=seed or 0, points=points)

        # Run each algorithm
        for algorithm in algorithms:
            try:
                start_time = time.time()

                if algorithm == 'brute_force':
                    path, wiener_index = self.brute_force_solver.solve(points)
                elif algorithm == 'divide_conquer':
                    path, wiener_index = self.divide_conquer_solver.solve_with_wiener(
                        points)
                elif algorithm == 'divide_conquer_median':
                    path, wiener_index = self.divide_conquer_solver.solve_with_wiener(
                        points, use_median_bisection=True)
                else:
                    self.logger.warning(f"Unknown algorithm: {algorithm}")
                    continue

                execution_time = time.time() - start_time

                result = AlgorithmResult(
                    algorithm_name=algorithm,
                    path=path,
                    wiener_index=wiener_index,
                    execution_time=execution_time
                )

                experiment.add_result(algorithm, result)
                self.logger.info(f"Algorithm {algorithm}: Wiener={
                                 wiener_index:.4f}, Time={execution_time:.4f}s")

            except Exception as e:
                self.logger.error(f"Error running {algorithm}: {e}")

        return experiment

    def run_comparison_study(self,
                             point_sizes: List[int],
                             trials_per_size: int = 10,
                             point_type: str = "convex",
                             algorithms: Optional[List[str]] = None,
                             save_results: bool = True,
                             generate_plots: bool = True) -> StudyResults:
        """
        Run a comprehensive comparison study across multiple point sizes and trials.

        Args:
            point_sizes: List of point counts to test
            trials_per_size: Number of random trials per point size
            point_type: Type of points to generate
            algorithms: Algorithms to compare
            save_results: Whether to save results to disk
            generate_plots: Whether to generate visualization plots

        Returns:
            StudyResults containing all experiment data and statistics
        """
        if algorithms is None:
            algorithms = ['brute_force', 'divide_conquer']

        self.logger.info(f"Starting comparison study: sizes={
                         point_sizes}, trials={trials_per_size}")

        experiments = []

        for n_points in point_sizes:
            self.logger.info(f"Running experiments for n={n_points}")

            for trial in range(trials_per_size):
                seed = trial  # Use trial number as seed for reproducibility
                experiment = self.run_single_experiment(
                    n_points=n_points,
                    point_type=point_type,
                    seed=seed,
                    algorithms=algorithms
                )
                experiments.append(experiment)

        # Analyze results
        study_results = self.statistical_analyzer.analyze_multi_algorithm_study(
            experiments)

        if save_results:
            # Save raw data
            filename = f"study_{point_type}_{
                time.strftime('%Y%m%d_%H%M%S')}.json"
            self.data_manager.save_multi_algorithm_results(
                experiments, filename)

            # Save analysis summary
            summary_filename = f"summary_{point_type}_{
                time.strftime('%Y%m%d_%H%M%S')}.json"
            self.data_manager.save_study_results(
                study_results, summary_filename)

            self.logger.info(f"Results saved to {
                             filename} and {summary_filename}")

        if generate_plots and self.enable_visualization:
            self._generate_study_plots(study_results, point_type)

        return study_results

    def run_quick_test(self, n_points: int = 6) -> Dict[str, Any]:
        """
        Run a quick test to verify all components are working.

        Args:
            n_points: Number of points for the test

        Returns:
            Dictionary with test results
        """
        self.logger.info(f"Running quick test with {n_points} points")

        experiment = self.run_single_experiment(
            n_points=n_points,
            point_type="convex",
            seed=42,
            algorithms=['brute_force', 'divide_conquer']
        )

        results = {
            'n_points': n_points,
            'algorithms_tested': list(experiment.results.keys()),
            'results': {}
        }

        for algorithm, result in experiment.results.items():
            results['results'][algorithm] = {
                'wiener_index': result.wiener_index,
                'execution_time': result.execution_time,
                'path_length': len(result.path)
            }

        # Calculate approximation ratio if both algorithms ran
        if 'brute_force' in experiment.results and 'divide_conquer' in experiment.results:
            ratio = experiment.get_approximation_ratio(
                'divide_conquer', 'brute_force')
            results['approximation_ratio'] = ratio

        self.logger.info("Quick test completed successfully")
        return results

    def _generate_study_plots(self, study_results: StudyResults, point_type: str) -> None:
        """Generate comprehensive visualization plots for study results."""
        try:
            plot_dir = Path(f"wiener_comparison_plots_{point_type}")
            plot_dir.mkdir(exist_ok=True)

            # Generate comprehensive study overview
            self.visualizer.plot_study_results(
                study_results,
                save_path=str(plot_dir / "study_overview.png")
            )

            # Generate algorithm performance comparison
            self.visualizer.plot_algorithm_comparison(
                study_results,
                save_path=str(plot_dir / "algorithm_comparison.png")
            )

            # Generate execution time analysis
            self.visualizer.plot_execution_times(
                study_results,
                save_path=str(plot_dir / "execution_times.png")
            )

            # Generate approximation ratio analysis if multiple algorithms
            if len(study_results.statistics) > 1:
                self.visualizer.plot_approximation_ratios(
                    study_results,
                    save_path=str(plot_dir / "approximation_ratios.png")
                )

            # Generate comprehensive report
            self.visualizer.create_comprehensive_report(
                study_results,
                output_dir=str(plot_dir / "detailed_analysis")
            )

            self.logger.info(
                f"Comprehensive visualization report generated in {plot_dir}")

        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")


def main():
    """
    Main function demonstrating the refactored orchestrator usage.
    """
    # Initialize orchestrator
    orchestrator = WienerAnalysisOrchestrator()

    # Run quick test
    print("Running quick test...")
    test_results = orchestrator.run_quick_test(n_points=6)
    print(f"Quick test results: {test_results}")

    # Run small comparison study
    print("\nRunning small comparison study...")
    study_results = orchestrator.run_comparison_study(
        point_sizes=[5, 6, 7],
        trials_per_size=3,
        point_type="convex"
    )

    print(f"\nStudy completed with {
          len(study_results.experiments)} experiments")
    print("Algorithm performance summary:")
    for algorithm, stats in study_results.statistics.items():
        print(f"  {algorithm}:")
        print(f"    Avg Wiener Index: {stats.wiener_index_mean:.4f}")
        print(f"    Avg Execution Time: {stats.execution_time_mean:.4f}s")


if __name__ == "__main__":
    main()
