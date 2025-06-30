import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from multiprocessing import cpu_count

from core.point import Point
from core.wiener_index_calculator import WienerIndexCalculator
from generators.point_generator import PointGenerator
from solvers.brute_force_solver import BruteForceSolver
from solvers.divide_conquer_solver import DivideConquerSolver


@dataclass
class ExperimentResult:
    """Container for single experiment results."""
    n_points: int
    seed: int
    points: List[Point]
    dc_path: List[Point]
    optimal_path: Optional[List[Point]]
    dc_wiener: float
    optimal_wiener: Optional[float]
    approximation_ratio: Optional[float]
    dc_time: float
    optimal_time: Optional[float]


@dataclass
class StatisticalSummary:
    """Container for statistical summary of multiple experiments."""
    n_points: int
    num_experiments: int
    dc_wiener_mean: float
    dc_wiener_std: float
    dc_wiener_min: float
    dc_wiener_max: float
    optimal_wiener_mean: Optional[float]
    optimal_wiener_std: Optional[float]
    optimal_wiener_min: Optional[float]
    optimal_wiener_max: Optional[float]
    ratio_mean: Optional[float]
    ratio_std: Optional[float]
    ratio_min: Optional[float]
    ratio_max: Optional[float]
    dc_time_mean: float
    optimal_time_mean: Optional[float]
    perfect_solutions: int
    good_solutions: int
    poor_solutions: int


class StatisticalAnalyzer:
    """Handles statistical analysis and comparison of algorithms."""

    def __init__(self, use_parallel: bool = True, max_workers: Optional[int] = None):
        self.use_parallel = use_parallel
        self.max_workers = max_workers or min(cpu_count(), 8)
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.point_generator = PointGenerator()
        self.wiener_calculator = WienerIndexCalculator()
        self.brute_force_solver = BruteForceSolver(
            use_parallel=use_parallel, max_workers=self.max_workers)
        self.dc_solver = DivideConquerSolver()

    def run_single_experiment(self, n_points: int, seed: int, is_convex: bool = False) -> ExperimentResult:
        """Run a single experiment comparing algorithms."""
        # Generate points
        if is_convex:
            points = self.point_generator.generate_convex_points(
                n_points, seed=seed)
        else:
            points = self.point_generator.generate_general_points(
                n_points, seed=seed)

        # Run divide and conquer
        start_time = time.time()
        dc_path = self.dc_solver.solve(points)
        dc_time = time.time() - start_time
        dc_wiener = self.wiener_calculator.calculate_path_wiener_index(dc_path)

        # Run brute force if feasible
        optimal_path = None
        optimal_wiener = None
        optimal_time = None
        approximation_ratio = None

        if n_points <= 10:  # Brute force threshold
            try:
                start_time = time.time()
                optimal_path = self.brute_force_solver.solve(points)
                optimal_time = time.time() - start_time
                optimal_wiener = self.wiener_calculator.calculate_path_wiener_index(
                    optimal_path)
                approximation_ratio = dc_wiener / optimal_wiener
            except MemoryError:
                self.logger.warning(f"Brute force skipped for seed {
                                    seed} (memory limit)")

        return ExperimentResult(
            n_points=n_points,
            seed=seed,
            points=points,
            dc_path=dc_path,
            optimal_path=optimal_path,
            dc_wiener=dc_wiener,
            optimal_wiener=optimal_wiener,
            approximation_ratio=approximation_ratio,
            dc_time=dc_time,
            optimal_time=optimal_time
        )

    def run_experiments(self, n_points: int, num_seeds: int, is_convex: bool = False) -> List[ExperimentResult]:
        """Run multiple experiments for a given point count."""
        self.logger.info(f"Running {num_seeds} experiments for {
                         n_points} points...")

        results = []
        for seed in range(num_seeds):
            if (seed + 1) % 20 == 0:
                self.logger.info(
                    f"  Completed {seed + 1}/{num_seeds} experiments...")

            result = self.run_single_experiment(n_points, seed, is_convex)
            results.append(result)

        return results

    def calculate_statistics(self, results: List[ExperimentResult]) -> StatisticalSummary:
        """Calculate statistical summary from experiment results."""
        if not results:
            raise ValueError("No results to analyze")

        n_points = results[0].n_points
        num_experiments = len(results)

        # Extract data arrays
        dc_wieners = np.array([r.dc_wiener for r in results])
        dc_times = np.array([r.dc_time for r in results])

        # Handle optional optimal results
        optimal_wieners = [
            r.optimal_wiener for r in results if r.optimal_wiener is not None]
        optimal_times = [
            r.optimal_time for r in results if r.optimal_time is not None]
        ratios = [
            r.approximation_ratio for r in results if r.approximation_ratio is not None]

        # Calculate basic statistics
        stats = StatisticalSummary(
            n_points=n_points,
            num_experiments=num_experiments,
            dc_wiener_mean=float(np.mean(dc_wieners)),
            dc_wiener_std=float(np.std(dc_wieners)),
            dc_wiener_min=float(np.min(dc_wieners)),
            dc_wiener_max=float(np.max(dc_wieners)),
            dc_time_mean=float(np.mean(dc_times)),
            optimal_wiener_mean=None,
            optimal_wiener_std=None,
            optimal_wiener_min=None,
            optimal_wiener_max=None,
            ratio_mean=None,
            ratio_std=None,
            ratio_min=None,
            ratio_max=None,
            optimal_time_mean=None,
            perfect_solutions=0,
            good_solutions=0,
            poor_solutions=0
        )

        # Add optimal statistics if available
        if optimal_wieners:
            optimal_array = np.array(optimal_wieners)
            stats.optimal_wiener_mean = float(np.mean(optimal_array))
            stats.optimal_wiener_std = float(np.std(optimal_array))
            stats.optimal_wiener_min = float(np.min(optimal_array))
            stats.optimal_wiener_max = float(np.max(optimal_array))

        if optimal_times:
            stats.optimal_time_mean = float(np.mean(optimal_times))

        if ratios:
            ratio_array = np.array(ratios)
            stats.ratio_mean = float(np.mean(ratio_array))
            stats.ratio_std = float(np.std(ratio_array))
            stats.ratio_min = float(np.min(ratio_array))
            stats.ratio_max = float(np.max(ratio_array))

            # Count solution quality
            stats.perfect_solutions = int(np.sum(ratio_array <= 1.001))
            stats.good_solutions = int(np.sum(ratio_array <= 1.1))
            stats.poor_solutions = int(np.sum(ratio_array > 1.5))

        return stats

    def find_interesting_cases(self, results: List[ExperimentResult]) -> Dict[str, ExperimentResult]:
        """Find best, worst, and median cases from results."""
        # Filter results with approximation ratios
        valid_results = [
            r for r in results if r.approximation_ratio is not None]

        if not valid_results:
            return {}

        # Sort by approximation ratio
        sorted_results = sorted(
            valid_results, key=lambda r: r.approximation_ratio)

        interesting_cases = {}

        if len(sorted_results) > 0:
            interesting_cases['best'] = sorted_results[0]
            interesting_cases['worst'] = sorted_results[-1]

            # Find median case
            median_idx = len(sorted_results) // 2
            interesting_cases['median'] = sorted_results[median_idx]

        return interesting_cases

    def log_statistics(self, stats: StatisticalSummary) -> None:
        """Log statistical summary."""
        self.logger.info(f"Statistics for {stats.n_points} points ({
                         stats.num_experiments} experiments):")
        self.logger.info(f"  D&C Wiener Index: {stats.dc_wiener_mean:.4f} ± {
                         stats.dc_wiener_std:.4f}")
        self.logger.info(f"  D&C Time: {stats.dc_time_mean:.4f}s")

        if stats.optimal_wiener_mean is not None:
            self.logger.info(f"  Optimal Wiener Index: {stats.optimal_wiener_mean:.4f} ± {
                             stats.optimal_wiener_std:.4f}")
            self.logger.info(f"  Optimal Time: {stats.optimal_time_mean:.4f}s")
            self.logger.info(f"  Approximation Ratio: {
                             stats.ratio_mean:.4f} ± {stats.ratio_std:.4f}")
            self.logger.info(
                f"  Range: [{stats.ratio_min:.4f}, {stats.ratio_max:.4f}]")
            self.logger.info(f"  Perfect solutions (≤1.001): {stats.perfect_solutions}/{stats.num_experiments} "
                             f"({100*stats.perfect_solutions/stats.num_experiments:.1f}%)")
            self.logger.info(f"  Good solutions (≤1.1): {stats.good_solutions}/{stats.num_experiments} "
                             f"({100*stats.good_solutions/stats.num_experiments:.1f}%)")
            self.logger.info(f"  Poor solutions (>1.5): {stats.poor_solutions}/{stats.num_experiments} "
                             f"({100*stats.poor_solutions/stats.num_experiments:.1f}%)")
            self.logger.info(f"  Speedup factor: {
                             stats.optimal_time_mean/stats.dc_time_mean:.1f}x")

    def comprehensive_analysis(self, point_counts: List[int], num_seeds: int = 100,
                               is_convex: bool = False) -> Dict[int, Tuple[StatisticalSummary, List[ExperimentResult]]]:
        """Run comprehensive analysis across multiple point counts."""
        self.logger.info("=== Comprehensive Statistical Analysis ===")
        self.logger.info(f"Testing point counts: {point_counts}")
        self.logger.info(f"Seeds per test: {num_seeds}")
        self.logger.info(f"Point type: {'Convex' if is_convex else 'General'}")

        all_results = {}

        for n_points in point_counts:
            self.logger.info(f"{'='*50}")
            self.logger.info(f"Analyzing {n_points} points...")
            self.logger.info(f"{'='*50}")

            # Run experiments
            results = self.run_experiments(n_points, num_seeds, is_convex)

            # Calculate statistics
            stats = self.calculate_statistics(results)

            # Log results
            self.log_statistics(stats)

            all_results[n_points] = (stats, results)

        # Log overall summary
        self.logger.info(f"\n{'='*50}")
        self.logger.info("Overall Summary:")
        self.logger.info(f"{'='*50}")

        for n_points, (stats, _) in all_results.items():
            self.logger.info(f"\n{n_points} points:")
            self.logger.info(f"  Approximation Ratio: {stats.ratio_mean:.4f} ± {stats.ratio_std:.4f}"
                             if stats.ratio_mean else "  No optimal comparison available")
            self.logger.info(f"  D&C Time: {stats.dc_time_mean:.4f}s")

        return all_results
