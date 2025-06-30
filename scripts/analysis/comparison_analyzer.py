"""
Enhanced Comparison Analyzer for Wiener Index Algorithm Performance Analysis

This module provides a specialized framework for comparing Wiener index algorithms
with focus on approximation ratios and performance binning.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict

from core.point import Point
from core.wiener_index_calculator import WienerIndexCalculator
from generators.point_generator import PointGenerator
from utils.data_manager import DataManager
from visualization.visualizer import Visualizer


@dataclass
class WienerIndexResult:
    """Container for Wiener index algorithm execution results."""
    algorithm_name: str
    solution: List[Point]
    wiener_index: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'algorithm_name': self.algorithm_name,
            'solution': [p.to_dict() for p in self.solution],
            'wiener_index': self.wiener_index,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }


@dataclass
class WienerComparisonCase:
    """Container for a single Wiener index comparison case."""
    case_id: str
    points: List[Point]
    results: Dict[str, WienerIndexResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: WienerIndexResult):
        """Add an algorithm result to this case."""
        self.results[result.algorithm_name] = result

    def get_optimal_wiener_index(self) -> Optional[float]:
        """Get the optimal (minimum) Wiener index from exact algorithms."""
        exact_results = [r for r in self.results.values()
                         if r.metadata.get('is_exact', False)]
        if not exact_results:
            return None
        return min(r.wiener_index for r in exact_results)

    def get_approximation_ratio(self, algorithm_name: str) -> Optional[float]:
        """Calculate approximation ratio for an algorithm against optimal."""
        if algorithm_name not in self.results:
            return None

        optimal_wiener = self.get_optimal_wiener_index()
        if optimal_wiener is None or optimal_wiener == 0:
            return None

        alg_wiener = self.results[algorithm_name].wiener_index
        return alg_wiener / optimal_wiener

    def is_optimal_solution(self, algorithm_name: str, tolerance: float = 1e-6) -> bool:
        """Check if algorithm found optimal solution within tolerance."""
        ratio = self.get_approximation_ratio(algorithm_name)
        return ratio is not None and ratio <= (1.0 + tolerance)


class WienerIndexAlgorithmInterface(ABC):
    """Abstract interface for Wiener index algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the algorithm name."""
        pass

    @abstractmethod
    def solve(self, points: List[Point]) -> List[Point]:
        """Solve the problem and return the solution."""
        pass

    @property
    @abstractmethod
    def is_exact(self) -> bool:
        """Return True if this is an exact algorithm."""
        pass

    @property
    def max_feasible_size(self) -> Optional[int]:
        """Return maximum feasible problem size, or None if no limit."""
        return None


class WienerIndexAlgorithmAdapter(WienerIndexAlgorithmInterface):
    """Adapter for Wiener index algorithms."""

    def __init__(self, algorithm_func: Callable[[List[Point]], List[Point]],
                 name: str, is_exact: bool = False, max_size: Optional[int] = None):
        self._algorithm_func = algorithm_func
        self._name = name
        self._is_exact = is_exact
        self._max_size = max_size

    @property
    def name(self) -> str:
        return self._name

    def solve(self, points: List[Point]) -> List[Point]:
        """Solve the problem and return the solution."""
        try:
            # Ensure we're passing Point objects
            if points and isinstance(points[0], dict):
                point_objects = [Point(p['x'], p['y']) for p in points]
                return self._algorithm_func(point_objects)
            else:
                return self._algorithm_func(points)
        except Exception as e:
            print(f"Error in {self._name}: {e}")
            raise

    @property
    def is_exact(self) -> bool:
        return self._is_exact

    @property
    def max_feasible_size(self) -> Optional[int]:
        return self._max_size


class WienerIndexComparisonAnalyzer:
    """
    Specialized analyzer for comparing Wiener index algorithms with focus on
    approximation ratios and performance binning.
    """

    def __init__(self,
                 point_generator: PointGenerator,
                 data_manager: Optional[DataManager] = None,
                 visualizer: Optional[Visualizer] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the Wiener index comparison analyzer."""
        self.wiener_calculator = WienerIndexCalculator()
        self.point_generator = point_generator
        self.data_manager = data_manager
        self.visualizer = visualizer
        self.logger = logger or logging.getLogger(__name__)

        self.algorithms: Dict[str, WienerIndexAlgorithmInterface] = {}
        self.comparison_cases: List[WienerComparisonCase] = []

        # Define approximation ratio bins
        self.ratio_bins = [
            (1.0, "Optimal"),
            (1.01, "Near-optimal (≤1%)"),
            (1.05, "Very good (≤5%)"),
            (1.10, "Good (≤10%)"),
            (1.20, "Acceptable (≤20%)"),
            (1.50, "Poor (≤50%)"),
            (float('inf'), "Very poor (>50%)")
        ]

    def register_algorithm(self, algorithm: WienerIndexAlgorithmInterface):
        """Register an algorithm for comparison."""
        self.algorithms[algorithm.name] = algorithm
        self.logger.info(f"Registered algorithm: {
                         algorithm.name} (exact: {algorithm.is_exact})")

    def run_algorithm_on_case(self, algorithm: WienerIndexAlgorithmInterface,
                              points: List[Point]) -> WienerIndexResult:
        """Run a single algorithm on a single test case."""
        start_time = time.time()

        try:
            solution = algorithm.solve(points)
            execution_time = time.time() - start_time
            wiener_index = self.wiener_calculator.calculate_wiener_index(
                solution)

            return WienerIndexResult(
                algorithm_name=algorithm.name,
                solution=solution,
                wiener_index=wiener_index,
                execution_time=execution_time,
                metadata={'success': True, 'is_exact': algorithm.is_exact}
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Algorithm {algorithm.name} failed: {str(e)}")

            return WienerIndexResult(
                algorithm_name=algorithm.name,
                solution=[],
                wiener_index=float('inf'),
                execution_time=execution_time,
                metadata={'success': False, 'error': str(
                    e), 'is_exact': algorithm.is_exact}
            )

    def run_comparison_study(self,
                             point_counts: List[int],
                             num_seeds: int = 100,
                             algorithms: Optional[List[str]] = None,
                             **generator_kwargs) -> List[WienerComparisonCase]:
        """Run a comprehensive comparison study focused on Wiener index performance."""
        self.logger.info("=== Starting Wiener Index Comparison Study ===")
        self.logger.info(f"Point counts: {point_counts}")
        self.logger.info(f"Seeds per test: {num_seeds}")
        self.logger.info(
            f"Algorithms: {algorithms or list(self.algorithms.keys())}")

        if algorithms is None:
            algorithms = list(self.algorithms.keys())

        all_cases = []

        for n_points in point_counts:
            self.logger.info(f"Testing {n_points} points across {
                             num_seeds} seeds...")

            for seed in range(num_seeds):
                if (seed + 1) % 20 == 0:
                    self.logger.info(
                        f"  Completed {seed + 1}/{num_seeds} seeds...")

                # Generate test case
                points = self.point_generator.generate_points(
                    n_points, seed=seed, **generator_kwargs)

                case_id = f"n{n_points}_s{seed}"
                case = WienerComparisonCase(case_id=case_id, points=points)
                case.metadata.update({
                    'n_points': n_points,
                    'seed': seed,
                    'generator_kwargs': generator_kwargs
                })

                # Run algorithms
                for alg_name in algorithms:
                    if alg_name not in self.algorithms:
                        continue

                    algorithm = self.algorithms[alg_name]

                    # Check feasibility
                    if (algorithm.max_feasible_size is not None and
                            len(points) > algorithm.max_feasible_size):
                        self.logger.debug(
                            f"Skipping {alg_name} (size limit exceeded)")
                        continue

                    result = self.run_algorithm_on_case(algorithm, points)
                    case.add_result(result)

                all_cases.append(case)

        self.comparison_cases.extend(all_cases)
        self.logger.info(f"Comparison study completed. Total cases: {
                         len(all_cases)}")
        return all_cases

    def analyze_approximation_ratios(self, cases: Optional[List[WienerComparisonCase]] = None) -> Dict[str, Any]:
        """Analyze approximation ratios with binning and detailed statistics."""
        if cases is None:
            cases = self.comparison_cases

        if not cases:
            self.logger.warning("No cases to analyze")
            return {}

        # Group cases by point count
        grouped_cases = defaultdict(list)
        for case in cases:
            n_points = case.metadata.get('n_points', 'unknown')
            grouped_cases[n_points].append(case)

        analysis_results = {}

        for n_points, point_cases in grouped_cases.items():
            self.logger.info(f"\n=== Analyzing {len(point_cases)} cases for {
                             n_points} points ===")

            # Filter cases that have at least one exact algorithm result
            valid_cases = [
                case for case in point_cases if case.get_optimal_wiener_index() is not None]

            if not valid_cases:
                self.logger.warning(
                    f"No valid cases with exact solutions for {n_points} points")
                continue

            self.logger.info(f"Valid cases with exact solutions: {
                             len(valid_cases)}")

            point_analysis = {'n_points': n_points,
                              'total_cases': len(valid_cases)}

            # Analyze each non-exact algorithm
            heuristic_algorithms = [name for name in self.algorithms.keys()
                                    if not self.algorithms[name].is_exact]

            for alg_name in heuristic_algorithms:
                # Collect ratios for this algorithm
                ratios = []
                optimal_count = 0
                failed_cases = 0

                for case in valid_cases:
                    if alg_name not in case.results:
                        failed_cases += 1
                        continue

                    if not case.results[alg_name].metadata.get('success', False):
                        failed_cases += 1
                        continue

                    ratio = case.get_approximation_ratio(alg_name)
                    if ratio is not None and ratio != float('inf'):
                        ratios.append(ratio)
                        if case.is_optimal_solution(alg_name):
                            optimal_count += 1

                if not ratios:
                    self.logger.warning(f"No valid ratios for {alg_name}")
                    continue

                ratios = np.array(ratios)

                # Calculate statistics
                stats = {
                    'algorithm_name': alg_name,
                    'total_cases': len(valid_cases),
                    'successful_cases': len(ratios),
                    'failed_cases': failed_cases,
                    'success_rate': len(ratios) / len(valid_cases),
                    'optimal_solutions': optimal_count,
                    'optimal_rate': optimal_count / len(ratios) if ratios.size > 0 else 0,
                    'ratio_mean': np.mean(ratios),
                    'ratio_median': np.median(ratios),
                    'ratio_std': np.std(ratios),
                    'ratio_min': np.min(ratios),
                    'ratio_max': np.max(ratios),
                    'ratio_q25': np.percentile(ratios, 25),
                    'ratio_q75': np.percentile(ratios, 75)
                }

                # Bin the ratios
                bin_counts = {}
                for i, (threshold, label) in enumerate(self.ratio_bins):
                    if i == 0:
                        # First bin is exactly 1.0 (optimal)
                        count = np.sum(ratios <= 1.0 + 1e-6)
                    else:
                        prev_threshold = self.ratio_bins[i-1][0]
                        count = np.sum((ratios > prev_threshold)
                                       & (ratios <= threshold))

                    bin_counts[label] = {
                        'count': int(count),
                        'percentage': count / len(ratios) * 100 if len(ratios) > 0 else 0
                    }

                stats['ratio_bins'] = bin_counts
                point_analysis[alg_name] = stats

                # Log detailed results
                self.logger.info(f"\n  {alg_name} Results:")
                self.logger.info(f"    Success rate: {stats['success_rate']:.1%} ({
                                 stats['successful_cases']}/{stats['total_cases']})")
                self.logger.info(f"    Optimal solutions: {stats['optimal_rate']:.1%} ({
                                 stats['optimal_solutions']}/{stats['successful_cases']})")
                self.logger.info(f"    Approximation ratio: {
                                 stats['ratio_mean']:.4f} ± {stats['ratio_std']:.4f}")
                self.logger.info(
                    f"    Ratio range: [{stats['ratio_min']:.4f}, {stats['ratio_max']:.4f}]")
                self.logger.info(f"    Median ratio: {
                                 stats['ratio_median']:.4f}")

                self.logger.info("    Ratio distribution:")
                for label, bin_data in bin_counts.items():
                    self.logger.info(f"      {label}: {bin_data['count']} ({
                                     bin_data['percentage']:.1f}%)")

            analysis_results[n_points] = point_analysis

        return analysis_results

    def compare_divide_conquer_algorithms(self, cases: Optional[List[WienerComparisonCase]] = None) -> Dict[str, Any]:
        """Compare divide and conquer algorithms to determine which performs better."""
        if cases is None:
            cases = self.comparison_cases

        # Get divide and conquer algorithms
        dc_algorithms = [name for name in self.algorithms.keys()
                         if 'divide' in name.lower() or 'dc' in name.lower()]

        if len(dc_algorithms) < 2:
            self.logger.warning(
                "Need at least 2 divide and conquer algorithms for comparison")
            return {}

        self.logger.info(f"\n=== Comparing Divide and Conquer Algorithms ===")
        self.logger.info(f"Algorithms: {dc_algorithms}")

        # Group by point count
        grouped_cases = defaultdict(list)
        for case in cases:
            n_points = case.metadata.get('n_points', 'unknown')
            grouped_cases[n_points].append(case)

        comparison_results = {}

        for n_points, point_cases in grouped_cases.items():
            self.logger.info(f"\n--- Comparing for {n_points} points ---")

            # Filter cases where both algorithms succeeded
            valid_cases = []
            for case in point_cases:
                if all(alg in case.results and case.results[alg].metadata.get('success', False)
                       for alg in dc_algorithms):
                    valid_cases.append(case)

            if not valid_cases:
                self.logger.warning(
                    f"No cases where all DC algorithms succeeded for {n_points} points")
                continue

            self.logger.info(f"Valid comparison cases: {len(valid_cases)}")

            # Compare algorithms pairwise
            comparison_matrix = {}

            for i, alg1 in enumerate(dc_algorithms):
                for j, alg2 in enumerate(dc_algorithms):
                    if i >= j:  # Only compare each pair once
                        continue

                    alg1_better = 0
                    alg2_better = 0
                    ties = 0
                    wiener_diffs = []

                    for case in valid_cases:
                        wiener1 = case.results[alg1].wiener_index
                        wiener2 = case.results[alg2].wiener_index
                        diff = wiener1 - wiener2
                        wiener_diffs.append(diff)

                        if abs(diff) < 1e-6:  # Tie
                            ties += 1
                        elif diff < 0:  # alg1 better (lower Wiener index)
                            alg1_better += 1
                        else:  # alg2 better
                            alg2_better += 1

                    wiener_diffs = np.array(wiener_diffs)

                    comparison_stats = {
                        f'{alg1}_better': alg1_better,
                        f'{alg2}_better': alg2_better,
                        'ties': ties,
                        f'{alg1}_win_rate': alg1_better / len(valid_cases),
                        f'{alg2}_win_rate': alg2_better / len(valid_cases),
                        'tie_rate': ties / len(valid_cases),
                        # positive means alg2 better
                        'mean_wiener_diff': np.mean(wiener_diffs),
                        'median_wiener_diff': np.median(wiener_diffs),
                        'std_wiener_diff': np.std(wiener_diffs)
                    }

                    comparison_matrix[f"{alg1}_vs_{alg2}"] = comparison_stats

                    # Log results
                    self.logger.info(f"\n  {alg1} vs {alg2}:")
                    self.logger.info(f"    {alg1} better: {alg1_better} ({
                                     alg1_better/len(valid_cases):.1%})")
                    self.logger.info(f"    {alg2} better: {alg2_better} ({
                                     alg2_better/len(valid_cases):.1%})")
                    self.logger.info(
                        f"    Ties: {ties} ({ties/len(valid_cases):.1%})")
                    self.logger.info(f"    Mean Wiener difference: {
                                     np.mean(wiener_diffs):.4f}")

            comparison_results[n_points] = {
                'algorithms': dc_algorithms,
                'valid_cases': len(valid_cases),
                'comparisons': comparison_matrix
            }

        return comparison_results

    def generate_performance_summary(self, approximation_analysis: Dict[str, Any],
                                     dc_comparison: Dict[str, Any]) -> str:
        """Generate a comprehensive performance summary report."""
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("WIENER INDEX ALGORITHM PERFORMANCE SUMMARY")
        summary_lines.append("=" * 80)

        # Approximation ratio summary
        summary_lines.append("\n📊 APPROXIMATION RATIO ANALYSIS")
        summary_lines.append("-" * 40)

        for n_points, analysis in approximation_analysis.items():
            if n_points == 'unknown':
                continue

            summary_lines.append(
                f"\n🔸 {n_points} Points ({analysis['total_cases']} test cases):")

            # Find best performing algorithm
            best_alg = None
            best_optimal_rate = -1
            best_mean_ratio = float('inf')

            alg_performances = []
            for alg_name, stats in analysis.items():
                if alg_name in ['n_points', 'total_cases']:
                    continue

                alg_performances.append((
                    alg_name,
                    stats['optimal_rate'],
                    stats['ratio_mean'],
                    stats['success_rate']
                ))

                if stats['optimal_rate'] > best_optimal_rate or \
                   (stats['optimal_rate'] == best_optimal_rate and stats['ratio_mean'] < best_mean_ratio):
                    best_alg = alg_name
                    best_optimal_rate = stats['optimal_rate']
                    best_mean_ratio = stats['ratio_mean']

            # Sort by optimal rate (descending), then by mean ratio (ascending)
            alg_performances.sort(key=lambda x: (-x[1], x[2]))

            for alg_name, optimal_rate, mean_ratio, success_rate in alg_performances:
                marker = "🏆" if alg_name == best_alg else "  "
                summary_lines.append(f"  {marker} {alg_name}:")
                summary_lines.append(f"     Optimal solutions: {
                                     optimal_rate:.1%}")
                summary_lines.append(f"     Avg. ratio: {mean_ratio:.4f}")
                summary_lines.append(f"     Success rate: {success_rate:.1%}")

        # Divide and conquer comparison summary
        if dc_comparison:
            summary_lines.append(
                f"\n⚔️  DIVIDE & CONQUER ALGORITHM COMPARISON")
            summary_lines.append("-" * 40)

            for n_points, comparison in dc_comparison.items():
                if n_points == 'unknown':
                    continue

                summary_lines.append(
                    f"\n🔸 {n_points} Points ({comparison['valid_cases']} head-to-head cases):")

                algorithms = comparison['algorithms']
                if len(algorithms) == 2:
                    alg1, alg2 = algorithms
                    comp_key = f"{alg1}_vs_{alg2}"
                    if comp_key in comparison['comparisons']:
                        stats = comparison['comparisons'][comp_key]

                        # Determine winner
                        alg1_rate = stats[f'{alg1}_win_rate']
                        alg2_rate = stats[f'{alg2}_win_rate']

                        if alg1_rate > alg2_rate:
                            winner = alg1
                            win_rate = alg1_rate
                        elif alg2_rate > alg1_rate:
                            winner = alg2
                            win_rate = alg2_rate
                        else:
                            winner = "TIE"
                            win_rate = max(alg1_rate, alg2_rate)

                        summary_lines.append(
                            f"  🏆 Winner: {winner} ({win_rate:.1%} win rate)")
                        summary_lines.append(
                            f"     {alg1}: {alg1_rate:.1%} wins")
                        summary_lines.append(
                            f"     {alg2}: {alg2_rate:.1%} wins")
                        summary_lines.append(
                            f"     Ties: {stats['tie_rate']:.1%}")

        # Overall recommendations
        summary_lines.append(f"\n💡 RECOMMENDATIONS")
        summary_lines.append("-" * 40)

        # Find overall best algorithm across all point counts
        overall_performance = defaultdict(list)
        for n_points, analysis in approximation_analysis.items():
            if n_points == 'unknown':
                continue
            for alg_name, stats in analysis.items():
                if alg_name in ['n_points', 'total_cases']:
                    continue
                overall_performance[alg_name].append(stats['optimal_rate'])

        # Calculate average optimal rates
        avg_optimal_rates = {}
        for alg_name, rates in overall_performance.items():
            avg_optimal_rates[alg_name] = np.mean(rates)

        best_overall = max(avg_optimal_rates.items(), key=lambda x: x[1])
        # summary_lines.append(f"🌟 Best overall algorithm: {best_overall[0]} ({
        #                      best_overall[1]:.1%} avg optimal rate)")
        summary_lines.append("\n" + "=" * 80)

        return "\n".join(summary_lines)

    def save_results(self, filename: str, cases: Optional[List[WienerComparisonCase]] = None):
        """Save comparison results to file."""
        if cases is None:
            cases = self.comparison_cases

        # Convert to serializable format
        serializable_cases = []
        for case in cases:
            case_data = {
                'case_id': case.case_id,
                'points': [p.to_dict() for p in case.points],
                'results': {name: result.to_dict() for name, result in case.results.items()},
                'metadata': case.metadata
            }
            serializable_cases.append(case_data)

        try:
            import json
            with open(filename, 'w') as f:
                json.dump(serializable_cases, f, indent=2)
            self.logger.info(f"Saved {len(cases)} cases to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")


# Factory function for Wiener index comparison
def create_wiener_index_analyzer(point_generator: PointGenerator,
                                 data_manager: Optional[DataManager] = None,
                                 visualizer: Optional[Visualizer] = None,
                                 logger: Optional[logging.Logger] = None) -> WienerIndexComparisonAnalyzer:
    """Create a specialized Wiener index comparison analyzer."""
    return WienerIndexComparisonAnalyzer(
        point_generator=point_generator,
        data_manager=data_manager,
        visualizer=visualizer,
        logger=logger
    )


def register_wiener_algorithms(analyzer: WienerIndexComparisonAnalyzer,
                               brute_force_func: Callable[[List[Point]], List[Point]],
                               divide_conquer_func: Callable[[List[Point]], List[Point]],
                               divide_conquer_alt_func: Callable[[List[Point]], List[Point]],
                               algorithm_names: Optional[Dict[str, str]] = None):
    """
    Register Wiener index algorithms with the analyzer.

    Args:
        analyzer: The comparison analyzer
        brute_force_func: Brute force algorithm function
        divide_conquer_func: First divide and conquer algorithm
        divide_conquer_alt_func: Second divide and conquer algorithm with different partition strategy
        algorithm_names: Optional custom names for algorithms
    """

    if algorithm_names is None:
        algorithm_names = {
            'brute_force': 'Brute Force',
            'divide_conquer': 'Divide & Conquer (Standard)',
            'divide_conquer_alt': 'Divide & Conquer (Alternative)'
        }

    # Register brute force (exact algorithm)
    bf_algorithm = WienerIndexAlgorithmAdapter(
        algorithm_func=brute_force_func,
        name=algorithm_names['brute_force'],
        is_exact=True,
        max_size=8  # Reasonable limit for brute force
    )
    analyzer.register_algorithm(bf_algorithm)

    # Register first divide and conquer
    dc_algorithm = WienerIndexAlgorithmAdapter(
        algorithm_func=divide_conquer_func,
        name=algorithm_names['divide_conquer'],
        is_exact=False
    )
    analyzer.register_algorithm(dc_algorithm)

    # Register second divide and conquer with alternative partition strategy
    dc_alt_algorithm = WienerIndexAlgorithmAdapter(
        algorithm_func=divide_conquer_alt_func,
        name=algorithm_names['divide_conquer_alt'],
        is_exact=False
    )
    analyzer.register_algorithm(dc_alt_algorithm)
    analyzer.register_algorithm(dc_alt_algorithm)
