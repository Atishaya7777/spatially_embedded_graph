import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from multiprocessing import cpu_count

from core.point import Point
from core.wiener_index_calculator import WienerIndexCalculator


@dataclass
class AlgorithmResult:
    """Container for single algorithm result."""
    algorithm_name: str
    path: List[Point]
    wiener_index: float
    execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'algorithm_name': self.algorithm_name,
            'path': [p.to_dict() for p in self.path],
            'wiener_index': self.wiener_index,
            'execution_time': self.execution_time
        }


@dataclass
class MultiAlgorithmExperiment:
    """Container for multi-algorithm experiment results."""
    n_points: int
    seed: int
    points: List[Point]
    results: Dict[str, AlgorithmResult] = field(default_factory=dict)
    
    def add_result(self, algorithm_name: str, result: AlgorithmResult) -> None:
        """Add result for an algorithm."""
        self.results[algorithm_name] = result
    
    def get_approximation_ratio(self, algorithm1: str, algorithm2: str) -> Optional[float]:
        """Calculate approximation ratio between two algorithms."""
        if algorithm1 in self.results and algorithm2 in self.results:
            wiener1 = self.results[algorithm1].wiener_index
            wiener2 = self.results[algorithm2].wiener_index
            return wiener1 / wiener2 if wiener2 > 0 else None
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'n_points': self.n_points,
            'seed': self.seed,
            'points': [p.to_dict() for p in self.points],
            'results': {name: result.to_dict() for name, result in self.results.items()}
        }


@dataclass
class AlgorithmStatistics:
    """Statistical summary for a single algorithm."""
    algorithm_name: str
    num_experiments: int
    wiener_index_mean: float
    wiener_index_std: float
    wiener_index_min: float
    wiener_index_max: float
    execution_time_mean: float
    execution_time_std: float
    execution_time_min: float
    execution_time_max: float


@dataclass
class StudyResults:
    """Container for comprehensive study results."""
    experiments: List[MultiAlgorithmExperiment]
    statistics: Dict[str, AlgorithmStatistics]
    
    def get_experiments_with_algorithm(self, algorithm_name: str) -> List[MultiAlgorithmExperiment]:
        """Get all experiments that include results for the specified algorithm."""
        return [exp for exp in self.experiments if algorithm_name in exp.results]


# Legacy data classes for backward compatibility
@dataclass
class ExperimentResult:
    """Container for single experiment results (legacy)."""
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
    """Container for statistical summary of multiple experiments (legacy)."""
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
        self.wiener_calculator = WienerIndexCalculator()

    def run_single_experiment(self, points: List[Point], solver, 
                             algorithm_name: str = "unknown",
                             solver_kwargs: Optional[Dict[str, Any]] = None) -> AlgorithmResult:
        """Run a single algorithm on a point set."""
        solver_kwargs = solver_kwargs or {}
        
        start_time = time.time()
        path = solver.solve(points, **solver_kwargs)
        execution_time = time.time() - start_time
        
        wiener_index = self.wiener_calculator.calculate_wiener_index(path)
        
        return AlgorithmResult(
            algorithm_name=algorithm_name,
            path=path,
            wiener_index=wiener_index,
            execution_time=execution_time
        )
    
    def run_multi_algorithm_experiment(self, points: List[Point], 
                                     solvers: List[Any],
                                     solver_names: List[str],
                                     solver_kwargs_list: Optional[List[Dict[str, Any]]] = None,
                                     seed: Optional[int] = None) -> MultiAlgorithmExperiment:
        """Run multiple algorithms on the same point set."""
        if solver_kwargs_list is None:
            solver_kwargs_list = [{} for _ in solvers]
        
        experiment = MultiAlgorithmExperiment(
            n_points=len(points),
            seed=seed or 0,
            points=points
        )
        
        for solver, name, kwargs in zip(solvers, solver_names, solver_kwargs_list):
            try:
                result = self.run_single_experiment(points, solver, name, kwargs)
                experiment.add_result(name, result)
                self.logger.debug(f"Algorithm {name}: Wiener={result.wiener_index:.4f}, Time={result.execution_time:.4f}s")
            except Exception as e:
                self.logger.warning(f"Algorithm {name} failed: {str(e)}")
        
        return experiment
    
    def run_comparison_study(self, point_sets: List[List[Point]],
                           solvers: List[Any],
                           solver_names: List[str],
                           solver_kwargs_list: Optional[List[Dict[str, Any]]] = None) -> StudyResults:
        """Run comparison study across multiple point sets."""
        experiments = []
        
        for i, points in enumerate(point_sets):
            experiment = self.run_multi_algorithm_experiment(
                points, solvers, solver_names, solver_kwargs_list, seed=i
            )
            experiments.append(experiment)
            
            if (i + 1) % 20 == 0:
                self.logger.info(f"Completed {i + 1}/{len(point_sets)} experiments...")
        
        # Calculate statistics
        statistics = self._calculate_statistics(experiments, solver_names)
        
        return StudyResults(experiments=experiments, statistics=statistics)
    
    def run_comprehensive_study(self, point_sets: List[List[Point]],
                              solvers: List[Any],
                              solver_names: List[str],
                              max_brute_force_size: int = 10) -> StudyResults:
        """Run comprehensive study with conditional brute force."""
        experiments = []
        
        for i, points in enumerate(point_sets):
            # Determine which solvers to use based on point set size
            active_solvers = []
            active_names = []
            active_kwargs = []
            
            for solver, name in zip(solvers, solver_names):
                if name == 'brute_force' and len(points) > max_brute_force_size:
                    continue  # Skip brute force for large point sets
                active_solvers.append(solver)
                active_names.append(name)
                active_kwargs.append({})
            
            experiment = self.run_multi_algorithm_experiment(
                points, active_solvers, active_names, active_kwargs, seed=i
            )
            experiments.append(experiment)
            
            if (i + 1) % 20 == 0:
                self.logger.info(f"Completed {i + 1}/{len(point_sets)} experiments...")
        
        # Calculate statistics
        statistics = self._calculate_statistics(experiments, solver_names)
        
        return StudyResults(experiments=experiments, statistics=statistics)
    
    def _calculate_statistics(self, experiments: List[MultiAlgorithmExperiment], 
                            algorithm_names: List[str]) -> Dict[str, AlgorithmStatistics]:
        """Calculate statistics for each algorithm."""
        statistics = {}
        
        for algorithm_name in algorithm_names:
            # Get all results for this algorithm
            results = []
            for exp in experiments:
                if algorithm_name in exp.results:
                    results.append(exp.results[algorithm_name])
            
            if not results:
                continue
            
            # Calculate statistics
            wiener_indices = [r.wiener_index for r in results]
            execution_times = [r.execution_time for r in results]
            
            statistics[algorithm_name] = AlgorithmStatistics(
                algorithm_name=algorithm_name,
                num_experiments=len(results),
                wiener_index_mean=np.mean(wiener_indices),
                wiener_index_std=np.std(wiener_indices),
                wiener_index_min=np.min(wiener_indices),
                wiener_index_max=np.max(wiener_indices),
                execution_time_mean=np.mean(execution_times),
                execution_time_std=np.std(execution_times),
                execution_time_min=np.min(execution_times),
                execution_time_max=np.max(execution_times)
            )
        
        return statistics

    # Legacy methods for backward compatibility
    def run_single_experiment_legacy(self, n_points: int, seed: int, is_convex: bool = False) -> ExperimentResult:
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

    def analyze_multi_algorithm_study(self, experiments: List[MultiAlgorithmExperiment]) -> StudyResults:
        """
        Analyze results from a multi-algorithm study and generate comprehensive statistics.
        
        Args:
            experiments: List of MultiAlgorithmExperiment objects
            
        Returns:
            StudyResults with comprehensive analysis
        """
        if not experiments:
            raise ValueError("No experiments provided for analysis")
        
        # Get all algorithm names from experiments
        all_algorithms = set()
        for exp in experiments:
            all_algorithms.update(exp.results.keys())
        
        algorithm_names = list(all_algorithms)
        
        # Calculate statistics using existing method
        statistics = self._calculate_statistics(experiments, algorithm_names)
        
        return StudyResults(experiments=experiments, statistics=statistics)
