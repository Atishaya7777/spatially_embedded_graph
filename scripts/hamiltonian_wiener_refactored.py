"""
Hamiltonian Wiener Index Analysis - Main orchestration module.

This module provides the main entry point for running Wiener index analysis
using the modular components from the project.
"""

from typing import List, Dict, Any, Optional
import logging

# Import modular components
from core.point import Point
from generators.point_generator import PointGenerator
from solvers.brute_force_solver import BruteForceSolver
from solvers.divide_conquer_solver import DivideConquerSolver
from analysis.statistical_analyzer import StatisticalAnalyzer, ExperimentResult
from visualization.visualizer import Visualizer
from utils.logger_setup import setup_logger
from utils.data_manager import DataManager


class WienerAnalysisOrchestrator:
    """Main orchestrator for Wiener index analysis experiments."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the orchestrator with all necessary components."""
        self.logger = logger or setup_logger("wiener_analysis", level=logging.INFO)
        
        # Initialize core components
        self.point_generator = PointGenerator()
        self.data_manager = DataManager()
        self.visualizer = Visualizer()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Initialize solvers
        self.brute_force_solver = BruteForceSolver()
        self.divide_conquer_solver = DivideConquerSolver()
        
        self.logger.info("WienerAnalysisOrchestrator initialized")
    
    def compare_algorithms(self, points: List[Point], use_parallel: bool = True, 
                          max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Compare divide-and-conquer with brute force (if feasible)."""
        self.logger.info(f"Analyzing {len(points)} points...")
        
        results = {}
        
        # Run divide and conquer
        self.logger.info("Running Divide & Conquer algorithm...")
        dc_result = self.statistical_analyzer.run_single_experiment(
            points, self.divide_conquer_solver, algorithm_name="divide_conquer"
        )
        results['divide_conquer'] = dc_result
        
        # Run median divide and conquer
        self.logger.info("Running median Divide & Conquer algorithm...")
        median_dc_result = self.statistical_analyzer.run_single_experiment(
            points, self.divide_conquer_solver, algorithm_name="median_divide_conquer",
            solver_kwargs={'use_median_bisection': True}
        )
        results['median_divide_conquer'] = median_dc_result
        
        # Run brute force if feasible
        if len(points) <= 10:
            try:
                self.logger.info("Running Brute Force algorithm...")
                bf_result = self.statistical_analyzer.run_single_experiment(
                    points, self.brute_force_solver, algorithm_name="brute_force",
                    solver_kwargs={'use_parallel': use_parallel, 'max_workers': max_workers}
                )
                results['brute_force'] = bf_result
                
                # Calculate approximation ratios
                dc_ratio = dc_result.wiener_index / bf_result.wiener_index
                median_dc_ratio = median_dc_result.wiener_index / bf_result.wiener_index
                
                self.logger.info(f"Approximation Ratio for normal divide and conquer: {dc_ratio:.4f}")
                self.logger.info(f"Approximation Ratio for median divide and conquer: {median_dc_ratio:.4f}")
                
                # Visualize comparison
                self.visualizer.visualize_comparison(
                    points, dc_result.path, bf_result.path,
                    dc_result.wiener_index, bf_result.wiener_index, dc_ratio
                )
                
            except MemoryError:
                self.logger.warning("Brute force skipped (memory limit exceeded)")
        else:
            self.logger.info("Brute force skipped (too many points for exhaustive search)")
            # Visualize only divide and conquer
            self.visualizer.visualize_path(points, dc_result.path, dc_result.wiener_index)
        
        return results
    
    def run_comprehensive_study(self, is_convex: bool = False, 
                               point_counts: List[int] = [6, 7, 8, 9, 10], 
                               num_seeds: int = 100,
                               visualize_interesting: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive statistical analysis across multiple point counts and seeds.
        """
        self.logger.info("=== Comprehensive Statistical Analysis ===")
        self.logger.info(f"Testing point counts: {point_counts}")
        self.logger.info(f"Seeds per test: {num_seeds}")
        
        all_results = {}
        all_experiments = []
        
        for n_points in point_counts:
            self.logger.info(f"{'='*50}")
            self.logger.info(f"Testing {n_points} points across {num_seeds} seeds...")
            self.logger.info(f"{'='*50}")
            
            # Generate point sets for all seeds
            point_sets = []
            for seed in range(num_seeds):
                if is_convex:
                    points = self.point_generator.generate_convex_hull_points(n_points, seed=seed)
                else:
                    points = self.point_generator.generate_random_points(n_points, seed=seed)
                point_sets.append(points)
            
            # Run statistical analysis
            study_results = self.statistical_analyzer.run_comprehensive_study(
                point_sets=point_sets,
                solvers=[self.divide_conquer_solver, self.brute_force_solver],
                solver_names=['divide_conquer', 'brute_force'],
                max_brute_force_size=10
            )
            
            all_results[n_points] = study_results
            all_experiments.extend(study_results.experiments)
            
            # Log statistics
            self._log_study_statistics(n_points, study_results)
        
        # Save results
        self.data_manager.save_experiment_results(all_experiments, "comprehensive_study")
        
        # Find and visualize interesting cases
        if visualize_interesting:
            interesting_cases = self._find_interesting_cases(all_results)
            self._visualize_interesting_cases(interesting_cases)
        
        self.logger.info("Comprehensive study completed.")
        return all_results
    
    def run_targeted_study(self, is_convex: bool = True, 
                          points_counts: List[int] = list(range(6, 11)), 
                          num_seeds: int = 100,
                          visualize_interesting: bool = True) -> Dict[str, Any]:
        """
        Run targeted study for divide-and-conquer performance analysis.
        """
        self.logger.info("=== Testing Divide-and-Conquer Wiener Index Minimization ===")
        self.logger.info(f"Point counts: {list(points_counts)}")
        self.logger.info(f"Seeds per test: {num_seeds}")
        
        all_results = {}
        
        for n_points in points_counts:
            self.logger.info(f"\nTesting {n_points} points across {num_seeds} seeds...")
            
            # Generate point sets
            point_sets = []
            for seed in range(num_seeds):
                if is_convex:
                    points = self.point_generator.generate_convex_hull_points(n_points, seed=seed)
                else:
                    points = self.point_generator.generate_random_points(n_points, seed=seed)
                point_sets.append(points)
            
            # Run analysis comparing both divide and conquer variants
            solvers = [self.divide_conquer_solver, self.divide_conquer_solver]
            solver_names = ['divide_conquer', 'median_divide_conquer']
            solver_kwargs_list = [
                {},  # Normal divide and conquer
                {'use_median_bisection': True}  # Median divide and conquer
            ]
            
            # Add brute force for smaller point sets
            if n_points <= 8:
                solvers.append(self.brute_force_solver)
                solver_names.append('brute_force')
                solver_kwargs_list.append({})
            
            study_results = self.statistical_analyzer.run_comparison_study(
                point_sets=point_sets,
                solvers=solvers,
                solver_names=solver_names,
                solver_kwargs_list=solver_kwargs_list
            )
            
            all_results[n_points] = study_results
            self._log_targeted_study_statistics(n_points, study_results, num_seeds)
        
        # Visualize interesting cases if requested
        if visualize_interesting:
            self._visualize_targeted_cases(all_results, points_counts, num_seeds)
        
        self.logger.info("\n=== Test completed ===")
        return all_results
    
    def _log_study_statistics(self, n_points: int, study_results) -> None:
        """Log statistics for comprehensive study."""
        stats = study_results.statistics
        
        self.logger.info(f"Statistics for {n_points} points:")
        if 'brute_force' in stats:
            dc_stats = stats['divide_conquer']
            bf_stats = stats['brute_force']
            
            # Calculate approximation ratios
            ratios = [exp.get_approximation_ratio('divide_conquer', 'brute_force') 
                     for exp in study_results.experiments 
                     if exp.get_approximation_ratio('divide_conquer', 'brute_force') is not None]
            
            if ratios:
                import numpy as np
                perfect_count = sum(1 for r in ratios if r <= 1.001)
                good_count = sum(1 for r in ratios if r <= 1.1)
                poor_count = sum(1 for r in ratios if r > 1.5)
                
                self.logger.info(f"  Approximation Ratio: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
                self.logger.info(f"  Range: [{np.min(ratios):.4f}, {np.max(ratios):.4f}]")
                self.logger.info(f"  Perfect solutions (≤1.001): {perfect_count}/{len(ratios)} ({100*perfect_count/len(ratios):.1f}%)")
                self.logger.info(f"  Good solutions (≤1.1): {good_count}/{len(ratios)} ({100*good_count/len(ratios):.1f}%)")
                self.logger.info(f"  Poor solutions (>1.5): {poor_count}/{len(ratios)} ({100*poor_count/len(ratios):.1f}%)")
            
            self.logger.info(f"  Avg D&C time: {dc_stats['execution_time_mean']:.4f}s")
            self.logger.info(f"  Avg Optimal time: {bf_stats['execution_time_mean']:.4f}s")
            self.logger.info(f"  Speedup factor: {bf_stats['execution_time_mean']/dc_stats['execution_time_mean']:.1f}x")
        else:
            dc_stats = stats['divide_conquer']
            self.logger.info(f"  D&C Wiener indices: {dc_stats['wiener_index_mean']:.4f} ± {dc_stats['wiener_index_std']:.4f}")
            self.logger.info(f"  Avg D&C time: {dc_stats['execution_time_mean']:.4f}s")
    
    def _log_targeted_study_statistics(self, n_points: int, study_results, num_seeds: int) -> None:
        """Log statistics for targeted study."""
        stats = study_results.statistics
        
        print(f"  Summary for {n_points} points:")
        
        if 'brute_force' in stats:
            # Calculate approximation ratios
            ratios = [exp.get_approximation_ratio('divide_conquer', 'brute_force') 
                     for exp in study_results.experiments 
                     if exp.get_approximation_ratio('divide_conquer', 'brute_force') is not None]
            
            if ratios:
                import numpy as np
                perfect_count = sum(1 for r in ratios if r <= 1.001)
                good_count = sum(1 for r in ratios if r <= 1.1)
                poor_count = sum(1 for r in ratios if r > 1.5)
                
                print(f"    Approximation ratio: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
                print(f"    Range: [{np.min(ratios):.4f}, {np.max(ratios):.4f}]")
                print(f"    Perfect solutions (≤1.001): {perfect_count}/{num_seeds} ({100*perfect_count/num_seeds:.1f}%)")
                print(f"    Good solutions (≤1.1): {good_count}/{num_seeds} ({100*good_count/num_seeds:.1f}%)")
                print(f"    Poor solutions (>1.5): {poor_count}/{num_seeds} ({100*poor_count/num_seeds:.1f}%)")
                print(f"    Avg D&C time: {stats['divide_conquer']['execution_time_mean']:.4f}s")
                print(f"    Avg Optimal time: {stats['brute_force']['execution_time_mean']:.4f}s")
                print(f"    Speedup factor: {stats['brute_force']['execution_time_mean']/stats['divide_conquer']['execution_time_mean']:.1f}x")
        else:
            dc_stats = stats['divide_conquer']
            print(f"  D&C Wiener indices: {dc_stats['wiener_index_mean']:.4f} ± {dc_stats['wiener_index_std']:.4f}")
            print(f"  Avg D&C time: {dc_stats['execution_time_mean']:.4f}s")
    
    def _find_interesting_cases(self, all_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find interesting cases from comprehensive study results."""
        interesting_cases = []
        
        for n_points, study_results in all_results.items():
            if 'brute_force' not in study_results.statistics:
                continue
                
            # Calculate ratios for all experiments
            experiments_with_ratios = []
            for exp in study_results.experiments:
                ratio = exp.get_approximation_ratio('divide_conquer', 'brute_force')
                if ratio is not None:
                    experiments_with_ratios.append((exp, ratio))
            
            if not experiments_with_ratios:
                continue
            
            # Sort by ratio
            experiments_with_ratios.sort(key=lambda x: x[1])
            
            # Find best, worst, and median cases
            n_cases = len(experiments_with_ratios)
            best_case = experiments_with_ratios[0]
            worst_case = experiments_with_ratios[-1]
            median_idx = n_cases // 2
            median_case = experiments_with_ratios[median_idx]
            
            for case_type, (exp, ratio) in [('best', best_case), ('worst', worst_case), ('median', median_case)]:
                interesting_cases.append({
                    'n_points': n_points,
                    'case_type': case_type,
                    'seed': exp.seed,
                    'points': exp.points,
                    'dc_path': exp.results['divide_conquer'].path,
                    'optimal_path': exp.results['brute_force'].path,
                    'dc_wiener': exp.results['divide_conquer'].wiener_index,
                    'optimal_wiener': exp.results['brute_force'].wiener_index,
                    'ratio': ratio
                })
        
        return interesting_cases
    
    def _visualize_interesting_cases(self, interesting_cases: List[Dict[str, Any]]) -> None:
        """Visualize interesting cases."""
        self.logger.info(f"Visualizing {len(interesting_cases)} interesting cases...")
        
        # Save interesting cases
        json_file, pickle_file = self.data_manager.save_interesting_cases(interesting_cases)
        self.logger.info(f"Interesting cases saved to: {json_file} and {pickle_file}")
        
        # Group cases by point count and type
        grouped_cases = {}
        for case in interesting_cases:
            n_points = case['n_points']
            case_type = case['case_type']
            
            if n_points not in grouped_cases:
                grouped_cases[n_points] = {}
            if case_type not in grouped_cases[n_points]:
                grouped_cases[n_points][case_type] = []
            
            grouped_cases[n_points][case_type].append(case)
        
        # Visualize representative cases
        count = 0
        max_cases = 10
        for n_points in sorted(grouped_cases.keys()):
            for case_type in ['best', 'worst', 'median']:
                if case_type in grouped_cases[n_points] and count < max_cases:
                    case = grouped_cases[n_points][case_type][0]
                    
                    self.logger.info(f"\n{case_type.title()} case for {n_points} points (seed {case['seed']}): "
                          f"Ratio = {case['ratio']:.4f}")
                    
                    self.visualizer.visualize_comparison(
                        case['points'],
                        case['dc_path'],
                        case['optimal_path'],
                        case['dc_wiener'],
                        case['optimal_wiener'],
                        case['ratio']
                    )
                    count += 1
    
    def _visualize_targeted_cases(self, all_results: Dict[str, Any], 
                                 points_counts: List[int], num_seeds: int) -> None:
        """Visualize cases from targeted study."""
        print(f"\nVisualizing best, median, and worst cases for each point count...")
        
        for n_points in points_counts:
            if n_points not in all_results:
                continue
                
            study_results = all_results[n_points]
            
            # Get experiments with brute force results
            experiments_with_ratios = []
            for exp in study_results.experiments:
                if 'brute_force' in exp.results:
                    ratio = exp.get_approximation_ratio('divide_conquer', 'brute_force')
                    if ratio is not None:
                        experiments_with_ratios.append((exp, ratio))
            
            if not experiments_with_ratios:
                print(f"Skipping visualization for {n_points} points (no optimal solutions computed)")
                continue
            
            print(f"\n{'='*60}")
            print(f"VISUALIZING {n_points} POINTS")
            print(f"{'='*60}")
            
            # Sort by ratio
            experiments_with_ratios.sort(key=lambda x: x[1])
            
            # Get best, median, and worst cases
            n_cases = len(experiments_with_ratios)
            best_cases = experiments_with_ratios[:min(3, n_cases)]
            worst_cases = experiments_with_ratios[max(0, n_cases-3):]
            
            median_idx = n_cases // 2
            median_start = max(0, median_idx - 1)
            median_end = min(n_cases, median_idx + 2)
            median_cases = experiments_with_ratios[median_start:median_end]
            
            # Visualize each category
            categories = [
                ("BEST", best_cases),
                ("MEDIAN", median_cases),
                ("WORST", worst_cases)
            ]
            
            for category_name, cases in categories:
                if not cases:
                    continue
                
                print(f"\n{'-'*40}")
                print(f"{category_name} CASES FOR {n_points} POINTS")
                print(f"{'-'*40}")
                
                for i, (exp, ratio) in enumerate(cases):
                    print(f"\n{category_name} Case {i+1}/{len(cases)}: Seed {exp.seed}, "
                          f"Ratio: {ratio:.4f}")
                    print(f"D&C Wiener: {exp.results['divide_conquer'].wiener_index:.4f}, "
                          f"Optimal Wiener: {exp.results['brute_force'].wiener_index:.4f}")
                    
                    self.visualizer.visualize_comparison(
                        points=exp.points,
                        dc_path=exp.results['divide_conquer'].path,
                        optimal_path=exp.results['brute_force'].path,
                        dc_wiener=exp.results['divide_conquer'].wiener_index,
                        optimal_wiener=exp.results['brute_force'].wiener_index,
                        approximation_ratio=ratio
                    )


def main():
    """Main entry point for the analysis."""
    # Set up orchestrator
    orchestrator = WienerAnalysisOrchestrator()
    
    # Run targeted study
    results = orchestrator.run_targeted_study(
        is_convex=False,
        points_counts=list(range(6, 12)),
        num_seeds=50,
        visualize_interesting=True
    )
    
    # Optionally run comprehensive study
    # comprehensive_results = orchestrator.run_comprehensive_study(
    #     is_convex=False,
    #     point_counts=[6, 7, 8, 9, 10],
    #     num_seeds=100,
    #     visualize_interesting=True
    # )
    
    orchestrator.logger.info("Analysis complete. Check the logs for details.")


if __name__ == "__main__":
    main()
