#!/usr/bin/env python3
"""
Wiener Index Analysis Framework - Standalone Main Entry Point

A comprehensive implementation for analyzing Wiener indices in Hamiltonian cycles
on spatially embedded graphs. This standalone file includes all necessary
components for point generation, algorithm solving, and analysis.

Author: Research Team
Date: July 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import random
import time
import logging
import itertools
import argparse
import sys
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from pathlib import Path
import json
import os
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class Point:
    """Represents a 2D point with x, y coordinates."""
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point."""
        return euclidean((self.x, self.y), (other.x, other.y))
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9


@dataclass
class AlgorithmResult:
    """Stores the result of running an algorithm."""
    path: List[Point]
    wiener_index: float
    execution_time: float
    algorithm_name: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExperimentResult:
    """Stores results from a single experiment with multiple algorithms."""
    n_points: int
    seed: int
    points: List[Point]
    results: Dict[str, AlgorithmResult]
    point_type: str = "convex"
    
    def get_approximation_ratio(self, algorithm_name: str, baseline: str = 'brute_force') -> Optional[float]:
        """Calculate approximation ratio relative to baseline algorithm."""
        if algorithm_name not in self.results or baseline not in self.results:
            return None
            
        baseline_wiener = self.results[baseline].wiener_index
        algorithm_wiener = self.results[algorithm_name].wiener_index
        
        if baseline_wiener == 0:
            return None
            
        return algorithm_wiener / baseline_wiener


# =============================================================================
# Point Generation
# =============================================================================

class PointGenerator:
    """Generates different types of point sets for analysis."""
    
    def generate_points(self, n_points: int, point_type: str = "convex", seed: Optional[int] = None) -> List[Point]:
        """
        Generate n_points of the specified type.
        
        Args:
            n_points: Number of points to generate
            point_type: Type of points ('convex', 'general', 'grid', 'circular')
            seed: Random seed for reproducibility
            
        Returns:
            List of generated points
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if point_type == "convex":
            return self._generate_convex_points(n_points)
        elif point_type == "general":
            return self._generate_general_points(n_points)
        elif point_type == "grid":
            return self._generate_grid_points(n_points)
        elif point_type == "circular":
            return self._generate_circular_points(n_points)
        else:
            raise ValueError(f"Unknown point type: {point_type}")
    
    def _generate_convex_points(self, n_points: int) -> List[Point]:
        """Generate points in convex position."""
        angles = sorted(np.random.uniform(0, 2*np.pi, n_points))
        radius = 50 + np.random.uniform(-10, 10, n_points)
        center_x, center_y = 50, 50
        
        points = []
        for angle, r in zip(angles, radius):
            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)
            points.append(Point(x, y))
        
        return points
    
    def _generate_general_points(self, n_points: int) -> List[Point]:
        """Generate random points in general position."""
        points = []
        for _ in range(n_points):
            x = np.random.uniform(0, 100)
            y = np.random.uniform(0, 100)
            points.append(Point(x, y))
        return points
    
    def _generate_grid_points(self, n_points: int) -> List[Point]:
        """Generate points on a grid."""
        side = int(np.ceil(np.sqrt(n_points)))
        points = []
        
        for i in range(min(n_points, side * side)):
            x = (i % side) * 10 + np.random.uniform(-1, 1)
            y = (i // side) * 10 + np.random.uniform(-1, 1)
            points.append(Point(x, y))
        
        return points[:n_points]
    
    def _generate_circular_points(self, n_points: int) -> List[Point]:
        """Generate points arranged in a circle."""
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        radius = 50
        center_x, center_y = 50, 50
        
        points = []
        for angle in angles:
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            points.append(Point(x, y))
        
        return points


# =============================================================================
# Wiener Index Calculator
# =============================================================================

class WienerIndexCalculator:
    """Calculates Wiener index for a given Hamiltonian cycle."""
    
    @staticmethod
    def calculate_wiener_index(path: List[Point]) -> float:
        """
        Calculate the Wiener index for a Hamiltonian cycle.
        
        Args:
            path: List of points representing the cycle
            
        Returns:
            Wiener index value
        """
        if len(path) < 2:
            return 0.0
        
        n = len(path)
        total_wiener = 0.0
        
        # Calculate shortest path distances in the cycle
        for i in range(n):
            for j in range(i + 1, n):
                # Distance going forward in cycle
                forward_dist = WienerIndexCalculator._cycle_distance(path, i, j)
                # Distance going backward in cycle
                backward_dist = WienerIndexCalculator._cycle_distance(path, j, i)
                
                # Take minimum distance
                min_dist = min(forward_dist, backward_dist)
                total_wiener += min_dist
        
        return total_wiener
    
    @staticmethod
    def _cycle_distance(path: List[Point], start: int, end: int) -> float:
        """Calculate distance along cycle from start to end index."""
        n = len(path)
        distance = 0.0
        current = start
        
        while current != end:
            next_idx = (current + 1) % n
            distance += path[current].distance_to(path[next_idx])
            current = next_idx
        
        return distance


# =============================================================================
# Algorithm Solvers
# =============================================================================

class BruteForceSolver:
    """Brute force solver for optimal Hamiltonian cycle."""
    
    def __init__(self):
        self.calculator = WienerIndexCalculator()
    
    def solve(self, points: List[Point], use_parallel: bool = True) -> Tuple[List[Point], float]:
        """
        Find optimal Hamiltonian cycle using brute force.
        
        Args:
            points: List of points to find cycle for
            use_parallel: Whether to use parallel processing
            
        Returns:
            Tuple of (optimal_path, optimal_wiener_index)
        """
        if len(points) <= 1:
            return points, 0.0
        
        if len(points) == 2:
            wiener = self.calculator.calculate_wiener_index(points)
            return points, wiener
        
        # Fix first point to avoid rotational symmetry
        fixed_point = points[0]
        remaining_points = points[1:]
        
        if use_parallel and len(remaining_points) > 3:
            return self._parallel_solve(fixed_point, remaining_points)
        else:
            return self._sequential_solve(fixed_point, remaining_points)
    
    def _sequential_solve(self, fixed_point: Point, remaining_points: List[Point]) -> Tuple[List[Point], float]:
        """Sequential brute force solution."""
        best_path = None
        best_wiener = float('inf')
        
        for perm in itertools.permutations(remaining_points):
            path = [fixed_point] + list(perm)
            wiener = self.calculator.calculate_wiener_index(path)
            
            if wiener < best_wiener:
                best_wiener = wiener
                best_path = path
        
        return best_path, best_wiener
    
    def _parallel_solve(self, fixed_point: Point, remaining_points: List[Point]) -> Tuple[List[Point], float]:
        """Parallel brute force solution."""
        num_workers = min(cpu_count(), 8)
        
        # Generate all permutations and split into chunks
        perms = list(itertools.permutations(remaining_points))
        chunk_size = max(1, len(perms) // num_workers)
        chunks = [perms[i:i + chunk_size] for i in range(0, len(perms), chunk_size)]
        
        logging.info(f"Evaluating {len(perms)} permutations using {num_workers} workers...")
        
        start_time = time.time()
        
        # Process chunks in parallel
        with Pool(num_workers) as pool:
            args = [(fixed_point, chunk) for chunk in chunks]
            results = pool.starmap(self._evaluate_chunk, args)
        
        elapsed = time.time() - start_time
        logging.info(f"Parallel brute force completed in {elapsed:.2f} seconds")
        
        # Find best result across all chunks
        best_path, best_wiener = min(results, key=lambda x: x[1])
        return best_path, best_wiener
    
    def _evaluate_chunk(self, fixed_point: Point, chunk: List[Tuple]) -> Tuple[List[Point], float]:
        """Evaluate a chunk of permutations."""
        best_path = None
        best_wiener = float('inf')
        
        for perm in chunk:
            path = [fixed_point] + list(perm)
            wiener = self.calculator.calculate_wiener_index(path)
            
            if wiener < best_wiener:
                best_wiener = wiener
                best_path = path
        
        return best_path, best_wiener


class DivideConquerSolver:
    """Divide and conquer solver for Hamiltonian cycle approximation."""
    
    def __init__(self, max_depth: int = 10, base_case_size: int = 4):
        self.max_depth = max_depth
        self.base_case_size = base_case_size
        self.calculator = WienerIndexCalculator()
        self.brute_force_solver = BruteForceSolver()
    
    def solve_with_wiener(self, points: List[Point], use_median_bisection: bool = False) -> Tuple[List[Point], float]:
        """
        Solve using divide and conquer approach.
        
        Args:
            points: List of points to find cycle for
            use_median_bisection: Whether to use median bisection instead of bbox
            
        Returns:
            Tuple of (path, wiener_index)
        """
        if len(points) <= self.base_case_size:
            return self.brute_force_solver.solve(points, use_parallel=True)
        
        path = self._divide_conquer_recursive(points, use_median_bisection, 0)
        wiener = self.calculator.calculate_wiener_index(path)
        
        return path, wiener
    
    def _divide_conquer_recursive(self, points: List[Point], use_median: bool, depth: int) -> List[Point]:
        """Recursive divide and conquer implementation."""
        if len(points) <= self.base_case_size or depth >= self.max_depth:
            path, _ = self.brute_force_solver.solve(points, use_parallel=True)
            return path
        
        # Divide points into two groups
        if use_median:
            left_group, right_group = self._median_bisection(points)
        else:
            left_group, right_group = self._bbox_bisection(points)
        
        # Recursively solve subproblems
        left_path = self._divide_conquer_recursive(left_group, use_median, depth + 1)
        right_path = self._divide_conquer_recursive(right_group, use_median, depth + 1)
        
        # Merge solutions
        return self._merge_paths(left_path, right_path)
    
    def _bbox_bisection(self, points: List[Point]) -> Tuple[List[Point], List[Point]]:
        """Divide points using bounding box bisection."""
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        
        if x_range >= y_range:
            # Split by x-coordinate
            mid_x = (max(xs) + min(xs)) / 2
            left = [p for p in points if p.x <= mid_x]
            right = [p for p in points if p.x > mid_x]
        else:
            # Split by y-coordinate
            mid_y = (max(ys) + min(ys)) / 2
            left = [p for p in points if p.y <= mid_y]
            right = [p for p in points if p.y > mid_y]
        
        # Ensure both groups have points
        if len(left) == 0 or len(right) == 0:
            mid = len(points) // 2
            return points[:mid], points[mid:]
        
        return left, right
    
    def _median_bisection(self, points: List[Point]) -> Tuple[List[Point], List[Point]]:
        """Divide points using median bisection."""
        # Try both x and y coordinates, choose the one with more balanced split
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        
        x_median = np.median(xs)
        y_median = np.median(ys)
        
        x_left = [p for p in points if p.x <= x_median]
        x_right = [p for p in points if p.x > x_median]
        
        y_left = [p for p in points if p.y <= y_median]
        y_right = [p for p in points if p.y > y_median]
        
        # Choose split that's more balanced
        x_balance = abs(len(x_left) - len(x_right))
        y_balance = abs(len(y_left) - len(y_right))
        
        if x_balance <= y_balance:
            return x_left, x_right
        else:
            return y_left, y_right
    
    def _merge_paths(self, left_path: List[Point], right_path: List[Point]) -> List[Point]:
        """Merge two paths into a single cycle."""
        if not left_path:
            return right_path
        if not right_path:
            return left_path
        
        # Find best connection points between the two paths
        best_cost = float('inf')
        best_connection = None
        
        for i in range(len(left_path)):
            for j in range(len(right_path)):
                # Try connecting left[i] to right[j] and right[j+1] back to left[i+1]
                cost = self._connection_cost(left_path, right_path, i, j)
                if cost < best_cost:
                    best_cost = cost
                    best_connection = (i, j)
        
        if best_connection is None:
            return left_path + right_path
        
        i, j = best_connection
        
        # Build merged path
        merged = (left_path[:i+1] + 
                 right_path[j:] + 
                 right_path[:j] + 
                 left_path[i+1:])
        
        return merged
    
    def _connection_cost(self, left_path: List[Point], right_path: List[Point], i: int, j: int) -> float:
        """Calculate cost of connecting paths at given indices."""
        # Cost of connecting left[i] -> right[j] and right[j-1] -> left[i+1]
        cost = 0.0
        
        # Connection from left to right
        cost += left_path[i].distance_to(right_path[j])
        
        # Connection from right back to left
        prev_j = (j - 1) % len(right_path)
        next_i = (i + 1) % len(left_path)
        cost += right_path[prev_j].distance_to(left_path[next_i])
        
        return cost


# =============================================================================
# Analysis and Orchestration
# =============================================================================

class WienerAnalysisOrchestrator:
    """Main orchestrator for running Wiener index analysis experiments."""
    
    def __init__(self, output_dir: str = "results", log_dir: str = "logs", enable_visualization: bool = True):
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.enable_visualization = enable_visualization
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.point_generator = PointGenerator()
        self.brute_force_solver = BruteForceSolver()
        self.divide_conquer_solver = DivideConquerSolver()
        
        # Setup logging
        self.logger = self._setup_logging()
        self.logger.info("WienerAnalysisOrchestrator initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"wiener_analysis_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized. Log file: {log_file}")
        return logger
    
    def run_single_experiment(self, n_points: int, point_type: str = "convex", 
                            seed: Optional[int] = None, 
                            algorithms: Optional[List[str]] = None) -> ExperimentResult:
        """
        Run a single experiment comparing algorithms on one point set.
        
        Args:
            n_points: Number of points to generate
            point_type: Type of points ('convex', 'general', etc.)
            seed: Random seed for reproducibility
            algorithms: List of algorithms to run
            
        Returns:
            ExperimentResult containing all results
        """
        if algorithms is None:
            algorithms = ['brute_force', 'divide_conquer']
        
        self.logger.info(f"Starting experiment: n={n_points}, type={point_type}, seed={seed}")
        
        # Generate points
        points = self.point_generator.generate_points(n_points, point_type, seed)
        experiment = ExperimentResult(
            n_points=n_points, 
            seed=seed or 0, 
            points=points,
            results={},
            point_type=point_type
        )
        
        # Run each algorithm
        for algorithm in algorithms:
            try:
                start_time = time.time()
                
                if algorithm == 'brute_force':
                    path, wiener_index = self.brute_force_solver.solve(points)
                elif algorithm == 'divide_conquer':
                    path, wiener_index = self.divide_conquer_solver.solve_with_wiener(points)
                elif algorithm == 'divide_conquer_median':
                    path, wiener_index = self.divide_conquer_solver.solve_with_wiener(
                        points, use_median_bisection=True)
                else:
                    self.logger.warning(f"Unknown algorithm: {algorithm}")
                    continue
                
                execution_time = time.time() - start_time
                
                result = AlgorithmResult(
                    path=path,
                    wiener_index=wiener_index,
                    execution_time=execution_time,
                    algorithm_name=algorithm,
                    metadata={'success': True}
                )
                
                experiment.results[algorithm] = result
                self.logger.info(f"Algorithm {algorithm}: Wiener={wiener_index:.4f}, Time={execution_time:.4f}s")
                
            except Exception as e:
                self.logger.error(f"Algorithm {algorithm} failed: {e}")
                experiment.results[algorithm] = AlgorithmResult(
                    path=[],
                    wiener_index=float('inf'),
                    execution_time=0.0,
                    algorithm_name=algorithm,
                    metadata={'success': False, 'error': str(e)}
                )
        
        return experiment
    
    def run_comprehensive_study(self, point_counts: List[int], point_types: List[str], 
                              num_trials: int = 3, algorithms: Optional[List[str]] = None) -> List[ExperimentResult]:
        """
        Run a comprehensive study across multiple configurations.
        
        Args:
            point_counts: List of point counts to test
            point_types: List of point types to test
            num_trials: Number of trials per configuration
            algorithms: List of algorithms to run
            
        Returns:
            List of all experiment results
        """
        if algorithms is None:
            algorithms = ['brute_force', 'divide_conquer', 'divide_conquer_median']
        
        all_results = []
        total_experiments = len(point_counts) * len(point_types) * num_trials
        
        self.logger.info(f"Starting comprehensive study: {total_experiments} experiments")
        
        experiment_count = 0
        for n_points in point_counts:
            for point_type in point_types:
                for trial in range(num_trials):
                    experiment_count += 1
                    self.logger.info(f"Experiment {experiment_count}/{total_experiments}: "
                                   f"{n_points} points, {point_type}, trial {trial}")
                    
                    result = self.run_single_experiment(
                        n_points=n_points,
                        point_type=point_type,
                        seed=trial,
                        algorithms=algorithms
                    )
                    
                    all_results.append(result)
        
        self.logger.info(f"Comprehensive study completed: {len(all_results)} experiments")
        return all_results
    
    def save_results(self, results: List[ExperimentResult], filename: str = None) -> str:
        """Save experiment results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"wiener_analysis_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_result = {
                'n_points': result.n_points,
                'seed': result.seed,
                'point_type': result.point_type,
                'points': [(p.x, p.y) for p in result.points],
                'results': {}
            }
            
            for alg_name, alg_result in result.results.items():
                serializable_result['results'][alg_name] = {
                    'path': [(p.x, p.y) for p in alg_result.path],
                    'wiener_index': alg_result.wiener_index,
                    'execution_time': alg_result.execution_time,
                    'algorithm_name': alg_result.algorithm_name,
                    'metadata': alg_result.metadata
                }
            
            serializable_results.append(serializable_result)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
        return str(filepath)


# =============================================================================
# Visualization
# =============================================================================

class Visualizer:
    """Creates visualizations for Wiener index analysis results."""
    
    def __init__(self):
        self.colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    def plot_experiment_comparison(self, experiment: ExperimentResult, save_path: str = None) -> None:
        """Plot comparison of algorithms for a single experiment."""
        n_algorithms = len(experiment.results)
        if n_algorithms == 0:
            return
        
        fig, axes = plt.subplots(1, n_algorithms, figsize=(5*n_algorithms, 5))
        if n_algorithms == 1:
            axes = [axes]
        
        fig.suptitle(f'Algorithm Comparison: {experiment.n_points} points, {experiment.point_type} type', 
                     fontsize=14, fontweight='bold')
        
        for idx, (alg_name, result) in enumerate(experiment.results.items()):
            ax = axes[idx]
            
            # Plot points
            points = experiment.points
            x_coords = [p.x for p in points]
            y_coords = [p.y for p in points]
            
            ax.scatter(x_coords, y_coords, c='black', s=60, alpha=0.7, zorder=3)
            
            # Plot path if available
            if result.path:
                path_x = [p.x for p in result.path] + [result.path[0].x]  # Close the cycle
                path_y = [p.y for p in result.path] + [result.path[0].y]
                ax.plot(path_x, path_y, color=self.colors[idx % len(self.colors)], 
                       linewidth=2, alpha=0.8, zorder=2)
                
                # Mark start point
                ax.scatter(result.path[0].x, result.path[0].y, c='lime', s=100, 
                          marker='s', zorder=4, edgecolors='black', linewidth=1)
            
            # Set title with metrics
            title = f'{alg_name}\n'
            if result.wiener_index != float('inf'):
                title += f'Wiener: {result.wiener_index:.2f}\n'
            title += f'Time: {result.execution_time:.4f}s'
            
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_performance_summary(self, results: List[ExperimentResult], save_path: str = None) -> None:
        """Plot performance summary across all experiments."""
        if not results:
            return
        
        # Collect data for plotting
        algorithms = set()
        for result in results:
            algorithms.update(result.results.keys())
        
        algorithms = sorted(list(algorithms))
        
        # Group results by point count
        point_counts = sorted(set(r.n_points for r in results))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Wiener Index Analysis Performance Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Average Wiener Index by Point Count
        for alg in algorithms:
            avg_wieners = []
            for n in point_counts:
                wieners = [r.results[alg].wiener_index for r in results 
                          if r.n_points == n and alg in r.results 
                          and r.results[alg].wiener_index != float('inf')]
                avg_wieners.append(np.mean(wieners) if wieners else 0)
            
            ax1.plot(point_counts, avg_wieners, marker='o', label=alg, linewidth=2)
        
        ax1.set_xlabel('Number of Points')
        ax1.set_ylabel('Average Wiener Index')
        ax1.set_title('Average Wiener Index vs Point Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average Execution Time by Point Count
        for alg in algorithms:
            avg_times = []
            for n in point_counts:
                times = [r.results[alg].execution_time for r in results 
                        if r.n_points == n and alg in r.results]
                avg_times.append(np.mean(times) if times else 0)
            
            ax2.plot(point_counts, avg_times, marker='s', label=alg, linewidth=2)
        
        ax2.set_xlabel('Number of Points')
        ax2.set_ylabel('Average Execution Time (s)')
        ax2.set_title('Execution Time vs Point Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Approximation Ratios (if brute force available)
        if 'brute_force' in algorithms:
            for alg in algorithms:
                if alg == 'brute_force':
                    continue
                
                ratios_by_count = []
                for n in point_counts:
                    ratios = []
                    for r in results:
                        if (r.n_points == n and 'brute_force' in r.results and alg in r.results):
                            bf_wiener = r.results['brute_force'].wiener_index
                            alg_wiener = r.results[alg].wiener_index
                            if bf_wiener > 0 and alg_wiener != float('inf'):
                                ratios.append(alg_wiener / bf_wiener)
                    
                    ratios_by_count.append(np.mean(ratios) if ratios else 1.0)
                
                ax3.plot(point_counts, ratios_by_count, marker='^', label=alg, linewidth=2)
            
            ax3.set_xlabel('Number of Points')
            ax3.set_ylabel('Approximation Ratio')
            ax3.set_title('Approximation Ratio vs Point Count')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 4: Success Rate by Algorithm
        success_rates = {}
        for alg in algorithms:
            total = sum(1 for r in results if alg in r.results)
            successful = sum(1 for r in results if alg in r.results and 
                           r.results[alg].metadata.get('success', True))
            success_rates[alg] = (successful / total * 100) if total > 0 else 0
        
        bars = ax4.bar(success_rates.keys(), success_rates.values(), 
                      color=[self.colors[i % len(self.colors)] for i in range(len(success_rates))])
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_title('Algorithm Success Rate')
        ax4.set_ylim(0, 105)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates.values()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# =============================================================================
# Analysis and Reporting
# =============================================================================

def print_analysis_summary(results: List[ExperimentResult]) -> None:
    """Print a comprehensive analysis summary."""
    if not results:
        print("No results to analyze.")
        return
    
    print("\n" + "="*80)
    print("WIENER INDEX ANALYSIS SUMMARY")
    print("="*80)
    
    total_experiments = len(results)
    print(f"Total experiments: {total_experiments}")
    
    # Collect algorithm statistics
    algorithms = set()
    for result in results:
        algorithms.update(result.results.keys())
    
    algorithms = sorted(list(algorithms))
    
    print(f"Algorithms tested: {', '.join(algorithms)}")
    
    # Statistics by algorithm
    print("\nAlgorithm Performance:")
    print("-" * 60)
    
    for alg in algorithms:
        alg_results = []
        execution_times = []
        wiener_indices = []
        success_count = 0
        
        for result in results:
            if alg in result.results:
                alg_result = result.results[alg]
                alg_results.append(alg_result)
                
                if alg_result.metadata.get('success', True):
                    success_count += 1
                    execution_times.append(alg_result.execution_time)
                    if alg_result.wiener_index != float('inf'):
                        wiener_indices.append(alg_result.wiener_index)
        
        total_count = len(alg_results)
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        avg_time = np.mean(execution_times) if execution_times else 0
        avg_wiener = np.mean(wiener_indices) if wiener_indices else 0
        
        print(f"{alg}:")
        print(f"  Success Rate: {success_rate:.1f}% ({success_count}/{total_count})")
        print(f"  Avg Execution Time: {avg_time:.4f}s")
        print(f"  Avg Wiener Index: {avg_wiener:.4f}")
        
        if wiener_indices:
            print(f"  Wiener Range: {min(wiener_indices):.4f} - {max(wiener_indices):.4f}")
        print()
    
    # Approximation ratios if brute force is available
    if 'brute_force' in algorithms:
        print("Approximation Ratios (relative to brute force):")
        print("-" * 50)
        
        for alg in algorithms:
            if alg == 'brute_force':
                continue
            
            ratios = []
            for result in results:
                if ('brute_force' in result.results and alg in result.results):
                    bf_result = result.results['brute_force']
                    alg_result = result.results[alg]
                    
                    if (bf_result.wiener_index > 0 and 
                        alg_result.wiener_index != float('inf') and
                        bf_result.metadata.get('success', True) and
                        alg_result.metadata.get('success', True)):
                        
                        ratio = alg_result.wiener_index / bf_result.wiener_index
                        ratios.append(ratio)
            
            if ratios:
                avg_ratio = np.mean(ratios)
                std_ratio = np.std(ratios)
                min_ratio = min(ratios)
                max_ratio = max(ratios)
                
                print(f"{alg}:")
                print(f"  Average ratio: {avg_ratio:.4f} ± {std_ratio:.4f}")
                print(f"  Range: {min_ratio:.4f} - {max_ratio:.4f}")
                print(f"  Optimal solutions: {sum(1 for r in ratios if r <= 1.001)} / {len(ratios)}")
                print()
    
    # Point type analysis
    point_types = sorted(set(r.point_type for r in results))
    if len(point_types) > 1:
        print("Performance by Point Type:")
        print("-" * 40)
        
        for pt in point_types:
            pt_results = [r for r in results if r.point_type == pt]
            print(f"{pt}: {len(pt_results)} experiments")
    
    print("\n" + "="*80)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main function demonstrating the Wiener Index Analysis framework."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Wiener Index Analysis Framework')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run a quick test with minimal parameters')
    parser.add_argument('--point-counts', nargs='+', type=int, default=[4, 5, 6, 7],
                       help='Point counts to test (default: 4 5 6 7)')
    parser.add_argument('--trials', type=int, default=2,
                       help='Number of trials per configuration (default: 2)')
    parser.add_argument('--output-dir', type=str, default='wiener_results',
                       help='Output directory for results (default: wiener_results)')
    
    args = parser.parse_args()
    
    print("🔬 Wiener Index Analysis Framework")
    print("=" * 50)
    
    # Adjust configuration based on arguments
    if args.quick_test:
        print("⚡ Running quick test mode...")
        config = {
            'point_counts': [4, 5],  # Minimal point counts for quick test
            'point_types': ['convex'],  # Only convex for quick test
            'num_trials': 1,  # Single trial for quick test
            'algorithms': ['brute_force', 'divide_conquer']  # Essential algorithms only
        }
    else:
        config = {
            'point_counts': args.point_counts,
            'point_types': ['convex', 'general'],
            'num_trials': args.trials,
            'algorithms': ['brute_force', 'divide_conquer', 'divide_conquer_median']
        }
    
    # Initialize orchestrator
    orchestrator = WienerAnalysisOrchestrator(
        output_dir=args.output_dir,
        log_dir="wiener_logs",
        enable_visualization=True
    )
    
    # Initialize visualizer
    visualizer = Visualizer()
    
    print(f"Configuration:")
    print(f"  Point counts: {config['point_counts']}")
    print(f"  Point types: {config['point_types']}")
    print(f"  Trials per config: {config['num_trials']}")
    print(f"  Algorithms: {config['algorithms']}")
    print()
    
    try:
        # Run comprehensive study
        results = orchestrator.run_comprehensive_study(
            point_counts=config['point_counts'],
            point_types=config['point_types'],
            num_trials=config['num_trials'],
            algorithms=config['algorithms']
        )
        
        # Save results
        results_file = orchestrator.save_results(results)
        
        # Print analysis summary
        print_analysis_summary(results)
        
        # Create visualizations
        print("📊 Generating visualizations...")
        
        # Plot performance summary
        summary_plot_path = orchestrator.output_dir / "performance_summary.png"
        visualizer.plot_performance_summary(results, str(summary_plot_path))
        
        # Plot a few example experiments
        example_results = results[:min(3, len(results))]
        for i, result in enumerate(example_results):
            example_plot_path = orchestrator.output_dir / f"experiment_example_{i+1}.png"
            visualizer.plot_experiment_comparison(result, str(example_plot_path))
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"   Results saved to: {results_file}")
        print(f"   Visualizations saved to: {orchestrator.output_dir}")
        
        if args.quick_test:
            print(f"   Quick test mode: reduced scope for faster execution")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
