import math
import time
from itertools import permutations
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional
import logging

from core.point import Point
from core.wiener_index_calculator import WienerIndexCalculator


class BruteForceSolver:
    """Brute force solver for finding optimal Hamiltonian paths."""

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize the brute force solver.

        Args:
            max_workers: Maximum number of parallel workers. If None, uses min(cpu_count(), 8)
        """
        self.max_workers = max_workers if max_workers is not None else min(
            cpu_count(), 8)
        self.logger = logging.getLogger(__name__)

    def solve(self, points: List[Point], use_parallel: bool = True) -> Tuple[List[Point], float]:
        """
        Find optimal Hamiltonian path by brute force.

        Args:
            points: List of points to find optimal path for
            use_parallel: Whether to use parallel processing for larger point sets

        Returns:
            Tuple of (optimal_path, optimal_wiener_index)
        """
        if len(points) <= 1:
            return points, 0.0 if len(points) <= 1 else WienerIndexCalculator.calculate_wiener_index(points)

        return self._parallel_solve(points)

        # Use parallel processing for larger point sets
        # if use_parallel and len(points) >= 8:
        #     return self._parallel_solve(points)
        # else:
        #     return self._sequential_solve(points)

    def _sequential_solve(self, points: List[Point]) -> Tuple[List[Point], float]:
        """
        Sequential brute force solution.

        Args:
            points: List of points to solve

        Returns:
            Tuple of (optimal_path, optimal_wiener_index)
        """
        self.logger.info(f"Running sequential brute force on {
                         len(points)} points...")

        start_time = time.time()
        best_path = None
        best_wiener = float('inf')

        # Try all possible permutations
        for perm in permutations(points):
            path = list(perm)
            wiener = WienerIndexCalculator.calculate_wiener_index(path)
            if wiener < best_wiener:
                best_wiener = wiener
                best_path = path

        end_time = time.time()
        self.logger.info(f"Sequential brute force completed in {
                         end_time - start_time:.2f} seconds")

        return best_path, best_wiener

    def _parallel_solve(self, points: List[Point]) -> Tuple[List[Point], float]:
        """
        Parallel brute force solution.

        Args:
            points: List of points to solve

        Returns:
            Tuple of (optimal_path, optimal_wiener_index)
        """
        n = len(points)
        self.logger.info(f"Evaluating {math.factorial(n):,} permutations using {
                         self.max_workers} workers...")

        # Generate all permutation indices
        all_permutations = list(permutations(range(n)))

        # Split permutations into chunks for parallel processing
        chunk_size = max(1, len(all_permutations) //
                         (self.max_workers * 4))  # 4 chunks per worker
        chunks = [all_permutations[i:i + chunk_size]
                  for i in range(0, len(all_permutations), chunk_size)]

        # Prepare data for parallel processing
        chunk_data = [(chunk, points) for chunk in chunks]

        start_time = time.time()

        # Process chunks in parallel
        with Pool(self.max_workers) as pool:
            results = pool.map(self._evaluate_permutation_chunk, chunk_data)

        end_time = time.time()
        self.logger.info(f"Parallel brute force completed in {
                         end_time - start_time:.2f} seconds")

        # Find the best result across all chunks
        best_path, best_wiener = min(results, key=lambda x: x[1])

        return best_path, best_wiener

    @staticmethod
    def _evaluate_permutation_chunk(chunk_data) -> Tuple[List[Point], float]:
        """
        Evaluate a chunk of permutations - designed for parallel processing.

        Args:
            chunk_data: Tuple of (permutations_chunk, points)

        Returns:
            Tuple of (best_path, best_wiener_index) for this chunk
        """
        permutations_chunk, points = chunk_data
        best_wiener = float('inf')
        best_path = None

        for perm_indices in permutations_chunk:
            path = [points[i] for i in perm_indices]
            wiener = WienerIndexCalculator.calculate_wiener_index(path)
            if wiener < best_wiener:
                best_wiener = wiener
                best_path = path

        return best_path, best_wiener

    def solve_simple(self, points: List[Point], use_parallel: bool = True, 
                    max_workers: Optional[int] = None) -> List[Point]:
        """
        Find optimal Hamiltonian path by brute force - simplified interface.

        Args:
            points: List of points to find optimal path for
            use_parallel: Whether to use parallel processing for larger point sets
            max_workers: Maximum number of workers (ignored, uses instance setting)

        Returns:
            Optimal path as list of points
        """
        optimal_path, _ = self.solve(points, use_parallel)
        return optimal_path
