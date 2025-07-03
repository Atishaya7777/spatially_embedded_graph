"""
Divide and conquer solver for Wiener index minimization.
"""
from typing import List, Tuple
import logging

from core.point import Point
from core.wiener_index_calculator import WienerIndexCalculator
from .brute_force_solver import BruteForceSolver


class DivideConquerSolver:
    """Divide and conquer solver for Hamiltonian path optimization."""

    def __init__(self, max_depth: int = 10, base_case_size: int = 4):
        """
        Initialize the divide and conquer solver.

        Args:
            max_depth: Maximum recursion depth to prevent infinite recursion
            base_case_size: Size threshold for switching to brute force
        """
        self.max_depth = max_depth
        self.base_case_size = base_case_size
        self.brute_force_solver = BruteForceSolver()
        self.logger = logging.getLogger(__name__)

    def solve(self, points: List[Point], use_median_bisection: bool = False) -> List[Point]:
        """
        Solve using divide and conquer approach.

        Args:
            points: List of points to find path for
            use_median_bisection: Whether to use median-based bisection

        Returns:
            Hamiltonian path as list of points
        """
        if use_median_bisection:
            return self._solve_median(points, depth=0)
        else:
            return self._solve_bbox(points, depth=0)

    def solve_with_wiener(self, points: List[Point], use_median_bisection: bool = False) -> Tuple[List[Point], float]:
        """
        Solve using divide and conquer approach and return path with Wiener index.

        Args:
            points: List of points to find path for
            use_median_bisection: Whether to use median-based bisection

        Returns:
            Tuple of (Hamiltonian path, Wiener index)
        """
        path = self.solve(points, use_median_bisection)
        wiener_index = WienerIndexCalculator.calculate_wiener_index(path)
        return path, wiener_index

    def _solve_bbox(self, points: List[Point], depth: int = 0) -> List[Point]:
        """
        Divide and conquer using bounding box bisection.

        Args:
            points: Points to solve
            depth: Current recursion depth

        Returns:
            Hamiltonian path
        """
        # Base case: small sets solved by brute force
        if len(points) <= self.base_case_size:
            path, _ = self.brute_force_solver.solve(points, use_parallel=False)
            return path

        # Prevent infinite recursion
        if depth > self.max_depth:
            path, _ = self.brute_force_solver.solve(points, use_parallel=False)
            return path

        # Find bisecting line
        line_a, line_b, line_c = self._find_bisecting_line(points)

        # Partition points
        left_points, right_points = self._partition_points(
            points, line_a, line_b, line_c)

        # Recursively solve subproblems
        left_path = self._solve_bbox(left_points, depth + 1)
        right_path = self._solve_bbox(right_points, depth + 1)

        # Find optimal connection
        final_path, _ = self._connect_paths(
            left_path, right_path
        )

        return final_path

    def _connect_paths(self, path1: List[Point], path2: List[Point]) -> tuple[List[Point], float]:
        """
        Find the optimal way to connect two Hamiltonian paths.

        Tests all four possible connections between two paths and returns
        the one with the minimum Wiener index.

        Args:
            path1: First Hamiltonian path
            path2: Second Hamiltonian path

        Returns:
            Tuple of (best_connected_path, wiener_index)
        """
        if not path1:
            return path2, WienerIndexCalculator.calculate_wiener_index(path2)
        if not path2:
            return path1, WienerIndexCalculator.calculate_wiener_index(path1)

        # Four possible ways to connect the paths
        connections = []

        # 1. end of path1 to start of path2
        conn1 = path1 + path2
        connections.append(
            (conn1, WienerIndexCalculator.calculate_wiener_index(conn1)))

        # 2. end of path1 to end of path2 (reverse path2)
        conn2 = path1 + path2[::-1]
        connections.append(
            (conn2, WienerIndexCalculator.calculate_wiener_index(conn2)))

        # 3. start of path1 to start of path2 (reverse path1)
        conn3 = path1[::-1] + path2
        connections.append(
            (conn3, WienerIndexCalculator.calculate_wiener_index(conn3)))

        # 4. start of path1 to end of path2 (reverse both)
        conn4 = path1[::-1] + path2[::-1]
        connections.append(
            (conn4, WienerIndexCalculator.calculate_wiener_index(conn4)))

        # Return the connection with minimum Wiener index
        return min(connections, key=lambda x: x[1])

    def _solve_median(self, points: List[Point], depth: int = 0) -> List[Point]:
        """
        Divide and conquer using median-based bisection.

        Args:
            points: Points to solve
            depth: Current recursion depth

        Returns:
            Hamiltonian path
        """
        # Base case: small sets solved by brute force
        if len(points) <= self.base_case_size:
            path, _ = self.brute_force_solver.solve(points, use_parallel=False)
            return path

        # Prevent infinite recursion
        if depth > self.max_depth:
            path, _ = self.brute_force_solver.solve(points, use_parallel=False)
            return path

        # Find median-based bisecting line
        line_a, line_b, line_c = self._find_median_bisecting_line(points)

        # Partition points
        left_points, right_points = self._partition_points(
            points, line_a, line_b, line_c)

        # Recursively solve subproblems
        left_path = self._solve_median(left_points, depth + 1)
        right_path = self._solve_median(right_points, depth + 1)

        # Find optimal connection
        final_path, _ = self._connect_paths(
            left_path, right_path)

        return final_path

    def _find_bisecting_line(self, points: List[Point]) -> Tuple[float, float, float]:
        """
        Find a line that roughly bisects the point set based on bounding box.

        Args:
            points: List of points to bisect

        Returns:
            Tuple (a, b, c) such that ax + by + c = 0
        """
        if len(points) < 2:
            return 1.0, 0.0, 0.0

        # Find the bounding box
        min_x = min(p.x for p in points)
        max_x = max(p.x for p in points)
        min_y = min(p.y for p in points)
        max_y = max(p.y for p in points)

        # Choose bisection direction based on larger dimension
        width = max_x - min_x
        height = max_y - min_y

        if width >= height:
            # Vertical line bisection
            mid_x = (min_x + max_x) / 2
            return 1.0, 0.0, -mid_x
        else:
            # Horizontal line bisection
            mid_y = (min_y + max_y) / 2
            return 0.0, 1.0, -mid_y

    def _find_median_bisecting_line(self, points: List[Point]) -> Tuple[float, float, float]:
        """
        Find a line that bisects the point set using the median point.

        Args:
            points: List of points to bisect

        Returns:
            Tuple (a, b, c) such that ax + by + c = 0
        """
        if len(points) < 2:
            return 1.0, 0.0, 0.0

        # Sort points by x and y coordinates to find median
        sorted_by_x = sorted(points, key=lambda p: p.x)
        sorted_by_y = sorted(points, key=lambda p: p.y)

        median_x = sorted_by_x[len(points) // 2].x
        median_y = sorted_by_y[len(points) // 2].y

        min_x = min(p.x for p in points)
        max_x = max(p.x for p in points)
        min_y = min(p.y for p in points)
        max_y = max(p.y for p in points)

        width = max_x - min_x
        height = max_y - min_y

        if width >= height:
            # Vertical line at median x
            return 1.0, 0.0, -median_x
        else:
            # Horizontal line at median y
            return 0.0, 1.0, -median_y

    def _partition_points(self, points: List[Point], line_a: float, line_b: float, line_c: float) -> Tuple[List[Point], List[Point]]:
        """
        Partition points based on which side of the line they're on.

        Args:
            points: Points to partition
            line_a, line_b, line_c: Line coefficients ax + by + c = 0

        Returns:
            Tuple of (left_points, right_points)
        """
        left_points = []
        right_points = []

        for point in points:
            # Calculate which side of line ax + by + c = 0 the point is on
            value = line_a * point.x + line_b * point.y + line_c
            if value <= 0:
                left_points.append(point)
            else:
                right_points.append(point)

        # Ensure both partitions are non-empty
        if len(left_points) == 0:
            left_points.append(right_points.pop())
        elif len(right_points) == 0:
            right_points.append(left_points.pop())

        return left_points, right_points
