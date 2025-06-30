import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from scipy.spatial import ConvexHull
import random
from typing import List, Tuple, Optional, Dict, Any
import math
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import json
import pickle
import os
from datetime import datetime
import logging


class Point:
    def __init__(self, x: float, y: float, id: int = None):
        self.x = x
        self.y = y
        self.id = id if id is not None else f"({x:.1f},{y:.1f})"

    def __repr__(self):
        return f"Point({self.x:.1f}, {self.y:.1f})"

    def distance_to(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Point to dictionary for JSON serialization."""
        return {'x': self.x, 'y': self.y, 'id': self.id}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Point':
        """Create Point from dictionary."""
        return cls(data['x'], data['y'], data['id'])


def setup_logging(log_dir: str = "wiener_analysis_logs") -> logging.Logger:
    """Set up logging configuration."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"wiener_analysis_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def euclidean_distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return p1.distance_to(p2)


def calculate_wiener_index(path: List[Point]) -> float:
    """Calculate the Wiener index of a Hamiltonian path."""
    n = len(path)
    wiener_sum = 0.0

    # For each pair of vertices in the path
    for i in range(n):
        for j in range(i + 1, n):
            # Distance between vertices i and j in the path is the sum of edge weights
            path_distance = 0.0
            for k in range(i, j):
                path_distance += euclidean_distance(path[k], path[k + 1])
            wiener_sum += path_distance

    return wiener_sum


def evaluate_permutation_chunk(chunk_data):
    """Evaluate a chunk of permutations - designed for parallel processing."""
    permutations_chunk, points = chunk_data
    best_wiener = float('inf')
    best_path = None

    for perm_indices in permutations_chunk:
        path = [points[i] for i in perm_indices]
        wiener = calculate_wiener_index(path)
        if wiener < best_wiener:
            best_wiener = wiener
            best_path = path

    return best_path, best_wiener


def parallel_brute_force_optimal_path(points: List[Point], max_workers: int = None) -> Tuple[List[Point], float]:
    """Find optimal Hamiltonian path using parallel brute force."""
    if len(points) <= 1:
        return points, 0.0 if len(points) <= 1 else calculate_wiener_index(points)

    if max_workers is None:
        max_workers = min(cpu_count(), 8)  # Limit to reasonable number

    n = len(points)
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating {math.factorial(n):,} permutations using {
                max_workers} workers...")

    # Generate all permutation indices
    all_permutations = list(permutations(range(n)))

    # Split permutations into chunks for parallel processing
    chunk_size = max(1, len(all_permutations) //
                     (max_workers * 4))  # 4 chunks per worker
    chunks = [all_permutations[i:i + chunk_size]
              for i in range(0, len(all_permutations), chunk_size)]

    # Prepare data for parallel processing
    chunk_data = [(chunk, points) for chunk in chunks]

    start_time = time.time()

    # Process chunks in parallel
    with Pool(max_workers) as pool:
        results = pool.map(evaluate_permutation_chunk, chunk_data)

    end_time = time.time()
    logger.info(f"Parallel brute force completed in {
                end_time - start_time:.2f} seconds")

    # Find the best result across all chunks
    best_path, best_wiener = min(results, key=lambda x: x[1])

    return best_path, best_wiener


def brute_force_optimal_path(points: List[Point]) -> List[Point]:
    """Find optimal Hamiltonian path by brute force for small sets."""
    if len(points) <= 1:
        return points

    best_path = None
    best_wiener = float('inf')

    # Try all possible permutations
    for perm in permutations(points):
        path = list(perm)
        wiener = calculate_wiener_index(path)
        if wiener < best_wiener:
            best_wiener = wiener
            best_path = path

    return best_path


def find_bisecting_line(points: List[Point]) -> Tuple[float, float, float]:
    """
    Find a line that roughly bisects the point set.
    Returns (a, b, c) such that ax + by + c = 0
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


def find_median_point_bisecting_line(points: List[Point]) -> Tuple[float, float, float]:
    """
    Find a line that bisects the point set using the median point.
    Returns (a, b, c) such that ax + by + c = 0
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


def partition_points(points: List[Point], line_a: float, line_b: float, line_c: float) -> Tuple[List[Point], List[Point]]:
    """Partition points based on which side of the line they're on."""
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


def connect_paths(path1: List[Point], path2: List[Point]) -> Tuple[List[Point], float]:
    """
    Find the optimal way to connect two Hamiltonian paths.
    Returns the best connected path and its Wiener index.
    """
    if not path1:
        return path2, calculate_wiener_index(path2)
    if not path2:
        return path1, calculate_wiener_index(path1)

    # Four possible ways to connect the paths
    connections = []

    # 1. end of path1 to start of path2
    conn1 = path1 + path2
    connections.append((conn1, calculate_wiener_index(conn1)))

    # 2. end of path1 to end of path2 (reverse path2)
    conn2 = path1 + path2[::-1]
    connections.append((conn2, calculate_wiener_index(conn2)))

    # 3. start of path1 to start of path2 (reverse path1)
    conn3 = path1[::-1] + path2
    connections.append((conn3, calculate_wiener_index(conn3)))

    # 4. start of path1 to end of path2 (reverse both)
    conn4 = path1[::-1] + path2[::-1]
    connections.append((conn4, calculate_wiener_index(conn4)))

    # Return the connection with minimum Wiener index
    return min(connections, key=lambda x: x[1])


def divide_conquer_wiener(points: List[Point], depth: int = 0, max_depth: int = 10) -> List[Point]:
    """
    Divide and conquer algorithm for Wiener index minimization.
    """
    # Base case: small sets solved by brute force
    if len(points) <= 4:  # Increased threshold for better optimization
        return brute_force_optimal_path(points)

    # Prevent infinite recursion
    if depth > max_depth:
        return brute_force_optimal_path(points)

    # Find bisecting line
    line_a, line_b, line_c = find_bisecting_line(points)

    # Partition points
    left_points, right_points = partition_points(
        points, line_a, line_b, line_c)

    # Recursively solve subproblems
    left_path = divide_conquer_wiener(left_points, depth + 1, max_depth)
    right_path = divide_conquer_wiener(right_points, depth + 1, max_depth)

    # Find optimal connection
    final_path, _ = connect_paths(left_path, right_path)

    return final_path


def divide_conquer_median_wiener(points: List[Point], depth: int = 0, max_depth: int = 10) -> List[Point]:
    """
    Divide and conquer algorithm for Wiener index minimization, optimized for median path.
    """
    # Base case: small sets solved by brute force
    if len(points) <= 4:  # Increased threshold for better optimization
        return brute_force_optimal_path(points)

    # Prevent infinite recursion
    if depth > max_depth:
        return brute_force_optimal_path(points)

    # Find bisecting line
    line_a, line_b, line_c = find_median_point_bisecting_line(points)

    # Partition points
    left_points, right_points = partition_points(
        points, line_a, line_b, line_c)

    # Recursively solve subproblems
    left_path = divide_conquer_median_wiener(left_points, depth + 1, max_depth)
    right_path = divide_conquer_median_wiener(
        right_points, depth + 1, max_depth)

    # Find optimal connection
    final_path, _ = connect_paths(left_path, right_path)

    return final_path


def generate_convex_points(n: int, seed: int = None) -> List[Point]:
    """Generate n points that form a convex set with varying distances."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Generate points on/near the boundary of an ellipse with random radii
    angles = sorted(np.random.uniform(0, 2*np.pi, n))
    points = []

    # Random ellipse parameters
    a = np.random.uniform(3, 7)  # semi-major axis
    b = np.random.uniform(2, 5)  # semi-minor axis
    center_x = np.random.uniform(-2, 2)
    center_y = np.random.uniform(-2, 2)

    for i, angle in enumerate(angles):
        # Add some randomness to the radius to create varying distances
        radius_factor = np.random.uniform(0.7, 1.3)
        x = center_x + a * radius_factor * np.cos(angle)
        y = center_y + b * radius_factor * np.sin(angle)

        # Add small random perturbation to avoid perfect regularity
        x += np.random.normal(0, 0.2)
        y += np.random.normal(0, 0.2)

        points.append(Point(x, y, i))

    return points


def generate_general_points(n: int, seed: int = None) -> List[Point]:
    """Generate n random points in a general position."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    points = []
    for i in range(n):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        points.append(Point(x, y, i))

    return points


def save_interesting_cases(interesting_cases: List[Dict[str, Any]], output_dir: str = "wiener_analysis_logs"):
    """Save interesting cases to files for later visualization."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON (human-readable)
    json_file = os.path.join(output_dir, f"interesting_cases_{timestamp}.json")

    # Convert Point objects to dictionaries for JSON serialization
    json_cases = []
    for case in interesting_cases:
        json_case = case.copy()
        json_case['points'] = [p.to_dict() for p in case['points']]
        json_case['dc_path'] = [p.to_dict() for p in case['dc_path']]
        if case['optimal_path']:
            json_case['optimal_path'] = [p.to_dict()
                                         for p in case['optimal_path']]
        else:
            json_case['optimal_path'] = None
        json_cases.append(json_case)

    with open(json_file, 'w') as f:
        json.dump(json_cases, f, indent=2)

    # Save as pickle (preserves exact Python objects)
    pickle_file = os.path.join(
        output_dir, f"interesting_cases_{timestamp}.pkl")
    with open(pickle_file, 'wb') as f:
        pickle.dump(interesting_cases, f)

    logger = logging.getLogger(__name__)
    logger.info(f"Saved {len(interesting_cases)} interesting cases to:")
    logger.info(f"  JSON: {json_file}")
    logger.info(f"  Pickle: {pickle_file}")

    return json_file, pickle_file


def load_interesting_cases(file_path: str) -> List[Dict[str, Any]]:
    """Load interesting cases from a file."""
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            json_cases = json.load(f)

        # Convert dictionaries back to Point objects
        cases = []
        for json_case in json_cases:
            case = json_case.copy()
            case['points'] = [Point.from_dict(p) for p in json_case['points']]
            case['dc_path'] = [Point.from_dict(p)
                               for p in json_case['dc_path']]
            if json_case['optimal_path']:
                case['optimal_path'] = [Point.from_dict(
                    p) for p in json_case['optimal_path']]
            else:
                case['optimal_path'] = None
            cases.append(case)
        return cases

    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    else:
        raise ValueError("File must be either .json or .pkl")


def visualize_path(points: List[Point], dc_path: List[Point], dc_wiener: float = None):
    """Visualize the points and the divide-and-conquer path."""
    plt.figure(figsize=(10, 6))
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]

    # Plot original points
    plt.scatter(x_coords, y_coords, c='blue', s=100, alpha=0.7, zorder=3)

    # Draw the D&C path
    if dc_path:
        path_x = [p.x for p in dc_path]
        path_y = [p.y for p in dc_path]
        plt.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.8, zorder=2)

        # Mark start and end
        plt.scatter(dc_path[0].x, dc_path[0].y, c='green', s=200, marker='s',
                    label='Start', zorder=4)
        plt.scatter(dc_path[-1].x, dc_path[-1].y, c='red', s=200, marker='^',
                    label='End', zorder=4)

        # Label points with path order
        for i, point in enumerate(dc_path):
            plt.annotate(f'{i}', (point.x, point.y), xytext=(5, 5),
                         textcoords='offset points', fontsize=10, weight='bold')

    # Title and labels
    title = 'Divide & Conquer Path Visualization'
    if dc_wiener is not None:
        title += f' (Wiener: {dc_wiener:.2f})'
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def visualize_comparison(points: List[Point], dc_path: List[Point], optimal_path: List[Point] = None,
                         dc_wiener: float = None, optimal_wiener: float = None, approximation_ratio: float = None):
    """Visualize the points and compare different algorithms."""

    # Determine number of subplots
    n_plots = 3 if optimal_path is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 6))

    if n_plots == 2:
        axes = [axes[0], axes[1]]  # Ensure axes is always a list

    # Extract coordinates
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]

    # Plot 1: Original points with convex hull
    axes[0].scatter(x_coords, y_coords, c='blue', s=100, alpha=0.7, zorder=3)

    # Draw convex hull
    if len(points) >= 3:
        hull_points = np.array([(p.x, p.y) for p in points])
        hull = ConvexHull(hull_points)
        for simplex in hull.simplices:
            axes[0].plot(hull_points[simplex, 0],
                         hull_points[simplex, 1], 'k--', alpha=0.5)

    # Label points
    for i, point in enumerate(points):
        axes[0].annotate(f'{i}', (point.x, point.y), xytext=(5, 5),
                         textcoords='offset points', fontsize=10)

    axes[0].set_title('Original Points with Convex Hull')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')

    # Plot 2: Divide & Conquer solution
    axes[1].scatter(x_coords, y_coords, c='blue', s=100, alpha=0.7, zorder=3)

    # Draw the D&C path
    if dc_path:
        path_x = [p.x for p in dc_path]
        path_y = [p.y for p in dc_path]
        axes[1].plot(path_x, path_y, 'r-', linewidth=2, alpha=0.8, zorder=2)

        # Mark start and end
        axes[1].scatter(dc_path[0].x, dc_path[0].y, c='green', s=200, marker='s',
                        label='Start', zorder=4)
        axes[1].scatter(dc_path[-1].x, dc_path[-1].y, c='red', s=200, marker='^',
                        label='End', zorder=4)

        # Label points with path order
        for i, point in enumerate(dc_path):
            axes[1].annotate(f'{i}', (point.x, point.y), xytext=(5, 5),
                             textcoords='offset points', fontsize=10, weight='bold')

    title_2 = f'Divide & Conquer'
    if dc_wiener is not None:
        title_2 += f' (Wiener: {dc_wiener:.2f})'
    axes[1].set_title(title_2)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_aspect('equal')

    # Plot 3: Optimal solution (if available)
    if optimal_path is not None:
        axes[2].scatter(x_coords, y_coords, c='blue',
                        s=100, alpha=0.7, zorder=3)

        # Draw the optimal path
        path_x = [p.x for p in optimal_path]
        path_y = [p.y for p in optimal_path]
        axes[2].plot(path_x, path_y, 'g-', linewidth=2, alpha=0.8, zorder=2)

        # Mark start and end
        axes[2].scatter(optimal_path[0].x, optimal_path[0].y, c='green', s=200, marker='s',
                        label='Start', zorder=4)
        axes[2].scatter(optimal_path[-1].x, optimal_path[-1].y, c='red', s=200, marker='^',
                        label='End', zorder=4)

        # Label points with path order
        for i, point in enumerate(optimal_path):
            axes[2].annotate(f'{i}', (point.x, point.y), xytext=(5, 5),
                             textcoords='offset points', fontsize=10, weight='bold')

        title_3 = f'Optimal (Brute Force)'
        if optimal_wiener is not None:
            title_3 += f' (Wiener: {optimal_wiener:.2f})'
        axes[2].set_title(title_3)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        axes[2].set_aspect('equal')

    # Main title with approximation ratio
    main_title = f"Wiener Index Minimization ({len(points)} points)"
    if approximation_ratio is not None:
        main_title += f" - Approximation Ratio: {approximation_ratio:.4f}"

    plt.suptitle(main_title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_from_file(file_path: str, max_cases: int = 10):
    """Load and visualize interesting cases from a saved file."""
    cases = load_interesting_cases(file_path)

    print(f"Loaded {len(cases)} interesting cases from {file_path}")

    # Group cases by point count and type
    grouped_cases = {}
    for case in cases:
        n_points = case['n_points']
        case_type = case['case_type']

        if n_points not in grouped_cases:
            grouped_cases[n_points] = {}
        if case_type not in grouped_cases[n_points]:
            grouped_cases[n_points][case_type] = []

        grouped_cases[n_points][case_type].append(case)

    # Visualize cases
    count = 0
    for n_points in sorted(grouped_cases.keys()):
        for case_type in ['best', 'worst', 'median']:
            if case_type in grouped_cases[n_points] and count < max_cases:
                # Take first (best) case of each type
                case = grouped_cases[n_points][case_type][0]

                print(f"\n{case_type.title()} case for {n_points} points (seed {case['seed']}): "
                      f"Ratio = {case['ratio']:.4f}")

                visualize_comparison(
                    case['points'],
                    case['dc_path'],
                    case['optimal_path'],
                    case['dc_wiener'],
                    case['optimal_wiener'],
                    case['ratio']
                )
                count += 1


def compare_algorithms(points: List[Point], use_parallel: bool = True, max_workers: int = None):
    """Compare divide-and-conquer with brute force (if feasible)."""
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing {len(points)} points...")

    # Divide and conquer solution
    logger.info("Running Divide & Conquer algorithm...")
    start_time = time.time()
    dc_path = divide_conquer_wiener(points)
    dc_time = time.time() - start_time
    dc_wiener = calculate_wiener_index(dc_path)
    logger.info(f"Divide & Conquer completed in {dc_time:.4f} seconds")
    logger.info(f"Divide & Conquer Wiener Index: {dc_wiener:.4f}")

    logger.info("Running median Divide & Conquer algorithm...")
    start_time = time.time()
    median_dc_path = divide_conquer_median_wiener(points)
    median_dc_time = time.time() - start_time
    median_dc_wiener = calculate_wiener_index(median_dc_path)
    logger.info(f"Median Divide & Conquer completed in {
                median_dc_time:.4f} seconds")

    # Brute force solution (only for manageable sets)
    optimal_path = None
    optimal_wiener = None
    approximation_ratio = None

    if len(points) <= 10:  # Increased threshold but still reasonable
        try:
            logger.info("Running Brute Force algorithm...")
            if use_parallel and len(points) >= 8:
                optimal_path, optimal_wiener = parallel_brute_force_optimal_path(
                    points, max_workers)
            else:
                start_time = time.time()
                optimal_path = brute_force_optimal_path(points)
                bf_time = time.time() - start_time
                optimal_wiener = calculate_wiener_index(optimal_path)
                logger.info(f"Brute Force completed in {bf_time:.4f} seconds")

            logger.info(f"Brute Force Optimal Wiener Index: {
                        optimal_wiener:.4f}")
            approximation_ratio = dc_wiener / optimal_wiener

            logger.info(f"Approximation Ratio for normal divide and conquer: {
                        approximation_ratio:.4f}")
            logger.info(f"Approximation Ratio for median divide and conquer: {
                        median_dc_wiener / optimal_wiener:.4f}")

        except MemoryError:
            logger.warning("Brute force skipped (memory limit exceeded)")
    else:
        logger.info(
            "Brute force skipped (too many points for exhaustive search)")

    # Visualize solutions
    visualize_comparison(points, dc_path, optimal_path,
                         dc_wiener, optimal_wiener, approximation_ratio)

    return dc_path, dc_wiener, optimal_path, optimal_wiener


def test_divide_conquer_wiener(is_convex: bool = True, points_counts: List[int] = range(6, 11), num_seeds: int = 100,
                               visualize_interesting: bool = True):
    '''
    Run tests for divide-and-conquer Wiener index minimization.
    Does not log the results, just runs the code.
    Returns results object and interesting cases for further analysis.
    '''
    print("=== Testing Divide-and-Conquer Wiener Index Minimization ===")
    print(f"Point counts: {list(points_counts)}")
    print(f"Seeds per test: {num_seeds}")

    all_results = {}
    interesting_cases = []

    for n_points in points_counts:
        print(f"\nTesting {n_points} points across {num_seeds} seeds...")

        results = {
            'dc_wieners': [],
            'optimal_wieners': [],
            'approximation_ratios': [],
            'dc_times': [],
            'optimal_times': [],
            'seeds': [],
            'point_sets': [],
            'dc_paths': [],
            'optimal_paths': []
        }

        perfect_count = 0
        good_count = 0
        poor_count = 0

        for seed in range(num_seeds):
            if (seed + 1) % 20 == 0:
                print(f"  Completed {seed + 1}/{num_seeds} seeds...")

            # Generate points
            if (is_convex):
                points = generate_convex_points(n_points, seed=seed)
            else:
                points = generate_general_points(n_points, seed=seed)

            # Run divide and conquer with timing
            start_time = time.time()
            dc_path = divide_conquer_wiener(points)
            dc_time = time.time() - start_time
            dc_wiener = calculate_wiener_index(dc_path)

            start_time = time.time()
            median_dc_path = divide_conquer_median_wiener(points)
            median_dc_time = time.time() - start_time
            median_dc_wiener = calculate_wiener_index(median_dc_path)

            # Store basic results
            results['dc_wieners'].append(dc_wiener)
            results['dc_times'].append(dc_time)
            results['seeds'].append(seed)
            results['point_sets'].append(points)
            results['dc_paths'].append(dc_path)

            # Compare with brute force if feasible
            if n_points <= 8:
                start_time = time.time()
                if n_points >= 8:
                    optimal_path, optimal_wiener = parallel_brute_force_optimal_path(
                        points, max_workers=cpu_count())
                else:
                    optimal_path = brute_force_optimal_path(points)
                    optimal_wiener = calculate_wiener_index(optimal_path)
                optimal_time = time.time() - start_time

                results['optimal_wieners'].append(optimal_wiener)
                results['optimal_times'].append(optimal_time)
                results['optimal_paths'].append(optimal_path)

                approximation_ratio = dc_wiener / optimal_wiener
                median_approximation_ratio = median_dc_wiener / optimal_wiener
                results['approximation_ratios'].append(approximation_ratio)

                # Count solution quality
                if approximation_ratio <= 1.001:
                    perfect_count += 1
                if approximation_ratio <= 1.1:
                    good_count += 1
                if approximation_ratio > 1.5:
                    poor_count += 1

                if seed < 5:
                    print(f"    Seed {seed}: D&C Wiener: {dc_wiener:.4f}, "
                          f"Optimal: {optimal_wiener:.4f}, Ratio: {approximation_ratio:.4f}")
                    print(f"    Median D&C Wiener: {median_dc_wiener:.4f}, "
                          f"Median Ratio: {median_approximation_ratio:.4f}, "
                          )
            else:
                # For larger point sets, store None for optimal results
                results['optimal_wieners'].append(None)
                results['optimal_times'].append(None)
                results['optimal_paths'].append(None)
                results['approximation_ratios'].append(None)

                print(f"    Seed {seed}: D&C Wiener Index: {dc_wiener:.4f} "
                      f"(Brute force skipped due to point count)")

        # Calculate statistics
        dc_wieners = np.array(results['dc_wieners'])
        dc_times = np.array(results['dc_times'])

        stats = {
            'n_points': n_points,
            'dc_wiener_mean': np.mean(dc_wieners),
            'dc_wiener_std': np.std(dc_wieners),
            'dc_wiener_min': np.min(dc_wieners),
            'dc_wiener_max': np.max(dc_wieners),
            'dc_time_mean': np.mean(dc_times),
        }

        # Print summary statistics for this point count
        if n_points <= 10 and any(r is not None for r in results['approximation_ratios']):
            ratios = np.array(
                [r for r in results['approximation_ratios'] if r is not None])
            optimal_wieners = np.array(
                [w for w in results['optimal_wieners'] if w is not None])
            optimal_times = np.array(
                [t for t in results['optimal_times'] if t is not None])

            # Update stats with optimal information
            stats.update({
                'optimal_wiener_mean': np.mean(optimal_wieners),
                'optimal_wiener_std': np.std(optimal_wieners),
                'optimal_wiener_min': np.min(optimal_wieners),
                'optimal_wiener_max': np.max(optimal_wieners),
                'ratio_mean': np.mean(ratios),
                'ratio_std': np.std(ratios),
                'ratio_min': np.min(ratios),
                'ratio_max': np.max(ratios),
                'optimal_time_mean': np.mean(optimal_times),
                'perfect_solutions': perfect_count,
                'good_solutions': good_count,
                'poor_solutions': poor_count
            })

            print(f"  Summary for {n_points} points:")
            print(f"    Approximation ratio: {
                  np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
            print(f"    Range: [{np.min(ratios):.4f}, {np.max(ratios):.4f}]")
            print(f"    Perfect solutions (≤1.001): {perfect_count}/{num_seeds} "
                  f"({100*perfect_count/num_seeds:.1f}%)")
            print(f"    Good solutions (≤1.1): {good_count}/{num_seeds} "
                  f"({100*good_count/num_seeds:.1f}%)")
            print(f"    Poor solutions (>1.5): {poor_count}/{num_seeds} "
                  f"({100*poor_count/num_seeds:.1f}%)")
            print(f"    Avg D&C time: {stats['dc_time_mean']:.4f}s")
            print(f"    Avg Optimal time: {stats['optimal_time_mean']:.4f}s")
            print(f"    Speedup factor: {
                  stats['optimal_time_mean']/stats['dc_time_mean']:.1f}x")

            # Find interesting cases for this point count
            ratios_array = np.array(ratios)
            if len(ratios_array) > 0:
                best_case_idx = np.argmin(ratios_array)
                worst_case_idx = np.argmax(ratios_array)
                median_case_idx = np.argmin(
                    np.abs(ratios_array - np.median(ratios_array)))

                # Add to interesting cases list
                valid_indices = [i for i, r in enumerate(
                    results['approximation_ratios']) if r is not None]
                for case_type, local_idx in [('best', best_case_idx), ('worst', worst_case_idx), ('median', median_case_idx)]:
                    global_idx = valid_indices[local_idx]
                    interesting_cases.append({
                        'n_points': n_points,
                        'case_type': case_type,
                        'seed': results['seeds'][global_idx],
                        'points': results['point_sets'][global_idx],
                        'dc_path': results['dc_paths'][global_idx],
                        'optimal_path': results['optimal_paths'][global_idx],
                        'dc_wiener': results['dc_wieners'][global_idx],
                        'optimal_wiener': results['optimal_wieners'][global_idx],
                        'ratio': results['approximation_ratios'][global_idx]
                    })
        else:
            print(f"  D&C Wiener indices: {
                  np.mean(dc_wieners):.4f} ± {np.std(dc_wieners):.4f}")
            print(f"  Avg D&C time: {stats['dc_time_mean']:.4f}s")

        all_results[n_points] = {'stats': stats, 'results': results}

    print("\n=== Test completed ===")
    print(f"Total interesting cases found: {len(interesting_cases)}")

    # Visualize interesting cases if requested
    if visualize_interesting and interesting_cases:
        print(f"\nVisualizing best 5, median 5, and worst 5 cases for each point count...")

        # Organize cases by point count for systematic visualization
        for n_points in points_counts:
            # Get all cases for this point count that have optimal solutions
            point_cases = []
            for seed in range(num_seeds):
                if n_points <= 10:  # Only visualize cases where we have optimal solutions
                    idx = seed
                    if idx < len(all_results[n_points]['results']['approximation_ratios']):
                        ratio = all_results[n_points]['results']['approximation_ratios'][idx]
                        if ratio is not None:
                            point_cases.append({
                                'n_points': n_points,
                                'seed': seed,
                                'points': all_results[n_points]['results']['point_sets'][idx],
                                'dc_path': all_results[n_points]['results']['dc_paths'][idx],
                                'optimal_path': all_results[n_points]['results']['optimal_paths'][idx],
                                'dc_wiener': all_results[n_points]['results']['dc_wieners'][idx],
                                'optimal_wiener': all_results[n_points]['results']['optimal_wieners'][idx],
                                'ratio': ratio
                            })

            if not point_cases:
                print(f"Skipping visualization for {
                      n_points} points (no optimal solutions computed)")
                continue

            print(f"\n{'='*60}")
            print(f"VISUALIZING {n_points} POINTS")
            print(f"{'='*60}")

            # Sort cases by approximation ratio
            point_cases.sort(key=lambda x: x['ratio'])

            # Get best 5, median 5, and worst 5
            n_cases = len(point_cases)
            best_5 = point_cases[:min(3, n_cases)]
            worst_5 = point_cases[max(0, n_cases-3):]

            # For median 5, get cases around the median
            median_idx = n_cases // 2
            median_start = max(0, median_idx - 1)
            median_end = min(n_cases, median_idx + 2)
            median_5 = point_cases[median_start:median_end]

            # Visualize each category
            categories = [
                ("BEST", best_5),
                ("MEDIAN", median_5),
                ("WORST", worst_5)
            ]

            for category_name, cases in categories:
                if not cases:
                    continue

                print(f"\n{'-'*40}")
                print(f"{category_name} 5 CASES FOR {n_points} POINTS")
                print(f"{'-'*40}")

                for i, case in enumerate(cases):
                    print(f"\n{category_name} Case {i+1}/{len(cases)}: Seed {case['seed']}, "
                          f"Ratio: {case['ratio']:.4f}")
                    print(f"D&C Wiener: {case['dc_wiener']:.4f}, "
                          f"Optimal Wiener: {case['optimal_wiener']:.4f}")

                    visualize_comparison(
                        points=case['points'],
                        dc_path=case['dc_path'],
                        optimal_path=case['optimal_path'],
                        dc_wiener=case['dc_wiener'],
                        optimal_wiener=case['optimal_wiener'],
                        approximation_ratio=case['ratio']
                    )

                # Print category statistics
                ratios = [case['ratio'] for case in cases]
                dc_wieners = [case['dc_wiener'] for case in cases]
                optimal_wieners = [case['optimal_wiener'] for case in cases]

                print(f"\n{category_name} Category Statistics:")
                print(f"  Approximation Ratios: {np.mean(ratios):.4f} ± {np.std(ratios):.4f} "
                      f"[{np.min(ratios):.4f}, {np.max(ratios):.4f}]")
                print(f"  D&C Wiener Indices: {
                      np.mean(dc_wieners):.4f} ± {np.std(dc_wieners):.4f}")
                print(f"  Optimal Wiener Indices: {np.mean(optimal_wieners):.4f} ± {
                      np.std(optimal_wieners):.4f}")

    return all_results, interesting_cases


def comprehensive_test(is_convex: bool = False, point_counts: List[int] = [6, 7, 8, 9, 10], num_seeds: int = 100, ):
    """
    Run comprehensive tests across multiple seeds and point counts.
    Log all results and save interesting cases.
    """
    logger = logging.getLogger(__name__)
    logger.info("=== Comprehensive Statistical Analysis ===")
    logger.info(f"Testing point counts: {point_counts}")
    logger.info(f"Seeds per test: {num_seeds}")
    logger.info(f"Using up to {cpu_count()} CPU cores for parallel processing")

    all_results = {}
    interesting_cases = []

    for n_points in point_counts:
        logger.info(f"{'='*50}")
        logger.info(f"Testing {n_points} points across {num_seeds} seeds...")
        logger.info(f"{'='*50}")

        results = {
            'dc_wieners': [],
            'optimal_wieners': [],
            'approximation_ratios': [],
            'dc_times': [],
            'optimal_times': [],
            'seeds': [],
            'point_sets': [],
            'dc_paths': [],
            'optimal_paths': []
        }

        for seed in range(num_seeds):
            if (seed + 1) % 20 == 0:
                logger.info(f"  Completed {seed + 1}/{num_seeds} seeds...")

            # Generate points
            if (is_convex):
                points = generate_convex_points(n_points, seed=seed)
            else:
                points = generate_general_points(n_points, seed=seed)

            # Divide and conquer
            start_time = time.time()
            dc_path = divide_conquer_wiener(points)
            dc_time = time.time() - start_time
            dc_wiener = calculate_wiener_index(dc_path)

            # Optimal solution
            start_time = time.time()
            if n_points >= 8:
                optimal_path, optimal_wiener = parallel_brute_force_optimal_path(
                    points, max_workers=cpu_count())
                optimal_time = time.time() - start_time
            else:
                optimal_path = brute_force_optimal_path(points)
                optimal_time = time.time() - start_time
                optimal_wiener = calculate_wiener_index(optimal_path)

            # Calculate metrics
            approximation_ratio = dc_wiener / optimal_wiener

            # Store results
            results['dc_wieners'].append(dc_wiener)
            results['optimal_wieners'].append(optimal_wiener)
            results['approximation_ratios'].append(approximation_ratio)
            results['dc_times'].append(dc_time)
            results['optimal_times'].append(optimal_time)
            results['seeds'].append(seed)
            results['point_sets'].append(points)
            results['dc_paths'].append(dc_path)
            results['optimal_paths'].append(optimal_path)

        # Calculate statistics
        dc_wieners = np.array(results['dc_wieners'])
        optimal_wieners = np.array(results['optimal_wieners'])
        ratios = np.array(results['approximation_ratios'])
        dc_times = np.array(results['dc_times'])
        optimal_times = np.array(results['optimal_times'])

        stats = {
            'n_points': n_points,
            'dc_wiener_mean': np.mean(dc_wieners),
            'dc_wiener_std': np.std(dc_wieners),
            'dc_wiener_min': np.min(dc_wieners),
            'dc_wiener_max': np.max(dc_wieners),
            'optimal_wiener_mean': np.mean(optimal_wieners),
            'optimal_wiener_std': np.std(optimal_wieners),
            'optimal_wiener_min': np.min(optimal_wieners),
            'optimal_wiener_max': np.max(optimal_wieners),
            'ratio_mean': np.mean(ratios),
            'ratio_std': np.std(ratios),
            'ratio_min': np.min(ratios),
            'ratio_max': np.max(ratios),
            'dc_time_mean': np.mean(dc_times),
            'optimal_time_mean': np.mean(optimal_times),
            'perfect_solutions': np.sum(ratios <= 1.001),
            'good_solutions': np.sum(ratios <= 1.1),
            'poor_solutions': np.sum(ratios > 1.5)
        }

        all_results[n_points] = {'stats': stats, 'results': results}

        # Log statistics
        logger.info(f"Statistics for {n_points} points:")
        logger.info(f"  Approximation Ratio: {
                    stats['ratio_mean']:.4f} ± {stats['ratio_std']:.4f}")
        logger.info(f"  Range: [{stats['ratio_min']:.4f}, {
                    stats['ratio_max']:.4f}]")
        logger.info(f"  Perfect solutions (≤1.001): {stats['perfect_solutions']}/{
                    num_seeds} ({100*stats['perfect_solutions']/num_seeds:.1f}%)")
        logger.info(f"  Good solutions (≤1.1): {
                    stats['good_solutions']}/{num_seeds} ({100*stats['good_solutions']/num_seeds:.1f}%)")
        logger.info(f"  Poor solutions (>1.5): {
                    stats['poor_solutions']}/{num_seeds} ({100*stats['poor_solutions']/num_seeds:.1f}%)")
        logger.info(f"  Avg D&C time: {stats['dc_time_mean']:.4f}s")
        logger.info(f"  Avg Optimal time: {stats['optimal_time_mean']:.4f}s")
        logger.info(f"  Speedup factor: {
                    stats['optimal_time_mean']/stats['dc_time_mean']:.1f}x")

        # Identify interesting cases for saving
        ratios_array = np.array(results['approximation_ratios'])

        # Find interesting cases
        best_case_idx = np.argmin(ratios_array)
        worst_case_idx = np.argmax(ratios_array)
        median_case_idx = np.argmin(
            np.abs(ratios_array - np.median(ratios_array)))

        # Add to interesting cases list
        for case_type, idx in [('best', best_case_idx), ('worst', worst_case_idx), ('median', median_case_idx)]:
            interesting_cases.append({
                'n_points': n_points,
                'case_type': case_type,
                'seed': results['seeds'][idx],
                'points': results['point_sets'][idx],
                'dc_path': results['dc_paths'][idx],
                'optimal_path': results['optimal_paths'][idx],
                'dc_wiener': results['dc_wieners'][idx],
                'optimal_wiener': results['optimal_wieners'][idx],
                'ratio': results['approximation_ratios'][idx]
            })

    # Log overall summary
    logger.info(f"\n{'='*50}")
    logger.info("Overall Summary Statistics:")
    logger.info(f"{'='*50}")

    for n_points, data in all_results.items():
        stats = data['stats']
        logger.info(f"\n{n_points} points:")
        logger.info(f"  D&C Wiener Index: {stats['dc_wiener_mean']:.4f} ± {
                    stats['dc_wiener_std']:.4f}")
        logger.info(f"  Optimal Wiener Index: {stats['optimal_wiener_mean']:.4f} ± {
                    stats['optimal_wiener_std']:.4f}")
        logger.info(f"  Approximation Ratio: {
                    stats['ratio_mean']:.4f} ± {stats['ratio_std']:.4f}")
        logger.info(f"  Avg D&C Time: {stats['dc_time_mean']:.4f}s")
        logger.info(f"  Avg Optimal Time: {stats['optimal_time_mean']:.4f}s")
    logger.info(f"\n{'='*50}")
    logger.info(f"Total interesting cases found: {len(interesting_cases)}")
    # Save interesting cases to files
    output_dir = "wiener_analysis_interesting_cases"
    json_file, pickle_file = save_interesting_cases(
        interesting_cases, output_dir)
    logger.info(f"Interesting cases saved to: {json_file} and {pickle_file}")
    logger.info("Visualizing interesting cases...")
    visualize_from_file(json_file, max_cases=10)
    logger.info("Comprehensive test completed.")
    return all_results, interesting_cases


if __name__ == "__main__":
    # Set up logging
    logger = setup_logging()

    # Run comprehensive test with default parameters
    # results, interesting_cases = comprehensive_test(
    #     point_counts=[6, 7, 8, 9, 10, 11], num_seeds=100)

    results, interesting_cases = test_divide_conquer_wiener(
        is_convex=False,
        points_counts=range(6, 20),
        num_seeds=100,
        visualize_interesting=True
    )

    # Optionally visualize a specific case
    if interesting_cases:
        visualize_comparison(
            interesting_cases[0]['points'],
            interesting_cases[0]['dc_path'],
            interesting_cases[0]['optimal_path'],
            interesting_cases[0]['dc_wiener'],
            interesting_cases[0]['optimal_wiener'],
            interesting_cases[0]['ratio']
        )
    logger.info("Analysis complete. Check the logs for details.")
