import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from typing import List, Optional

from core.point import Point


class Visualizer:
    """Visualizer for Hamiltonian paths and point configurations."""

    @staticmethod
    def visualize_path(points: List[Point], path: List[Point],
                       wiener_index: Optional[float] = None,
                       title: str = "Hamiltonian Path") -> None:
        """
        Visualize a single Hamiltonian path.

        Args:
            points: Original points
            path: Hamiltonian path to visualize
            wiener_index: Optional Wiener index to display
            title: Title for the plot
        """
        plt.figure(figsize=(10, 6))
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]

        # Plot original points
        plt.scatter(x_coords, y_coords, c='blue', s=100, alpha=0.7, zorder=3)

        # Draw the path
        if path:
            path_x = [p.x for p in path]
            path_y = [p.y for p in path]
            plt.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.8, zorder=2)

            # Mark start and end
            plt.scatter(path[0].x, path[0].y, c='green', s=200, marker='s',
                        label='Start', zorder=4)
            plt.scatter(path[-1].x, path[-1].y, c='red', s=200, marker='^',
                        label='End', zorder=4)

            # Label points with path order
            for i, point in enumerate(path):
                plt.annotate(f'{i}', (point.x, point.y), xytext=(5, 5),
                             textcoords='offset points', fontsize=10, weight='bold')

        # Title and labels
        plot_title = title
        if wiener_index is not None:
            plot_title += f' (Wiener: {wiener_index:.2f})'

        plt.title(plot_title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_comparison_cases(points: List[Point],
                                   dc_path: List[Point],
                                   optimal_path: Optional[List[Point]] = None,
                                   dc_wiener: Optional[float] = None,
                                   optimal_wiener: Optional[float] = None,
                                   approximation_ratio: Optional[float] = None) -> None:
        """
        Visualize comparison between divide-and-conquer and optimal solutions.

        Args:
            points: Original points
            dc_path: Divide and conquer path
            optimal_path: Optimal path (if available)
            dc_wiener: Divide and conquer Wiener index
            optimal_wiener: Optimal Wiener index
            approximation_ratio: Approximation ratio
        """
        Visualizer.visualize_comparison(
            points, dc_path, optimal_path, dc_wiener, optimal_wiener, approximation_ratio)

    @staticmethod
    def visualize_comparison(points: List[Point],
                             dc_path: List[Point],
                             optimal_path: Optional[List[Point]] = None,
                             dc_wiener: Optional[float] = None,
                             optimal_wiener: Optional[float] = None,
                             approximation_ratio: Optional[float] = None) -> None:
        """
        Visualize comparison between divide-and-conquer and optimal solutions.

        Args:
            points: Original points
            dc_path: Divide and conquer path
            optimal_path: Optimal path (if available)
            dc_wiener: Divide and conquer Wiener index
            optimal_wiener: Optimal Wiener index
            approximation_ratio: Approximation ratio
        """
        # Determine number of subplots
        n_plots = 3 if optimal_path is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 6))

        if n_plots == 2:
            axes = [axes[0], axes[1]]  # Ensure axes is always a list
        elif n_plots == 1:
            axes = [axes]  # Handle single plot case

        # Extract coordinates
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]

        # Plot 1: Original points with convex hull
        axes[0].scatter(x_coords, y_coords, c='blue',
                        s=100, alpha=0.7, zorder=3)

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
        axes[1].scatter(x_coords, y_coords, c='blue',
                        s=100, alpha=0.7, zorder=3)

        # Draw the D&C path
        if dc_path:
            path_x = [p.x for p in dc_path]
            path_y = [p.y for p in dc_path]
            axes[1].plot(path_x, path_y, 'r-',
                         linewidth=2, alpha=0.8, zorder=2)

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
            axes[2].plot(path_x, path_y, 'g-',
                         linewidth=2, alpha=0.8, zorder=2)

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

    @staticmethod
    def visualize_multiple_paths(points: List[Point],
                                 paths: List[List[Point]],
                                 labels: List[str],
                                 wiener_indices: Optional[List[float]] = None,
                                 title: str = "Path Comparison") -> None:
        """
        Visualize multiple Hamiltonian paths on the same point set.

        Args:
            points: Original points
            paths: List of Hamiltonian paths to visualize
            labels: Labels for each path
            wiener_indices: Optional Wiener indices for each path
            title: Title for the plot
        """
        fig, axes = plt.subplots(1, len(paths), figsize=(6*len(paths), 6))

        if len(paths) == 1:
            axes = [axes]

        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]

        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']

        for i, (path, label) in enumerate(zip(paths, labels)):
            ax = axes[i]

            # Plot original points
            ax.scatter(x_coords, y_coords, c='blue',
                       s=100, alpha=0.7, zorder=3)

            # Draw the path
            if path:
                path_x = [p.x for p in path]
                path_y = [p.y for p in path]
                color = colors[i % len(colors)]
                ax.plot(path_x, path_y, color=color,
                        linewidth=2, alpha=0.8, zorder=2)

                # Mark start and end
                ax.scatter(path[0].x, path[0].y, c='green', s=200, marker='s',
                           label='Start', zorder=4)
                ax.scatter(path[-1].x, path[-1].y, c='red', s=200, marker='^',
                           label='End', zorder=4)

                # Label points with path order
                for j, point in enumerate(path):
                    ax.annotate(f'{j}', (point.x, point.y), xytext=(5, 5),
                                textcoords='offset points', fontsize=10, weight='bold')

            # Title and labels
            plot_title = label
            if wiener_indices and i < len(wiener_indices) and wiener_indices[i] is not None:
                plot_title += f' (Wiener: {wiener_indices[i]:.2f})'

            ax.set_title(plot_title)
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_aspect('equal')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_from_file(file_path: str, max_cases: int = 10) -> None:
        """
        Load and visualize interesting cases from a saved file.

        Args:
            file_path: Path to the saved file (JSON or pickle)
            max_cases: Maximum number of cases to visualize
        """
        cases = Visualizer.load_interesting_cases(file_path)

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

                    Visualizer.visualize_comparison(
                        case['points'],
                        case['dc_path'],
                        case['optimal_path'],
                        case['dc_wiener'],
                        case['optimal_wiener'],
                        case['ratio']
                    )
                    count += 1

    @staticmethod
    def plot_performance_statistics(results: dict,
                                    title: str = "Algorithm Performance Analysis") -> None:
        """
        Plot performance statistics across different point counts.

        Args:
            results: Dictionary containing results from comprehensive tests
            title: Title for the plot
        """
        point_counts = sorted(results.keys())

        # Extract statistics
        dc_means = [results[n]['stats']['dc_wiener_mean']
                    for n in point_counts]
        optimal_means = [results[n]['stats']['optimal_wiener_mean']
                         for n in point_counts]
        ratio_means = [results[n]['stats']['ratio_mean'] for n in point_counts]
        ratio_stds = [results[n]['stats']['ratio_std'] for n in point_counts]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Wiener indices comparison
        axes[0, 0].plot(point_counts, dc_means, 'r-o',
                        label='Divide & Conquer', linewidth=2)
        axes[0, 0].plot(point_counts, optimal_means, 'g-s',
                        label='Optimal', linewidth=2)
        axes[0, 0].set_xlabel('Number of Points')
        axes[0, 0].set_ylabel('Mean Wiener Index')
        axes[0, 0].set_title('Wiener Index Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Approximation ratio
        axes[0, 1].errorbar(point_counts, ratio_means, yerr=ratio_stds,
                            fmt='b-o', capsize=5, linewidth=2)
        axes[0, 1].axhline(y=1.0, color='g', linestyle='--',
                           alpha=0.7, label='Optimal')
        axes[0, 1].set_xlabel('Number of Points')
        axes[0, 1].set_ylabel('Approximation Ratio')
        axes[0, 1].set_title('Approximation Ratio (mean ± std)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Solution quality distribution
        perfect_ratios = [results[n]['stats']['perfect_solutions'] /
                          len(results[n]['results']['seeds']) * 100 for n in point_counts]
        good_ratios = [results[n]['stats']['good_solutions'] /
                       len(results[n]['results']['seeds']) * 100 for n in point_counts]

        axes[1, 0].plot(point_counts, perfect_ratios, 'g-o',
                        label='Perfect (≤1.001)', linewidth=2)
        axes[1, 0].plot(point_counts, good_ratios, 'b-s',
                        label='Good (≤1.1)', linewidth=2)
        axes[1, 0].set_xlabel('Number of Points')
        axes[1, 0].set_ylabel('Percentage of Solutions')
        axes[1, 0].set_title('Solution Quality Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Timing comparison
        dc_times = [results[n]['stats']['dc_time_mean'] for n in point_counts]
        optimal_times = [results[n]['stats']['optimal_time_mean']
                         for n in point_counts]

        axes[1, 1].semilogy(point_counts, dc_times, 'r-o',
                            label='Divide & Conquer', linewidth=2)
        axes[1, 1].semilogy(point_counts, optimal_times,
                            'g-s', label='Optimal', linewidth=2)
        axes[1, 1].set_xlabel('Number of Points')
        axes[1, 1].set_ylabel('Time (seconds, log scale)')
        axes[1, 1].set_title('Runtime Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
