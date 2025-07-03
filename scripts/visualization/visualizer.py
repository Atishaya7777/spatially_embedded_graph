import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.spatial import ConvexHull, distance_matrix
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from core.point import Point

# Import these here to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from analysis.statistical_analyzer import StudyResults, MultiAlgorithmExperiment


class Visualizer:
    """
    Enhanced Visualizer for Hamiltonian paths, algorithm comparisons, and study results.
    
    This class integrates all visualization functionality previously scattered across
    multiple visualize_*.py files into a single, modular component.
    """

    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize the visualizer with styling options.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        self.figsize = figsize
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))  # Color palette for algorithms

    # Core path visualization methods
    def visualize_path(self, points: List[Point], path: List[Point],
                       wiener_index: Optional[float] = None,
                       title: str = "Hamiltonian Path",
                       save_path: Optional[str] = None) -> None:
        """
        Visualize a single Hamiltonian path.

        Args:
            points: Original points
            path: Hamiltonian path to visualize
            wiener_index: Optional Wiener index to display
            title: Title for the plot
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=self.figsize)
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

        plt.title(plot_title, fontsize=14, fontweight='bold')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def visualize_algorithm_comparison(self, experiment: 'MultiAlgorithmExperiment', 
                                     save_path: Optional[str] = None) -> None:
        """
        Visualize comparison between multiple algorithms on the same point set.
        
        Args:
            experiment: MultiAlgorithmExperiment containing results from multiple algorithms
            save_path: Optional path to save the plot
        """
        num_algorithms = len(experiment.results)
        if num_algorithms == 0:
            return
            
        # Create subplots
        cols = min(num_algorithms, 3)
        rows = (num_algorithms + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
        
        if num_algorithms == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        # Plot each algorithm's result
        for idx, (algorithm_name, result) in enumerate(experiment.results.items()):
            ax = axes[idx]
            
            # Plot points
            x_coords = [p.x for p in experiment.points]
            y_coords = [p.y for p in experiment.points]
            ax.scatter(x_coords, y_coords, c='blue', s=100, alpha=0.7, zorder=3)
            
            # Plot path
            if result.path:
                path_x = [p.x for p in result.path]
                path_y = [p.y for p in result.path]
                ax.plot(path_x, path_y, linewidth=2, alpha=0.8, 
                       color=self.colors[idx % len(self.colors)], zorder=2)
                
                # Mark start and end
                ax.scatter(result.path[0].x, result.path[0].y, c='green', s=150, 
                          marker='s', zorder=4)
                ax.scatter(result.path[-1].x, result.path[-1].y, c='red', s=150, 
                          marker='^', zorder=4)
            
            # Set title with metrics
            title = f'{algorithm_name.replace("_", " ").title()}\n'
            title += f'Wiener: {result.wiener_index:.3f}\n'
            title += f'Time: {result.execution_time:.3f}s'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
        
        # Hide extra subplots
        for idx in range(num_algorithms, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(f'Algorithm Comparison ({experiment.n_points} points, seed {experiment.seed})', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_study_results(self, study_results: 'StudyResults', 
                          save_path: Optional[str] = None) -> None:
        """
        Create comprehensive plots for study results showing algorithm performance.
        
        Args:
            study_results: StudyResults containing experiments and statistics
            save_path: Optional path to save the plot
        """
        algorithms = list(study_results.statistics.keys())
        if not algorithms:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data for plotting
        point_sizes = []
        algorithm_data = {alg: {'wiener': [], 'time': []} for alg in algorithms}
        
        for exp in study_results.experiments:
            if exp.results:
                point_sizes.append(exp.n_points)
                for alg_name, result in exp.results.items():
                    if alg_name in algorithm_data:
                        algorithm_data[alg_name]['wiener'].append(result.wiener_index)
                        algorithm_data[alg_name]['time'].append(result.execution_time)
        
        # Group by point size for better visualization
        size_groups = {}
        for i, size in enumerate(point_sizes):
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(i)
        
        # Plot 1: Wiener Index by Point Size
        ax1 = axes[0, 0]
        for alg_name, color in zip(algorithms, self.colors):
            sizes = []
            wiener_means = []
            wiener_stds = []
            
            for size in sorted(size_groups.keys()):
                indices = size_groups[size]
                wiener_values = [algorithm_data[alg_name]['wiener'][i] for i in indices 
                               if i < len(algorithm_data[alg_name]['wiener'])]
                if wiener_values:
                    sizes.append(size)
                    wiener_means.append(np.mean(wiener_values))
                    wiener_stds.append(np.std(wiener_values))
            
            if sizes:
                ax1.errorbar(sizes, wiener_means, yerr=wiener_stds, 
                           label=alg_name.replace('_', ' ').title(), 
                           color=color, marker='o', capsize=5)
        
        ax1.set_xlabel('Number of Points')
        ax1.set_ylabel('Wiener Index')
        ax1.set_title('Wiener Index by Point Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Execution Time by Point Size
        ax2 = axes[0, 1]
        for alg_name, color in zip(algorithms, self.colors):
            sizes = []
            time_means = []
            time_stds = []
            
            for size in sorted(size_groups.keys()):
                indices = size_groups[size]
                time_values = [algorithm_data[alg_name]['time'][i] for i in indices 
                             if i < len(algorithm_data[alg_name]['time'])]
                if time_values:
                    sizes.append(size)
                    time_means.append(np.mean(time_values))
                    time_stds.append(np.std(time_values))
            
            if sizes:
                ax2.errorbar(sizes, time_means, yerr=time_stds,
                           label=alg_name.replace('_', ' ').title(),
                           color=color, marker='s', capsize=5)
        
        ax2.set_xlabel('Number of Points')
        ax2.set_ylabel('Execution Time (seconds)')
        ax2.set_title('Execution Time by Point Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Algorithm Statistics Summary
        ax3 = axes[1, 0]
        alg_names = []
        wiener_means = []
        time_means = []
        
        for alg_name, stats in study_results.statistics.items():
            alg_names.append(alg_name.replace('_', ' ').title())
            wiener_means.append(stats.wiener_index_mean)
            time_means.append(stats.execution_time_mean)
        
        x_pos = np.arange(len(alg_names))
        ax3.bar(x_pos, wiener_means, color=self.colors[:len(alg_names)], alpha=0.7)
        ax3.set_xlabel('Algorithm')
        ax3.set_ylabel('Mean Wiener Index')
        ax3.set_title('Average Wiener Index by Algorithm')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(alg_names, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance Scatter
        ax4 = axes[1, 1]
        for alg_name, color in zip(algorithms, self.colors):
            wiener_vals = algorithm_data[alg_name]['wiener']
            time_vals = algorithm_data[alg_name]['time']
            if wiener_vals and time_vals:
                ax4.scatter(time_vals, wiener_vals, 
                          label=alg_name.replace('_', ' ').title(),
                          color=color, alpha=0.6, s=50)
        
        ax4.set_xlabel('Execution Time (seconds)')
        ax4.set_ylabel('Wiener Index')
        ax4.set_title('Quality vs Speed Trade-off')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_approximation_ratios(self, study_results: 'StudyResults',
                                baseline_algorithm: str = 'brute_force',
                                save_path: Optional[str] = None) -> None:
        """
        Plot approximation ratios relative to a baseline algorithm.
        
        Args:
            study_results: StudyResults containing experiments
            baseline_algorithm: Algorithm to use as baseline (optimal)
            save_path: Optional path to save the plot
        """
        algorithms = [alg for alg in study_results.statistics.keys() 
                     if alg != baseline_algorithm]
        if not algorithms:
            return
            
        plt.figure(figsize=self.figsize)
        
        for alg_name, color in zip(algorithms, self.colors):
            ratios = []
            point_sizes = []
            
            for exp in study_results.experiments:
                if (baseline_algorithm in exp.results and alg_name in exp.results):
                    baseline_wiener = exp.results[baseline_algorithm].wiener_index
                    alg_wiener = exp.results[alg_name].wiener_index
                    if baseline_wiener > 0:
                        ratio = alg_wiener / baseline_wiener
                        ratios.append(ratio)
                        point_sizes.append(exp.n_points)
            
            if ratios:
                plt.scatter(point_sizes, ratios, 
                          label=alg_name.replace('_', ' ').title(),
                          color=color, alpha=0.6, s=50)
        
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Optimal')
        plt.xlabel('Number of Points')
        plt.ylabel('Approximation Ratio')
        plt.title(f'Approximation Ratios (vs {baseline_algorithm.replace("_", " ").title()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def create_comprehensive_report(self, study_results: 'StudyResults',
                                   output_dir: str = "visualization_report") -> None:
        """
        Create a comprehensive visualization report with multiple plots.
        
        Args:
            study_results: StudyResults to visualize
            output_dir: Directory to save all plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate all plots
        self.plot_study_results(study_results, 
                               save_path=str(output_path / "study_overview.png"))
        
        self.plot_approximation_ratios(study_results,
                                     save_path=str(output_path / "approximation_ratios.png"))
        
        # Plot individual interesting cases
        self._plot_interesting_cases(study_results, output_path)
        
        print(f"Comprehensive visualization report saved to {output_path}")

    def _plot_interesting_cases(self, study_results: 'StudyResults', 
                               output_path: Path) -> None:
        """Plot interesting cases (best and worst performance)."""
        algorithms = list(study_results.statistics.keys())
        if len(algorithms) < 2:
            return
            
        # Find best and worst cases for comparison
        best_case = None
        worst_case = None
        best_ratio = float('inf')
        worst_ratio = 0
        
        for exp in study_results.experiments:
            if len(exp.results) >= 2:
                # Calculate approximation ratio between first two algorithms
                alg_names = list(exp.results.keys())
                alg1_wiener = exp.results[alg_names[0]].wiener_index
                alg2_wiener = exp.results[alg_names[1]].wiener_index
                
                if alg1_wiener > 0:
                    ratio = max(alg1_wiener, alg2_wiener) / min(alg1_wiener, alg2_wiener)
                    
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_case = exp
                    
                    if ratio > worst_ratio:
                        worst_ratio = ratio
                        worst_case = exp
        
        # Plot best and worst cases
        if best_case:
            self.visualize_algorithm_comparison(
                best_case, 
                save_path=str(output_path / f"best_case_n{best_case.n_points}_seed{best_case.seed}.png")
            )
        
        if worst_case and worst_case != best_case:
            self.visualize_algorithm_comparison(
                worst_case,
                save_path=str(output_path / f"worst_case_n{worst_case.n_points}_seed{worst_case.seed}.png")
            )

    # Legacy compatibility methods
    def visualize_comparison_cases(self, points: List[Point],
                                   dc_path: List[Point],
                                   optimal_path: Optional[List[Point]] = None,
                                   dc_wiener: Optional[float] = None,
                                   optimal_wiener: Optional[float] = None,
                                   approximation_ratio: Optional[float] = None) -> None:
        """Legacy method for compatibility."""
        self.visualize_comparison(points, dc_path, optimal_path, dc_wiener, optimal_wiener, approximation_ratio)

    def visualize_comparison(self, points: List[Point],
                             dc_path: List[Point],
                             optimal_path: Optional[List[Point]] = None,
                             dc_wiener: Optional[float] = None,
                             optimal_wiener: Optional[float] = None,
                             approximation_ratio: Optional[float] = None) -> None:
        """Legacy comparison visualization method."""
        plt.figure(figsize=(15, 5))
        
        # Plot divide and conquer result
        plt.subplot(1, 3 if optimal_path else 2, 1)
        self._plot_single_path(points, dc_path, 
                              title=f"Divide & Conquer\nWiener: {dc_wiener:.3f}" if dc_wiener else "Divide & Conquer")
        
        # Plot optimal result if available
        if optimal_path:
            plt.subplot(1, 3, 2)
            self._plot_single_path(points, optimal_path,
                                  title=f"Optimal\nWiener: {optimal_wiener:.3f}" if optimal_wiener else "Optimal")
            
            # Plot comparison
            plt.subplot(1, 3, 3)
            plt.axis('off')
            comparison_text = "Comparison:\n"
            if dc_wiener and optimal_wiener:
                comparison_text += f"D&C Wiener: {dc_wiener:.3f}\n"
                comparison_text += f"Optimal Wiener: {optimal_wiener:.3f}\n"
            if approximation_ratio:
                comparison_text += f"Approximation Ratio: {approximation_ratio:.3f}"
            plt.text(0.5, 0.5, comparison_text, ha='center', va='center', 
                    fontsize=12, transform=plt.gca().transAxes)
        else:
            plt.subplot(1, 2, 2)
            plt.axis('off')
            info_text = "Algorithm Information:\n"
            if dc_wiener:
                info_text += f"Wiener Index: {dc_wiener:.3f}"
            plt.text(0.5, 0.5, info_text, ha='center', va='center',
                    fontsize=12, transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.show()

    def _plot_single_path(self, points: List[Point], path: List[Point], title: str = ""):
        """Helper method to plot a single path."""
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        plt.scatter(x_coords, y_coords, c='blue', s=50, alpha=0.7, zorder=3)
        
        if path:
            path_x = [p.x for p in path]
            path_y = [p.y for p in path]
            plt.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.8, zorder=2)
            
            # Mark start and end
            plt.scatter(path[0].x, path[0].y, c='green', s=100, marker='s', zorder=4)
            plt.scatter(path[-1].x, path[-1].y, c='red', s=100, marker='^', zorder=4)
        
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

    # Enhanced methods for the orchestrator
    def plot_algorithm_comparison(self, study_results: 'StudyResults', 
                                save_path: Optional[str] = None) -> None:
        """Plot algorithm comparison for orchestrator compatibility."""
        self.plot_study_results(study_results, save_path)

    def plot_execution_times(self, study_results: 'StudyResults',
                           save_path: Optional[str] = None) -> None:
        """Plot execution times for orchestrator compatibility."""
        algorithms = list(study_results.statistics.keys())
        plt.figure(figsize=self.figsize)
        
        # Create execution time comparison
        alg_names = []
        times = []
        time_stds = []
        
        for alg_name, stats in study_results.statistics.items():
            alg_names.append(alg_name.replace('_', ' ').title())
            times.append(stats.execution_time_mean)
            time_stds.append(stats.execution_time_std)
        
        plt.bar(alg_names, times, yerr=time_stds, capsize=5, 
               color=self.colors[:len(alg_names)], alpha=0.7)
        plt.xlabel('Algorithm')
        plt.ylabel('Mean Execution Time (seconds)')
        plt.title('Algorithm Execution Time Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
