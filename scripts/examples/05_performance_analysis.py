#!/usr/bin/env python3
"""
Example 05: Performance Analysis
================================

This example demonstrates detailed performance analysis capabilities,
including benchmarking, timing analysis, and scalability measurement.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import gc
import threading
import resource
from collections import defaultdict
import numpy as np

from generators.point_generator import PointGenerator
from solvers.brute_force_solver import BruteForceSolver
from solvers.divide_conquer_solver import DivideConquerSolver
from visualization.visualizer import Visualizer
from utils.logger_setup import setup_logger
from utils.data_manager import DataManager

class PerformanceProfiler:
    """Simple performance profiling for algorithm analysis."""
    
    def __init__(self, logger):
        self.logger = logger
        self.profiles = {}
        
    def profile_execution(self, func, *args, **kwargs):
        """Profile function execution with timing and basic metrics."""
        # Memory before (using resource module - Unix only)
        try:
            mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except:
            mem_before = 0
        
        # Execute with timing
        gc.collect()  # Clean garbage before timing
        start_time = time.perf_counter()
        start_cpu = time.process_time()
        
        result = func(*args, **kwargs)
        
        end_cpu = time.process_time()
        end_time = time.perf_counter()
        
        # Memory after
        try:
            mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except:
            mem_after = 0
        
        profile = {
            'wall_time': end_time - start_time,
            'cpu_time': end_cpu - start_cpu,
            'memory_before_kb': mem_before,
            'memory_after_kb': mem_after,
            'memory_delta_kb': mem_after - mem_before,
            'result': result
        }
        
        return profile

def benchmark_point_generation(generator, logger, num_iterations=100):
    """Benchmark point generation methods."""
    logger.info("Benchmarking point generation methods...")
    
    generation_benchmarks = {}
    sizes = [5, 8, 10, 12, 15]
    
    for size in sizes:
        size_benchmarks = {}
        
        # Convex position generation
        start_time = time.perf_counter()
        for i in range(num_iterations):
            points = generator.generate_convex_position_points(size, seed=i)
        convex_time = (time.perf_counter() - start_time) / num_iterations
        size_benchmarks['convex_generation'] = convex_time
        
        # General position generation  
        start_time = time.perf_counter()
        for i in range(num_iterations):
            points = generator.generate_general_position_points(size, seed=i)
        general_time = (time.perf_counter() - start_time) / num_iterations
        size_benchmarks['general_generation'] = general_time
        
        generation_benchmarks[size] = size_benchmarks
        
        logger.info(f"  n={size}: Convex {convex_time:.6f}s, General {general_time:.6f}s")
    
    return generation_benchmarks

def analyze_algorithm_complexity(performance_data, algorithm_name, logger):
    """Analyze time complexity from performance data."""
    sizes = []
    times = []
    
    for size, profiles in performance_data.items():
        avg_time = np.mean([p['wall_time'] for p in profiles])
        sizes.append(size)
        times.append(avg_time)
    
    # Fit different complexity models
    complexity_fits = {}
    
    if len(sizes) >= 3:
        # Linear O(n)
        try:
            linear_coeffs = np.polyfit(sizes, times, 1)
            linear_pred = np.polyval(linear_coeffs, sizes)
            linear_r2 = 1 - np.sum((times - linear_pred)**2) / np.sum((times - np.mean(times))**2)
            complexity_fits['O(n)'] = {'r_squared': linear_r2, 'coeffs': linear_coeffs}
        except:
            complexity_fits['O(n)'] = {'r_squared': 0, 'coeffs': None}
        
        # Quadratic O(n²)
        try:
            quad_coeffs = np.polyfit(sizes, times, 2)
            quad_pred = np.polyval(quad_coeffs, sizes)
            quad_r2 = 1 - np.sum((times - quad_pred)**2) / np.sum((times - np.mean(times))**2)
            complexity_fits['O(n²)'] = {'r_squared': quad_r2, 'coeffs': quad_coeffs}
        except:
            complexity_fits['O(n²)'] = {'r_squared': 0, 'coeffs': None}
        
        # Cubic O(n³)
        try:
            cubic_coeffs = np.polyfit(sizes, times, 3)
            cubic_pred = np.polyval(cubic_coeffs, sizes)
            cubic_r2 = 1 - np.sum((times - cubic_pred)**2) / np.sum((times - np.mean(times))**2)
            complexity_fits['O(n³)'] = {'r_squared': cubic_r2, 'coeffs': cubic_coeffs}
        except:
            complexity_fits['O(n³)'] = {'r_squared': 0, 'coeffs': None}
        
        # n log n
        try:
            log_times = [t for s, t in zip(sizes, times)]
            log_sizes = [s * np.log(s) for s in sizes]
            nlogn_coeffs = np.polyfit(log_sizes, log_times, 1)
            nlogn_pred = np.polyval(nlogn_coeffs, log_sizes)
            nlogn_r2 = 1 - np.sum((times - nlogn_pred)**2) / np.sum((times - np.mean(times))**2)
            complexity_fits['O(n log n)'] = {'r_squared': nlogn_r2, 'coeffs': nlogn_coeffs}
        except:
            complexity_fits['O(n log n)'] = {'r_squared': 0, 'coeffs': None}
    
    # Find best fit
    best_complexity = max(complexity_fits.keys(), 
                         key=lambda k: complexity_fits[k]['r_squared'])
    
    logger.info(f"{algorithm_name} complexity analysis:")
    for complexity, fit in complexity_fits.items():
        logger.info(f"  {complexity}: R² = {fit['r_squared']:.4f}")
    logger.info(f"  Best fit: {best_complexity} (R² = {complexity_fits[best_complexity]['r_squared']:.4f})")
    
    return {
        'fits': complexity_fits,
        'best_complexity': best_complexity,
        'best_r_squared': complexity_fits[best_complexity]['r_squared']
    }

def main():
    """Demonstrate comprehensive performance analysis."""
    
    # Setup
    logger = setup_logger("performance_analysis_example")
    visualizer = Visualizer(logger=logger)
    data_manager = DataManager()
    profiler = PerformanceProfiler(logger)
    
    logger.info("=" * 60)
    logger.info("Performance Analysis Example")
    logger.info("=" * 60)
    
    # Initialize algorithms
    algorithms = {
        'Brute Force': BruteForceSolver(),
        'Divide & Conquer': DivideConquerSolver()
    }
    
    generator = PointGenerator()
    
    # 1. DETAILED EXECUTION PROFILING
    logger.info("\n🔍 DETAILED EXECUTION PROFILING")
    logger.info("=" * 40)
    
    profile_results = {}
    test_sizes = [5, 6, 7, 8, 9, 10]
    
    for alg_name, algorithm in algorithms.items():
        logger.info(f"\nProfiling {alg_name}...")
        alg_profiles = {}
        
        for size in test_sizes:
            size_profiles = []
            
            for trial in range(5):  # Multiple trials for statistical accuracy
                points = generator.generate_convex_position_points(size, seed=trial*10)
                
                profile = profiler.profile_execution(algorithm.solve, points)
                size_profiles.append(profile)
            
            alg_profiles[size] = size_profiles
            
            # Log average metrics
            avg_wall_time = np.mean([p['wall_time'] for p in size_profiles])
            avg_cpu_time = np.mean([p['cpu_time'] for p in size_profiles])
            avg_memory = np.mean([p['memory_delta_mb'] for p in size_profiles])
            
            logger.info(f"  n={size}: Wall {avg_wall_time:.4f}s, "
                       f"CPU {avg_cpu_time:.4f}s, Mem {avg_memory:.2f}MB")
        
        profile_results[alg_name] = alg_profiles
    
    # 2. MEMORY STRESS TESTING
    logger.info("\n💾 MEMORY STRESS TESTING")
    logger.info("=" * 30)
    
    memory_results = {}
    for alg_name, algorithm in algorithms.items():
        if alg_name == 'Brute Force':
            # Limit Brute Force to smaller sizes due to O(n³) complexity
            max_size = 10
        else:
            max_size = 12
            
        memory_results[alg_name] = run_memory_stress_test(
            algorithm, max_size=max_size, logger=logger
        )
    
    # 3. SCALABILITY ANALYSIS
    logger.info("\n📈 SCALABILITY ANALYSIS")
    logger.info("=" * 30)
    
    scalability_analysis = {}
    
    for alg_name in algorithms.keys():
        sizes = []
        times = []
        
        for size, profiles in profile_results[alg_name].items():
            avg_time = np.mean([p['wall_time'] for p in profiles])
            sizes.append(size)
            times.append(avg_time)
        
        # Fit polynomial to determine complexity
        if len(sizes) >= 3:
            # Try different polynomial degrees
            fits = {}
            for degree in [1, 2, 3]:
                try:
                    coeffs = np.polyfit(sizes, times, degree)
                    poly = np.poly1d(coeffs)
                    r_squared = 1 - (np.sum((times - poly(sizes))**2) / 
                                   np.sum((times - np.mean(times))**2))
                    fits[degree] = {'coeffs': coeffs, 'r_squared': r_squared}
                except:
                    fits[degree] = {'coeffs': None, 'r_squared': 0}
            
            # Choose best fit
            best_degree = max(fits.keys(), key=lambda k: fits[k]['r_squared'])
            best_fit = fits[best_degree]
            
            scalability_analysis[alg_name] = {
                'sizes': sizes,
                'times': times,
                'best_degree': best_degree,
                'best_r_squared': best_fit['r_squared'],
                'complexity_estimate': f"O(n^{best_degree})" if best_degree > 0 else "O(1)"
            }
            
            logger.info(f"{alg_name}:")
            logger.info(f"  Best fit: {scalability_analysis[alg_name]['complexity_estimate']}")
            logger.info(f"  R²: {best_fit['r_squared']:.4f}")
    
    # 4. BOTTLENECK IDENTIFICATION
    logger.info("\n🔧 BOTTLENECK IDENTIFICATION")
    logger.info("=" * 35)
    
    bottleneck_analysis = {}
    
    for alg_name, alg_profiles in profile_results.items():
        # Analyze where time is spent
        wall_times = []
        cpu_times = []
        memory_deltas = []
        
        for size_profiles in alg_profiles.values():
            for profile in size_profiles:
                wall_times.append(profile['wall_time'])
                cpu_times.append(profile['cpu_time'])
                memory_deltas.append(profile['memory_delta_mb'])
        
        cpu_efficiency = np.mean([c/w for c, w in zip(cpu_times, wall_times) if w > 0])
        memory_efficiency = np.std(memory_deltas) / (np.mean(memory_deltas) + 1e-6)
        
        bottleneck_analysis[alg_name] = {
            'cpu_efficiency': cpu_efficiency,
            'memory_variability': memory_efficiency,
            'primary_bottleneck': 'CPU' if cpu_efficiency > 0.8 else 'I/O or Memory'
        }
        
        logger.info(f"{alg_name}:")
        logger.info(f"  CPU Efficiency: {cpu_efficiency:.2f}")
        logger.info(f"  Memory Variability: {memory_efficiency:.2f}")
        logger.info(f"  Primary Bottleneck: {bottleneck_analysis[alg_name]['primary_bottleneck']}")
    
    # 5. PERFORMANCE OPTIMIZATION RECOMMENDATIONS
    logger.info("\n⚡ OPTIMIZATION RECOMMENDATIONS")
    logger.info("=" * 40)
    
    for alg_name, analysis in bottleneck_analysis.items():
        logger.info(f"\n{alg_name}:")
        
        if analysis['cpu_efficiency'] < 0.5:
            logger.info("  • Consider parallel processing or algorithmic improvements")
        
        if analysis['memory_variability'] > 1.0:
            logger.info("  • Memory usage is inconsistent - consider memory pooling")
        
        if alg_name == 'Brute Force':
            logger.info("  • O(n³) complexity limits scalability")
            logger.info("  • Consider switching to Divide & Conquer for n > 8")
        
        if alg_name == 'Divide & Conquer':
            logger.info("  • Good scalability characteristics")
            logger.info("  • Consider optimizing recursion depth for very large inputs")
    
    # 6. GENERATE PERFORMANCE VISUALIZATIONS
    logger.info("\n📊 GENERATING PERFORMANCE VISUALIZATIONS")
    logger.info("=" * 45)
    
    # Execution time comparison
    visualizer.plot_performance_analysis(
        profile_results,
        title="Algorithm Performance Analysis",
        save_path="examples_output/05_performance_analysis.png"
    )
    
    # Memory usage patterns
    visualizer.plot_memory_usage(
        memory_results,
        title="Memory Usage Patterns",
        save_path="examples_output/05_memory_analysis.png"
    )
    
    # Scalability visualization
    visualizer.plot_scalability_analysis(
        scalability_analysis,
        title="Algorithm Scalability Analysis",
        save_path="examples_output/05_scalability_analysis.png"
    )
    
    # 7. BENCHMARK COMPARISON REPORT
    logger.info("\n📋 GENERATING BENCHMARK REPORT")
    logger.info("=" * 35)
    
    # Point generation benchmarks
    generation_benchmarks = benchmark_algorithm_variants(logger)
    
    # Compile comprehensive report
    performance_report = {
        'execution_profiles': profile_results,
        'memory_analysis': memory_results,
        'scalability_analysis': scalability_analysis,
        'bottleneck_analysis': bottleneck_analysis,
        'generation_benchmarks': generation_benchmarks,
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'platform': os.name
        }
    }
    
    # Save detailed report
    report_path = "examples_output/05_performance_report.json"
    data_manager.save_analysis_results(performance_report, report_path)
    
    logger.info(f"Performance report saved to: {report_path}")
    
    # Export performance data for external analysis
    csv_exports = data_manager.export_performance_data(
        performance_report,
        output_dir="examples_output/05_performance_csv"
    )
    
    logger.info("Performance CSV exports:")
    for export_name, path in csv_exports.items():
        logger.info(f"  {export_name}: {path}")
    
    logger.info("\n✅ Performance analysis complete!")
    logger.info("=" * 40)
    logger.info("📁 Results saved to examples_output/05_*")
    logger.info("\nKey Performance Insights:")
    logger.info("• Detailed execution profiling with CPU, memory, and timing metrics")
    logger.info("• Memory stress testing to identify usage patterns")
    logger.info("• Scalability analysis with complexity estimation")
    logger.info("• Bottleneck identification for optimization guidance")
    logger.info("• System-specific benchmarks and recommendations")
    logger.info("\nThis analysis provides the foundation for performance")
    logger.info("optimization and algorithm selection decisions.")

if __name__ == "__main__":
    main()
