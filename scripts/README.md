# Spatially Embedded Graph - Wiener Index Analysis

This project provides a modular, decoupled architecture for analyzing Wiener indices of Hamiltonian paths in spatially embedded graphs.

## Architecture Overview

The codebase has been refactored into a clean, modular architecture with proper separation of concerns:

```
scripts/
├── core/                          # Core data structures and calculations
│   ├── point.py                   # Point class and utilities
│   └── wiener_index_calculator.py # Wiener index computation
├── generators/                    # Point set generation
│   └── point_generator.py         # Various point set generators
├── solvers/                       # Algorithm implementations
│   ├── brute_force_solver.py      # Exhaustive search solver
│   └── divide_conquer_solver.py   # Divide & conquer solver
├── analysis/                      # Statistical analysis and comparison
│   ├── comparison_analyzer.py     # Algorithm comparison framework
│   └── statistical_analyzer.py   # Statistical analysis tools
├── visualization/                 # Plotting and visualization
│   └── visualizer.py             # Path and result visualization
├── utils/                         # Utility modules
│   ├── data_manager.py           # Data saving/loading
│   └── logger_setup.py           # Logging configuration
├── hamiltonian_wiener.py         # Main orchestration module
└── main_demo.py                  # Demo and examples
```

## Key Features

### 1. Modular Design
- **Core**: Fundamental data structures (`Point`, `WienerIndexCalculator`)
- **Generators**: Pluggable point set generators (convex, random, grid, circular)
- **Solvers**: Algorithm implementations with consistent interfaces
- **Analysis**: Statistical analysis and comparison frameworks
- **Visualization**: Comprehensive plotting and visualization tools
- **Utils**: Data management and logging utilities

### 2. Algorithm Implementations
- **Divide & Conquer**: Fast heuristic with bbox and median bisection variants
- **Brute Force**: Exhaustive search with parallel processing support
- **Extensible**: Easy to add new algorithms through consistent interfaces

### 3. Analysis Capabilities
- **Single Algorithm Analysis**: Performance profiling of individual algorithms
- **Comparative Analysis**: Head-to-head algorithm comparisons
- **Statistical Studies**: Large-scale statistical analysis across multiple parameters
- **Interesting Case Detection**: Automatic identification of best/worst/median cases

### 4. Visualization
- **Path Visualization**: Display Hamiltonian paths with point ordering
- **Comparison Plots**: Side-by-side algorithm comparison
- **Statistical Plots**: Performance analysis and approximation ratio distributions
- **Convex Hull Display**: Show point set geometry

## Usage

### Quick Start

```python
from hamiltonian_wiener import WienerAnalysisOrchestrator

# Initialize orchestrator
orchestrator = WienerAnalysisOrchestrator()

# Run targeted study
results = orchestrator.run_targeted_study(
    is_convex=False,
    points_counts=[6, 7, 8],
    num_seeds=50,
    visualize_interesting=True
)
```

### Individual Components

```python
# Point generation
from generators.point_generator import PointGenerator
generator = PointGenerator()
points = generator.generate_convex_hull_points(8, seed=42)

# Algorithm execution
from solvers.divide_conquer_solver import DivideConquerSolver
from solvers.brute_force_solver import BruteForceSolver

dc_solver = DivideConquerSolver()
bf_solver = BruteForceSolver()

dc_path = dc_solver.solve(points)
bf_path = bf_solver.solve_simple(points)

# Analysis
from analysis.statistical_analyzer import StatisticalAnalyzer
analyzer = StatisticalAnalyzer()
result = analyzer.run_single_experiment(points, dc_solver, "divide_conquer")

# Visualization
from visualization.visualizer import Visualizer
visualizer = Visualizer()
visualizer.visualize_comparison(points, dc_path, bf_path)
```

### Running Analysis

1. **Demo and Examples**:
   ```bash
   python main_demo.py
   ```

2. **Full Analysis**:
   ```bash
   python hamiltonian_wiener.py
   ```

3. **Using Make**:
   ```bash
   make run          # Run main analysis
   make setup        # Set up environment
   make clean        # Clean generated files
   ```

## Configuration

### Point Generation Types
- `convex`: Points forming a convex hull
- `random`: Randomly distributed points
- `grid`: Grid-based point placement
- `circular`: Points on circle/ellipse

### Algorithm Parameters
- **Divide & Conquer**: `max_depth`, `base_case_size`, `use_median_bisection`
- **Brute Force**: `use_parallel`, `max_workers`

### Study Parameters
- `point_counts`: List of point set sizes to analyze
- `num_seeds`: Number of random seeds per size
- `visualize_interesting`: Enable/disable visualization
- `is_convex`: Use convex vs. random point sets

## Output

### Data Files
- **JSON**: Human-readable experiment results
- **Pickle**: Python object serialization for detailed analysis
- **Logs**: Detailed execution logs with timestamps

### Visualizations
- **Path plots**: Hamiltonian paths with point ordering
- **Comparison plots**: Algorithm performance side-by-side
- **Statistical summaries**: Approximation ratio distributions

### Statistics
- **Approximation ratios**: D&C vs. optimal performance
- **Execution times**: Algorithm runtime comparisons
- **Solution quality**: Perfect/good/poor solution counts
- **Speedup factors**: Performance improvements

## Extensions

### Adding New Algorithms
1. Implement solver with `solve(points: List[Point]) -> List[Point]` interface
2. Add to orchestrator solver list
3. Update analysis configurations

### Adding New Point Generators
1. Add method to `PointGenerator` class
2. Update `generate_points()` dispatcher
3. Add configuration options

### Adding New Analysis Types
1. Extend `StatisticalAnalyzer` with new methods
2. Update orchestrator to call new analysis
3. Add visualization support if needed

## Dependencies

- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `scipy`: Scientific computing (convex hulls)
- `typing`: Type hints for better code quality

## Performance Notes

- **Brute force**: Limited to ~10 points due to factorial complexity
- **Divide & conquer**: Scales to 100+ points efficiently
- **Parallel processing**: Automatic for brute force on larger sets
- **Memory management**: Careful handling of large experiment sets

## Best Practices

1. **Use seeds**: For reproducible experiments
2. **Start small**: Test with few points/seeds before large studies
3. **Save results**: Use DataManager for experiment persistence
4. **Monitor memory**: Large studies can consume significant memory
5. **Use logging**: Enable detailed logging for debugging

This refactored architecture provides a clean, extensible foundation for Wiener index analysis while maintaining all original functionality in a more maintainable form.
