# Examples Directory

This directory contains comprehensive examples demonstrating the full power and capabilities of the refactored Wiener Index Analysis framework.

## Examples Overview

### 1. **Basic Usage Examples**
- `01_quick_start.py` - Basic framework usage and quick test
- `02_single_algorithm.py` - Running individual algorithms
- `03_algorithm_comparison.py` - Comparing multiple algorithms

### 2. **Advanced Analysis Examples**
- `04_comprehensive_study.py` - Large-scale comparative studies
- `05_performance_analysis.py` - Performance benchmarking and optimization
- `06_statistical_analysis.py` - Advanced statistical analysis

### 3. **Visualization Examples**
- `07_visualization_showcase.py` - Complete visualization capabilities
- `08_custom_plots.py` - Custom visualization and reporting
- `09_publication_ready.py` - Publication-quality figures

### 4. **Research Examples**
- `10_scalability_study.py` - Algorithm scalability analysis
- `11_approximation_analysis.py` - Approximation ratio studies
- `12_point_type_comparison.py` - Convex vs. general point analysis

### 5. **Integration Examples**
- `13_batch_processing.py` - Batch processing multiple datasets
- `14_parameter_sweep.py` - Parameter sensitivity analysis
- `15_custom_algorithm.py` - Adding new algorithms to the framework

### 6. **Real-World Applications**
- `16_research_workflow.py` - Complete research workflow example
- `17_reproducible_results.py` - Reproducible research practices
- `18_data_export.py` - Data export and external tool integration

## How to Run Examples

Each example is self-contained and can be run independently:

```bash
cd examples/
python3 01_quick_start.py
```

Or use the Makefile targets:
```bash
make run-examples    # Run all examples
make example-basic   # Run basic examples (01-03)
make example-advanced # Run advanced examples (04-06)
make example-viz     # Run visualization examples (07-09)
```

## Framework Architecture

These examples demonstrate the modular architecture:

```
Core Components:
├── generators/     - Point generation (convex, random, custom)
├── solvers/       - Algorithm implementations (extensible)
├── analysis/      - Statistical analysis and comparison
├── visualization/ - Enhanced plotting and reporting
├── utils/         - Data management and logging
└── Orchestrator   - Coordinates all components
```

## Requirements

All examples use the same dependencies as the main framework:
- numpy
- matplotlib
- scipy
- networkx
- (optional) tqdm for progress bars

## Contributing

To add new examples:
1. Follow the naming convention: `##_descriptive_name.py`
2. Include comprehensive docstrings
3. Add to this README
4. Update the Makefile if needed
