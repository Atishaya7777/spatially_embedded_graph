#!/usr/bin/env python3
"""
Example 04: Comprehensive Study
===============================

This example demonstrates how to conduct large-scale comparative studies
using the framework's orchestration capabilities, suitable for research
publications and comprehensive analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from hamiltonian_wiener import HamiltonianWienerAnalyzer
from utils.logger_setup import setup_logger
from utils.data_manager import DataManager
import json

def main():
    """Demonstrate comprehensive research study capabilities."""
    
    # Setup
    logger = setup_logger("comprehensive_study_example", level="INFO")
    data_manager = DataManager()
    
    logger.info("=" * 60)
    logger.info("Comprehensive Study Example")
    logger.info("=" * 60)
    
    # Study parameters - extensive configuration for research
    study_config = {
        "study_name": "comprehensive_wiener_analysis_2024",
        "algorithms": ["brute_force", "divide_conquer"],
        "point_types": ["convex", "general"],
        "sizes": list(range(5, 12)),  # n=5 to n=11
        "trials_per_config": 15,      # More trials for statistical significance
        "seeds": list(range(0, 150, 10)),  # 15 different seed ranges
        "analysis_depth": "comprehensive",
        "generate_plots": True,
        "save_intermediate": True,
        "statistical_tests": True
    }
    
    logger.info("Study Configuration:")
    for key, value in study_config.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize the main analyzer
    analyzer = HamiltonianWienerAnalyzer(
        logger=logger,
        enable_visualization=True,
        enable_analysis=True
    )
    
    # Run comprehensive study
    logger.info("\n🚀 LAUNCHING COMPREHENSIVE STUDY")
    logger.info("=" * 50)
    
    study_results = {}
    total_experiments = (len(study_config["algorithms"]) * 
                        len(study_config["point_types"]) * 
                        len(study_config["sizes"]) * 
                        study_config["trials_per_config"])
    
    logger.info(f"Total experiments to run: {total_experiments:,}")
    logger.info("This is a comprehensive study - it may take several minutes...")
    
    experiment_count = 0
    
    for point_type in study_config["point_types"]:
        logger.info(f"\n📊 ANALYZING {point_type.upper()} POINTS")
        logger.info("-" * 40)
        
        point_type_results = {}
        
        for algorithm in study_config["algorithms"]:
            logger.info(f"\n🔧 Testing {algorithm.replace('_', ' ').title()} Algorithm")
            
            # Run algorithm across all sizes
            results = analyzer.run_comparative_analysis(
                algorithms=[algorithm],
                sizes=study_config["sizes"],
                point_type=point_type,
                num_trials=study_config["trials_per_config"],
                seeds=study_config["seeds"][:study_config["trials_per_config"]],
                detailed_analysis=True
            )
            
            point_type_results[algorithm] = results[algorithm]
            
            # Log progress
            experiments_this_alg = len(study_config["sizes"]) * study_config["trials_per_config"]
            experiment_count += experiments_this_alg
            logger.info(f"  Completed {experiments_this_alg} experiments "
                       f"({experiment_count}/{total_experiments} total)")
        
        study_results[point_type] = point_type_results
        
        # Generate intermediate analysis
        logger.info(f"\n📈 Generating {point_type} analysis...")
        analyzer.generate_comprehensive_report(
            point_type_results,
            output_prefix=f"examples_output/04_study_{point_type}",
            include_visualizations=True,
            statistical_analysis=True
        )
    
    # Cross-algorithm analysis
    logger.info("\n🔍 CROSS-ALGORITHM ANALYSIS")
    logger.info("=" * 40)
    
    # Compare algorithms across point types
    cross_analysis = analyzer.perform_cross_analysis(
        study_results,
        algorithms=study_config["algorithms"],
        point_types=study_config["point_types"]
    )
    
    # Statistical significance testing
    logger.info("\n📊 STATISTICAL SIGNIFICANCE TESTING")
    logger.info("-" * 40)
    
    statistical_results = analyzer.perform_statistical_tests(
        study_results,
        significance_level=0.05
    )
    
    for comparison, test_result in statistical_results.items():
        logger.info(f"{comparison}:")
        logger.info(f"  p-value: {test_result['p_value']:.6f}")
        logger.info(f"  Significant: {'Yes' if test_result['significant'] else 'No'}")
        logger.info(f"  Effect size: {test_result['effect_size']:.4f}")
    
    # Performance scalability analysis
    logger.info("\n⚡ SCALABILITY ANALYSIS")
    logger.info("-" * 30)
    
    scalability_analysis = analyzer.analyze_scalability(
        study_results,
        algorithms=study_config["algorithms"]
    )
    
    for algorithm, scaling in scalability_analysis.items():
        logger.info(f"{algorithm.replace('_', ' ').title()}:")
        logger.info(f"  Complexity class: {scaling['complexity_class']}")
        logger.info(f"  Growth rate: {scaling['growth_rate']:.2f}")
        logger.info(f"  R² fit: {scaling['r_squared']:.4f}")
        logger.info(f"  Scalability rating: {scaling['scalability_rating']}/5")
    
    # Generate comprehensive study report
    logger.info("\n📋 GENERATING COMPREHENSIVE REPORT")
    logger.info("-" * 40)
    
    # Prepare report data
    report_data = {
        "study_metadata": {
            "study_name": study_config["study_name"],
            "total_experiments": total_experiments,
            "algorithms_tested": study_config["algorithms"],
            "point_types": study_config["point_types"],
            "size_range": f"{min(study_config['sizes'])}-{max(study_config['sizes'])}",
            "trials_per_config": study_config["trials_per_config"]
        },
        "results": study_results,
        "cross_analysis": cross_analysis,
        "statistical_tests": statistical_results,
        "scalability_analysis": scalability_analysis
    }
    
    # Save comprehensive data
    study_data_path = "examples_output/04_comprehensive_study_data.json"
    with open(study_data_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    logger.info(f"Study data saved to: {study_data_path}")
    
    # Generate publication-ready visualizations
    logger.info("\n🎨 GENERATING PUBLICATION-READY VISUALIZATIONS")
    logger.info("-" * 50)
    
    publication_plots = analyzer.generate_publication_figures(
        study_results,
        output_dir="examples_output/04_publication_figures",
        style="publication",
        formats=["png", "pdf", "svg"]
    )
    
    logger.info("Generated publication figures:")
    for plot_name, plot_path in publication_plots.items():
        logger.info(f"  {plot_name}: {plot_path}")
    
    # Generate executive summary
    logger.info("\n📄 EXECUTIVE SUMMARY")
    logger.info("=" * 30)
    
    summary = analyzer.generate_executive_summary(report_data)
    
    logger.info("\nKEY FINDINGS:")
    for finding in summary["key_findings"]:
        logger.info(f"• {finding}")
    
    logger.info("\nRECOMMENDations:")
    for recommendation in summary["recommendations"]:
        logger.info(f"• {recommendation}")
    
    logger.info(f"\nPERFORMANCE INSIGHTS:")
    for insight in summary["performance_insights"]:
        logger.info(f"• {insight}")
    
    # Save executive summary
    summary_path = "examples_output/04_executive_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nExecutive summary saved to: {summary_path}")
    
    # Data export for external tools
    logger.info("\n📤 EXPORTING DATA FOR EXTERNAL TOOLS")
    logger.info("-" * 40)
    
    # Export to CSV for R/MATLAB/Excel analysis
    csv_exports = data_manager.export_study_to_csv(
        study_results,
        output_dir="examples_output/04_csv_exports"
    )
    
    logger.info("CSV exports created:")
    for export_name, export_path in csv_exports.items():
        logger.info(f"  {export_name}: {export_path}")
    
    # Generate LaTeX tables for publication
    latex_tables = analyzer.generate_latex_tables(
        study_results,
        output_dir="examples_output/04_latex_tables"
    )
    
    logger.info("\nLaTeX tables generated:")
    for table_name, table_path in latex_tables.items():
        logger.info(f"  {table_name}: {table_path}")
    
    logger.info("\n✅ Comprehensive study complete!")
    logger.info("=" * 50)
    logger.info("📁 All results saved to examples_output/04_*")
    logger.info("\nStudy Deliverables:")
    logger.info("• Raw data and metadata (JSON)")
    logger.info("• Statistical analysis results")
    logger.info("• Publication-ready figures (PNG, PDF, SVG)")
    logger.info("• Executive summary and recommendations")
    logger.info("• CSV exports for external analysis")
    logger.info("• LaTeX tables for academic papers")
    logger.info("\nThis comprehensive study demonstrates the framework's capability")
    logger.info("for large-scale research projects with statistical rigor and")
    logger.info("publication-quality output.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive Wiener index study")
    parser.add_argument("--quick", action="store_true", 
                       help="Run a quicker version with fewer trials")
    parser.add_argument("--max-size", type=int, default=11,
                       help="Maximum point set size to test")
    
    args = parser.parse_args()
    
    # Adjust parameters for quick run
    if args.quick:
        # Override for demonstration purposes
        study_config = {
            "sizes": list(range(5, min(8, args.max_size + 1))),
            "trials_per_config": 5,
            "seeds": list(range(0, 50, 10))
        }
        print("Running in quick mode with reduced parameters for demonstration")
    
    main()
