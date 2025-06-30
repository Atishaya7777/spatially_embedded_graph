import json
import pickle
import os
from datetime import datetime
from typing import List, Dict, Any
import logging

from core.point import Point
from analysis.statistical_analyzer import ExperimentResult


class DataManager:
    """Handles saving and loading of experiment data and interesting cases."""

    def __init__(self, output_dir: str = "wiener_analysis_data"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def save_experiment_results(self, results: List[ExperimentResult],
                                experiment_name: str = "experiment") -> tuple[str, str]:
        """Save experiment results to JSON and pickle files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare JSON-serializable data
        json_data = []
        for result in results:
            json_result = {
                'n_points': result.n_points,
                'seed': result.seed,
                'points': [p.to_dict() for p in result.points],
                'dc_path': [p.to_dict() for p in result.dc_path],
                'optimal_path': [p.to_dict() for p in result.optimal_path] if result.optimal_path else None,
                'dc_wiener': result.dc_wiener,
                'optimal_wiener': result.optimal_wiener,
                'approximation_ratio': result.approximation_ratio,
                'dc_time': result.dc_time,
                'optimal_time': result.optimal_time
            }
            json_data.append(json_result)

        # Save as JSON
        json_file = os.path.join(
            self.output_dir, f"{experiment_name}_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)

        # Save as pickle
        pickle_file = os.path.join(
            self.output_dir, f"{experiment_name}_{timestamp}.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump(results, f)

        self.logger.info(f"Saved {len(results)} experiment results to:")
        self.logger.info(f"  JSON: {json_file}")
        self.logger.info(f"  Pickle: {pickle_file}")

        return json_file, pickle_file

    def load_experiment_results(self, file_path: str) -> List[ExperimentResult]:
        """Load experiment results from file."""
        if file_path.endswith('.json'):
            return self._load_from_json(file_path)
        elif file_path.endswith('.pkl'):
            return self._load_from_pickle(file_path)
        else:
            raise ValueError("File must be either .json or .pkl")

    def _load_from_json(self, file_path: str) -> List[ExperimentResult]:
        """Load experiment results from JSON file."""
        with open(file_path, 'r') as f:
            json_data = json.load(f)

        results = []
        for json_result in json_data:
            # Convert point dictionaries back to Point objects
            points = [Point.from_dict(p) for p in json_result['points']]
            dc_path = [Point.from_dict(p) for p in json_result['dc_path']]
            optimal_path = ([Point.from_dict(p) for p in json_result['optimal_path']]
                            if json_result['optimal_path'] else None)

            result = ExperimentResult(
                n_points=json_result['n_points'],
                seed=json_result['seed'],
                points=points,
                dc_path=dc_path,
                optimal_path=optimal_path,
                dc_wiener=json_result['dc_wiener'],
                optimal_wiener=json_result['optimal_wiener'],
                approximation_ratio=json_result['approximation_ratio'],
                dc_time=json_result['dc_time'],
                optimal_time=json_result['optimal_time']
            )
            results.append(result)

        return results

    def _load_from_pickle(self, file_path: str) -> List[ExperimentResult]:
        """Load experiment results from pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save_interesting_cases(self, interesting_cases: List[Dict[str, Any]],
                               filename_prefix: str = "interesting_cases") -> tuple[str, str]:
        """Save interesting cases to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert to JSON-serializable format
        json_cases = []
        for case in interesting_cases:
            json_case = case.copy()

            # Convert Point objects to dictionaries
            if 'points' in case:
                json_case['points'] = [p.to_dict() for p in case['points']]
            if 'dc_path' in case:
                json_case['dc_path'] = [p.to_dict() for p in case['dc_path']]
            if 'optimal_path' in case and case['optimal_path']:
                json_case['optimal_path'] = [p.to_dict()
                                             for p in case['optimal_path']]

            json_cases.append(json_case)

        # Save as JSON
        json_file = os.path.join(
            self.output_dir, f"{filename_prefix}_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(json_cases, f, indent=2)

        # Save as pickle
        pickle_file = os.path.join(
            self.output_dir, f"{filename_prefix}_{timestamp}.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump(interesting_cases, f)

        self.logger.info(
            f"Saved {len(interesting_cases)} interesting cases to:")
        self.logger.info(f"  JSON: {json_file}")
        self.logger.info(f"  Pickle: {pickle_file}")

        return json_file, pickle_file

    def load_interesting_cases(self, file_path: str) -> List[Dict[str, Any]]:
        """Load interesting cases from file."""
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                json_cases = json.load(f)

            # Convert dictionaries back to Point objects
            cases = []
            for json_case in json_cases:
                case = json_case.copy()

                if 'points' in json_case:
                    case['points'] = [Point.from_dict(
                        p) for p in json_case['points']]
                if 'dc_path' in json_case:
                    case['dc_path'] = [Point.from_dict(
                        p) for p in json_case['dc_path']]
                if 'optimal_path' in json_case and json_case['optimal_path']:
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

    def get_latest_file(self, pattern: str) -> str:
        """Get the most recent file matching the pattern."""
        import glob

        files = glob.glob(os.path.join(self.output_dir, f"{pattern}*"))
        if not files:
            raise FileNotFoundError(
                f"No files found matching pattern: {pattern}")

        # Sort by modification time and return the latest
        latest_file = max(files, key=os.path.getmtime)
        return latest_file

    def list_files(self, pattern: str = "*") -> List[str]:
        """List all files in the output directory matching the pattern."""
        import glob

        files = glob.glob(os.path.join(self.output_dir, pattern))
        return sorted(files, key=os.path.getmtime, reverse=True)

    def export_summary_csv(self, results_by_points: Dict[int, List[ExperimentResult]],
                           filename: str = "experiment_summary.csv") -> str:
        """Export experiment summary to CSV file."""
        import csv

        csv_file = os.path.join(self.output_dir, filename)

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            header = [
                'n_points', 'num_experiments',
                'dc_wiener_mean', 'dc_wiener_std', 'dc_wiener_min', 'dc_wiener_max',
                'optimal_wiener_mean', 'optimal_wiener_std', 'optimal_wiener_min', 'optimal_wiener_max',
                'ratio_mean', 'ratio_std', 'ratio_min', 'ratio_max',
                'dc_time_mean', 'optimal_time_mean',
                'perfect_solutions', 'good_solutions', 'poor_solutions'
            ]
            writer.writerow(header)

            # Write data for each point count
            for n_points, results in sorted(results_by_points.items()):
                if not results:
                    continue

                # Calculate statistics
                from analysis.statistical_analyzer import StatisticalAnalyzer
                analyzer = StatisticalAnalyzer()
                stats = analyzer.calculate_statistics(results)

                row = [
                    stats.n_points, stats.num_experiments,
                    stats.dc_wiener_mean, stats.dc_wiener_std, stats.dc_wiener_min, stats.dc_wiener_max,
                    stats.optimal_wiener_mean, stats.optimal_wiener_std,
                    stats.optimal_wiener_min, stats.optimal_wiener_max,
                    stats.ratio_mean, stats.ratio_std, stats.ratio_min, stats.ratio_max,
                    stats.dc_time_mean, stats.optimal_time_mean,
                    stats.perfect_solutions, stats.good_solutions, stats.poor_solutions
                ]
                writer.writerow(row)

        self.logger.info(f"Exported summary to CSV: {csv_file}")
        return csv_file
