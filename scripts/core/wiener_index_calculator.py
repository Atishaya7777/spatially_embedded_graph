"""
Wiener index calculation utilities.
"""
from typing import List
from .point import Point


class WienerIndexCalculator:
    """Calculator for Wiener index of Hamiltonian paths."""

    @staticmethod
    def calculate_wiener_index(path: List[Point]) -> float:
        """
        Calculate the Wiener index of a Hamiltonian path.

        The Wiener index is the sum of distances between all pairs of vertices
        in the path, where distance is measured along the path edges.

        Args:
            path: List of points representing a Hamiltonian path

        Returns:
            Wiener index of the path
        """
        n = len(path)
        wiener_sum = 0.0

        # For each pair of vertices in the path
        for i in range(n):
            for j in range(i + 1, n):
                # Distance between vertices i and j in the path is the sum of edge weights
                path_distance = 0.0
                for k in range(i, j):
                    path_distance += Point.euclidean_distance(
                        path[k], path[k + 1])
                wiener_sum += path_distance

        return wiener_sum
