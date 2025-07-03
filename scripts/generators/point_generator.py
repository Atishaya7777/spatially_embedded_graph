import random
import numpy as np
from typing import List, Optional
from core.point import Point


class PointGenerator:
    """Generator for creating various point configurations."""

    def generate_points(self,
                        n: int,
                        point_type: str = "convex",
                        seed: Optional[int] = None,
                        **kwargs
                        ) -> List[Point]:
        """
        Generate a list of points based on the specified type.

        Args:
            n: Number of points to generate
            point_type: Type of point configuration ('convex', 'general', 'grid', 'circular')
            seed: Random seed for reproducibility
            **kwargs: Additional parameters for specific point types

        Returns:
            List of generated points
        """
        if point_type == "convex":
            return self.generate_convex_points(n, seed)
        elif point_type == "general":
            return self.generate_general_points(n, seed)
        elif point_type == "grid":
            return self.generate_grid_points(kwargs.get('rows', 5), kwargs.get('cols', 5), kwargs.get('spacing', 1.0))
        elif point_type == "circular":
            return self.generate_circular_points(n, kwargs.get('radius', 5.0), kwargs.get('center', (0, 0)))
        else:
            raise ValueError(f"Unknown point type: {point_type}")

    @staticmethod
    def generate_convex_points(n: int, seed: Optional[int] = None) -> List[Point]:
        """
        Generate n points that form a convex set with varying distances.

        Args:
            n: Number of points to generate
            seed: Random seed for reproducibility

        Returns:
            List of points forming a convex set
        """
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

    @staticmethod
    def generate_general_points(n: int, seed: Optional[int] = None) -> List[Point]:
        """
        Generate n random points in a general position.

        Args:
            n: Number of points to generate
            seed: Random seed for reproducibility

        Returns:
            List of randomly distributed points
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        points = []
        for i in range(n):
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)
            points.append(Point(x, y, i))

        return points

    @staticmethod
    def generate_grid_points(rows: int, cols: int, spacing: float = 1.0) -> List[Point]:
        """
        Generate points arranged in a grid pattern.

        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            spacing: Distance between adjacent points

        Returns:
            List of points in grid formation
        """
        points = []
        for i in range(rows):
            for j in range(cols):
                x = j * spacing
                y = i * spacing
                points.append(Point(x, y, i * cols + j))

        return points

    @staticmethod
    def generate_circular_points(n: int, radius: float = 5.0, center: tuple = (0, 0)) -> List[Point]:
        """
        Generate n points arranged in a circle.

        Args:
            n: Number of points to generate
            radius: Radius of the circle
            center: Center coordinates (x, y)

        Returns:
            List of points arranged in a circle
        """
        points = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points.append(Point(x, y, i))

        return points

    def generate_convex_hull_points(self, n: int, seed: Optional[int] = None) -> List[Point]:
        """
        Generate n points that form a convex hull configuration.
        Alias for generate_convex_points for compatibility.
        """
        return self.generate_convex_points(n, seed)

    def generate_random_points(self, n: int, seed: Optional[int] = None) -> List[Point]:
        """
        Generate n random points in general position.
        Alias for generate_general_points for compatibility.
        """
        return self.generate_general_points(n, seed)
