import math
from typing import Dict, Any


class Point:
    """Represents a 2D point with coordinates and optional identifier."""

    def __init__(self, x: float, y: float, id: int = 0):
        """
        Initialize a Point.

        Args:
            x: X coordinate
            y: Y coordinate 
            id: Optional identifier for the point
        """
        self.x = x
        self.y = y
        self.id = id if id is not None else f"({x:.1f},{y:.1f})"

    def __repr__(self) -> str:
        """String representation of the point."""
        return f"Point({self.x:.1f}, {self.y:.1f})"

    def __eq__(self, other) -> bool:
        """Check equality with another point."""
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        """Make Point hashable for use in sets and as dict keys."""
        return hash((self.x, self.y))

    def distance_to(self, other: 'Point') -> float:
        """
        Calculate Euclidean distance to another point.

        Args:
            other: Another Point object

        Returns:
            Euclidean distance between the points
        """
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Point to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the point
        """
        return {'x': self.x, 'y': self.y, 'id': self.id}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Point':
        """
        Create Point from dictionary.

        Args:
            data: Dictionary containing x, y, and id keys

        Returns:
            Point object created from dictionary data
        """
        return cls(data['x'], data['y'], data['id'])

    @staticmethod
    def euclidean_distance(p1: 'Point', p2: 'Point') -> float:
        """
        Static method to calculate Euclidean distance between two points.

        Args:
            p1: First point
            p2: Second point

        Returns:
            Euclidean distance between the points
        """
        return p1.distance_to(p2)
