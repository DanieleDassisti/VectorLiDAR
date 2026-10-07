import numpy as np
from typing import List, Dict, Any, Tuple
from app.core.vector2d import ray_segment_intersection_vectorized

class Environment:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        # List of segments: each is a numpy array of shape (2, 2) [[x1, y1], [x2, y2]]
        self.segments: List[np.ndarray] = []
        self.default_walls()

    def clear(self):
        """Clear all obstacles, keeping only the boundary walls."""
        self.segments = []
        self.default_walls()

    def default_walls(self):
        """Create the bounding outer walls of the environment."""
        # Top, Right, Bottom, Left borders
        w, h = self.width, self.height
        self.add_wall(0, 0, w, 0)
        self.add_wall(w, 0, w, h)
        self.add_wall(w, h, 0, h)
        self.add_wall(0, h, 0, 0)

    def add_wall(self, x1: float, y1: float, x2: float, y2: float):
        """Add a custom line segment wall to the environment."""
        self.segments.append(np.array([[x1, y1], [x2, y2]], dtype=np.float64))

    def get_segments_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return starts and ends of all segments as numpy arrays.
        Useful for vectorized calculations.
        Returns:
            starts: shape (M, 2)
            ends: shape (M, 2)
        """
        M = len(self.segments)
        starts = np.zeros((M, 2))
        ends = np.zeros((M, 2))
        for idx, seg in enumerate(self.segments):
            starts[idx] = seg[0]
            ends[idx] = seg[1]
        return starts, ends

    def raycast(self, origin: Tuple[float, float], angles: np.ndarray, max_range: float) -> np.ndarray:
        """
        Cast multiple rays from an origin point at specific angles.
        Returns an array of hit distances.
        """
        if not self.segments:
            return np.full_like(angles, max_range)

        # origin array (2,)
        orig_arr = np.array(origin, dtype=np.float64)
        
        # direction unit vectors (N, 2)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
        # segment starts and ends (M, 2)
        starts, ends = self.get_segments_arrays()
        
        # Run the vectorized intersection code
        return ray_segment_intersection_vectorized(orig_arr, dirs, starts, ends, max_range)

    def load_preset(self, name: str):
        """Load a predefined map layout."""
        self.clear()
        w, h = self.width, self.height

        if name == "office":
            # Simple office layout with corridors and rooms
            # Main central corridor vertical wall
            self.add_wall(w * 0.4, 0, w * 0.4, h * 0.7)
            self.add_wall(w * 0.6, h * 0.3, w * 0.6, h)
            
            # Horizontal partitions
            self.add_wall(0, h * 0.3, w * 0.4, h * 0.3)
            self.add_wall(w * 0.6, h * 0.3, w, h * 0.3)
            self.add_wall(0, h * 0.65, w * 0.25, h * 0.65)
            self.add_wall(w * 0.6, h * 0.7, w * 0.85, h * 0.7)
            
            # Floating desks/columns (blocks)
            self.add_box_obstacle(w * 0.15, h * 0.1, 40, 40)
            self.add_box_obstacle(w * 0.8, h * 0.5, 50, 50)
            self.add_box_obstacle(w * 0.15, h * 0.8, 60, 40)
            
        elif name == "maze":
            # A classic robotic grid maze layout
            cols, rows = 6, 5
            cw, ch = w / cols, h / rows
            # Add selective grid walls to create a maze
            # Vertical walls
            self.add_wall(cw * 1, 0, cw * 1, ch * 2)
            self.add_wall(cw * 2, ch * 1, cw * 2, ch * 3)
            self.add_wall(cw * 3, ch * 2, cw * 3, h)
            self.add_wall(cw * 4, 0, cw * 4, ch * 2)
            self.add_wall(cw * 4, ch * 3, cw * 4, ch * 5)
            self.add_wall(cw * 5, ch * 1, cw * 5, ch * 4)

            # Horizontal walls
            self.add_wall(0, ch * 2, cw * 1, ch * 2)
            self.add_wall(cw * 1, ch * 1, cw * 3, ch * 1)
            self.add_wall(cw * 2, ch * 3, cw * 4, ch * 3)
            self.add_wall(cw * 1, ch * 4, cw * 3, ch * 4)
            self.add_wall(cw * 4, ch * 2, w, ch * 2)

        elif name == "arena":
            # An open yard with multiple round/box obstacles
            # Middle circle-like polygon obstacle
            self.add_polygon_obstacle(w * 0.5, h * 0.5, 60, 8)
            # Four corner obstacles
            self.add_box_obstacle(w * 0.2, h * 0.2, 50, 50)
            self.add_box_obstacle(w * 0.8, h * 0.2, 50, 50)
            self.add_box_obstacle(w * 0.2, h * 0.8, 50, 50)
            self.add_box_obstacle(w * 0.8, h * 0.8, 50, 50)
            
            # Diagonal separator walls
            self.add_wall(w * 0.1, h * 0.5, w * 0.3, h * 0.5)
            self.add_wall(w * 0.7, h * 0.5, w * 0.9, h * 0.5)

    def add_box_obstacle(self, cx: float, cy: float, size_x: float, size_y: float):
        """Add a rectangular bounding box obstacle."""
        hx, hy = size_x / 2, size_y / 2
        # Four sides
        self.add_wall(cx - hx, cy - hy, cx + hx, cy - hy)
        self.add_wall(cx + hx, cy - hy, cx + hx, cy + hy)
        self.add_wall(cx + hx, cy + hy, cx - hx, cy + hy)
        self.add_wall(cx - hx, cy + hy, cx - hx, cy - hy)

    def add_polygon_obstacle(self, cx: float, cy: float, radius: float, sides: int = 6):
        """Add a regular polygon obstacle approximating a circle."""
        angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        points = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]
        for i in range(sides):
            p1 = points[i]
            p2 = points[(i + 1) % sides]
            self.add_wall(p1[0], p1[1], p2[0], p2[1])

    def serialize(self) -> List[List[float]]:
        """Serialize segments to lists for JSON transmission."""
        return [[float(p[0][0]), float(p[0][1]), float(p[1][0]), float(p[1][1])] for p in self.segments]
