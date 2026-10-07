import numpy as np
from typing import Tuple, List, Dict, Any
from app.core.vector2d import polar_to_cartesian

class SLAMGridMap:
    def __init__(self, width: float, height: float, config: Dict[str, Any]):
        self.cell_size = int(config.get("cell_size", 8))
        self.width = int(width)
        self.height = int(height)
        
        # Grid dimensions
        self.cols = self.width // self.cell_size
        self.rows = self.height // self.cell_size
        
        # Log-odds representation of the occupancy grid
        # Initialize with 0.0 log-odds (corresponds to p = 0.5 prior)
        self.log_odds = np.zeros((self.rows, self.cols), dtype=np.float64)
        
        # Log-odds update values
        self.log_odds_free = float(config.get("log_odds_free", -0.4))
        self.log_odds_occ = float(config.get("log_odds_occ", 0.85))
        self.min_log_odds = float(config.get("min_log_odds", -2.0))
        self.max_log_odds = float(config.get("max_log_odds", 3.5))
        
        # Probability cache
        self.probabilities = np.full((self.rows, self.cols), 0.5, dtype=np.float64)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (x, y) to grid indices (col, row)."""
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Convert grid indices (col, row) to world coordinates of cell center."""
        x = (col + 0.5) * self.cell_size
        y = (row + 0.5) * self.cell_size
        return x, y

    def in_bounds(self, col: int, row: int) -> bool:
        """Check if grid cell is within boundaries."""
        return 0 <= col < self.cols and 0 <= row < self.rows

    def update_map(self, robot_x: float, robot_y: float, robot_theta: float, 
                   angles: np.ndarray, ranges: np.ndarray, max_range: float):
        """
        Update occupancy grid using a LiDAR scan from the robot's estimated pose.
        Uses Bresenham's line algorithm to trace rays.
        """
        start_col, start_row = self.world_to_grid(robot_x, robot_y)
        if not self.in_bounds(start_col, start_row):
            return

        # Prepare lists of cells to update to avoid multiple updates per cell in a single step
        # This keeps the SLAM calculations clean and fast
        cells_free = set()
        cells_occupied = set()

        for angle, dist in zip(angles, ranges):
            # Calculate beam endpoint in world space
            end_x, end_y = polar_to_cartesian(robot_x, robot_y, dist, robot_theta + angle)
            end_col, end_row = self.world_to_grid(end_x, end_y)
            
            # Trace the line using Bresenham's algorithm
            beam_cells = self.bresenham_line(start_col, start_row, end_col, end_row)
            
            # All cells along the beam are free
            # The last cell is the obstacle hit point if the range is less than max_range
            is_hit = dist < (max_range - 1.0)
            
            if len(beam_cells) > 0:
                if is_hit:
                    # Endpoint is occupied
                    endpoint = beam_cells[-1]
                    if self.in_bounds(endpoint[0], endpoint[1]):
                        cells_occupied.add(endpoint)
                    # Remaining cells are free
                    for cell in beam_cells[:-1]:
                        if self.in_bounds(cell[0], cell[1]):
                            cells_free.add(cell)
                else:
                    # Entire ray is free
                    for cell in beam_cells:
                        if self.in_bounds(cell[0], cell[1]):
                            cells_free.add(cell)

        # Apply log-odds updates
        # Ensure occupied takes precedence if a cell was marked both (e.g. noise boundaries)
        cells_free = cells_free - cells_occupied
        
        for col, row in cells_free:
            self.log_odds[row, col] = max(self.min_log_odds, self.log_odds[row, col] + self.log_odds_free)
            
        for col, row in cells_occupied:
            self.log_odds[row, col] = min(self.max_log_odds, self.log_odds[row, col] + self.log_odds_occ)

        # Update probability matrix: p = 1 - 1 / (1 + exp(log_odds))
        self.probabilities = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """
        Bresenham's Line Generation Algorithm.
        Returns all grid coordinates along a line from (x0, y0) to (x1, y1).
        """
        # Clamp endpoints within grid size to prevent infinite drawing
        x0 = max(0, min(self.cols - 1, x0))
        y0 = max(0, min(self.rows - 1, y0))
        x1 = max(0, min(self.cols - 1, x1))
        y1 = max(0, min(self.rows - 1, y1))

        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
                
        return points

    def get_explored_ratio(self) -> float:
        """Return percentage of grid that has been observed (prob != 0.5)."""
        # Count cells that deviate from the 0.5 prior
        explored = np.count_nonzero(np.abs(self.log_odds) > 0.05)
        return float(explored / (self.cols * self.rows))

    def get_serialized_grid(self) -> List[int]:
        """
        Serialize occupancy grid to 1D list of ints (0-100 probability) for web visualizer.
        Use: 0 for free space, 50 for unknown, 100 for wall.
        """
        # Convert probabilities (0.0 - 1.0) to integers (0 - 100)
        grid_1d = (self.probabilities * 100).astype(np.uint8).flatten()
        return grid_1d.tolist()
