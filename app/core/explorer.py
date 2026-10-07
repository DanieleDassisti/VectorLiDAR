import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.ndimage import label, center_of_mass
from app.core.vector2d import distance

class FrontierExplorer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cell_size = int(config.get("slam", {}).get("cell_size", 8))
        self.min_cluster_size = 3  # Ignore clusters with fewer than 3 cells to avoid noise

    def detect_frontiers(self, grid_probabilities: np.ndarray) -> List[Tuple[int, int]]:
        """
        Identify frontier cells (known-free cells adjacent to unknown cells).
        Uses vectorized NumPy matrix shifts for high performance.
        Returns:
            List of (row, col) grid coordinates representing frontiers.
        """
        # Threshold definitions:
        # Free: P < 0.4
        # Unknown: 0.45 <= P <= 0.55
        free_mask = grid_probabilities < 0.4
        unknown_mask = (grid_probabilities >= 0.45) & (grid_probabilities <= 0.55)
        
        # Shift mask in 4 directions to find cells with unknown neighbors
        shift_u = np.zeros_like(unknown_mask)
        shift_u[:-1, :] = unknown_mask[1:, :]
        
        shift_d = np.zeros_like(unknown_mask)
        shift_d[1:, :] = unknown_mask[:-1, :]
        
        shift_l = np.zeros_like(unknown_mask)
        shift_l[:, :-1] = unknown_mask[:, 1:]
        
        shift_r = np.zeros_like(unknown_mask)
        shift_r[:, 1:] = unknown_mask[:, :-1]
        
        has_unknown_neighbor = shift_u | shift_d | shift_l | shift_r
        
        # Frontiers are free cells that have at least one unknown neighbor
        frontier_mask = free_mask & has_unknown_neighbor
        
        # Get indices where mask is True
        rows, cols = np.where(frontier_mask)
        return list(zip(rows, cols))

    def get_exploration_target(
        self,
        robot_pose_world: Tuple[float, float, float],
        grid_probabilities: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """
        Processes frontiers, clusters them, and returns the centroid of the best cluster
        in world coordinates (x, y) to navigate towards.
        """
        # 1. Detect frontiers
        frontier_cells = self.detect_frontiers(grid_probabilities)
        if not frontier_cells:
            return None
            
        # 2. Create binary map of frontiers
        rows, cols = grid_probabilities.shape
        frontier_grid = np.zeros((rows, cols), dtype=np.uint8)
        for r, c in frontier_cells:
            frontier_grid[r, c] = 1
            
        # 3. Label connected components (clusters of adjacent frontier cells)
        # Using 8-connectivity for labeling
        labeled_grid, num_features = label(frontier_grid, structure=np.ones((3, 3)))
        if num_features == 0:
            return None
            
        # 4. Find centroids of valid clusters (size >= min_cluster_size)
        robot_x, robot_y, _ = robot_pose_world
        
        best_target: Optional[Tuple[float, float]] = None
        min_dist_to_robot = float('inf')
        
        for i in range(1, num_features + 1):
            cluster_mask = (labeled_grid == i)
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size < self.min_cluster_size:
                continue  # Filter out noise clusters
                
            # Get centroid of the cluster (r, c)
            centroid = center_of_mass(cluster_mask)
            centroid_row, centroid_col = float(centroid[0]), float(centroid[1])
            
            # Convert to world coordinates
            tx = (centroid_col + 0.5) * self.cell_size
            ty = (centroid_row + 0.5) * self.cell_size
            
            # Rate target: We prefer the closest frontier to minimize travel distance
            dist = distance((robot_x, robot_y), (tx, ty))
            
            # Ignore target if it's too close to the robot
            if dist < 25.0:
                continue
                
            if dist < min_dist_to_robot:
                min_dist_to_robot = dist
                best_target = (tx, ty)
                
        return best_target
