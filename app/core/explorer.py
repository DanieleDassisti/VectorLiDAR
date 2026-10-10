import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.ndimage import label
from app.core.vector2d import distance

class FrontierExplorer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cell_size = int(config.get("slam", {}).get("cell_size", 8))
        self.min_cluster_size = 3  # Ignore clusters with fewer than 3 cells to avoid noise

    def detect_frontiers_mask(self, grid_probabilities: np.ndarray) -> np.ndarray:
        """
        Identify frontier cells boolean mask (known-free cells adjacent to unknown cells).
        Uses in-place shift operations for maximum efficiency without intermediate allocations.
        """
        # Threshold definitions:
        # Free: P < 0.4
        # Unknown: 0.45 <= P <= 0.55
        free_mask = grid_probabilities < 0.4
        unknown_mask = (grid_probabilities >= 0.45) & (grid_probabilities <= 0.55)
        
        # Shift mask in 4 directions in-place
        has_unknown_neighbor = np.zeros_like(unknown_mask)
        has_unknown_neighbor[:-1, :] |= unknown_mask[1:, :]
        has_unknown_neighbor[1:, :] |= unknown_mask[:-1, :]
        has_unknown_neighbor[:, :-1] |= unknown_mask[:, 1:]
        has_unknown_neighbor[:, 1:] |= unknown_mask[:, :-1]
        
        return free_mask & has_unknown_neighbor

    def detect_frontiers(self, grid_probabilities: np.ndarray) -> List[Tuple[int, int]]:
        """
        Identify frontier cells (known-free cells adjacent to unknown cells).
        Returns:
            List of (row, col) grid coordinates representing frontiers.
        """
        frontier_mask = self.detect_frontiers_mask(grid_probabilities)
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
        Uses fast vectorized bincount weighted sums to compute all cluster centroids.
        """
        # 1. Detect frontiers mask directly
        frontier_mask = self.detect_frontiers_mask(grid_probabilities)
        if not np.any(frontier_mask):
            return None
            
        # 2. Label connected components (clusters of adjacent frontier cells)
        labeled_grid, num_features = label(frontier_mask, structure=np.ones((3, 3)))
        if num_features == 0:
            return None
            
        # 3. Vectorized cluster sizes and centroids via bincount
        r_idx, c_idx = np.where(labeled_grid > 0)
        lbls = labeled_grid[r_idx, c_idx]
        
        counts = np.bincount(lbls, minlength=num_features + 1)
        sum_r = np.bincount(lbls, weights=r_idx, minlength=num_features + 1)
        sum_c = np.bincount(lbls, weights=c_idx, minlength=num_features + 1)
        
        valid_indices = np.where(counts[1:] >= self.min_cluster_size)[0] + 1
        if len(valid_indices) == 0:
            return None

        robot_x, robot_y, _ = robot_pose_world
        best_target: Optional[Tuple[float, float]] = None
        min_dist_to_robot = float('inf')
        cell_s = self.cell_size
        
        for i in valid_indices:
            centroid_row = sum_r[i] / counts[i]
            centroid_col = sum_c[i] / counts[i]
            
            # Convert to world coordinates
            tx = (centroid_col + 0.5) * cell_s
            ty = (centroid_row + 0.5) * cell_s
            
            # Rate target: We prefer the closest frontier to minimize travel distance
            dist = distance((robot_x, robot_y), (tx, ty))
            
            # Ignore target if it's too close to the robot
            if dist < 25.0:
                continue
                
            if dist < min_dist_to_robot:
                min_dist_to_robot = dist
                best_target = (tx, ty)
                
        return best_target
