import heapq
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from app.core.vector2d import normalize_angle, distance

class PathPlanner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_tolerance = float(config.get("target_tolerance", 12.0))
        
        # DWA params
        planner_cfg = config.get("planner", {})
        self.dwa_dt = float(planner_cfg.get("dwa_dt", 0.1))
        self.dwa_predict_time = float(planner_cfg.get("dwa_predict_time", 1.2))
        self.dwa_v_samples = int(planner_cfg.get("dwa_velocity_samples", 8))
        self.dwa_w_samples = int(planner_cfg.get("dwa_omega_samples", 16))
        
        weights = planner_cfg.get("weights", {})
        self.w_heading = float(weights.get("heading", 2.2))
        self.w_clearance = float(weights.get("clearance", 1.0))
        self.w_velocity = float(weights.get("velocity", 1.0))

        # Thresholds
        self.prob_occ_threshold = float(config.get("slam", {}).get("threshold_occupied", 0.65))
        self.cell_size = int(config.get("slam", {}).get("cell_size", 8))
        
        # Inflation size (robot radius in cells)
        robot_radius = float(config.get("robot", {}).get("radius", 15.0))
        self.inflation_cells = int(np.ceil(robot_radius / self.cell_size))

    def compute_c_space(self, grid_probabilities: np.ndarray) -> np.ndarray:
        """
        Compute C-space (Configuration Space) by inflating occupied cells.
        Uses a quick morphological dilation approximation in NumPy.
        """
        rows, cols = grid_probabilities.shape
        c_space = (grid_probabilities > self.prob_occ_threshold).astype(np.uint8)
        
        # Find coordinates of all occupied cells
        occ_rows, occ_cols = np.where(c_space == 1)
        
        # Inflate obstacles
        inflated = np.copy(c_space)
        for dr in range(-self.inflation_cells, self.inflation_cells + 1):
            for dc in range(-self.inflation_cells, self.inflation_cells + 1):
                if dr == 0 and dc == 0:
                    continue
                # Circular inflation condition: dr^2 + dc^2 <= radius^2
                if dr**2 + dc**2 > self.inflation_cells**2:
                    continue
                
                shifted_rows = np.clip(occ_rows + dr, 0, rows - 1)
                shifted_cols = np.clip(occ_cols + dc, 0, cols - 1)
                inflated[shifted_rows, shifted_cols] = 1
                
        return inflated

    def a_star(self, start_world: Tuple[float, float], goal_world: Tuple[float, float], 
               grid_probabilities: np.ndarray) -> List[Tuple[float, float]]:
        """
        A* Global Path Planning on the inflated occupancy grid.
        Returns a list of path points in world coordinates (x, y).
        """
        rows, cols = grid_probabilities.shape
        
        # Convert world coordinates to grid indices
        start_c = int(start_world[0] / self.cell_size)
        start_r = int(start_world[1] / self.cell_size)
        goal_c = int(goal_world[0] / self.cell_size)
        goal_r = int(goal_world[1] / self.cell_size)

        # Clip endpoints to map boundaries
        start_c, start_r = np.clip(start_c, 0, cols - 1), np.clip(start_r, 0, rows - 1)
        goal_c, goal_r = np.clip(goal_c, 0, cols - 1), np.clip(goal_r, 0, rows - 1)

        # Inflate grid map to get dynamic configuration space
        c_space = self.compute_c_space(grid_probabilities)

        # If start or goal is inside an obstacle, find nearest free cell
        if c_space[start_r, start_c] == 1:
            start_c, start_r = self.find_nearest_free_cell(start_c, start_r, c_space)
        if c_space[goal_r, goal_c] == 1:
            goal_c, goal_r = self.find_nearest_free_cell(goal_c, goal_r, c_space)

        # Priority Queue holds (f_score, (r, c))
        open_set = []
        heapq.heappush(open_set, (0.0, (start_r, start_c)))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        g_score = { (r, c): float('inf') for r in range(rows) for c in range(cols) }
        g_score[(start_r, start_c)] = 0.0
        
        f_score = { (r, c): float('inf') for r in range(rows) for c in range(cols) }
        f_score[(start_r, start_c)] = self.heuristic((start_r, start_c), (goal_r, goal_c))

        # 8-connected grid offsets
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), # orthogonal
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414) # diagonal
        ]

        while open_set:
            _, current = heapq.heappop(open_set)
            curr_r, curr_c = current

            if curr_r == goal_r and curr_c == goal_c:
                # Reconstruct path
                grid_path = []
                temp = current
                while temp in came_from:
                    grid_path.append(temp)
                    temp = came_from[temp]
                grid_path.append((start_r, start_c))
                grid_path.reverse()
                
                # Convert grid path back to world coordinates
                world_path = []
                for r, c in grid_path:
                    # Target center of cell
                    wx = (c + 0.5) * self.cell_size
                    wy = (r + 0.5) * self.cell_size
                    world_path.append((wx, wy))
                return world_path

            for dr, dc, cost in neighbors:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if c_space[nr, nc] == 1:
                        continue  # Collision
                    
                    tentative_g = g_score[current] + cost
                    if tentative_g < g_score.get((nr, nc), float('inf')):
                        came_from[(nr, nc)] = current
                        g_score[(nr, nc)] = tentative_g
                        f = tentative_g + self.heuristic((nr, nc), (goal_r, goal_c))
                        f_score[(nr, nc)] = f
                        heapq.heappush(open_set, (f, (nr, nc)))
                        
        return [] # No path found

    def heuristic(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Euclidean distance heuristic for A*."""
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    def find_nearest_free_cell(self, start_c: int, start_r: int, c_space: np.ndarray) -> Tuple[int, int]:
        """BFS to find the nearest traversable cell if robot is stuck in an obstacle."""
        rows, cols = c_space.shape
        queue = [(start_r, start_c)]
        visited = {(start_r, start_c)}
        
        while queue:
            r, c = queue.pop(0)
            if c_space[r, c] == 0:
                return c, r
                
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return start_c, start_r

    def local_dwa_control(
        self,
        pose: Tuple[float, float, float],    # (x, y, theta)
        vel: Tuple[float, float],            # (v, w)
        path: List[Tuple[float, float]],
        grid_probabilities: np.ndarray
    ) -> Tuple[float, float]:
        """
        Dynamic Window Approach (DWA) local planner.
        Finds optimal (v, w) command to track path while avoiding obstacles.
        """
        x, y, theta = pose
        curr_v, curr_w = vel
        
        # Find current sub-goal along the path
        if not path:
            return 0.0, 0.0
            
        target = path[-1]
        # Look ahead along path to find sub-target
        for pt in path:
            if distance((x, y), pt) > 30.0:
                target = pt
                break
                
        # If very close to end goal, use proportional braking controller
        dist_to_goal = distance((x, y), path[-1])
        if dist_to_goal < self.target_tolerance:
            return 0.0, 0.0
            
        # 1. Compute Dynamic Window
        max_speed = float(self.config["robot"]["max_speed"])
        max_omega = float(self.config["robot"]["max_omega"])
        max_accel = float(self.config["robot"]["max_accel"])
        max_alpha = float(self.config["robot"]["max_alpha"])
        
        # Max velocity based on acceleration constraints
        vs_min_v = max(-max_speed, curr_v - max_accel * self.dwa_dt)
        vs_max_v = min(max_speed, curr_v + max_accel * self.dwa_dt)
        vs_min_w = max(-max_omega, curr_w - max_alpha * self.dwa_dt)
        vs_max_w = min(max_omega, curr_w + max_alpha * self.dwa_dt)

        best_v = 0.0
        best_w = 0.0
        best_score = -float('inf')

        # Sample speeds
        v_samples = np.linspace(vs_min_v, vs_max_v, self.dwa_v_samples)
        w_samples = np.linspace(vs_min_w, vs_max_w, self.dwa_w_samples)

        # Precompute obstacle points for distance checks (c_space grid cell centers)
        c_space = self.compute_c_space(grid_probabilities)
        obs_rows, obs_cols = np.where(c_space == 1)
        # Convert occupied cell indices to world coords
        obs_x = (obs_cols + 0.5) * self.cell_size
        obs_y = (obs_rows + 0.5) * self.cell_size
        obs_coords = np.stack([obs_x, obs_y], axis=1) # shape (K, 2)

        # Test trajectories
        for v in v_samples:
            for w in w_samples:
                # 2. Predict Trajectory
                traj_x, traj_y, traj_theta = self.predict_trajectory(x, y, theta, v, w)
                
                # Check for collision
                min_dist = self.calc_clearance(traj_x, traj_y, obs_coords)
                if min_dist < float(self.config["robot"]["radius"]):
                    continue  # Trajectory collides
                    
                # 3. Calculate Scores
                # Heading score: Alignment of projected final heading with goal direction
                goal_theta = np.arctan2(target[1] - traj_y[-1], target[0] - traj_x[-1])
                heading_err = abs(normalize_angle(goal_theta - traj_theta[-1]))
                # Normalize to 0 (bad) to 1 (perfect alignment)
                heading_score = (np.pi - heading_err) / np.pi
                
                # Clearance score: distance to obstacles (normalize to range)
                clearance_score = min(min_dist, 80.0) / 80.0
                
                # Velocity score: linear speed preference
                velocity_score = v / max_speed if v >= 0 else 0.0

                score = (self.w_heading * heading_score + 
                         self.w_clearance * clearance_score + 
                         self.w_velocity * velocity_score)
                         
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w
                    
        return float(best_v), float(best_w)

    def predict_trajectory(self, x: float, y: float, theta: float, v: float, w: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project robot positions over dynamic window horizon."""
        steps = int(self.dwa_predict_time / self.dwa_dt)
        traj_x = np.zeros(steps)
        traj_y = np.zeros(steps)
        traj_theta = np.zeros(steps)
        
        curr_x, curr_y, curr_theta = x, y, theta
        for i in range(steps):
            if abs(w) < 1e-4:
                curr_x += v * np.cos(curr_theta) * self.dwa_dt
                curr_y += v * np.sin(curr_theta) * self.dwa_dt
            else:
                r = v / w
                theta_next = curr_theta + w * self.dwa_dt
                curr_x += r * (np.sin(theta_next) - np.sin(curr_theta))
                curr_y += -r * (np.cos(theta_next) - np.cos(curr_theta))
                curr_theta = normalize_angle(theta_next)
                
            traj_x[i] = curr_x
            traj_y[i] = curr_y
            traj_theta[i] = curr_theta
            
        return traj_x, traj_y, traj_theta

    def calc_clearance(self, traj_x: np.ndarray, traj_y: np.ndarray, obs_coords: np.ndarray) -> float:
        """Find the minimum distance from the trajectory to any obstacle coordinate."""
        if obs_coords.size == 0:
            return 999.0
            
        # Trajectory coordinates (steps, 2)
        traj_coords = np.stack([traj_x, traj_y], axis=1) # shape (S, 2)
        
        # Calculate distances between all trajectory steps and all obstacles
        # Uses broadcasting to calculate shape (S, K) distances
        diff = traj_coords[:, np.newaxis, :] - obs_coords[np.newaxis, :, :] # shape (S, K, 2)
        dists = np.sqrt(np.sum(diff**2, axis=-1)) # shape (S, K)
        
        return float(np.min(dists))
