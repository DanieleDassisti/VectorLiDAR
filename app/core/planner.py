import heapq
import math
from collections import deque
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.ndimage import binary_dilation
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
        
        # Precompute circular dilation structuring element
        if self.inflation_cells > 0:
            r = self.inflation_cells
            y, x = np.ogrid[-r:r+1, -r:r+1]
            self._dilation_structure = (x * x + y * y <= r * r)
        else:
            self._dilation_structure = np.ones((1, 1), dtype=bool)

    def compute_c_space(self, grid_probabilities: np.ndarray) -> np.ndarray:
        """
        Compute C-space (Configuration Space) by inflating occupied cells.
        Uses fast SciPy binary morphological dilation.
        """
        c_space = grid_probabilities > self.prob_occ_threshold
        if self.inflation_cells > 0:
            return binary_dilation(c_space, structure=self._dilation_structure).astype(np.uint8)
        return c_space.astype(np.uint8)

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
        start_c, start_r = int(np.clip(start_c, 0, cols - 1)), int(np.clip(start_r, 0, rows - 1))
        goal_c, goal_r = int(np.clip(goal_c, 0, cols - 1)), int(np.clip(goal_r, 0, rows - 1))

        # Inflate grid map to get dynamic configuration space
        c_space = self.compute_c_space(grid_probabilities)

        # If start or goal is inside an obstacle, find nearest free cell
        if c_space[start_r, start_c] == 1:
            start_c, start_r = self.find_nearest_free_cell(start_c, start_r, c_space)
        if c_space[goal_r, goal_c] == 1:
            goal_c, goal_r = self.find_nearest_free_cell(goal_c, goal_r, c_space)

        # Priority Queue holds (f_score, (r, c))
        open_set = []
        h_start = self.heuristic((start_r, start_c), (goal_r, goal_c))
        heapq.heappush(open_set, (h_start, (start_r, start_c)))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        g_score = np.full((rows, cols), np.inf, dtype=np.float64)
        g_score[start_r, start_c] = 0.0
        
        f_score = np.full((rows, cols), np.inf, dtype=np.float64)
        f_score[start_r, start_c] = h_start

        # 8-connected grid offsets
        neighbors = (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), # orthogonal
            (-1, -1, 1.41421356), (-1, 1, 1.41421356), (1, -1, 1.41421356), (1, 1, 1.41421356) # diagonal
        )

        while open_set:
            f, current = heapq.heappop(open_set)
            curr_r, curr_c = current

            if f > f_score[curr_r, curr_c]:
                continue

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
                cell_s = self.cell_size
                return [((c + 0.5) * cell_s, (r + 0.5) * cell_s) for r, c in grid_path]

            for dr, dc, cost in neighbors:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if c_space[nr, nc] == 1:
                        continue  # Collision
                    
                    tentative_g = g_score[curr_r, curr_c] + cost
                    if tentative_g < g_score[nr, nc]:
                        came_from[(nr, nc)] = current
                        g_score[nr, nc] = tentative_g
                        f_val = tentative_g + self.heuristic((nr, nc), (goal_r, goal_c))
                        f_score[nr, nc] = f_val
                        heapq.heappush(open_set, (f_val, (nr, nc)))
                        
        return [] # No path found

    def heuristic(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Euclidean distance heuristic for A* using fast math.hypot."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def find_nearest_free_cell(self, start_c: int, start_r: int, c_space: np.ndarray) -> Tuple[int, int]:
        """BFS to find the nearest traversable cell if robot is stuck in an obstacle using deque."""
        rows, cols = c_space.shape
        queue = deque([(start_r, start_c)])
        visited = {(start_r, start_c)}
        
        while queue:
            r, c = queue.popleft()
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

        # Spatial filtering: only keep obstacles within reach of the DWA lookahead horizon
        if obs_coords.size > 0:
            horizon = max_speed * self.dwa_predict_time + float(self.config["robot"]["radius"]) + 20.0
            nearby_mask = (np.abs(obs_coords[:, 0] - x) <= horizon) & (np.abs(obs_coords[:, 1] - y) <= horizon)
            obs_coords = obs_coords[nearby_mask]

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
                goal_theta = math.atan2(target[1] - traj_y[-1], target[0] - traj_x[-1])
                heading_err = abs(normalize_angle(goal_theta - traj_theta[-1]))
                # Normalize to 0 (bad) to 1 (perfect alignment)
                heading_score = (math.pi - heading_err) / math.pi
                
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
        
        # Calculate squared distances between all trajectory steps and all obstacles
        diff = traj_coords[:, np.newaxis, :] - obs_coords[np.newaxis, :, :] # shape (S, K, 2)
        dists_sq = np.sum(diff**2, axis=-1) # shape (S, K)
        
        return float(np.sqrt(np.min(dists_sq)))
