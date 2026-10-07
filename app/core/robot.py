import numpy as np
from typing import Tuple, Dict, Any, List
from app.core.vector2d import normalize_angle, distance

class Robot:
    def __init__(self, x: float, y: float, theta: float, config: Dict[str, Any]):
        self.config = config
        
        # Ground truth pose
        self.x = float(x)
        self.y = float(y)
        self.theta = float(normalize_angle(theta))
        
        # Estimated pose (Odometry)
        self.x_odo = float(x)
        self.y_odo = float(y)
        self.theta_odo = float(normalize_angle(theta))
        
        # Target velocities (commands)
        self.cmd_v = 0.0      # Target linear velocity (pixels/s)
        self.cmd_w = 0.0      # Target angular velocity (rad/s)
        
        # Current actual velocities (after acceleration limits)
        self.v = 0.0
        self.w = 0.0
        
        # Physical constraints
        self.radius = float(config.get("radius", 15.0))
        self.max_speed = float(config.get("max_speed", 120.0))
        self.max_omega = float(config.get("max_omega", 3.14))
        self.max_accel = float(config.get("max_accel", 80.0))
        self.max_alpha = float(config.get("max_alpha", 4.0))
        
        # Odometry noise configurations
        noise_cfg = config.get("odometry_noise", {})
        self.noise_xy = float(noise_cfg.get("xy", 0.05))
        self.noise_theta = float(noise_cfg.get("theta", 0.01))

    def set_velocity(self, linear: float, angular: float):
        """Set commanded target velocities, clipped to physical limits."""
        self.cmd_v = np.clip(linear, -self.max_speed, self.max_speed)
        self.cmd_w = np.clip(angular, -self.max_omega, self.max_omega)

    def update(self, dt: float, segments: List[np.ndarray]) -> bool:
        """
        Update the robot state for a time step dt.
        Applies acceleration limits, computes differential kinematics, 
        checks for collisions with walls, and updates ground truth and odometry.
        
        Returns True if a collision occurred.
        """
        # Apply acceleration limits to linear velocity v
        dv = self.cmd_v - self.v
        max_dv = self.max_accel * dt
        self.v += np.clip(dv, -max_dv, max_dv)
        
        # Apply angular acceleration limits to angular velocity w
        dw = self.cmd_w - self.w
        max_dw = self.max_alpha * dt
        self.w += np.clip(dw, -max_dw, max_dw)

        # Differential drive kinematics equations
        if abs(self.w) < 1e-4:
            dx = self.v * np.cos(self.theta) * dt
            dy = self.v * np.sin(self.theta) * dt
            dtheta = 0.0
            
            # Odometry increments (simulating noise)
            v_noise = self.v * (1.0 + np.random.normal(0, self.noise_xy))
            dx_odo = v_noise * np.cos(self.theta_odo) * dt
            dy_odo = v_noise * np.sin(self.theta_odo) * dt
            dtheta_odo = np.random.normal(0, self.noise_theta) * dt
        else:
            # Curved trajectory motion model
            r = self.v / self.w
            theta_new = self.theta + self.w * dt
            dx = r * (np.sin(theta_new) - np.sin(self.theta))
            dy = -r * (np.cos(theta_new) - np.cos(self.theta))
            dtheta = self.w * dt
            
            # Odometry increments with noise
            w_noise = self.w * (1.0 + np.random.normal(0, self.noise_theta))
            v_noise = self.v * (1.0 + np.random.normal(0, self.noise_xy))
            r_odo = v_noise / w_noise
            theta_odo_new = self.theta_odo + w_noise * dt
            dx_odo = r_odo * (np.sin(theta_odo_new) - np.sin(self.theta_odo))
            dy_odo = -r_odo * (np.cos(theta_odo_new) - np.cos(self.theta_odo))
            dtheta_odo = w_noise * dt + np.random.normal(0, self.noise_theta * 0.1) * dt

        # Collision detection against environment segments
        new_x = self.x + dx
        new_y = self.y + dy
        new_theta = normalize_angle(self.theta + dtheta)
        
        collided = False
        # Simplified segment collision: check distance from new_x, new_y to each wall
        for seg in segments:
            p1 = seg[0]
            p2 = seg[1]
            # Find closest point on segment to (new_x, new_y)
            seg_len_sq = np.sum((p2 - p1) ** 2)
            if seg_len_sq == 0:
                d = distance((new_x, new_y), (p1[0], p1[1]))
            else:
                t = max(0.0, min(1.0, np.dot([new_x - p1[0], new_y - p1[1]], p2 - p1) / seg_len_sq))
                proj = p1 + t * (p2 - p1)
                d = distance((new_x, new_y), (proj[0], proj[1]))
                
            if d < self.radius:
                collided = True
                break
                
        if not collided:
            # Apply motion
            self.x = new_x
            self.y = new_y
            self.theta = new_theta
            
            # Update odometry
            self.x_odo += dx_odo
            self.y_odo += dy_odo
            self.theta_odo = normalize_angle(self.theta_odo + dtheta_odo)
        else:
            # If hit wall, stop linear speed, slide slightly or bounce
            self.v = 0.0
            
        return collided

    def get_pose(self) -> Tuple[float, float, float]:
        """Return ground truth pose (x, y, theta)."""
        return self.x, self.y, self.theta

    def get_odometry_pose(self) -> Tuple[float, float, float]:
        """Return estimated odometry pose (x_odo, y_odo, theta_odo)."""
        return self.x_odo, self.y_odo, self.theta_odo

    def reset_odometry(self):
        """Sync odometry with ground truth pose."""
        self.x_odo = self.x
        self.y_odo = self.y
        self.theta_odo = self.theta
