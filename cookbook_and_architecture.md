# LiDAR Robotics Sandbox: Architecture & Cookbook

Welcome to the Developer Guide, Architecture Documentation, and Cookbook for the **LiDAR Robotics Sandbox**. This document provides an in-depth breakdown of the mathematical models, systems architecture, and code customization guides ("recipes") to help you understand, extend, or explain this project to recruiters.

---

## 1. System Architecture

The project is structured as a decoupled, real-time client-server application:

```
+------------------------------------------------------------+
|                     PYTHON BACKEND                         |
|                                                            |
|  +------------------+                   +---------------+  |
|  | Simulation Loop  | <--- Physics ---> | Environment   |  |
|  |  (asyncio @ 20Hz)|                   |  (True Walls) |  |
|  +------------------+                   +---------------+  |
|           |                                       |        |
|     Robot Kinematics                         Raycasting    |
|           |                                 (NumPy Matrix) |
|           v                                       v        |
|  +------------------+                   +---------------+  |
|  | SLAM Grid Map    | <--- Occupancy -- | LiDAR Scanner |  |
|  | (Bayesian Odds)  |                   +---------------+  |
|  +------------------+                                      |
|           |                                                |
|     Path Planners                                          |
|  (A* & DWA Local)                                          |
|           |                                                |
|           v                                                |
|  +------------------+                                      |
|  | Frontier Explorer|                                      |
|  |  (SciPy Cluster) |                                      |
|  +------------------+                                      |
|           |                                                |
|     State Broadcast                                        |
|           v                                                |
|  +------------------------------------------------------+  |
|  | FastAPI WebSocket Server                             |  |
|  +------------------------------------------------------+  |
+---------------------------|--------------------------------+
                            |
                 Bi-directional WebSockets
                            |
+---------------------------v--------------------------------+
|                     FRONTEND UI                            |
|                                                            |
|  +------------------------------------------------------+  |
|  |                    WebSocket Client                  |  |
|  +------------------------------------------------------+  |
|        |                  |                     |          |
|    Telemetry          True Sim               SLAM Map      |
|    Readouts           Canvas (2D)          Canvas (2D)     |
+------------------------------------------------------------+
```

### Async Simulation Loop (`app/main.py`)
The server runs an asynchronous event loop (`asyncio`) ticking at $20\text{ Hz}$ ($dt = 0.05\text{ s}$). Each tick executes the following pipeline:
1. Integrates robot kinematics and checks for wall collisions.
2. Performs vectorized raycasting on the true environment geometry.
3. Updates the SLAM occupancy grid map using the estimated odometry pose.
4. (Optional) Detects frontiers and schedules autonomous goal coordinates.
5. (Optional) Executes A* global path planning and DWA local obstacle avoidance commands.
6. Serializes the active state and broadcasts it via WebSockets as a JSON payload.

### Real-Time WebSocket Protocol
#### State Broadcast Payload (Server -> Client)
```json
{
  "true_pose": [x, y, theta],       // Ground truth coordinates
  "odo_pose": [x_odo, y_odo, th],    // Estimated coordinate frame
  "v": linear_velocity,             // Current speed (px/s)
  "w": angular_velocity,            // Current angular speed (rad/s)
  "lidar_ranges": [r1, r2, ...],     // Scanning ray lengths
  "lidar_angles": [a1, a2, ...],     // Scanning ray angles (radians)
  "slam_grid": [p1, p2, p3, ...],    // Occupancy probability matrix (flat 1D, 0-100)
  "grid_cols": 100,                  // Grid dimensions
  "grid_rows": 75,
  "grid_cell_size": 8,
  "path": [[x1, y1], [x2, y2], ...], // A* path nodes
  "target": [gx, gy],                // Goal coordinate or null
  "explored_ratio": 0.42,            // Explorer coverage %
  "auto_explore": true,              // Autonomy state
  "true_walls": [[x1, y1, x2, y2]]   // Bounding segments
}
```

---

## 2. Core Mathematics & Algorithm Designs

### Vectorized Cramer's Rule Raycaster (`app/core/vector2d.py`)
To compute intersections of $N$ beams against $M$ line segments in parallel, we broadcast variables to shape $(N, M, 2)$ and solve the system:

$$\mathbf{o} + t \cdot \mathbf{d} = \mathbf{a} + u \cdot (\mathbf{b} - \mathbf{a})$$

Where $\mathbf{o}$ is the robot origin, $\mathbf{d}$ is the ray unit vector, and $\mathbf{a}, \mathbf{b}$ are the segment ends.
Letting $\mathbf{v} = \mathbf{b} - \mathbf{a}$ and $\mathbf{p} = \mathbf{a} - \mathbf{o}$, we solve:

$$\begin{bmatrix} d_x & -v_x \\ d_y & -v_y \end{bmatrix} \begin{bmatrix} t \\ u \end{bmatrix} = \begin{bmatrix} p_x \\ p_y \end{bmatrix}$$

$$\text{Determinant } \Delta = -d_x v_y + d_y v_x$$

Using Cramer's Rule:

$$t = \frac{-p_x v_y + p_y v_x}{\Delta}, \quad u = \frac{d_x p_y - d_y p_x}{\Delta}$$

This is computed inside `ray_segment_intersection_vectorized` using fast NumPy array operations:
```python
det = -D[..., 0] * V[..., 1] + D[..., 1] * V[..., 0]  # Shape (N, M)
t_num = -P[..., 0] * V[..., 1] + P[..., 1] * V[..., 0] # Shape (1, M)
t_num_broadcast = np.broadcast_to(t_num, det.shape)   # Broadcast to (N, M)
# Solve and filter for valid hits (0 <= t <= range, 0 <= u <= 1)
```

### Log-Odds Occupancy Updates (`app/core/slam.py`)
Each cell in the occupancy grid stores log-odds values:

$$l(x) = \log \frac{P(x)}{1 - P(x)}$$

When a LiDAR beam sweeps, cells along the line of sight are traced using **Bresenham's Line Algorithm**. 
*   Traversed cells: $l_{new} = \max(l_{min}, l_{old} + l_{free})$ where $l_{free} = -0.4$.
*   Hit point cell: $l_{new} = \min(l_{max}, l_{old} + l_{occ})$ where $l_{occ} = 0.85$.

Conversion back to probability for rendering/planning is:

$$P(x) = 1.0 - \frac{1.0}{1.0 + \exp(l(x))}$$

### Frontier Morphological Detection (`app/core/explorer.py`)
To isolate boundaries between explored free space ($P < 0.40$) and unknown regions ($0.45 \le P \le 0.55$) in $O(1)$ operations, we shift the unknown mask in 4 cardinal directions using slicing:

```python
free_mask = grid_probabilities < 0.4
unknown_mask = (grid_probabilities >= 0.45) & (grid_probabilities <= 0.55)

# Shift matrix left, right, up, down
shift_u[:-1, :] = unknown_mask[1:, :]
shift_d[1:, :] = unknown_mask[:-1, :]
...
has_unknown_neighbor = shift_u | shift_d | shift_l | shift_r
frontier_mask = free_mask & has_unknown_neighbor
```
We then group adjacent frontier pixels using `scipy.ndimage.label` connected component analysis and navigate to the centroid of the nearest cluster.

---

## 3. Developer Cookbook (Recipes)

### Recipe 1: Adding a Custom Map Preset
To add a new map preset layout (e.g., a "Testing Ring"):
1.  Open [app/core/environment.py](app/core/environment.py).
2.  Add a new conditional block inside the `load_preset(self, name)` method:
    ```python
    elif name == "ring":
        # Add a circular layout in the center
        self.add_polygon_obstacle(w * 0.5, h * 0.5, radius=120, sides=12)
        # Add outer boundary partitions
        self.add_wall(w * 0.2, h * 0.2, w * 0.2, h * 0.8)
        self.add_wall(w * 0.8, h * 0.2, w * 0.8, h * 0.8)
    ```
3.  Add the new option to the dropdown menu in [static/index.html](static/index.html):
    ```html
    <select id="map-preset" class="cyber-select">
        ...
        <option value="ring">Robotics Ring Yard</option>
    </select>
    ```

### Recipe 2: Switching Kinematics Models (e.g. Omnidirectional Drive)
To change the robot kinematics from differential-drive (default) to omnidirectional drive (holonomic, allowing lateral motion):
1.  Open [app/core/robot.py](app/core/robot.py).
2.  Add lateral velocity variable: `self.cmd_vy = 0.0`.
3.  Modify `update(self, dt, segments)` to apply holonomic motion:
    ```python
    # Under standard holonomic drive equations:
    # robot moves in x, y directions independently
    dx = (self.v * np.cos(self.theta) - self.vy_actual * np.sin(self.theta)) * dt
    dy = (self.v * np.sin(self.theta) + self.vy_actual * np.cos(self.theta)) * dt
    dtheta = self.w * dt
    ```
4.  Modify controls in [app/main.py](app/main.py) and [static/app.js](static/app.js) to support passing sideways velocity commands.

### Recipe 3: Tuning DWA Planner Weights
If the robot gets too close to walls or is sluggish when steering, you can adjust DWA scoring coefficients:
1.  Open [config.yaml](config.yaml).
2.  Locate the `weights` section under `planner`:
    ```yaml
    weights:
      heading: 2.2         # Increase this to align the robot faster with path targets
      clearance: 1.0       # Increase this to make the robot steer further away from walls
      velocity: 1.0        # Increase this to prefer high speed trajectories
    ```
3.  Restart the server. The background reloader will automatically load the new weights.

### Recipe 4: Adding Simulated Sensor Dropouts
To simulate packet loss or LiDAR scanner dropouts (e.g., hardware faults):
1.  Open [app/main.py](app/main.py).
2.  Inside the `simulation_loop`, add a random dropout mask to `ranges`:
    ```python
    # 5% chance of scan dropouts
    if np.random.rand() < 0.05:
        ranges = np.full_like(ranges, lidar_max_range)
    ```
    This simulates real-world environmental occlusion or electrical noise in the sensor.

---

## 4. Troubleshooting

*   **Error: `ModuleNotFoundError: No module named 'app'`**
    Ensure you run the application using `python -m app.main` from the root directory of the project, rather than `python app/main.py`. This ensures correct package pathing in Python.
*   **Warning: `No supported WebSocket library detected.`**
    Install the websockets package in your active virtual environment:
    ```bash
    pip install websockets
    ```
*   **WebUI shows "OFFLINE"**
    Verify that the FastAPI server is running on port 8000. Check the console for traceback errors and make sure no other program is binding to port 8000.
