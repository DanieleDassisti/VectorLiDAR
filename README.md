# CyberLiDAR: 2D Robotics Simulation & Autonomous Exploration Sandbox

An advanced, high-performance 2D mobile robot simulator and visualization dashboard built from scratch. This project demonstrates core robotics algorithms: **vectorized 2D LiDAR raycasting**, **Differential Drive Kinematics**, **Occupancy Grid SLAM (Log-Odds Bayesian formulation)**, **A\* Path Planning**, **Dynamic Window Approach (DWA) obstacle avoidance**, and **Frontier-Based Autonomous Exploration**.

It features a high-concurrency Python backend (WebSockets + FastAPI) and a responsive, glowing neon-cyberpunk web UI interface (HTML5 Canvas 2D) designed as a standout portfolio piece for GitHub.

---

## Key Algorithmic Highlights

### 1. Vectorized NumPy Raycasting
To simulate the LiDAR scan sweep, $N$ beams are intersected against $M$ wall segments simultaneously. Instead of slow nested loops, the math is solved in parallel using NumPy broadcasting:

Let a ray start at origin $O$ with direction unit vector $D$, and a wall segment go from $A$ to $B$ with vector $V = B - A$. We solve for intersection parameters $t$ (ray distance) and $u$ (segment fraction) where $O + t \cdot D = A + u \cdot V$:

$$
t \cdot D - u \cdot V = A - O
$$

Using Cramer's Rule, the determinant is $\Delta = -D_x V_y + D_y V_x$. The parameters are computed as:

$$
t = \frac{-P_x V_y + P_y V_x}{\Delta}, \quad u = \frac{D_x P_y - D_y P_x}{\Delta} \quad (\text{where } P = A - O)
$$

An intersection is valid if $\Delta \neq 0$, $0 \le t \le R_{\max}$, and $0 \le u \le 1$. The closest hit distance across all segments is returned:

$$
\text{range}_i = \min_{j \in M} (t_{ij})
$$

### 2. Occupancy Grid Mapping (Log-Odds SLAM)
The robot constructs a map of its environment as it moves, using a **Bayesian Log-Odds** formulation. For each grid cell $m_i$, we track the log-odds representation of its occupancy probability:

$$
l_{i, t} = l_{i, t-1} + \text{InverseSensorModel}(m_i, z_t) - l_0
$$

Where the sensor model updates:
*   $\Delta l = -0.4$ (free space) for cells traversed along the ray path (traced via **Bresenham's Line Algorithm**).
*   $\Delta l = +0.85$ (occupied wall) for the cells containing the scan hits.

The log-odds value is mapped back to occupancy probability $P(m_i)$ for rendering and path planning:

$$
P(m_i) = 1.0 - \frac{1.0}{1.0 + \exp(l_i)}
$$

### 3. Frontier-Based Autonomous Exploration
To map the layout without manual controls, the robot detects **Frontiers** (known-free cells adjacent to unknown cells).
1.  **Vectorized Shift Detection**: Shift matrices to isolate border cells.
2.  **Connected Component Clustering**: Using `scipy.ndimage.label` to group adjacent frontier cells.
3.  **Target Extraction**: The centroid of the closest cluster (ranked by Euclidean distance) is selected as the next exploration target, prompting the path planner.

### 4. Hybrid Navigation (A* + DWA)
*   **Global Planner (A*)**: Computes the shortest path on the Configuration Space (C-Space) map, inflated to match the robot's physical radius.
*   **Local Planner (Dynamic Window Approach)**: Samples legal velocities $(v, \omega)$ within the robot's acceleration limits over a short horizon. Trajectories are rated based on:
    
    $$\text{Score} = \alpha \cdot \text{HeadingAlignment} + \beta \cdot \text{ObstacleClearance} + \gamma \cdot \text{Velocity}$$
    
    The highest-scoring velocity command is executed, achieving smooth, collision-free navigation.

---

## Project Structure

```
lidar-robotics-sandbox/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI server, websockets, simulator loop
│   └── core/
│       ├── __init__.py
│       ├── vector2d.py      # NumPy geometric intersection helpers
│       ├── environment.py   # Ground-truth wall layout & presets
│       ├── robot.py         # Kinematics and collision handling
│       ├── slam.py          # Bayesian Occupancy Grid Map
│       ├── planner.py       # A* global & DWA local planners
│       └── explorer.py      # Frontier detection & clustering
├── static/
│   ├── index.html           # Controls & viewport containers
│   ├── style.css            # Cyberpunk HUD styling
│   └── app.js               # WebSocket stream client & canvas renderer
├── tests/
│   └── test_simulation.py   # Pytest suite for core math & motion
├── config.yaml              # Hyperparameters (speed, noise, cells)
├── requirements.txt         # Package dependencies
├── Dockerfile               # Container build file
├── docker-compose.yml       # Docker compose orchestrator
└── README.md                # Documentation (this file)
```

---

## Getting Started

### Method 1: Local Python Setup (Recommended)

1.  Ensure you have **Python 3.10+** installed.
2.  Clone the repository:
    ```bash
    git clone https://github.com/DanieleDassisti/VectorLiDAR.git
    cd VectorLiDAR
    ```
3.  Create a virtual environment and install requirements:
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate

    pip install -r requirements.txt
    ```
4.  Run the application:
    ```bash
    python app/main.py
    ```
5.  Open your browser and visit: `http://localhost:8000`

### Method 2: Running with Docker (Zero Setup)

1.  Start the application container:
    ```bash
    docker-compose up --build
    ```
2.  Open your browser and navigate to `http://localhost:8000`.

---

## Running Unit Tests

To run the automated math and algorithm validation suite:

```bash
pytest
```

---

## License

This project is licensed under the MIT License. Feel free to use and extend!
