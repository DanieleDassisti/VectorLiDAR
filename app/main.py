import asyncio
import json
import logging
import math
import os
import yaml
import numpy as np
from typing import Set, Dict, Any, Optional, Tuple, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from app.core.environment import Environment
from app.core.robot import Robot
from app.core.slam import SLAMGridMap
from app.core.planner import PathPlanner
from app.core.explorer import FrontierExplorer

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LidarSandbox")

app = FastAPI(title="LiDAR SLAM & Autonomous Exploration Sandbox")

# Load configuration file
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Simulation state variables
env_w = float(config["environment"]["width"])
env_h = float(config["environment"]["height"])

env = Environment(env_w, env_h)
env.load_preset("office") # Default preset

# Robot spawn (x, y, theta)
robot = Robot(env_w * 0.5, env_h * 0.8, -np.pi / 2, config)
slam_map = SLAMGridMap(env_w, env_h, config)
planner = PathPlanner(config)
explorer = FrontierExplorer(config)

# Navigation goals
navigation_target: Optional[Tuple[float, float]] = None
global_path: List[Tuple[float, float]] = []
auto_explore = False
noise_enabled = True

# Active WebSockets clients
connected_clients: Set[WebSocket] = set()

# Configuration variables that can be dynamically updated
lidar_beam_count = int(config["lidar"]["beam_count"])
lidar_max_range = float(config["lidar"]["max_range"])
lidar_noise_std = float(config["lidar"]["noise_std"])

# Sim interval in seconds
dt = 0.05  # 20Hz update loop

# Replanning timer to prevent A* recalculations on every single frame
replan_counter = 0
REPLAN_INTERVAL = 10  # Replan A* path every 10 simulation steps (0.5s)

def get_lidar_angles() -> np.ndarray:
    """Get scanning angles based on configured parameters."""
    fov_rad = np.radians(float(config["lidar"]["fov"]))
    half_fov = fov_rad / 2
    return np.linspace(-half_fov, half_fov, lidar_beam_count)

# Precomputed LiDAR angle array
cached_lidar_angles = get_lidar_angles()

async def broadcast_state(true_pose: Tuple[float, float, float], angles: np.ndarray, ranges: np.ndarray):
    """Serialize and send simulation state to all connected WebSocket clients."""
    if not connected_clients:
        return
        
    # Serialize data
    state = {
        "true_pose": [float(true_pose[0]), float(true_pose[1]), float(true_pose[2])],
        "odo_pose": [float(robot.x_odo), float(robot.y_odo), float(robot.theta_odo)],
        "v": float(robot.v),
        "w": float(robot.w),
        "lidar_ranges": ranges.tolist(),
        "lidar_angles": angles.tolist(),
        "slam_grid": slam_map.get_serialized_grid(),
        "grid_cols": slam_map.cols,
        "grid_rows": slam_map.rows,
        "grid_cell_size": slam_map.cell_size,
        "path": [[float(p[0]), float(p[1])] for p in global_path],
        "target": [float(navigation_target[0]), float(navigation_target[1])] if navigation_target else None,
        "explored_ratio": float(slam_map.get_explored_ratio()),
        "auto_explore": auto_explore,
        "true_walls": env.serialize(),
    }
    
    payload = json.dumps(state)
    
    # Send payload to all clients
    tasks = [client.send_text(payload) for client in connected_clients]
    await asyncio.gather(*tasks, return_exceptions=True)

async def simulation_loop():
    """Continuous background physics and robotics computation loop."""
    global navigation_target, global_path, auto_explore, replan_counter
    logger.info("Simulation background loop started.")
    
    while True:
        try:
            # 1. Update true physics (robot kinematics and collision check)
            # Wall segments are directly cached in env.segments
            wall_segs = env.segments
            
            # 2. Get LiDAR ranges
            true_pose = robot.get_pose()
            angles = cached_lidar_angles
            ranges = env.raycast((true_pose[0], true_pose[1]), angles + true_pose[2], lidar_max_range)
            if noise_enabled:
                noise = np.random.normal(0, lidar_noise_std, size=ranges.shape)
                ranges = np.clip(ranges + noise, 0, lidar_max_range)

            # 3. Update SLAM map based on estimated odometry pose
            slam_map.update_map(
                robot.x_odo, robot.y_odo, robot.theta_odo,
                angles, ranges, lidar_max_range
            )
            
            # 4. Autonomous Frontier Exploration Target Selection
            if auto_explore:
                # Target is dynamically set to the nearest frontier centroid
                target = explorer.get_exploration_target(
                    (robot.x_odo, robot.y_odo, robot.theta_odo),
                    slam_map.probabilities
                )
                if target:
                    # If target changed, reset path to trigger immediate recalculation
                    if not navigation_target or math.hypot(target[0]-navigation_target[0], target[1]-navigation_target[1]) > 25.0:
                        navigation_target = target
                        global_path = []
                else:
                    # No frontiers left, exploration complete!
                    logger.info("Exploration complete! No frontiers detected.")
                    auto_explore = False
                    navigation_target = None
                    global_path = []
            
            # 5. Global Path Planning (A*)
            if navigation_target:
                replan_counter += 1
                dist_to_target = math.hypot(robot.x_odo - navigation_target[0], robot.y_odo - navigation_target[1])
                
                # Arrived at target check
                if dist_to_target < planner.target_tolerance:
                    navigation_target = None
                    global_path = []
                    robot.set_velocity(0.0, 0.0)
                else:
                    # Replan if no path exists, or replan counter expires
                    if len(global_path) == 0 or replan_counter >= REPLAN_INTERVAL:
                        replan_counter = 0
                        # Plan path on SLAM grid map (estimating start from robot's odometry)
                        path = planner.a_star(
                            (robot.x_odo, robot.y_odo),
                            navigation_target,
                            slam_map.probabilities
                        )
                        if path:
                            global_path = path
                        else:
                            # If planning fails, clear path and try to recover or stop
                            global_path = []
                            robot.set_velocity(0.0, 0.0)
            
            # 6. Local Path Planning & Obstacle Avoidance (DWA)
            if navigation_target and global_path:
                # Run DWA controller to get velocity commands
                v, w = planner.local_dwa_control(
                    (robot.x_odo, robot.y_odo, robot.theta_odo),
                    (robot.v, robot.w),
                    global_path,
                    slam_map.probabilities
                )
                robot.set_velocity(v, w)
            elif not auto_explore and not navigation_target:
                # Let user manual controls take over (velocities decay when not pressed)
                pass
                
            # Perform movement integration
            robot.update(dt, wall_segs)
            
            # 7. Broadcast updated state (reusing existing sensor scan)
            await broadcast_state(true_pose, angles, ranges)
            
        except Exception as e:
            logger.error(f"Error in simulation loop: {e}", exc_info=True)
            
        await asyncio.sleep(dt)

@app.on_event("startup")
async def startup_event():
    """Create simulation background task on FastAPI launch."""
    asyncio.create_task(simulation_loop())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle incoming WebSocket requests from the web UI."""
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"Client connected. Total clients: {len(connected_clients)}")
    
    global navigation_target, global_path, auto_explore, noise_enabled, lidar_beam_count, lidar_max_range
    
    try:
        # Send initial configuration / layout details to the client
        await websocket.send_text(json.dumps({
            "type": "init",
            "width": env_w,
            "height": env_h,
            "true_walls": env.serialize(),
        }))
        
        # Read incoming messages
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type")
            
            if msg_type == "set_target":
                # User click to set navigation goal
                auto_explore = False  # Manual target overrides auto exploration
                navigation_target = (float(msg["x"]), float(msg["y"]))
                global_path = []  # Force recalculation next step
                logger.info(f"Target navigation goal set to: {navigation_target}")
                
            elif msg_type == "manual_control":
                # User manual joystick control
                auto_explore = False
                navigation_target = None
                global_path = []
                robot.set_velocity(float(msg["v"]), float(msg["w"]))
                
            elif msg_type == "toggle_auto_exploration":
                auto_explore = bool(msg["value"])
                if not auto_explore:
                    navigation_target = None
                    global_path = []
                    robot.set_velocity(0.0, 0.0)
                logger.info(f"Autonomous exploration: {auto_explore}")
                
            elif msg_type == "add_wall":
                # Add customized wall
                env.add_wall(float(msg["x1"]), float(msg["y1"]), float(msg["x2"]), float(msg["y2"]))
                
            elif msg_type == "clear_walls":
                env.clear()
                
            elif msg_type == "load_preset":
                preset_name = msg["name"]
                env.load_preset(preset_name)
                logger.info(f"Loaded environment preset: {preset_name}")
                
            elif msg_type == "reset_odom":
                robot.reset_odometry()
                slam_map.log_odds.fill(0.0)
                slam_map.probabilities.fill(0.5)
                global_path = []
                navigation_target = None
                logger.info("Odometry and SLAM map reset.")
                
            elif msg_type == "update_config":
                old_beam_count = lidar_beam_count
                lidar_beam_count = int(msg.get("beam_count", lidar_beam_count))
                lidar_max_range = float(msg.get("max_range", lidar_max_range))
                noise_enabled = bool(msg.get("noise_enabled", noise_enabled))
                if lidar_beam_count != old_beam_count:
                    cached_lidar_angles = get_lidar_angles()
                logger.info(f"Sensor configuration updated: Beams={lidar_beam_count}, Range={lidar_max_range}, Noise={noise_enabled}")
                
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(connected_clients)}")
    except Exception as e:
        logger.error(f"WebSocket communication error: {e}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)

# HTML endpoint serving frontend
@app.get("/")
async def get_root():
    """Serves the main application landing page."""
    static_index = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    if os.path.exists(static_index):
        with open(static_index, "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>Frontend index.html not found!</h1>", status_code=404)

# Serve static dashboard files (js, css, images)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    # Start web server
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
