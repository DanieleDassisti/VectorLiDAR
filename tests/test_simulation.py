import numpy as np
import pytest
from app.core.vector2d import normalize_angle, distance, ray_segment_intersection, ray_segment_intersection_vectorized
from app.core.robot import Robot
from app.core.planner import PathPlanner

def test_angle_normalization():
    # Test positive and negative wraps
    assert np.allclose(normalize_angle(np.pi), np.pi)
    assert np.allclose(normalize_angle(2 * np.pi), 0.0)
    assert np.allclose(normalize_angle(-np.pi - 0.1), np.pi - 0.1)

def test_distance_calc():
    assert np.allclose(distance((0, 0), (3, 4)), 5.0)

def test_ray_intersection_single():
    ray_origin = np.array([0.0, 0.0])
    ray_dir = np.array([1.0, 0.0]) # Scanning right
    
    # Verticle wall at x = 10, from y = -5 to y = 5
    seg_start = np.array([10.0, -5.0])
    seg_end = np.array([10.0, 5.0])
    
    t = ray_segment_intersection(ray_origin, ray_dir, seg_start, seg_end, max_range=20.0)
    assert t is not None
    assert np.allclose(t, 10.0)

def test_vectorized_raycasting():
    origin = np.array([0.0, 0.0])
    
    # 3 beams scanning: left (-x), forward (+y), right (+x)
    dirs = np.array([
        [-1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0]
    ])
    
    # Two walls:
    # Wall 1: at x = 5 (intersects beam 2)
    # Wall 2: at y = 8 (intersects beam 1)
    starts = np.array([
        [5.0, -2.0],
        [-2.0, 8.0]
    ])
    ends = np.array([
        [5.0, 2.0],
        [2.0, 8.0]
    ])
    
    hits = ray_segment_intersection_vectorized(origin, dirs, starts, ends, max_range=20.0)
    
    # beam 0 (-x) intersects nothing: returns max_range
    # beam 1 (+y) intersects Wall 2 at y = 8: returns 8.0
    # beam 2 (+x) intersects Wall 1 at x = 5: returns 5.0
    assert np.allclose(hits[0], 20.0)
    assert np.allclose(hits[1], 8.0)
    assert np.allclose(hits[2], 5.0)

def test_robot_kinematics():
    config = {
        "radius": 15.0,
        "max_speed": 100.0,
        "max_omega": 3.0,
        "max_accel": 100.0,
        "max_alpha": 5.0,
        "odometry_noise": {"xy": 0.0, "theta": 0.0}
    }
    
    # Spawn robot facing north (+y in screen space, or let's use standard angles)
    # Let's say theta = 0 (facing +x)
    robot = Robot(x=100.0, y=100.0, theta=0.0, config=config)
    robot.set_velocity(50.0, 0.0)
    
    # Empty environment
    walls = []
    
    # Update simulation step (dt = 0.1s)
    # robot actual velocity ramps up immediately due to accel limit 100 px/s^2 (needs 0.5s to reach 50, so in 0.1s it reaches v = 10 px/s)
    robot.update(0.1, walls)
    
    # Assert robot has moved forward along +x
    assert robot.x > 100.0
    assert np.allclose(robot.y, 100.0)
    assert np.allclose(robot.theta, 0.0)

def test_a_star_planning():
    config = {
        "robot": {"radius": 0.0},
        "slam": {"cell_size": 10, "threshold_occupied": 0.65},
        "planner": {"target_tolerance": 5.0}
    }
    
    planner = PathPlanner(config)
    
    # 5x5 grid map
    # 0.0 = free space, 1.0 = occupied
    grid = np.zeros((5, 5))
    grid[2, 1:4] = 1.0  # Put a wall barrier in the middle row (row 2, cols 1,2,3)
    
    # A* search from (col 0, row 0) to (col 4, row 4) in world coords (cell_size = 10)
    # Center of cell (0,0) is (5,5). Goal cell (4,4) is (45,45)
    path = planner.a_star((5.0, 5.0), (45.0, 45.0), grid)
    
    # Path should be found and go around the barrier (since we only blocked col 1,2,3, it can go through col 0 or col 4)
    assert len(path) > 0
    # Goal reached
    assert np.allclose(path[-1], (45.0, 45.0))
