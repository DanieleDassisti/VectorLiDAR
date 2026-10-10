import math
import numpy as np
from typing import Tuple, Optional, Union

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points using fast C-level math.hypot."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def normalize_angle(angle: float) -> float:
    """Normalize angle to the range [-pi, pi] using fast scalar atan2."""
    return math.atan2(math.sin(angle), math.cos(angle))

def polar_to_cartesian(x: float, y: float, r: float, theta: float) -> Tuple[float, float]:
    """Convert polar coordinates (relative to origin x, y) to global Cartesian coordinates."""
    return x + r * math.cos(theta), y + r * math.sin(theta)

def ray_segment_intersection(
    ray_origin: np.ndarray,      # shape (2,)
    ray_dir: np.ndarray,         # shape (2,)
    segment_start: np.ndarray,   # shape (2,)
    segment_end: np.ndarray,     # shape (2,)
    max_range: float
) -> Optional[float]:
    """
    Find the intersection distance of a single ray with a line segment.
    Returns the distance 't' if intersecting, or None.
    
    Solve: ray_origin + t * ray_dir = segment_start + u * (segment_end - segment_start)
    """
    v1_x = ray_origin[0] - segment_start[0]
    v1_y = ray_origin[1] - segment_start[1]
    v2_x = segment_end[0] - segment_start[0]
    v2_y = segment_end[1] - segment_start[1]
    
    # v3 is perpendicular to ray_dir: (-ray_dir[1], ray_dir[0])
    v3_x = -ray_dir[1]
    v3_y = ray_dir[0]
    
    dot = v2_x * v3_x + v2_y * v3_y
    if abs(dot) < 1e-6:
        return None  # Parallel
        
    t = (v2_x * v1_y - v2_y * v1_x) / dot
    u = (v1_x * v3_x + v1_y * v3_y) / dot
    
    if 0.0 <= t <= max_range and 0.0 <= u <= 1.0:
        return float(t)
    return None

def ray_segment_intersection_vectorized(
    origin: np.ndarray,       # shape (2,)
    dirs: np.ndarray,         # shape (N, 2) - array of unit direction vectors for N beams
    starts: np.ndarray,       # shape (M, 2) - start points of M segments
    ends: np.ndarray,         # shape (M, 2) - end points of M segments
    max_range: float
) -> np.ndarray:              # shape (N,) - minimum hit distance for each ray
    """
    Vectorized ray-caster using NumPy.
    Computes intersections of N rays against M segments simultaneously.
    Optimized 2D broadcasting avoids intermediate 3D tensor allocations.
    """
    M = starts.shape[0]
    if M == 0:
        return np.full(dirs.shape[0], max_range, dtype=np.float64)

    # Segment vectors V and relative origins P of shape (M, 2)
    V = ends - starts
    P = starts - origin
    
    # Determinant of the system: det([D, -V]) = -D_x * V_y + D_y * V_x (shape (N, M))
    det = -dirs[:, 0:1] * V[:, 1] + dirs[:, 1:2] * V[:, 0]
    det_mask = np.abs(det) > 1e-8
    
    # Numerators for Cramer's rule
    # t_num has shape (M,), broadcasting along axis 0 across N rays
    t_num = -P[:, 0] * V[:, 1] + P[:, 1] * V[:, 0]
    # u_num has shape (N, M)
    u_num = dirs[:, 0:1] * P[:, 1] - dirs[:, 1:2] * P[:, 0]
    
    safe_det = np.where(det_mask, det, 1.0)
    t_val = t_num / safe_det
    u_val = u_num / safe_det
    
    valid_mask = det_mask & (t_val >= 0.0) & (t_val <= max_range) & (u_val >= 0.0) & (u_val <= 1.0)
    hits = np.where(valid_mask, t_val, max_range)
    
    return np.min(hits, axis=1)
