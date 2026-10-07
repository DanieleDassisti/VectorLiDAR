import numpy as np
from typing import Tuple, Optional, Union

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

def normalize_angle(angle: float) -> float:
    """Normalize angle to the range [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))

def polar_to_cartesian(x: float, y: float, r: float, theta: float) -> Tuple[float, float]:
    """Convert polar coordinates (relative to origin x, y) to global Cartesian coordinates."""
    return x + r * np.cos(theta), y + r * np.sin(theta)

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
    v1 = ray_origin - segment_start
    v2 = segment_end - segment_start
    v3 = np.array([-ray_dir[1], ray_dir[0]]) # Perpendicular to ray_dir
    
    dot = np.dot(v2, v3)
    if abs(dot) < 1e-6:
        return None  # Parallel
        
    t = np.cross(v2, v1) / dot
    u = np.dot(v1, v3) / dot
    
    if t >= 0.0 and t <= max_range and 0.0 <= u <= 1.0:
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
    This is extremely fast and showcases advanced NumPy usage.
    """
    N = dirs.shape[0]
    M = starts.shape[0]
    
    # Reshape arrays for broadcasting
    # We want to perform operations of shape (N, M, ...)
    
    # ray_dir (N, 1, 2)
    D = dirs[:, np.newaxis, :]
    # segment vectors (1, M, 2)
    V = (ends - starts)[np.newaxis, :, :]
    # segment start points (1, M, 2)
    A = starts[np.newaxis, :, :]
    # origin (1, 1, 2)
    O = origin[np.newaxis, np.newaxis, :]
    
    # We solve: O + t * D = A + u * V
    # Rearranging: t * D - u * V = A - O = P
    P = A - O  # shape (1, M, 2), broadcast to (N, M, 2)
    
    # Determinant of the system: det([D, -V]) = -D_x * V_y + D_y * V_x
    det = -D[..., 0] * V[..., 1] + D[..., 1] * V[..., 0]  # shape (N, M)
    
    # Avoid divide by zero
    det_mask = np.abs(det) > 1e-8
    
    # Solve for t (ray parameter) and u (segment parameter) using Cramer's rule
    # t_num = -P_x * V_y + P_y * V_x (shape (1, M))
    t_num = -P[..., 0] * V[..., 1] + P[..., 1] * V[..., 0]
    t_num_broadcast = np.broadcast_to(t_num, det.shape)
    
    # u_num = D_x * P_y - D_y * P_x (shape (N, M))
    u_num = D[..., 0] * P[..., 1] - D[..., 1] * P[..., 0]
    
    # Divide safely
    t_val = np.zeros((N, M))
    u_val = np.zeros((N, M))
    
    # Vectorized division
    t_val[det_mask] = t_num_broadcast[det_mask] / det[det_mask]
    u_val[det_mask] = u_num[det_mask] / det[det_mask]
    
    # Valid intersection mask:
    # 0 <= t <= max_range
    # 0 <= u <= 1
    # det_mask must be True (lines are not parallel)
    valid_mask = det_mask & (t_val >= 0.0) & (t_val <= max_range) & (u_val >= 0.0) & (u_val <= 1.0)
    
    # Where invalid, set distance to max_range
    hits = np.where(valid_mask, t_val, max_range)
    
    # Find the minimum hit distance for each of the N rays across all M segments
    min_hits = np.min(hits, axis=1)  # shape (N,)
    
    return min_hits
