def roll_angle(angle):
    angle = angle % 360
    if angle < 0:
        angle += 360
    return angle

def is_angle_in_fov(angle, theta_min, theta_max):
    """Check if an angle is within the FOV, accounting for wrap-around cases."""
    if theta_min < theta_max:
        return theta_min < angle < theta_max
    else:
        return angle > theta_min or angle < theta_max

def cross2d(v1, v2):
    return v1[0]*v2[1] - v1[1]*v2[0]

def dot2d(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1]

def lineseg_ray_intersection(A, B, dir_vec):
    """
    Determines if a line segment (A, B) intersects with a ray defined by its direction vector.
    
    Parameters:
    - A (tuple): The starting point (x, y) of the line segment.
    - B (tuple): The ending point (x, y) of the line segment.
    - dir_vec (tuple): The unit direction vector (dx, dy) representing the ray.
    
    Returns:
    - bool: True if the ray intersects the line segment, False otherwise.
    """
    # Compute the vector representation of the line segment from A to B
    R = (B[0] - A[0], B[1] - A[1])
    
    # Compute the determinant using the 2D cross product of the segment vector and ray direction
    denom = cross2d(R, dir_vec)
    
    # If the determinant is close to zero, the line segment and ray are parallel (no intersection)
    if abs(denom) < 1e-15:
        return False
    
    # Compute intersection parameters (alpha and beta)
    alpha = cross2d(dir_vec, A) / denom # Determines where along the segment the intersection occurs
    beta = cross2d(R, A) / denom    # Determines where along the ray the intersection occurs
    
    # Compute the intersection point (for reference, though not returned)
    pt = (A[0] + alpha*R[0], A[1] + alpha*R[1])
    
    # Check if the intersection occurs within the segment and in the forward direction of the ray
    if (0 < alpha < 1) and (beta > -1e-15):
        return True # Intersection detected
    return False  # No intersection

if __name__=="__main__":
    import numpy as np
    print(lineseg_ray_intersection((-1.5, 0.5), (-1.5, -0.5), (np.cos(np.deg2rad(162)), np.sin(np.deg2rad(162)))))