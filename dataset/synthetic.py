import numpy as np
from PIL import Image

def generate_gps():
    """Returns random (lat, lon) coordinates."""
    lat = np.random.uniform(-90, 90)
    lon = np.random.uniform(-180, 180)
    return lat, lon

def generate_pose():
    """Generates a random 4x4 SE(3) pose matrix."""
    # Random orthogonal matrix from QR decomposition
    H = np.random.randn(3, 3)
    Q, R = np.linalg.qr(H)
    # Ensure determinant is +1 to be a valid rotation
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    
    t = np.random.randn(3)
    
    pose = np.eye(4)
    pose[:3, :3] = Q
    pose[:3, 3] = t
    return pose

def is_valid_pose(pose, tol=1e-5):
    """Validates if a 4x4 matrix is a valid SE(3) transformation."""
    if pose.shape != (4, 4):
        return False
    # Check bottom row
    if not np.allclose(pose[3, :], [0, 0, 0, 1], atol=tol):
        return False
    # Check orthogonality of R
    R = pose[:3, :3]
    if not np.allclose(np.dot(R.T, R), np.eye(3), atol=tol):
        return False
    # Check det(R) == 1
    if not np.isclose(np.linalg.det(R), 1.0, atol=tol):
        return False
    return True

def generate_intrinsics():
    """Generates a dummy 3x3 camera matrix (intrinsics)."""
    fx, fy = 800.0, 800.0
    cx, cy = 112.0, 112.0
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

def generate_image(size=(224, 224)):
    """Generates a random PIL Image of given size."""
    # Generate random pixels, shape (H, W, 3)
    pixels = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(pixels)
