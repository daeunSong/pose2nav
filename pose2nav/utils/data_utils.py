import os
import cv2
from PIL import Image
import torch
import math

def imread(address: str):
    if not os.path.exists(address):
        raise FileNotFoundError(f"File not found: {address}")
    
    img = cv2.imread(address, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {address}")
    
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(img)

def cartesian_to_polar(x: float, y: float) -> torch.Tensor:
    """Convert cartesian coordinates to polar coordinates
    x: (float) x direction
    y: (float) y direction
    """
    # Calculating radius
    radius = math.sqrt(x * x + y * y)
    # Calculating angle (theta) in radian
    theta = math.atan2(y, x)
    return torch.Tensor([radius, theta]).float()