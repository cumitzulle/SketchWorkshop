"""
Bézier curve utilities for CLIPasso stroke optimization
"""

import torch
import numpy as np
from typing import List, Tuple, Optional

def bezier_curve(t: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Evaluate Bézier curve at parameter t
    
    Args:
        t: Parameter values (n,) in [0, 1]
        points: Control points (n_points, 2)
    
    Returns:
        Curve points (n, 2)
    """
    n = len(points) - 1
    result = torch.zeros(len(t), 2, device=points.device)
    
    for i in range(n + 1):
        # Bernstein polynomial
        coeff = binomial_coefficient(n, i) * (1 - t) ** (n - i) * t ** i
        result += coeff.unsqueeze(1) * points[i]
    
    return result

def binomial_coefficient(n: int, k: int) -> float:
    """Compute binomial coefficient C(n, k)"""
    from math import factorial
    return factorial(n) // (factorial(k) * factorial(n - k))

def generate_random_bezier_points(num_curves: int,
                                 num_points: int = 4,
                                 canvas_size: Tuple[int, int] = (512, 512),
                                 device: str = "cpu") -> List[torch.Tensor]:
    """
    Generate random Bézier curve control points
    
    Args:
        num_curves: Number of curves
        num_points: Number of control points per curve
        canvas_size: Canvas dimensions (width, height)
        device: Torch device
    
    Returns:
        List of point tensors
    """
    points_list = []
    width, height = canvas_size
    
    for _ in range(num_curves):
        points = torch.rand(num_points, 2, device=device)
        points[:, 0] = points[:, 0] * width
        points[:, 1] = points[:, 1] * height
        points_list.append(points)
    
    return points_list

def initialize_strokes_from_saliency(image_tensor: torch.Tensor,
                                    num_strokes: int,
                                    device: str = "cpu") -> List[torch.Tensor]:
    """
    Initialize strokes based on image saliency
    
    Args:
        image_tensor: Input image tensor
        num_strokes: Number of strokes
        device: Torch device
    
    Returns:
        List of point tensors
    """
    # Simplified saliency: use edges
    from torchvision.transforms.functional import gaussian_blur
    
    # Convert to grayscale
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.mean(dim=1, keepdim=True)
    
    # Edge detection
    blurred = gaussian_blur(image_tensor, kernel_size=5, sigma=1.0)
    
    # Sobel filter for edges
    sobel_x = torch.tensor([[[[1, 0, -1],
                              [2, 0, -2],
                              [1, 0, -1]]]], dtype=torch.float32, device=device)
    
    sobel_y = torch.tensor([[[[1, 2, 1],
                              [0, 0, 0],
                              [-1, -2, -1]]]], dtype=torch.float32, device=device)
    
    edges_x = torch.nn.functional.conv2d(blurred, sobel_x, padding=1)
    edges_y = torch.nn.functional.conv2d(blurred, sobel_y, padding=1)
    
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
    
    # Normalize to probability distribution
    edges = edges - edges.min()
    edges = edges / (edges.max() + 1e-8)
    
    # Sample points from edge distribution
    h, w = edges.shape[2], edges.shape[3]
    edges_flat = edges.view(-1)
    
    # Generate strokes at high-probability locations
    indices = torch.multinomial(edges_flat, num_strokes, replacement=False)
    
    points_list = []
    for idx in indices:
        y = (idx // w) / h
        x = (idx % w) / w
        
        # Create control points around sampled location
        points = torch.rand(4, 2, device=device)
        points = points * 0.1 + torch.tensor([[x, y]], device=device)
        
        # Scale to canvas
        points[:, 0] = points[:, 0] * w
        points[:, 1] = points[:, 1] * h
        
        points_list.append(points)
    
    return points_list

def optimize_bezier_points(points: torch.Tensor,
                          target_function,
                          num_iterations: int = 100,
                          learning_rate: float = 0.1) -> torch.Tensor:
    """
    Optimize Bézier points using gradient descent
    
    Args:
        points: Initial control points
        target_function: Loss function
        num_iterations: Number of iterations
        learning_rate: Learning rate
    
    Returns:
        Optimized points
    """
    points = points.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([points], lr=learning_rate)
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = target_function(points)
        loss.backward()
        optimizer.step()
    
    return points.detach()