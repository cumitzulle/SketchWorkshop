"""
DiffVG differentiable renderer wrapper for CLIPasso
"""

import torch
import sys
import os
from typing import List, Tuple, Optional

def setup_diffvg():
    """Setup DiffVG for differentiable rendering"""
    try:
        import diffvg
        DIFFVG_AVAILABLE = True
        
        # Test with simple rendering
        test_diffvg()
        
        return True
        
    except ImportError:
        print("DiffVG is not installed. Please install it from:")
        print("https://github.com/BachiLi/diffvg")
        return False
    except Exception as e:
        print(f"DiffVG setup error: {e}")
        return False

def test_diffvg():
    """Test DiffVG installation"""
    import diffvg
    
    # Create simple shapes
    circle_shape = diffvg.Circle(radius=torch.tensor(50.0),
                                 center=torch.tensor([256.0, 256.0]))
    
    # Create shape group
    shape_group = diffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        use_even_odd_rule=False
    )
    
    # Render
    width, height = 512, 512
    scene_args = diffvg.RenderFunction.serialize_scene(
        width, height,
        [circle_shape], [shape_group]
    )
    
    img = diffvg.RenderFunction.apply(width, height, 2, 2, 0, None, *scene_args)
    
    return img

def create_bezier_path(points: torch.Tensor,
                      is_closed: bool = False,
                      stroke_width: float = 2.0) -> Tuple:
    """
    Create a Bézier path for DiffVG rendering
    
    Args:
        points: Control points tensor (n_points, 2)
        is_closed: Whether path is closed
        stroke_width: Stroke width
    
    Returns:
        (shape, shape_group) tuple
    """
    import diffvg
    
    shape = diffvg.Path(
        num_control_points=torch.tensor([2] * (len(points) // 3)),
        points=points,
        is_closed=is_closed,
        stroke_width=torch.tensor(stroke_width)
    )
    
    shape_group = diffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=None,
        stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        use_even_odd_rule=False
    )
    
    return shape, shape_group

def render_paths(shapes: List,
                shape_groups: List,
                width: int = 512,
                height: int = 512,
                spp: int = 4,
                num_samples: int = 1) -> torch.Tensor:
    """
    Render paths using DiffVG
    
    Args:
        shapes: List of DiffVG shapes
        shape_groups: List of shape groups
        width: Output width
        height: Output height
        spp: Samples per pixel
        num_samples: Number of MSAA samples
    
    Returns:
        Rendered image tensor (H, W, 4)
    """
    import diffvg
    
    scene_args = diffvg.RenderFunction.serialize_scene(
        width, height,
        shapes, shape_groups,
        spp, num_samples,
        0, None
    )
    
    img = diffvg.RenderFunction.apply(
        width, height,
        spp, num_samples,
        0, None,
        *scene_args
    )
    
    return img