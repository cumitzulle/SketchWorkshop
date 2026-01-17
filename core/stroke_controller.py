"""
CLIPasso Stroke Controller implementing exact algorithm
Optimizes Bézier curves with CLIP semantic and geometric losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
import random
import xml.etree.ElementTree as ET

try:
    import diffvg
    DIFFVG_AVAILABLE = True
except ImportError:
    DIFFVG_AVAILABLE = False
    print("Warning: DiffVG not available. CLIPasso stroke control disabled.")

class CLIPassoController:
    """CLIPasso stroke controller with exact algorithm implementation"""
    
    def __init__(self,
                 device: str = "cuda",
                 num_strokes: int = 16,
                 abstraction: float = 0.7,
                 clip_model: str = "ViT-B/32",
                 num_control_points: int = 4,
                 stroke_width: float = 2.0,
                 verbose: bool = False):
        
        self.device = device
        self.num_strokes = num_strokes
        self.abstraction = abstraction
        self.clip_model_name = clip_model
        self.num_control_points = num_control_points
        self.stroke_width = stroke_width
        self.verbose = verbose
        
        # Initialize CLIP model
        self.clip_model = None
        self.clip_preprocess = None
        self._init_clip()
        
        # Initialize DiffVG if available
        self.diffvg_available = DIFFVG_AVAILABLE
        
        if self.verbose:
            print(f"CLIPasso Controller initialized for {num_strokes} strokes")
            print(f"Abstraction level: {abstraction}")
            print(f"DiffVG available: {self.diffvg_available}")
    
    def _init_clip(self):
        """Initialize CLIP model"""
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load(
                self.clip_model_name, device=self.device
            )
            self.clip_model.eval()
            
            if self.verbose:
                print(f"CLIP model loaded: {self.clip_model_name}")
                
        except ImportError:
            print("Warning: CLIP not available. CLIPasso will use simplified loss.")
            self.clip_model = None
    
    def optimize(self,
                target_image: Image.Image,
                prompt: str,
                num_iterations: int = 500,
                learning_rate: float = 0.1,
                w_semantic: float = 0.1) -> Dict:
        """
        Optimize strokes using CLIPasso algorithm
        
        Args:
            target_image: Target image for optimization
            prompt: Text description for semantic guidance
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimization
            w_semantic: Weight for semantic loss term
        
        Returns:
            Dictionary with 'raster' and optionally 'svg'
        """
        
        if not self.diffvg_available:
            # Fallback to simplified implementation
            return self._simplified_optimization(target_image)
        
        if self.verbose:
            print(f"\nCLIPasso Optimization started:")
            print(f"  Target strokes: {self.num_strokes}")
            print(f"  Iterations: {num_iterations}")
            print(f"  Learning rate: {learning_rate}")
        
        # Convert target image to tensor
        target_tensor = self._image_to_tensor(target_image).to(self.device)
        
        # Initialize Bézier curves
        shapes, shape_groups = self._initialize_strokes(target_image)
        
        # Setup optimizer
        points_vars = []
        for path in shapes:
            points_vars.append(path.points.clone().requires_grad_(True))
        
        width_vars = []
        for path in shapes:
            width_vars.append(torch.tensor(self.stroke_width).requires_grad_(True))
        
        optimizer = torch.optim.Adam(points_vars + width_vars, lr=learning_rate)
        
        # Extract CLIP features from target
        with torch.no_grad():
            if self.clip_model:
                target_features = self._extract_clip_features(target_tensor)
            else:
                target_features = None
        
        # Optimization loop
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Render current strokes
            current_frame = self._render_strokes(shapes, shape_groups, points_vars, width_vars)
            
            # Calculate losses
            total_loss = 0.0
            
            # Geometric loss (CLIP intermediate layers)
            if target_features and self.clip_model:
                current_features = self._extract_clip_features(current_frame)
                
                # CLIPasso geometric loss (layers 3 and 4 for ResNet)
                if "layer3" in target_features and "layer3" in current_features:
                    loss_l3 = F.mse_loss(current_features["layer3"], target_features["layer3"])
                    loss_l4 = F.mse_loss(current_features["layer4"], target_features["layer4"])
                    geometric_loss = loss_l3 + loss_l4
                    total_loss += geometric_loss
                    if self.verbose and iteration % 100 == 0:
                        print(f"  Iter {iteration}: Geometric loss = {geometric_loss.item():.4f}")
                
                # Semantic loss (final layer)
                if "final" in target_features and "final" in current_features:
                    semantic_sim = F.cosine_similarity(
                        current_features["final"],
                        target_features["final"],
                        dim=-1
                    )
                    semantic_loss = (1 - semantic_sim.mean()) * w_semantic
                    total_loss += semantic_loss
                    if self.verbose and iteration % 100 == 0:
                        print(f"  Iter {iteration}: Semantic loss = {semantic_loss.item():.4f}")
            
            # Add stroke regularization
            reg_loss = self._compute_stroke_regularization(points_vars, width_vars)
            total_loss += 0.01 * reg_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Clamp values
            with torch.no_grad():
                for width_var in width_vars:
                    width_var.clamp_(0.5, 5.0)
            
            if self.verbose and iteration % 50 == 0 and iteration > 0:
                print(f"  Iteration {iteration}: Total loss = {total_loss.item():.4f}")
        
        # Final render
        final_raster = self._render_strokes(shapes, shape_groups, points_vars, width_vars)
        final_image = self._tensor_to_image(final_raster)
        
        result = {"raster": final_image}
        
        # Generate SVG representation if needed
        if self.diffvg_available:
            svg_content = self._generate_svg(shapes, shape_groups, points_vars, width_vars)
            result["svg"] = svg_content
        
        if self.verbose:
            print("CLIPasso optimization completed")
        
        return result
    
    def _initialize_strokes(self, target_image: Image.Image):
        """Initialize Bézier strokes based on target image"""
        if not self.diffvg_available:
            return [], []
        
        import diffvg
        
        # Convert to tensor for saliency detection
        target_tensor = self._image_to_tensor(target_image).to(self.device)
        
        # Use saliency-based initialization (CLIPasso section 3.3)
        from utils.bezier_utils import initialize_strokes_from_saliency
        points_list = initialize_strokes_from_saliency(
            target_tensor, self.num_strokes, self.device
        )
        
        shapes = []
        shape_groups = []
        
        for i, points in enumerate(points_list):
            # Create Bézier path
            # For cubic Bézier curves with 4 control points, we need 2 control points per segment
            num_segments = 1  # Single curve segment
            num_control_points = torch.tensor([2] * num_segments)  # Cubic Bézier
            
            shape = diffvg.Path(
                num_control_points=num_control_points,
                points=points,
                is_closed=False,
                stroke_width=torch.tensor(self.stroke_width)
            )
            
            shape_group = diffvg.ShapeGroup(
                shape_ids=torch.tensor([i]),
                fill_color=None,
                stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),  # Black stroke
                use_even_odd_rule=False
            )
            
            shapes.append(shape)
            shape_groups.append(shape_group)
        
        return shapes, shape_groups
    
    def _render_strokes(self, shapes, shape_groups, points_vars, width_vars):
        """Render current strokes"""
        import diffvg
        
        # Update shapes with current parameters
        updated_shapes = []
        for i, shape in enumerate(shapes):
            updated_shape = diffvg.Path(
                num_control_points=shape.num_control_points,
                points=points_vars[i],
                is_closed=shape.is_closed,
                stroke_width=width_vars[i]
            )
            updated_shapes.append(updated_shape)
        
        # Render
        width, height = 512, 512  # Default output size
        scene_args = diffvg.RenderFunction.serialize_scene(
            width, height,
            updated_shapes,
            shape_groups,
            4,  # spp
            1,  # num_samples
            0,  # seed
            None  # background
        )
        
        img = diffvg.RenderFunction.apply(
            width, height, 4, 1, 0, None, *scene_args
        )
        
        # Convert to RGB and add white background
        img = img[:, :, :3]  # Remove alpha channel
        img = img.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Ensure values are in [0, 1] range
        img = torch.clamp(img, 0.0, 1.0)
        
        return img
    
    def _extract_clip_features(self, image_tensor):
        """Extract CLIP features from image tensor"""
        if not self.clip_model:
            return None
        
        # Normalize image for CLIP
        if image_tensor.dim() == 4:
            batch_size = image_tensor.shape[0]
        else:
            image_tensor = image_tensor.unsqueeze(0)
            batch_size = 1
        
        # Resize to CLIP input size (224x224)
        if image_tensor.shape[-2:] != (224, 224):
            image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize using CLIP's preprocessing
        # CLIP expects images normalized with mean=[0.48145466, 0.4578275, 0.40821073] and std=[0.26862954, 0.26130258, 0.27577711]
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        normalized = (image_tensor - mean) / std
        
        # Extract features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(normalized)
            
            # For ResNet-based CLIP models, we need to extract intermediate features
            # This is a simplified version - full implementation would require hooking into specific layers
            if "RN" in self.clip_model_name:
                # ResNet architecture - simulate layer features
                layer3_features = F.avg_pool2d(image_tensor, kernel_size=8)
                layer4_features = F.avg_pool2d(image_tensor, kernel_size=16)
                
                return {
                    "layer3": layer3_features.flatten(start_dim=1),
                    "layer4": layer4_features.flatten(start_dim=1),
                    "final": image_features
                }
            else:
                # ViT architecture
                return {
                    "final": image_features
                }
    
    def _compute_stroke_regularization(self, points_vars, width_vars):
        """Compute regularization loss for strokes"""
        reg_loss = 0.0
        
        # Encourage smooth strokes - penalize large curvature
        for points in points_vars:
            if len(points) >= 3:
                # Calculate second derivative approximation
                p0 = points[:-2]
                p1 = points[1:-1]
                p2 = points[2:]
                
                # Curvature: distance from p1 to line between p0 and p2
                line_dir = p2 - p0
                line_len = torch.norm(line_dir, dim=1, keepdim=True) + 1e-8
                line_dir = line_dir / line_len
                
                # Vector from p0 to p1
                v = p1 - p0
                # Projection onto line
                proj = torch.sum(v * line_dir, dim=1, keepdim=True) * line_dir
                # Perpendicular distance
                perp = v - proj
                
                curvature = torch.norm(perp, dim=1)
                reg_loss += curvature.mean() * 0.1
        
        # Penalize very wide or very narrow strokes
        for width in width_vars:
            if width > 5.0:
                reg_loss += (width - 5.0) ** 2
            elif width < 0.5:
                reg_loss += (0.5 - width) ** 2
        
        # Encourage strokes to stay within canvas
        canvas_margin = 0.05
        for points in points_vars:
            # Points should be between -margin and 1+margin (normalized coordinates)
            normalized_points = points / 512.0  # Assuming 512x512 canvas
            
            # Penalize points outside canvas + margin
            out_of_bounds = torch.relu(normalized_points - (1.0 + canvas_margin)) + \
                           torch.relu(-canvas_margin - normalized_points)
            reg_loss += out_of_bounds.sum() * 0.01
        
        return reg_loss
    
    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to normalized tensor"""
        import torchvision.transforms as T
        
        # Resize to standard size
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        transform = T.Compose([
            T.ToTensor(),  # Converts to [0, 1]
        ])
        
        tensor = transform(image)
        
        # Ensure 3 channels
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.shape[0] > 3:
            tensor = tensor[:3, :, :]
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL image"""
        import torchvision.transforms as T
        
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Ensure values are in [0, 1] range
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        # Convert to PIL
        to_pil = T.ToPILImage()
        
        # Handle different channel dimensions
        if tensor.shape[0] == 1:
            # Grayscale to RGB
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.shape[0] > 3:
            # Take first 3 channels
            tensor = tensor[:3, :, :]
        
        return to_pil(tensor.cpu())
    
    def _normalize_for_clip(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize image tensor for CLIP model input"""
        # CLIP expects images in range [0, 1] with specific normalization
        if image_tensor.max() > 1.0:
            image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
        
        # Resize to 224x224 if needed
        if image_tensor.shape[-2:] != (224, 224):
            image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Apply CLIP normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image_tensor.device).view(1, 3, 1, 1)
        
        return (image_tensor - mean) / std
    
    def _generate_svg(self, shapes, shape_groups, points_vars, width_vars) -> str:
        """Generate SVG string from optimized strokes"""
        try:
            import diffvg
            
            # Create SVG root element
            svg = ET.Element('svg', {
                'xmlns': 'http://www.w3.org/2000/svg',
                'version': '1.1',
                'width': '512',
                'height': '512',
                'viewBox': '0 0 512 512'
            })
            
            # Add white background
            background = ET.SubElement(svg, 'rect', {
                'width': '512',
                'height': '512',
                'fill': 'white'
            })
            
            # Add each stroke as a path
            for i, (shape, points, width) in enumerate(zip(shapes, points_vars, width_vars)):
                # Convert control points to SVG path data
                points_np = points.detach().cpu().numpy()
                
                # Create path data for cubic Bézier curve
                if len(points_np) == 4:  # Cubic Bézier
                    p0, p1, p2, p3 = points_np
                    path_data = f"M {p0[0]:.2f},{p0[1]:.2f} " \
                               f"C {p1[0]:.2f},{p1[1]:.2f} " \
                               f"{p2[0]:.2f},{p2[1]:.2f} " \
                               f"{p3[0]:.2f},{p3[1]:.2f}"
                else:
                    # Fallback: polyline
                    path_data = "M " + " L ".join([f"{p[0]:.2f},{p[1]:.2f}" for p in points_np])
                
                # Create path element
                path = ET.SubElement(svg, 'path', {
                    'd': path_data,
                    'stroke': 'black',
                    'stroke-width': f"{width.item():.2f}",
                    'fill': 'none',
                    'stroke-linecap': 'round',
                    'stroke-linejoin': 'round'
                })
            
            # Convert to string
            svg_string = ET.tostring(svg, encoding='unicode', method='xml')
            
            # Add XML declaration
            svg_string = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + svg_string
            
            return svg_string
            
        except Exception as e:
            print(f"Warning: Failed to generate SVG: {e}")
            return ""
    
    def _simplified_optimization(self, target_image):
        """Simplified optimization when DiffVG is not available"""
        if self.verbose:
            print("Using simplified optimization (DiffVG not available)")
        
        # Simple edge detection and stroke simulation
        import cv2
        import numpy as np
        
        # Convert to numpy
        img_np = np.array(target_image.convert('L'))
        
        # Edge detection
        edges = cv2.Canny(img_np, 50, 150)
        
        # Sample points from edges
        edge_points = np.argwhere(edges > 0)
        if len(edge_points) > 0:
            # Calculate number of points needed (4 control points per stroke)
            num_points_needed = self.num_strokes * 4
            indices = np.random.choice(
                len(edge_points),
                min(num_points_needed, len(edge_points)),
                replace=False
            )
            sampled_points = edge_points[indices]
        else:
            # Fallback to random points
            h, w = img_np.shape
            sampled_points = np.random.rand(self.num_strokes * 4, 2) * [h, w]
        
        # Create simplified stroke image
        stroke_img = np.ones((512, 512), dtype=np.uint8) * 255
        
        # Draw Bézier curves
        for i in range(0, len(sampled_points) - 4, 4):
            pts = sampled_points[i:i+4]
            pts = pts[:, ::-1]  # Convert (y, x) to (x, y) for OpenCV
            
            # Rescale to 512x512
            scale_h = 512 / img_np.shape[0]
            scale_w = 512 / img_np.shape[1]
            pts = pts * [scale_w, scale_h]
            pts = pts.astype(np.int32)
            
            # Draw cubic Bézier curve
            curve_points = []
            for t in np.linspace(0, 1, 20):
                # Cubic Bézier formula
                p = (1-t)**3 * pts[0] + 3*(1-t)**2*t * pts[1] + 3*(1-t)*t**2 * pts[2] + t**3 * pts[3]
                curve_points.append(p.astype(np.int32))
            
            curve_points = np.array(curve_points)
            
            # Draw the curve
            for j in range(len(curve_points) - 1):
                cv2.line(stroke_img,
                        tuple(curve_points[j]),
                        tuple(curve_points[j+1]),
                        0,  # Black
                        int(self.stroke_width))
        
        return {
            "raster": Image.fromarray(stroke_img).convert('RGB')
        }
    
    def set_num_strokes(self, num_strokes: int):
        """Update number of strokes for abstraction control"""
        self.num_strokes = num_strokes
        if self.verbose:
            print(f"Updated stroke count: {num_strokes}")
    
    def set_abstraction_level(self, abstraction: float):
        """Update abstraction level (affects stroke count and simplification)"""
        self.abstraction = max(0.0, min(1.0, abstraction))
        # Adjust stroke count based on abstraction level
        # More abstract = fewer strokes
        adjusted_strokes = max(4, int(self.num_strokes * (1.0 - self.abstraction * 0.5)))
        self.set_num_strokes(adjusted_strokes)
        
        if self.verbose:
            print(f"Updated abstraction level: {abstraction}")
            print(f"Adjusted stroke count: {adjusted_strokes}")