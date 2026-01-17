"""
DiffSketcher Attribute Controller implementing SDS loss for stroke attributes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from PIL import Image
import numpy as np

class DiffSketcherController:
    """DiffSketcher attribute controller using SDS loss"""
    
    def __init__(self,
                 device: str = "cuda",
                 sds_guidance_scale: float = 100.0,
                 verbose: bool = False):
        
        self.device = device
        self.sds_guidance_scale = sds_guidance_scale
        self.verbose = verbose
        
        # Load diffusion model for SDS loss
        self.diffusion_model = None
        self.scheduler = None
        self._load_diffusion_model()
        
        if self.verbose:
            print(f"DiffSketcher Controller initialized with SDS scale: {sds_guidance_scale}")
    
    def _load_diffusion_model(self):
        """Load diffusion model for SDS loss calculation"""
        try:
            from diffusers import StableDiffusionPipeline, DDIMScheduler
            
            self.diffusion_model = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            self.scheduler = DDIMScheduler.from_config(
                self.diffusion_model.scheduler.config
            )
            
            # Freeze model
            self.diffusion_model.vae.eval()
            self.diffusion_model.unet.eval()
            self.diffusion_model.text_encoder.eval()
            
        except Exception as e:
            print(f"Error loading diffusion model: {e}")
            raise
    
    def compute_sds_loss(self,
                        image_tensor: torch.Tensor,
                        prompt: str,
                        t: int = 500) -> torch.Tensor:
        """
        Compute Score Distillation Sampling loss
        
        Args:
            image_tensor: Input image tensor (1, 3, H, W) in [-1, 1]
            prompt: Text prompt for guidance
            t: Timestep for noise injection
        
        Returns:
            SDS loss value
        """
        # Encode image to latents
        with torch.no_grad():
            latents = self.diffusion_model.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * self.diffusion_model.vae.config.scaling_factor
        
        # Add noise
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        
        # Get text embeddings
        text_input = self.diffusion_model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.diffusion_model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.diffusion_model.text_encoder(
            text_input.input_ids.to(self.device)
        )[0]
        
        # Predict noise
        noise_pred = self.diffusion_model.unet(
            noisy_latents,
            t,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # SDS loss: (noise_pred - noise) * guidance_scale
        sds_loss = (noise_pred - noise) * self.sds_guidance_scale
        
        return sds_loss.mean()
    
    def optimize(self,
                sketch: Image.Image,
                prompt: str,
                target_width: float = 1.0,
                opacity: float = 1.0,
                width_variation: float = 0.5,
                num_iterations: int = 200) -> Dict:
        """
        Optimize stroke attributes using SDS loss
        
        Args:
            sketch: Input sketch image
            prompt: Text prompt for semantic guidance
            target_width: Target stroke width multiplier
            opacity: Target stroke opacity (0-1)
            width_variation: Width variation factor
            num_iterations: Number of optimization iterations
        
        Returns:
            Dictionary with optimized raster image
        """
        if self.verbose:
            print(f"Optimizing stroke attributes:")
            print(f"  Target width: {target_width}")
            print(f"  Opacity: {opacity}")
            print(f"  Width variation: {width_variation}")
        
        # Convert sketch to tensor
        sketch_tensor = self._image_to_tensor(sketch).to(self.device)
        
        # Create parameter tensors for optimization
        width_param = torch.tensor([target_width], device=self.device, requires_grad=True)
        opacity_param = torch.tensor([opacity], device=self.device, requires_grad=True)
        
        optimizer = torch.optim.Adam([width_param, opacity_param], lr=0.01)
        
        # Optimization loop
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Apply current parameters to sketch
            modified_sketch = self._apply_attributes(
                sketch_tensor,
                width_param.item(),
                opacity_param.item(),
                width_variation
            )
            
            # Compute SDS loss
            sds_loss = self.compute_sds_loss(modified_sketch, prompt)
            
            # Add regularization
            reg_loss = F.mse_loss(width_param, torch.tensor([target_width], device=self.device)) + \
                      F.mse_loss(opacity_param, torch.tensor([opacity], device=self.device))
            
            total_loss = sds_loss + 0.1 * reg_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Clamp values
            with torch.no_grad():
                width_param.clamp_(0.1, 5.0)
                opacity_param.clamp_(0.0, 1.0)
            
            if self.verbose and iteration % 50 == 0:
                print(f"  Iteration {iteration}: Loss = {total_loss.item():.4f}")
        
        # Apply final attributes
        final_sketch = self._apply_attributes(
            sketch_tensor,
            width_param.item(),
            opacity_param.item(),
            width_variation
        )
        
        # Convert back to PIL
        final_image = self._tensor_to_image(final_sketch)
        
        return {
            "raster": final_image,
            "params": {
                "final_width": width_param.item(),
                "final_opacity": opacity_param.item()
            }
        }
    
    def _apply_attributes(self,
                         image_tensor: torch.Tensor,
                         width: float,
                         opacity: float,
                         variation: float) -> torch.Tensor:
        """
        Apply stroke attributes to image tensor
        
        Args:
            image_tensor: Input image tensor
            width: Stroke width multiplier
            opacity: Stroke opacity
            variation: Width variation factor
        
        Returns:
            Modified image tensor
        """
        # Simulate stroke width variation
        # In practice, this would modify the actual stroke widths in vector format
        # For raster images, we apply image processing
        
        # Create width mask
        _, _, h, w = image_tensor.shape
        width_map = torch.ones(1, 1, h, w, device=self.device) * width
        
        # Add variation
        if variation > 0:
            noise = torch.randn_like(width_map) * variation * 0.5
            width_map = width_map + noise
            width_map = width_map.clamp(width * 0.5, width * 1.5)
        
        # Apply to image
        # Multiply by opacity
        result = image_tensor * opacity
        
        # Apply width effect (simulated with blur)
        # In vector rendering, this would be actual stroke width
        if width > 1.0:
            from torchvision.transforms.functional import gaussian_blur
            kernel_size = int(width * 2) * 2 + 1
            sigma = width * 0.5
            result = gaussian_blur(result, kernel_size=kernel_size, sigma=sigma)
        
        return result
    
    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to normalized tensor"""
        import torchvision.transforms as T
        
        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        
        return transform(image).unsqueeze(0)
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL image"""
        import torchvision.transforms as T
        
        # Denormalize
        tensor = tensor.squeeze(0).cpu()
        tensor = (tensor + 1.0) / 2.0
        tensor = tensor.clamp(0, 1)
        
        # Convert to PIL
        to_pil = T.ToPILImage()
        return to_pil(tensor)