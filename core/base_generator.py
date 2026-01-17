"""
Base Stable Diffusion generator with optional M3S integration
"""

import torch
from typing import Dict, Optional, Union, Tuple
from PIL import Image
import numpy as np

from diffusers import StableDiffusionPipeline, DDIMScheduler

class BaseGenerator:
    """Base generator using Stable Diffusion with M3S style injection"""
    
    def __init__(self,
                 device: str = "cuda",
                 model_id: str = "runwayml/stable-diffusion-v1-5",
                 cache_dir: str = "./cache",
                 verbose: bool = False):
        
        self.device = device
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.verbose = verbose
        
        # Load pipeline
        self.pipeline = None
        self._load_pipeline()
        
        if self.verbose:
            print(f"Base Generator loaded: {model_id}")
    
    def _load_pipeline(self):
        """Load Stable Diffusion pipeline"""
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                cache_dir=self.cache_dir,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Use DDIM scheduler for better inversion
            self.pipeline.scheduler = DDIMScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable attention slicing for memory efficiency
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
                
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise
    
    def generate(self,
                prompt: str,
                params: Dict) -> Image.Image:
        """
        Generate base image using Stable Diffusion
        
        Args:
            prompt: Text prompt
            params: Generation parameters
        
        Returns:
            PIL Image
        """
        if self.verbose:
            print(f"Generating base image: {prompt}")
        
        generator = None
        if params.get("seed") is not None:
            generator = torch.Generator(device=self.device).manual_seed(params["seed"])
        
        # Generate image
        with torch.autocast(self.device) if self.device == "cuda" else torch.no_grad():
            image = self.pipeline(
                prompt=prompt,
                negative_prompt="text, signature, watermark, blurry, low quality",
                num_inference_steps=params.get("num_inference_steps", 50),
                guidance_scale=params.get("guidance_scale", 7.5),
                width=params.get("width", 512),
                height=params.get("height", 512),
                generator=generator
            ).images[0]
        
        return image
    
    def generate_with_m3s_style(self,
                               prompt: str,
                               style_injector,
                               reference_path: str,
                               injection_params: Dict,
                               base_params: Dict) -> Image.Image:
        """
        Generate image with M3S style injection
        
        Args:
            prompt: Text prompt
            style_injector: M3SStyleInjector instance
            reference_path: Path to style reference image
            injection_params: M3S injection parameters
            base_params: Base generation parameters
        
        Returns:
            PIL Image with style injection
        """
        if self.verbose:
            print(f"Generating with M3S style: {reference_path}")
        
        # Load style reference
        from PIL import Image as PILImage
        style_image = PILImage.open(reference_path).convert("RGB")
        
        # Extract style features
        style_injector.extract_style_features(
            style_image=style_image,
            unet=self.pipeline.unet
        )
        
        # Register injection hooks
        eta = injection_params.get("eta", 1.0)
        style_injector.register_injection_hooks(
            unet=self.pipeline.unet,
            eta=eta
        )
        
        try:
            # Generate with style injection
            image = self.generate(prompt, base_params)
            
            return image
            
        finally:
            # Clean up hooks
            style_injector.remove_injection_hooks(self.pipeline.unet)