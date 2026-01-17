"""
Score Distillation Sampling (SDS) loss implementation
Based on DiffSketcher and DreamFusion papers
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from diffusers import StableDiffusionPipeline, DDIMScheduler

class SDSCalculator:
    """Calculate SDS loss for vector sketch optimization"""
    
    def __init__(self,
                 model_name: str = "runwayml/stable-diffusion-v1-5",
                 device: str = "cuda"):
        
        self.device = device
        
        # Load diffusion model
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        # Use DDIM scheduler
        self.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        # Freeze model
        self.pipeline.vae.eval()
        self.pipeline.unet.eval()
        self.pipeline.text_encoder.eval()
    
    def calculate_sds_loss(self,
                          image_tensor: torch.Tensor,
                          text_prompt: str,
                          guidance_scale: float = 100.0,
                          t: Optional[int] = None) -> torch.Tensor:
        """
        Calculate SDS loss for an image tensor
        
        Args:
            image_tensor: Input image tensor (1, 3, H, W) in [-1, 1]
            text_prompt: Text prompt for guidance
            guidance_scale: CFG guidance scale
            t: Timestep for noise injection (random if None)
        
        Returns:
            SDS loss tensor
        """
        # Encode image to latents
        with torch.no_grad():
            latents = self.pipeline.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * self.pipeline.vae.config.scaling_factor
        
        # Sample random timestep if not provided
        if t is None:
            t = torch.randint(
                low=int(0.05 * len(self.scheduler.timesteps)),
                high=int(0.95 * len(self.scheduler.timesteps)),
                size=(1,),
                device=self.device
            ).item()
        
        # Add noise
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        
        # Get text embeddings
        text_input = self.pipeline.tokenizer(
            [text_prompt],
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.pipeline.text_encoder(
            text_input.input_ids.to(self.device)
        )[0]
        
        # Get unconditional embeddings
        uncond_input = self.pipeline.tokenizer(
            [""],
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.pipeline.text_encoder(
            uncond_input.input_ids.to(self.device)
        )[0]
        
        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Predict noise with classifier-free guidance
        noisy_latents = noisy_latents.repeat(2, 1, 1, 1)
        noise_pred = self.pipeline.unet(
            noisy_latents,
            torch.tensor([t] * 2, device=self.device),
            encoder_hidden_states=text_embeddings
        ).sample
        
        # Apply classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # SDS loss gradient
        w_t = self._get_weighting(t)
        grad = w_t * (noise_pred - noise)
        
        return grad.mean()
    
    def _get_weighting(self, t: int) -> float:
        """Get weighting term w(t) for SDS loss"""
        # Simplified weighting based on timestep
        # In practice, this should follow the diffusion scheduler
        alpha_t = self.scheduler.alphas_cumprod[t]
        sigma_t = (1 - alpha_t) ** 0.5
        return (1 - alpha_t) / sigma_t