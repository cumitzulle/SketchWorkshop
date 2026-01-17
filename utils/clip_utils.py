"""
CLIP model utilities for CLIPasso semantic and geometric losses
"""

import torch
import clip
from typing import Tuple, Optional

class CLIPFeatureExtractor:
    """Extract features from CLIP model for semantic and geometric losses"""
    
    def __init__(self,
                 model_name: str = "ViT-B/32",
                 device: str = "cuda"):
        
        self.device = device
        self.model_name = model_name
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        
        # Store intermediate features
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture intermediate features"""
        
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # Register hooks for different layers
        if "ViT" in self.model_name:
            # Vision Transformer architecture
            for i, layer in enumerate(self.model.visual.transformer.resblocks):
                layer.register_forward_hook(get_hook(f"vit_layer_{i}"))
        else:
            # ResNet architecture (for CLIPasso geometric loss)
            for name, module in self.model.visual.named_modules():
                if "layer" in name:
                    module.register_forward_hook(get_hook(name))
    
    def extract_features(self, image: torch.Tensor) -> dict:
        """
        Extract features from image
        
        Args:
            image: Input image tensor (normalized)
        
        Returns:
            Dictionary of features
        """
        with torch.no_grad():
            self.features.clear()
            _ = self.model.encode_image(image)
        
        return self.features.copy()
    
    def get_semantic_features(self, image: torch.Tensor) -> torch.Tensor:
        """Get final semantic features (last layer)"""
        features = self.extract_features(image)
        
        if "ViT" in self.model_name:
            # Last layer of ViT
            last_layer_key = f"vit_layer_{len(self.model.visual.transformer.resblocks)-1}"
            if last_layer_key in features:
                return features[last_layer_key][:, 0, :]  # Class token
        else:
            # Last layer of ResNet
            return self.model.encode_image(image)
        
        raise ValueError("Could not extract semantic features")
    
    def get_geometric_features(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get geometric features from intermediate layers (CLIPasso layers 3 and 4)
        
        Returns:
            Tuple of (layer3_features, layer4_features)
        """
        features = self.extract_features(image)
        
        if "RN" in self.model_name:
            # ResNet architecture
            layer3_key = "layer3"
            layer4_key = "layer4"
            
            if layer3_key in features and layer4_key in features:
                return features[layer3_key], features[layer4_key]
        
        # Fallback: use specific ViT layers for geometric features
        if "ViT" in self.model_name:
            # Use middle layers for geometric information
            mid_layer = len(self.model.visual.transformer.resblocks) // 2
            layer3_key = f"vit_layer_{mid_layer-1}"
            layer4_key = f"vit_layer_{mid_layer}"
            
            if layer3_key in features and layer4_key in features:
                # Use patch embeddings (excluding class token)
                layer3_feat = features[layer3_key][:, 1:, :].mean(dim=1)
                layer4_feat = features[layer4_key][:, 1:, :].mean(dim=1)
                return layer3_feat, layer4_feat
        
        raise ValueError("Could not extract geometric features")
    
    def compute_semantic_loss(self,
                             image1: torch.Tensor,
                             image2: torch.Tensor) -> torch.Tensor:
        """Compute semantic loss (cosine distance between final features)"""
        feat1 = self.get_semantic_features(image1)
        feat2 = self.get_semantic_features(image2)
        
        # Cosine distance: 1 - cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(feat1, feat2, dim=-1)
        return 1 - cosine_sim.mean()
    
    def compute_geometric_loss(self,
                              image1: torch.Tensor,
                              image2: torch.Tensor) -> torch.Tensor:
        """Compute geometric loss (L2 distance between intermediate features)"""
        layer3_1, layer4_1 = self.get_geometric_features(image1)
        layer3_2, layer4_2 = self.get_geometric_features(image2)
        
        # L2 distance (CLIPasso equation 2)
        loss_l3 = torch.nn.functional.mse_loss(layer3_1, layer3_2)
        loss_l4 = torch.nn.functional.mse_loss(layer4_1, layer4_2)
        
        return loss_l3 + loss_l4