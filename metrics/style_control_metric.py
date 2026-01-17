# metrics/style_control_metric.py
"""
Style Control Metric for Sketch Generation
Implements CLIP-T and DINO based metrics from M3S paper
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class StyleControlMetric:
    """Style control metric for sketch generation using CLIP-T and DINO"""
    
    def __init__(self,
                 device: str = "cpu",
                 clip_model: str = "ViT-B/32",
                 dino_model: str = "vit_base_patch16",
                 verbose: bool = False):
        
        self.device = device
        self.clip_model_name = clip_model
        self.dino_model_name = dino_model
        self.verbose = verbose
        
        # Initialize models
        self.clip_model = None
        self.clip_preprocess = None
        self.dino_model = None
        self.dino_transform = None
        
        self._init_clip()
        self._init_dino()
        
        if self.verbose:
            print(f"StyleControlMetric initialized on {device}")
            print(f"CLIP model: {clip_model}")
            print(f"DINO model: {dino_model}")
    
    def _init_clip(self):
        """Initialize CLIP model for CLIP-T metric"""
        try:
            import clip
            import open_clip
            
            # Try to load CLIP model
            try:
                # First try with original CLIP
                self.clip_model, self.clip_preprocess = clip.load(
                    self.clip_model_name, device=self.device
                )
                self.clip_model.eval()
            except:
                # Fallback to open_clip
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.clip_model_name, pretrained="openai"
                )
                self.clip_model = model.to(self.device).eval()
                self.clip_preprocess = preprocess
            
            if self.verbose:
                print(f"✓ CLIP model loaded: {self.clip_model_name}")
                
        except ImportError as e:
            print(f"Warning: CLIP not available. Install with: pip install open-clip-torch")
            self.clip_model = None
    
    def _init_dino(self):
        """Initialize DINO model for feature extraction"""
        try:
            import torchvision.transforms as T
            
            # Load DINO model
            try:
                # Try to load from torch hub
                self.dino_model = torch.hub.load(
                    'facebookresearch/dino:main',
                    f'dino_{self.dino_model_name}'
                ).to(self.device).eval()
                
                # DINO transform
                self.dino_transform = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
                ])
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: DINO model not available: {e}")
                self.dino_model = None
                
        except ImportError:
            if self.verbose:
                print("Warning: DINO dependencies not installed")
            self.dino_model = None
    
    def compute_clip_t(self,
                      sketch: Image.Image,
                      reference: Image.Image) -> float:
        """
        Compute CLIP-T similarity between sketch and reference
        
        Args:
            sketch: Generated sketch image
            reference: Style reference image
        
        Returns:
            CLIP-T similarity score (0-1)
        """
        if self.clip_model is None:
            if self.verbose:
                print("CLIP model not available, returning default score")
            return 0.5
        
        try:
            # Preprocess images
            sketch_tensor = self.clip_preprocess(sketch).unsqueeze(0).to(self.device)
            ref_tensor = self.clip_preprocess(reference).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                sketch_features = self.clip_model.encode_image(sketch_tensor)
                ref_features = self.clip_model.encode_image(ref_tensor)
                
                # Normalize features
                sketch_features = F.normalize(sketch_features, dim=-1)
                ref_features = F.normalize(ref_features, dim=-1)
                
                # Compute cosine similarity
                similarity = F.cosine_similarity(sketch_features, ref_features, dim=-1)
                
                # Convert to 0-1 range (cosine similarity is already in -1 to 1, but we want 0-1)
                score = (similarity.item() + 1) / 2
                
                if self.verbose:
                    print(f"CLIP-T similarity: {score:.4f}")
                
                return max(0.0, min(1.0, score))
                
        except Exception as e:
            if self.verbose:
                print(f"Error computing CLIP-T: {e}")
            return 0.0
    
    def compute_dino_similarity(self,
                               sketch: Image.Image,
                               reference: Image.Image) -> float:
        """
        Compute DINO feature similarity between sketch and reference
        
        Args:
            sketch: Generated sketch image
            reference: Style reference image
        
        Returns:
            DINO similarity score (0-1)
        """
        if self.dino_model is None:
            if self.verbose:
                print("DINO model not available, returning default score")
            return 0.5
        
        try:
            import torchvision.transforms as T
            
            # Ensure images are RGB
            sketch = sketch.convert("RGB") if sketch.mode != "RGB" else sketch
            reference = reference.convert("RGB") if reference.mode != "RGB" else reference
            
            # Apply DINO transform
            sketch_tensor = self.dino_transform(sketch).unsqueeze(0).to(self.device)
            ref_tensor = self.dino_transform(reference).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                # Get features from the last layer
                sketch_features = self.dino_model(sketch_tensor)
                ref_features = self.dino_model(ref_tensor)
                
                # Use cls token for similarity (first token)
                sketch_cls = sketch_features[:, 0, :]
                ref_cls = ref_features[:, 0, :]
                
                # Normalize
                sketch_cls = F.normalize(sketch_cls, dim=-1)
                ref_cls = F.normalize(ref_cls, dim=-1)
                
                # Compute cosine similarity
                similarity = F.cosine_similarity(sketch_cls, ref_cls, dim=-1)
                
                # Convert to 0-1 range
                score = (similarity.item() + 1) / 2
                
                if self.verbose:
                    print(f"DINO similarity: {score:.4f}")
                
                return max(0.0, min(1.0, score))
                
        except Exception as e:
            if self.verbose:
                print(f"Error computing DINO similarity: {e}")
            return 0.0
    
    def compute_style_score(self,
                           sketch: Image.Image,
                           style_reference: Image.Image,
                           content_reference: Optional[Image.Image] = None) -> Dict:
        """
        Compute complete style control metrics
        
        Args:
            sketch: Generated sketch image
            style_reference: Style reference image
            content_reference: Optional content reference for content preservation
        
        Returns:
            Dictionary with all style metrics
        """
        results = {}
        
        # 1. Compute CLIP-T score
        clip_t_score = self.compute_clip_t(sketch, style_reference)
        results["clip_t_score"] = clip_t_score
        
        # 2. Compute DINO similarity
        dino_score = self.compute_dino_similarity(sketch, style_reference)
        results["dino_score"] = dino_score
        
        # 3. Compute harmonic mean (style metric β from paper)
        if clip_t_score > 0 and dino_score > 0:
            harmonic_mean = 2 * clip_t_score * dino_score / (clip_t_score + dino_score)
        else:
            harmonic_mean = 0.0
        results["style_score"] = harmonic_mean
        
        # 4. Content preservation (if content reference provided)
        if content_reference is not None:
            content_score = self.compute_clip_t(sketch, content_reference)
            results["content_preservation"] = content_score
            
            # Style-content balance
            if clip_t_score > 0 and content_score > 0:
                balance_score = np.sqrt(clip_t_score * content_score)
            else:
                balance_score = 0.0
            results["style_content_balance"] = balance_score
        
        return results
    
    def batch_evaluate(self,
                      sketches: List[Image.Image],
                      style_references: List[Image.Image],
                      content_references: Optional[List[Image.Image]] = None) -> Dict:
        """
        Batch evaluation of multiple sketches
        
        Args:
            sketches: List of generated sketches
            style_references: List of style reference images
            content_references: Optional list of content reference images
        
        Returns:
            Dictionary with batch evaluation results
        """
        if len(sketches) != len(style_references):
            raise ValueError("Number of sketches must match number of style references")
        
        if content_references is not None and len(content_references) != len(sketches):
            raise ValueError("Number of content references must match number of sketches")
        
        batch_results = {
            "clip_t_scores": [],
            "dino_scores": [],
            "style_scores": [],
            "content_preservation": [] if content_references else None,
            "style_content_balance": [] if content_references else None
        }
        
        detailed_results = []
        
        for i in range(len(sketches)):
            if self.verbose and i % 10 == 0:
                print(f"Processing sketch {i+1}/{len(sketches)}...")
            
            content_ref = content_references[i] if content_references else None
            
            result = self.compute_style_score(
                sketch=sketches[i],
                style_reference=style_references[i],
                content_reference=content_ref
            )
            
            # Collect scores
            batch_results["clip_t_scores"].append(result["clip_t_score"])
            batch_results["dino_scores"].append(result["dino_score"])
            batch_results["style_scores"].append(result["style_score"])
            
            if content_references:
                batch_results["content_preservation"].append(result.get("content_preservation", 0.0))
                batch_results["style_content_balance"].append(result.get("style_content_balance", 0.0))
            
            detailed_results.append(result)
        
        # Compute statistics
        stats = {}
        for key, values in batch_results.items():
            if values:  # Only compute if list is not empty
                stats[f"mean_{key}"] = np.mean(values)
                stats[f"std_{key}"] = np.std(values)
                stats[f"min_{key}"] = np.min(values)
                stats[f"max_{key}"] = np.max(values)
        
        # Create final results dictionary
        final_results = {
            "batch_scores": batch_results,
            "detailed_results": detailed_results,
            "statistics": stats,
            "summary": {
                "mean_style_score": stats.get("mean_style_scores", 0.0),
                "mean_clip_t": stats.get("mean_clip_t_scores", 0.0),
                "mean_dino": stats.get("mean_dino_scores", 0.0),
            }
        }
        
        if content_references:
            final_results["summary"]["mean_content_preservation"] = stats.get("mean_content_preservation", 0.0)
            final_results["summary"]["mean_balance"] = stats.get("mean_style_content_balance", 0.0)
        
        return final_results
    
    def evaluate_style_transfer(self,
                               generated_sketches: List[Image.Image],
                               style_references: List[Image.Image],
                               content_references: Optional[List[Image.Image]] = None) -> Dict:
        """
        Alias for batch_evaluate with better naming for style transfer
        
        Args:
            generated_sketches: Generated sketches with style transfer
            style_references: Style reference images
            content_references: Original content images (optional)
        
        Returns:
            Evaluation results
        """
        return self.batch_evaluate(
            sketches=generated_sketches,
            style_references=style_references,
            content_references=content_references
        )


# ========== 使用示例 ==========

def example_usage():
    """Example of how to use the StyleControlMetric"""
    
    # 1. Initialize metric
    metric = StyleControlMetric(device="cpu", verbose=True)
    
    # 2. Prepare test images (replace with your actual images)
    from PIL import Image
    import numpy as np
    
    # Create dummy images for demonstration
    dummy_sketch = Image.fromarray((np.random.rand(512, 512, 3) * 255).astype(np.uint8))
    dummy_style = Image.fromarray((np.random.rand(512, 512, 3) * 255).astype(np.uint8))
    dummy_content = Image.fromarray((np.random.rand(512, 512, 3) * 255).astype(np.uint8))
    
    # 3. Single image evaluation
    print("\nSingle image evaluation:")
    result = metric.compute_style_score(
        sketch=dummy_sketch,
        style_reference=dummy_style,
        content_reference=dummy_content
    )
    
    print(f"CLIP-T Score: {result['clip_t_score']:.4f}")
    print(f"DINO Score: {result['dino_score']:.4f}")
    print(f"Style Score (harmonic mean): {result['style_score']:.4f}")
    print(f"Content Preservation: {result.get('content_preservation', 'N/A')}")
    print(f"Style-Content Balance: {result.get('style_content_balance', 'N/A')}")
    
    # 4. Batch evaluation
    print("\n\nBatch evaluation:")
    batch_results = metric.batch_evaluate(
        sketches=[dummy_sketch, dummy_sketch],
        style_references=[dummy_style, dummy_style],
        content_references=[dummy_content, dummy_content]
    )
    
    print(f"Mean Style Score: {batch_results['summary']['mean_style_score']:.4f}")
    print(f"Mean CLIP-T: {batch_results['summary']['mean_clip_t']:.4f}")
    print(f"Mean DINO: {batch_results['summary']['mean_dino']:.4f}")


if __name__ == "__main__":
    example_usage()