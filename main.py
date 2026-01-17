"""
Sketch Generation Pipeline with Fine-Grained Control
Combining CLIPasso, DiffSketcher, and M3S algorithms
Main entry point with integrated pipeline to avoid import issues
"""

import argparse
import os
import sys
import json
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ========== DYNAMIC IMPORTS ==========

def dynamic_import(module_name, module_path):
    """Dynamically import a module by path"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Cannot find module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import core modules dynamically
try:
    base_generator_module = dynamic_import(
        "base_generator", 
        project_root / "core" / "base_generator.py"
    )
    BaseGenerator = base_generator_module.BaseGenerator
except Exception as e:
    print(f"Warning: Could not import BaseGenerator: {e}")
    BaseGenerator = None

try:
    style_injector_module = dynamic_import(
        "style_injector",
        project_root / "core" / "style_injector.py"
    )
    M3SStyleInjector = style_injector_module.M3SStyleInjector
except Exception as e:
    print(f"Warning: Could not import M3SStyleInjector: {e}")
    M3SStyleInjector = None

try:
    stroke_controller_module = dynamic_import(
        "stroke_controller",
        project_root / "core" / "stroke_controller.py"
    )
    CLIPassoController = stroke_controller_module.CLIPassoController
except Exception as e:
    print(f"Warning: Could not import CLIPassoController: {e}")
    CLIPassoController = None

try:
    attribute_controller_module = dynamic_import(
        "attribute_controller",
        project_root / "core" / "attribute_controller.py"
    )
    DiffSketcherController = attribute_controller_module.DiffSketcherController
except Exception as e:
    print(f"Warning: Could not import DiffSketcherController: {e}")
    DiffSketcherController = None

# Import DiffVG utilities
try:
    diffvg_wrapper_module = dynamic_import(
        "diffvg_wrapper",
        project_root / "utils" / "diffvg_wrapper.py"
    )
    setup_diffvg = diffvg_wrapper_module.setup_diffvg
except Exception as e:
    print(f"Warning: Could not import DiffVG wrapper: {e}")
    def setup_diffvg():
        print("DiffVG not available")
        return False

# ========== INTEGRATED PIPELINE CLASS ==========

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageEnhance
    import numpy as np
    HAS_TORCH = True
except ImportError as e:
    print(f"Warning: Torch/PIL not available: {e}")
    HAS_TORCH = False
    Image = None

class SketchGenerationPipeline:
    """Main pipeline orchestrating all modules in exact order"""
    
    def __init__(self,
                 device: str = "cpu",
                 model_id: str = "runwayml/stable-diffusion-v1-5",
                 cache_dir: str = "./cache",
                 verbose: bool = False):
        
        if not HAS_TORCH:
            raise ImportError("PyTorch and PIL are required for the pipeline")
        
        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, switching to CPU")
            device = "cpu"
        elif device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.verbose = verbose
        
        # Initialize modules
        self.base_generator = None
        self.style_injector = None
        self.stroke_controller = None
        self.attribute_controller = None
        
        # Track which modules are active
        self.active_modules = {
            "base_generator": True,
            "style_injector": False,
            "stroke_controller": False,
            "attribute_controller": False
        }
        
        if self.verbose:
            print("Initializing Sketch Generation Pipeline...")
            print(f"Device: {device}")
    
    def initialize_modules(self, control_params: Dict):
        """Initialize only the modules needed based on control parameters"""
        
        # Base generator is always needed
        if BaseGenerator is not None:
            self.base_generator = BaseGenerator(
                device=self.device,
                model_id=self.model_id,
                cache_dir=self.cache_dir,
                verbose=self.verbose
            )
        else:
            raise ImportError("BaseGenerator module not available")
        
        # Style injector (M3S)
        if "style_control" in control_params and M3SStyleInjector is not None:
            self.style_injector = M3SStyleInjector(
                device=self.device,
                verbose=self.verbose
            )
            self.active_modules["style_injector"] = True
            
            if self.verbose:
                print("M3S Style Injector initialized")
        
        # Stroke controller (CLIPasso)
        if "stroke_control" in control_params and CLIPassoController is not None:
            stroke_params = control_params["stroke_control"]
            self.stroke_controller = CLIPassoController(
                device=self.device,
                num_strokes=stroke_params.get("num_strokes", 16),
                abstraction=stroke_params.get("abstraction", 0.7),
                clip_model=stroke_params.get("clip_model", "ViT-B/32"),
                verbose=self.verbose
            )
            self.active_modules["stroke_controller"] = True
            
            if self.verbose:
                print("CLIPasso Stroke Controller initialized")
        
        # Attribute controller (DiffSketcher)
        if "attribute_control" in control_params and DiffSketcherController is not None:
            attr_params = control_params["attribute_control"]
            self.attribute_controller = DiffSketcherController(
                device=self.device,
                sds_guidance_scale=attr_params.get("sds_guidance_scale", 100.0),
                verbose=self.verbose
            )
            self.active_modules["attribute_controller"] = True
            
            if self.verbose:
                print("DiffSketcher Attribute Controller initialized")
    
    def generate(self,
                prompt: str,
                control_params: Optional[Dict] = None,
                seed: Optional[int] = None,
                num_inference_steps: int = 50,
                guidance_scale: float = 7.5,
                width: int = 512,
                height: int = 512) -> Dict:
        """
        Generate sketch following exact workflow order
        
        Returns:
            Dict containing 'raster' (PIL Image) and optionally 'svg' (string)
        """
        
        if control_params is None:
            control_params = {}
        
        # Initialize only needed modules
        self.initialize_modules(control_params)
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("Starting Generation Pipeline")
            print("=" * 60)
        
        result = {}
        current_image = None
        
        # ====================================================
        # STEP 1: Base Generation (Stable Diffusion)
        # ====================================================
        print("\n[1/4] Base Generation (Stable Diffusion)")
        
        base_params = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed
        }
        
        # Check if M3S style injection is needed at generation time
        style_control = control_params.get("style_control")
        if style_control and self.style_injector:
            # Generate with M3S style injection
            if style_control.get("type") == "reference" and "reference_path" in style_control:
                current_image = self.base_generator.generate_with_m3s_style(
                    prompt=prompt,
                    style_injector=self.style_injector,
                    reference_path=style_control["reference_path"],
                    injection_params=style_control,
                    base_params=base_params
                )
            else:
                # Textual style - enhance prompt
                style_desc = style_control.get("description", "")
                enhanced_prompt = f"{prompt}, {style_desc}, sketch, line drawing"
                current_image = self.base_generator.generate(enhanced_prompt, base_params)
        else:
            # Standard generation
            enhanced_prompt = f"{prompt}, sketch, line drawing, black and white"
            current_image = self.base_generator.generate(enhanced_prompt, base_params)
        
        if current_image is None:
            raise RuntimeError("Base generation failed")
        
        # ====================================================
        # STEP 2: Stroke Control (CLIPasso)
        # ====================================================
        if self.stroke_controller and current_image is not None:
            print("\n[2/4] Stroke Control (CLIPasso Algorithm)")
            
            stroke_params = control_params.get("stroke_control", {})
            
            # CLIPasso needs a target image - we use the generated image
            target_image = current_image
            
            # Optimize strokes using CLIPasso algorithm
            stroke_result = self.stroke_controller.optimize(
                target_image=target_image,
                prompt=prompt,
                num_iterations=stroke_params.get("num_iterations", 200),  # Reduced for speed
                learning_rate=stroke_params.get("learning_rate", 0.1),
                w_semantic=stroke_params.get("w_semantic", 0.1)
            )
            
            # Update current image
            if "raster" in stroke_result:
                current_image = stroke_result["raster"]
                if "svg" in stroke_result:
                    result["svg"] = stroke_result["svg"]
        
        # ====================================================
        # STEP 3: Attribute Control (DiffSketcher)
        # ====================================================
        if self.attribute_controller and current_image is not None:
            print("\n[3/4] Attribute Control (DiffSketcher Algorithm)")
            
            attr_params = control_params.get("attribute_control", {})
            
            # Optimize attributes using DiffSketcher SDS loss
            attr_result = self.attribute_controller.optimize(
                sketch=current_image,
                prompt=prompt,
                target_width=attr_params.get("base_width", 1.0),
                opacity=attr_params.get("opacity", 1.0),
                width_variation=attr_params.get("width_variation", 0.5),
                num_iterations=attr_params.get("num_iterations", 100)  # Reduced for speed
            )
            
            # Update current image
            if "raster" in attr_result:
                current_image = attr_result["raster"]
        
        # ====================================================
        # STEP 4: Final Processing
        # ====================================================
        print("\n[4/4] Final Processing")
        
        # Apply final enhancements
        if current_image is not None:
            final_image = self._final_postprocess(current_image)
            result["raster"] = final_image
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("Pipeline Execution Complete")
            print("=" * 60)
            active = [k for k, v in self.active_modules.items() if v]
            print(f"Active modules: {active}")
        
        return result
    
    def _final_postprocess(self, image: Image.Image) -> Image.Image:
        """Apply final post-processing"""
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Enhance contrast for better sketch visibility
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Optional: convert to true black and white
        # image = image.convert("L").convert("RGB")
        
        return image
    
    def get_module_status(self) -> Dict:
        """Get status of all modules"""
        return {
            "base_generator": "Ready" if self.base_generator else "Not loaded",
            "style_injector": "Ready" if self.style_injector else "Not loaded",
            "stroke_controller": "Ready" if self.stroke_controller else "Not loaded",
            "attribute_controller": "Ready" if self.attribute_controller else "Not loaded"
        }

# ========== ORIGINAL MAIN.PY FUNCTIONS ==========

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fine-grained controllable sketch generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation
  python main.py --prompt "a cat sitting on windowsill"
  
  # With stroke control (CLIPasso)
  python main.py --prompt "mountain" --stroke-count 16
  
  # With style control (M3S)
  python main.py --prompt "flower" --style-reference ./style.jpg
  
  # With attribute control (DiffSketcher)
  python main.py --prompt "tree" --stroke-width 1.5 --stroke-opacity 0.8
  
  # Full control
  python main.py --prompt "dragon" \\
                 --stroke-count 24 \\
                 --style-reference ./ink_style.jpg \\
                 --stroke-width 1.5 \\
                 --stroke-opacity 0.7 \\
                 --output-dir ./results \\
                 --seed 42
        """
    )
    
    # Required arguments
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text description for sketch generation")
    
    # Stroke control (CLIPasso)
    parser.add_argument("--stroke-count", type=int, default=None,
                       help="Target number of strokes (CLIPasso control)")
    parser.add_argument("--abstraction", type=float, default=0.7,
                       help="Abstraction level (0.0=detailed, 1.0=abstract)")
    
    # Style control (M3S)
    parser.add_argument("--style-reference", type=str, default=None,
                       help="Path to style reference image (M3S injection)")
    parser.add_argument("--style-text", type=str, default=None,
                       help="Textual style description")
    parser.add_argument("--style-strength", type=float, default=0.8,
                       help="Style injection strength (0.0-1.0)")
    parser.add_argument("--injection-layers", type=str, default="7,8",
                       help="UNet layers for K/V injection (comma-separated)")
    
    # Attribute control (DiffSketcher)
    parser.add_argument("--stroke-width", type=float, default=1.0,
                       help="Base stroke width multiplier")
    parser.add_argument("--stroke-opacity", type=float, default=1.0,
                       help="Stroke opacity (0.0-1.0)")
    parser.add_argument("--width-variation", type=float, default=0.5,
                       help="Stroke width variation (0.0-1.0)")
    
    # Generation parameters
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--num-steps", type=int, default=30,
                       help="Number of diffusion steps (reduced for speed)")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                       help="Classifier-free guidance scale")
    parser.add_argument("--width", type=int, default=512,
                       help="Output width")
    parser.add_argument("--height", type=int, default=512,
                       help="Output height")
    
    # Output settings
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Output directory")
    parser.add_argument("--output-name", type=str, default=None,
                       help="Custom output filename")
    parser.add_argument("--save-svg", action="store_true",
                       help="Save vector SVG in addition to raster")
    parser.add_argument("--save-config", action="store_true",
                       help="Save generation configuration")
    
    # System settings
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Compute device")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="Base diffusion model")
    parser.add_argument("--cache-dir", type=str, default="./cache",
                       help="Model cache directory")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    # Debug options
    parser.add_argument("--skip-diffvg", action="store_true",
                       help="Skip DiffVG initialization (use simplified mode)")
    
    return parser.parse_args()

def setup_environment(args):
    """Setup environment and device"""
    if args.device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("CUDA not available, using CPU")
        except:
            device = "cpu"
    else:
        device = args.device
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    return device

def generate_output_name(prompt, control_params, seed=None):
    """Generate descriptive output filename"""
    import re
    
    # Clean prompt for filename
    clean_prompt = re.sub(r'[^\w\s-]', '', prompt.lower())
    clean_prompt = re.sub(r'[-\s]+', '_', clean_prompt)[:30]
    
    # Add control parameters
    parts = [clean_prompt]
    
    if control_params.get("stroke_control"):
        sc = control_params["stroke_control"]
        parts.append(f"s{sc.get('num_strokes', 'auto')}")
    
    if control_params.get("style_control"):
        parts.append("styled")
    
    if control_params.get("attribute_control"):
        parts.append("attr")
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts.append(timestamp)
    
    # Add seed if provided
    if seed is not None:
        parts.append(f"seed{seed}")
    
    return "_".join(parts)

def main():
    args = parse_arguments()
    
    # Setup environment
    device = setup_environment(args)
    
    print("=" * 60)
    print("Sketch Generation Pipeline")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Prompt: {args.prompt}")
    
    # Check for required modules
    if BaseGenerator is None:
        print("ERROR: BaseGenerator module is required but not available")
        print("Please check that core/base_generator.py exists and has no syntax errors")
        sys.exit(1)
    
    # Parse injection layers
    injection_layers = [int(x.strip()) for x in args.injection_layers.split(",")]
    
    # Build control parameters
    control_params = {}
    
    # CLIPasso stroke control
    if args.stroke_count is not None:
        if CLIPassoController is None:
            print("Warning: CLIPassoController not available, skipping stroke control")
        else:
            control_params["stroke_control"] = {
                "num_strokes": args.stroke_count,
                "abstraction": args.abstraction,
                "clip_model": "ViT-B/32",
                "num_iterations": 200,  # Reduced for testing
                "learning_rate": 0.1,
                "w_semantic": 0.1
            }
    
    # M3S style control
    if args.style_reference or args.style_text:
        if M3SStyleInjector is None:
            print("Warning: M3SStyleInjector not available, skipping style control")
        else:
            style_params = {
                "strength": args.style_strength,
                "injection_layers": injection_layers,
                "content_guidance": 15.0,
                "style_guidance": 15.0,
                "lambda_blend": 0.1
            }
            
            if args.style_reference:
                style_params["type"] = "reference"
                style_params["reference_path"] = args.style_reference
            elif args.style_text:
                style_params["type"] = "textual"
                style_params["description"] = args.style_text
                
            control_params["style_control"] = style_params
    
    # DiffSketcher attribute control
    if args.stroke_width != 1.0 or args.stroke_opacity != 1.0 or args.width_variation != 0.5:
        if DiffSketcherController is None:
            print("Warning: DiffSketcherController not available, skipping attribute control")
        else:
            control_params["attribute_control"] = {
                "opacity": args.stroke_opacity,
                "base_width": args.stroke_width,
                "width_variation": args.width_variation,
                "sds_guidance_scale": 100.0,
                "num_iterations": 100  # Reduced for testing
            }
    
    # Setup DiffVG if needed and not skipped
    if (args.stroke_count is not None or args.save_svg) and not args.skip_diffvg:
        print("Setting up DiffVG for vector rendering...")
        try:
            diffvg_available = setup_diffvg()
            if not diffvg_available:
                print("DiffVG setup failed, using simplified mode")
                if "stroke_control" in control_params:
                    # Modify stroke control to use simplified mode
                    control_params["stroke_control"]["simplified_mode"] = True
                args.save_svg = False
        except Exception as e:
            print(f"Warning: DiffVG setup failed: {e}")
            print("Vector rendering will be disabled")
            if "stroke_control" in control_params:
                control_params["stroke_control"]["simplified_mode"] = True
            args.save_svg = False
    
    # Initialize pipeline
    try:
        pipeline = SketchGenerationPipeline(
            device=device,
            model_id=args.model,
            cache_dir=args.cache_dir,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        print("\nCommon issues:")
        print("1. PyTorch not installed: pip install torch torchvision")
        print("2. PIL not installed: pip install pillow")
        print("3. CUDA not available: use --device cpu")
        sys.exit(1)
    
    # Generation parameters
    generation_params = {
        "seed": args.seed,
        "num_inference_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "width": args.width,
        "height": args.height
    }
    
    if args.verbose and control_params:
        print("\nActive Control Modules:")
        for module in control_params.keys():
            print(f"  • {module}")
    
    # Generate sketch
    try:
        print("\n" + "=" * 60)
        print("Starting Sketch Generation...")
        print("=" * 60)
        
        result = pipeline.generate(
            prompt=args.prompt,
            control_params=control_params,
            **generation_params
        )
        
        # Generate output filename
        if args.output_name:
            base_name = args.output_name
        else:
            base_name = generate_output_name(
                args.prompt, control_params, args.seed
            )
        
        # Save raster image
        raster_path = os.path.join(args.output_dir, f"{base_name}.png")
        if "raster" in result:
            result["raster"].save(raster_path)
            print(f" Raster sketch saved: {raster_path}")
        else:
            print(" No raster output generated")
            sys.exit(1)
        
        # Save vector if requested and available
        if args.save_svg and "svg" in result and result["svg"]:
            svg_path = os.path.join(args.output_dir, f"{base_name}.svg")
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(result["svg"])
            print(f" Vector sketch saved: {svg_path}")
        
        # Save configuration if requested
        if args.save_config:
            config_path = os.path.join(args.output_dir, f"{base_name}_config.json")
            config_data = {
                "prompt": args.prompt,
                "control_params": control_params,
                "generation_params": generation_params,
                "outputs": {
                    "raster": raster_path,
                    "svg": svg_path if args.save_svg and "svg" in result else None
                },
                "timestamp": datetime.now().isoformat(),
                "pipeline_version": "1.0"
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
            print(f" Configuration saved: {config_path}")
        
        print("\n" + "=" * 60)
        print("Generation Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nGeneration failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        print("\nTroubleshooting tips:")
        print("1. Try with --device cpu if CUDA is having issues")
        print("2. Try without stroke control: remove --stroke-count")
        print("3. Check if models are downloaded correctly")
        print("4. Reduce image size with --width 256 --height 256")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
