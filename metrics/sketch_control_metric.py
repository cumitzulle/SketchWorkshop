import torch
import torch.nn.functional as F
import clip
import numpy as np
from PIL import Image
import warnings
from typing import Dict, List, Optional, Tuple

try:
    # 尝试导入DINO相关库
    import timm
    import torchvision.transforms as T
    HAS_DINO = True
except ImportError:
    HAS_DINO = False
    warnings.warn("DINO not available. Style metric will use CLIP-T only.")

class StyleControlMetric:
    """
    风格控制指标评估器
    指标 = 2 * CLIP-T * DINO / (CLIP-T + DINO) (谐波平均)
    """
    
    def __init__(self,
                 clip_model_name: str = "ViT-B/32",
                 dino_model_name: str = "vit_base_patch16_224.dino",
                 device: str = "cuda",
                 verbose: bool = False):
        """
        初始化
        
        Args:
            clip_model_name: CLIP模型名称
            dino_model_name: DINO模型名称
            device: 计算设备
            verbose: 是否输出详细信息
        """
        self.device = device
        self.verbose = verbose
        
        # 加载CLIP模型 (用于CLIP-T)
        self.clip_model, self.clip_preprocess = clip.load(
            clip_model_name, device=device
        )
        self.clip_model.eval()
        
        # 加载DINO模型
        self.dino_model = None
        self.dino_preprocess = None
        if HAS_DINO:
            self._load_dino_model(dino_model_name)
        else:
            if self.verbose:
                print("Warning: DINO model not available. Style metric will use CLIP-T only.")
        
        if self.verbose:
            print(f"StyleControlMetric initialized on {device}")
            print(f"  CLIP model: {clip_model_name}")
            print(f"  DINO model: {dino_model_name if HAS_DINO else 'Not available'}")
    
    def _load_dino_model(self, model_name: str):
        """加载DINO模型"""
        try:
            # 加载预训练的DINO模型
            self.dino_model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0  # 不要分类头
            ).to(self.device)
            self.dino_model.eval()
            
            # DINO预处理
            self.dino_preprocess = T.Compose([
                T.Resize(256, interpolation=3),  # Image.BICUBIC
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            print(f"Error loading DINO model: {e}")
            self.dino_model = None
            self.dino_preprocess = None
    
    def evaluate(self,
                generated_sketch: Image.Image,
                style_reference: Image.Image,
                compute_clip_only: bool = False) -> Dict[str, float]:
        """
        评估风格控制指标
        
        Args:
            generated_sketch: 生成的草图图像 (PIL Image)
            style_reference: 风格参考图像 (PIL Image)
            compute_clip_only: 是否只计算CLIP-T (当DINO不可用时)
        
        Returns:
            包含各项指标的字典
        """
        # 1. 计算CLIP-T相似度
        clip_t_score = self._compute_clip_t_similarity(
            generated_sketch, style_reference
        )
        
        # 2. 计算DINO相似度（如果可用）
        dino_score = None
        if self.dino_model is not None and not compute_clip_only:
            dino_score = self._compute_dino_similarity(
                generated_sketch, style_reference
            )
        
        # 3. 计算综合风格控制指标
        if dino_score is not None:
            # 使用谐波平均: 2 * CLIP-T * DINO / (CLIP-T + DINO)
            if clip_t_score > 0 or dino_score > 0:
                style_control_score = 2.0 * clip_t_score * dino_score / (clip_t_score + dino_score + 1e-8)
            else:
                style_control_score = 0.0
        else:
            # 只有CLIP-T可用
            style_control_score = clip_t_score
        
        # 准备结果
        results = {
            "style_control_score": float(style_control_score),
            "clip_t_score": float(clip_t_score)
        }
        
        if dino_score is not None:
            results["dino_score"] = float(dino_score)
        
        if self.verbose:
            print(f"  CLIP-T: {clip_t_score:.4f}")
            if dino_score is not None:
                print(f"  DINO: {dino_score:.4f}")
            print(f"  Style control score: {style_control_score:.4f}")
        
        return results
    
    def _compute_clip_t_similarity(self,
                                  sketch_img: Image.Image,
                                  style_img: Image.Image) -> float:
        """
        计算CLIP-T相似度
        
        Args:
            sketch_img: 生成的草图
            style_img: 风格参考图像
        
        Returns:
            CLIP-T余弦相似度 [0, 1]
        """
        # 预处理图像
        sketch_tensor = self._preprocess_for_clip(sketch_img)
        style_tensor = self._preprocess_for_clip(style_img)
        
        # 提取CLIP特征
        with torch.no_grad():
            sketch_features = self.clip_model.encode_image(sketch_tensor)
            style_features = self.clip_model.encode_image(style_tensor)
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(sketch_features, style_features, dim=-1)
        
        # 转换为[0, 1]范围
        score = (similarity.item() + 1.0) / 2.0
        
        return score
    
    def _compute_dino_similarity(self,
                               sketch_img: Image.Image,
                               style_img: Image.Image) -> float:
        """
        计算DINO相似度
        
        Args:
            sketch_img: 生成的草图
            style_img: 风格参考图像
        
        Returns:
            DINO余弦相似度 [0, 1]
        """
        if self.dino_model is None or self.dino_preprocess is None:
            return 0.0
        
        # 预处理图像
        sketch_tensor = self.dino_preprocess(sketch_img).unsqueeze(0).to(self.device)
        style_tensor = self.dino_preprocess(style_img).unsqueeze(0).to(self.device)
        
        # 提取DINO特征
        with torch.no_grad():
            sketch_features = self.dino_model(sketch_tensor)
            style_features = self.dino_model(style_tensor)
        
        # 归一化特征
        sketch_features = F.normalize(sketch_features, dim=-1)
        style_features = F.normalize(style_features, dim=-1)
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(sketch_features, style_features, dim=-1)
        
        # 转换为[0, 1]范围
        score = (similarity.item() + 1.0) / 2.0
        
        return score
    
    def _preprocess_for_clip(self, image: Image.Image) -> torch.Tensor:
        """
        为CLIP预处理图像
        
        Args:
            image: PIL图像
        
        Returns:
            预处理后的张量
        """
        # CLIP需要224x224输入
        if image.size != (224, 224):
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # 转换为张量并标准化
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def batch_evaluate(self,
                      generated_sketches: List[Image.Image],
                      style_references: List[Image.Image],
                      compute_clip_only: bool = False) -> Dict[str, List[float]]:
        """
        批量评估风格控制指标
        
        Args:
            generated_sketches: 生成的草图列表
            style_references: 风格参考图像列表
            compute_clip_only: 是否只计算CLIP-T
        
        Returns:
            包含所有指标的字典
        """
        results = {
            "style_control_scores": [],
            "clip_t_scores": [],
            "dino_scores": [] if self.dino_model is not None and not compute_clip_only else None
        }
        
        for i, (sketch, style_ref) in enumerate(zip(generated_sketches, style_references)):
            if self.verbose:
                print(f"\nEvaluating sample {i+1}/{len(generated_sketches)}")
            
            metrics = self.evaluate(sketch, style_ref, compute_clip_only)
            
            results["style_control_scores"].append(metrics["style_control_score"])
            results["clip_t_scores"].append(metrics["clip_t_score"])
            
            if "dino_score" in metrics and results["dino_scores"] is not None:
                results["dino_scores"].append(metrics["dino_score"])
        
        # 计算统计信息
        stats = {
            "mean_style_score": np.mean(results["style_control_scores"]),
            "std_style_score": np.std(results["style_control_scores"]),
            "mean_clip_t": np.mean(results["clip_t_scores"])
        }
        
        if results["dino_scores"] is not None and len(results["dino_scores"]) > 0:
            stats["mean_dino"] = np.mean(results["dino_scores"])
        
        results["statistics"] = stats
        
        return results
    
    def evaluate_style_transfer(self,
                               generated_sketches: List[Image.Image],
                               style_references: List[Image.Image],
                               content_references: Optional[List[Image.Image]] = None) -> Dict[str, any]:
        """
        评估风格转移效果（包含风格保持和内容保持的平衡）
        
        Args:
            generated_sketches: 生成的草图列表
            style_references: 风格参考图像列表
            content_references: 内容参考图像列表（可选）
        
        Returns:
            包含风格转移指标的字典
        """
        # 计算风格相似度
        style_metrics = self.batch_evaluate(generated_sketches, style_references)
        
        results = {
            "style_similarity": style_metrics
        }
        
        # 如果提供了内容参考，计算内容保持度
        if content_references is not None:
            content_scores = []
            for sketch, content_ref in zip(generated_sketches, content_references):
                # 使用CLIP计算内容相似度
                content_score = self._compute_clip_t_similarity(sketch, content_ref)
                content_scores.append(content_score)
            
            results["content_preservation"] = {
                "scores": content_scores,
                "mean_score": np.mean(content_scores),
                "std_score": np.std(content_scores)
            }
            
            # 计算风格-内容平衡
            style_scores = style_metrics["style_control_scores"]
            balance_scores = []
            
            for style_score, content_score in zip(style_scores, content_scores):
                # 使用几何平均作为平衡指标
                if style_score > 0 and content_score > 0:
                    balance = np.sqrt(style_score * content_score)
                else:
                    balance = 0.0
                balance_scores.append(balance)
            
            results["style_content_balance"] = {
                "scores": balance_scores,
                "mean_balance": np.mean(balance_scores),
                "std_balance": np.std(balance_scores)
            }
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """
        保存评估结果到JSON文件
        
        Args:
            results: 评估结果
            output_path: 输出文件路径
        """
        import json
        
        # 确保数值类型可以序列化
        serializable_results = {}
        for key, value in results.items():
            if key == "statistics" or key == "content_preservation" or key == "style_content_balance":
                serializable_results[key] = {k: float(v) if isinstance(v, (np.floating, float)) 
                                           else int(v) if isinstance(v, (np.integer, int))
                                           else v for k, v in value.items()}
            elif isinstance(value, list):
                serializable_results[key] = [float(v) if isinstance(v, (np.floating, float)) 
                                           else int(v) if isinstance(v, (np.integer, int))
                                           else v for v in value]
            else:
                serializable_results[key] = value
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"Style evaluation results saved to {output_path}")

# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 初始化评估器
    evaluator = StyleControlMetric(device="cpu", verbose=True)
    
    # 检查DINO是否可用
    if evaluator.dino_model is None:
        print("Warning: DINO model not loaded. Using CLIP-T only.")
    
    # 加载测试图像
    sketch_img = Image.open("styled_sketch.png").convert("RGB")
    style_img = Image.open("style_reference.png").convert("RGB")
    
    # 单个样本评估
    metrics = evaluator.evaluate(
        generated_sketch=sketch_img,
        style_reference=style_img
    )
    
    print("\n" + "="*60)
    print("Style Control Metrics:")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 批量评估示例
    """
    batch_results = evaluator.batch_evaluate(
        generated_sketches=[sketch1, sketch2, sketch3],
        style_references=[style1, style2, style3]
    )
    
    # 风格转移评估（包含内容保持）
    content_imgs = [content1, content2, content3]
    transfer_results = evaluator.evaluate_style_transfer(
        generated_sketches=[sketch1, sketch2, sketch3],
        style_references=[style1, style2, style3],
        content_references=content_imgs
    )
    
    evaluator.save_results(transfer_results, "style_transfer_results.json")
    """