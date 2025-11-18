import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

# 导入自定义模块
import sys
sys.path.append('src')
from models.student import create_student_model
from dataset import FusionDataset
from utils import load_checkpoint


# ============================================
# 融合质量评估指标
# ============================================

class FusionMetrics:
    """融合图像质量评估指标集合"""
    
    def __init__(self, device='cuda'):
        self.device = device
        # LPIPS for perceptual similarity
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        
    def compute_ssim(self, img1, img2):
        """计算SSIM"""
        # Convert to numpy and ensure grayscale
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()
        
        # Handle channel dimension
        if img1.ndim == 4:
            img1 = img1[0]
        if img2.ndim == 4:
            img2 = img2[0]
        
        # Convert to HWC if needed
        if img1.shape[0] == 3 or img1.shape[0] == 1:
            img1 = np.transpose(img1, (1, 2, 0))
        if img2.shape[0] == 3 or img2.shape[0] == 1:
            img2 = np.transpose(img2, (1, 2, 0))
        
        # Convert to grayscale if RGB
        if img1.shape[-1] == 3:
            img1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        if img2.shape[-1] == 3:
            img2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        
        return ssim(img1, img2, data_range=1.0)
    
    def compute_psnr(self, img1, img2):
        """计算PSNR"""
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()
        
        return psnr(img1, img2, data_range=1.0)
    
    def compute_lpips(self, img1, img2):
        """计算LPIPS (感知相似度)"""
        # LPIPS expects [-1, 1] range
        img1 = img1 * 2 - 1
        img2 = img2 * 2 - 1
        
        with torch.no_grad():
            dist = self.lpips_fn(img1, img2)
        
        return dist.item()
    
    def compute_en(self, img):
        """计算信息熵 (Entropy)"""
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # Convert to grayscale
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        elif img.shape[0] == 1:
            img = img[0]
        
        # Compute histogram
        hist, _ = np.histogram(img, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        
        # Compute entropy
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        
        return entropy
    
    def compute_sd(self, img):
        """计算标准差 (Standard Deviation)"""
        if isinstance(img, torch.Tensor):
            return torch.std(img).item()
        else:
            return np.std(img)
    
    def compute_sf(self, img):
        """计算空间频率 (Spatial Frequency)"""
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] == 3 or img.shape[0] == 1:
            img = np.transpose(img, (1, 2, 0))
        
        # Convert to grayscale
        if img.shape[-1] == 3:
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        elif img.shape[-1] == 1:
            img = img[:, :, 0]
        
        # Row frequency
        rf = np.sqrt(np.mean(np.diff(img, axis=0)**2))
        
        # Column frequency
        cf = np.sqrt(np.mean(np.diff(img, axis=1)**2))
        
        # Spatial frequency
        sf = np.sqrt(rf**2 + cf**2)
        
        return sf
    
    def compute_ag(self, img):
        """计算平均梯度 (Average Gradient)"""
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] == 3 or img.shape[0] == 1:
            img = np.transpose(img, (1, 2, 0))
        
        # Convert to grayscale
        if img.shape[-1] == 3:
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        elif img.shape[-1] == 1:
            img = (img[:, :, 0] * 255).astype(np.uint8)
        
        # Compute gradients
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        # Average gradient
        ag = np.mean(np.sqrt(gx**2 + gy**2))
        
        return ag
    
    def compute_all_metrics(self, pred, target):
        """计算所有指标"""
        metrics = {}
        
        # Reference-based metrics
        metrics['SSIM'] = self.compute_ssim(pred, target)
        metrics['PSNR'] = self.compute_psnr(pred, target)
        metrics['LPIPS'] = self.compute_lpips(pred, target)
        
        # No-reference metrics (on predicted image)
        metrics['EN'] = self.compute_en(pred)
        metrics['SD'] = self.compute_sd(pred)
        metrics['SF'] = self.compute_sf(pred)
        metrics['AG'] = self.compute_ag(pred)
        
        return metrics


# ============================================
# 评估函数
# ============================================

@torch.no_grad()
def evaluate_model(model, dataloader, metrics_calculator, device, save_dir=None):
    """
    评估模型性能
    
    Args:
        model: 待评估模型
        dataloader: 数据加载器
        metrics_calculator: 指标计算器
        device: 设备
        save_dir: 可选的结果保存目录
    
    Returns:
        avg_metrics: 平均指标字典
    """
    model.eval()
    
    all_metrics = {
        'SSIM': [],
        'PSNR': [],
        'LPIPS': [],
        'EN': [],
        'SD': [],
        'SF': [],
        'AG': []
    }
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)
    
    pbar = tqdm(dataloader, desc='Evaluating')
    
    for batch_idx, (ir, vis, teacher) in enumerate(pbar):
        ir = ir.to(device)
        vis = vis.to(device)
        teacher = teacher.to(device)
        
        # Forward pass
        pred = model(ir, vis)
        
        # Compute metrics for each image in batch
        for i in range(pred.size(0)):
            metrics = metrics_calculator.compute_all_metrics(
                pred[i:i+1], 
                teacher[i:i+1]
            )
            
            for key, value in metrics.items():
                all_metrics[key].append(value)
        
        # Save predictions if needed
        if save_dir and batch_idx < 10:  # Save first 10 batches
            for i in range(pred.size(0)):
                pred_img = pred[i].cpu().numpy().transpose(1, 2, 0)
                pred_img = (pred_img * 255).clip(0, 255).astype(np.uint8)
                
                save_path = os.path.join(
                    save_dir, 
                    'predictions', 
                    f'pred_{batch_idx:04d}_{i:02d}.png'
                )
                cv2.imwrite(save_path, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
        
        # Update progress bar
        current_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        pbar.set_postfix({
            'SSIM': f"{current_metrics['SSIM']:.4f}",
            'PSNR': f"{current_metrics['PSNR']:.2f}",
            'LPIPS': f"{current_metrics['LPIPS']:.4f}"
        })
    
    # Compute average metrics
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    std_metrics = {k: np.std(v) for k, v in all_metrics.items()}
    
    return avg_metrics, std_metrics


# ============================================
# 推理速度测试
# ============================================

def benchmark_speed(model, input_size=(1, 3, 512, 512), device='cuda', 
                   num_warmup=50, num_runs=200):
    """
    测试模型推理速度
    
    Args:
        model: 模型
        input_size: 输入尺寸
        device: 设备
        num_warmup: 预热次数
        num_runs: 测试次数
    
    Returns:
        速度统计
    """
    model.eval()
    
    ir = torch.randn(input_size).to(device)
    vis = torch.randn(input_size).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(ir, vis)
    
    # Synchronize
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure time
    print(f"Running {num_runs} iterations...")
    times = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_runs)):
            if device == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                _ = model(ir, vis)
                end.record()
                
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                import time
                start = time.time()
                _ = model(ir, vis)
                end = time.time()
                times.append((end - start) * 1000)
    
    stats = {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
        'fps': 1000 / np.mean(times)
    }
    
    return stats


# ============================================
# 主评估程序
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate Fusion Student Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--ir_root', type=str, default='./data/ir',
                       help='Path to IR images directory')
    parser.add_argument('--vis_root', type=str, default='./data/vi',
                       help='Path to VIS images directory')
    parser.add_argument('--teacher_root', type=str, default='./DDAEFuse_MSRS',
                       help='Path to Teacher fused images directory')
    parser.add_argument('--save_dir', type=str, default='eval_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--input_size', type=int, default=512,
                       help='Input image size')
    parser.add_argument('--in_channels', type=int, default=1,
                       help='Number of IR input channels')
    parser.add_argument('--benchmark_speed', action='store_true',
                       help='Run speed benchmark')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Loading model...")
    model = create_student_model(in_channels=args.in_channels, out_channels=3).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model)
    
    # Create dataset - 使用test split
    print("Loading test dataset...")
    from dataset import split_dataset
    
    train_list, val_list, test_list = split_dataset(
        args.ir_root,
        args.vis_root,
        args.teacher_root
    )
    
    test_dataset = FusionDataset(
        args.ir_root,
        args.vis_root,
        args.teacher_root,
        image_list=test_list,
        mode='val',
        input_size=args.input_size
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create metrics calculator
    metrics_calc = FusionMetrics(device=device)
    
    # Evaluate
    print("\nEvaluating model...")
    avg_metrics, std_metrics = evaluate_model(
        model, 
        test_loader, 
        metrics_calc, 
        device,
        save_dir=args.save_dir
    )
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print("\nQuality Metrics:")
    for key in ['SSIM', 'PSNR', 'LPIPS']:
        print(f"  {key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
    
    print("\nNo-Reference Metrics:")
    for key in ['EN', 'SD', 'SF', 'AG']:
        print(f"  {key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
    
    # Speed benchmark
    if args.benchmark_speed:
        print("\n" + "="*80)
        print("SPEED BENCHMARK")
        print("="*80)
        
        speed_stats = benchmark_speed(
            model, 
            input_size=(1, args.in_channels, args.input_size, args.input_size),
            device=device
        )
        
        print(f"\nInference Time Statistics:")
        print(f"  Mean: {speed_stats['mean']:.2f} ms")
        print(f"  Std: {speed_stats['std']:.2f} ms")
        print(f"  Min: {speed_stats['min']:.2f} ms")
        print(f"  Max: {speed_stats['max']:.2f} ms")
        print(f"  Median: {speed_stats['median']:.2f} ms")
        print(f"  FPS: {speed_stats['fps']:.2f}")
    
    # Save results to file
    results_file = os.path.join(args.save_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Quality Metrics:\n")
        for key in ['SSIM', 'PSNR', 'LPIPS']:
            f.write(f"  {key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}\n")
        
        f.write("\nNo-Reference Metrics:\n")
        for key in ['EN', 'SD', 'SF', 'AG']:
            f.write(f"  {key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}\n")
        
        if args.benchmark_speed:
            f.write("\nSpeed Benchmark:\n")
            f.write(f"  Mean: {speed_stats['mean']:.2f} ms\n")
            f.write(f"  FPS: {speed_stats['fps']:.2f}\n")
    
    print(f"\nResults saved to {results_file}")
    print("="*80)


if __name__ == '__main__':
    main()
