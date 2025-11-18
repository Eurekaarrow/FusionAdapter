import os
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt


# ============================================
# 随机种子设置
# ============================================

def set_seed(seed=42):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


# ============================================
# 检查点保存/加载
# ============================================

def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step,
                   best_val_loss, train_losses, val_losses):
    """
    保存训练检查点
    
    Args:
        path: 保存路径
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        global_step: 全局步数
        best_val_loss: 最佳验证损失
        train_losses: 训练损失字典
        val_losses: 验证损失字典
    """
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    加载训练检查点
    
    Args:
        path: 检查点路径
        model: 模型
        optimizer: 优化器 (可选)
        scheduler: 学习率调度器 (可选)
    
    Returns:
        checkpoint: 检查点字典
    """
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    return checkpoint


# ============================================
# 可视化工具
# ============================================

def tensor_to_image(tensor):
    """
    将tensor转换为numpy图像
    
    Args:
        tensor: [C, H, W] 范围 [0, 1]
    
    Returns:
        image: [H, W, C] 范围 [0, 255] uint8
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # 取第一张图
    
    image = tensor.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
    image = (image * 255).clip(0, 255).astype(np.uint8)
    
    return image


def visualize_results(ir, vis, teacher, pred, save_path=None):
    """
    可视化融合结果
    
    Args:
        ir: IR图像 tensor [1, C, H, W]
        vis: VIS图像 tensor [1, C, H, W]
        teacher: Teacher融合图像 tensor [1, C, H, W]
        pred: 预测融合图像 tensor [1, C, H, W]
        save_path: 保存路径 (可选)
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    images = [
        (ir, 'IR Image'),
        (vis, 'VIS Image'),
        (teacher, 'Teacher Fused'),
        (pred, 'Student Fused')
    ]
    
    for ax, (img, title) in zip(axes, images):
        img_np = tensor_to_image(img)
        
        # 如果是单通道，转换为灰度显示
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze()
            ax.imshow(img_np, cmap='gray')
        else:
            ax.imshow(img_np)
        
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================
# 推理时间测量
# ============================================

def measure_inference_time(model, input_size=(1, 3, 512, 512), device='cuda', 
                          num_warmup=10, num_runs=100):
    """
    测量模型推理时间
    
    Args:
        model: 模型
        input_size: 输入尺寸 (B, C, H, W)
        device: 设备
        num_warmup: 预热次数
        num_runs: 测试次数
    
    Returns:
        avg_time: 平均推理时间 (ms)
        std_time: 标准差 (ms)
    """
    model.eval()
    
    # 创建随机输入
    ir = torch.randn(input_size).to(device)
    vis = torch.randn(input_size).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(ir, vis)
    
    # 同步
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 测量时间
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
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
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time


# ============================================
# 模型分析工具
# ============================================

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    return total_params, trainable_params


def analyze_model(model, input_size=(1, 3, 512, 512), device='cuda'):
    """
    全面分析模型
    
    Args:
        model: 模型
        input_size: 输入尺寸
        device: 设备
    """
    print("="*80)
    print("MODEL ANALYSIS")
    print("="*80)
    
    # 参数量
    print("\n1. Parameter Count:")
    count_parameters(model)
    
    # FLOPs (需要thop库)
    try:
        from thop import profile, clever_format
        ir = torch.randn(input_size).to(device)
        vis = torch.randn(input_size).to(device)
        
        macs, params = profile(model, inputs=(ir, vis), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        
        print(f"\n2. Computational Complexity:")
        print(f"  FLOPs: {macs}")
        print(f"  Parameters: {params}")
    except ImportError:
        print("\n2. Computational Complexity:")
        print("  thop not installed. Run: pip install thop")
    
    # 推理时间
    print(f"\n3. Inference Time (input size: {input_size}):")
    avg_time, std_time = measure_inference_time(model, input_size, device)
    print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  FPS: {1000/avg_time:.2f}")
    
    # 内存占用
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        ir = torch.randn(input_size).to(device)
        vis = torch.randn(input_size).to(device)
        
        with torch.no_grad():
            _ = model(ir, vis)
        
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\n4. Memory Usage:")
        print(f"  Peak memory: {memory_allocated:.2f} MB")
    
    print("="*80)


# ============================================
# ONNX导出
# ============================================

def export_to_onnx(model, output_path, input_size=(1, 3, 512, 512), 
                  opset_version=11, device='cuda'):
    """
    导出模型到ONNX格式
    
    Args:
        model: PyTorch模型
        output_path: ONNX文件保存路径
        input_size: 输入尺寸
        opset_version: ONNX opset版本
        device: 设备
    """
    model.eval()
    
    # 创建dummy输入
    ir = torch.randn(input_size).to(device)
    vis = torch.randn(input_size).to(device)
    
    # 导出
    torch.onnx.export(
        model,
        (ir, vis),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['ir', 'vis'],
        output_names=['fused'],
        dynamic_axes={
            'ir': {0: 'batch_size'},
            'vis': {0: 'batch_size'},
            'fused': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX: {output_path}")
    
    # 验证ONNX模型
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed!")
    except ImportError:
        print("onnx not installed. Skipping validation.")
    except Exception as e:
        print(f"ONNX validation failed: {e}")


# ============================================
# 学习率查找器
# ============================================

class LRFinder:
    """
    学习率范围测试工具
    用于找到合适的初始学习率
    """
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.history = {'lr': [], 'loss': []}
        
    def range_test(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100):
        """
        执行学习率范围测试
        
        Args:
            train_loader: 训练数据加载器
            start_lr: 起始学习率
            end_lr: 结束学习率
            num_iter: 迭代次数
        """
        self.model.train()
        
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        lr = start_lr
        
        self.optimizer.param_groups[0]['lr'] = lr
        best_loss = float('inf')
        
        iterator = iter(train_loader)
        
        for iteration in range(num_iter):
            try:
                ir, vis, teacher = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                ir, vis, teacher = next(iterator)
            
            ir = ir.to(self.device)
            vis = vis.to(self.device)
            teacher = teacher.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            pred = self.model(ir, vis)
            loss, _ = self.criterion(pred, teacher)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Record
            self.history['lr'].append(lr)
            self.history['loss'].append(loss.item())
            
            # Update learning rate
            lr *= lr_mult
            self.optimizer.param_groups[0]['lr'] = lr
            
            # Stop if loss explodes
            if loss.item() > 4 * best_loss or torch.isnan(loss):
                break
            
            if loss.item() < best_loss:
                best_loss = loss.item()
        
        print(f"LR Finder completed. Tested {len(self.history['lr'])} learning rates.")
    
    def plot(self, skip_start=10, skip_end=5, save_path=None):
        """绘制学习率 vs 损失曲线"""
        lrs = self.history['lr'][skip_start:-skip_end]
        losses = self.history['loss'][skip_start:-skip_end]
        
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"LR Finder plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


# ============================================
# 早停机制
# ============================================

class EarlyStopping:
    """
    早停机制
    当验证损失不再改善时停止训练
    """
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Args:
            patience: 容忍epochs数
            min_delta: 最小改善量
            mode: 'min' 或 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        Args:
            score: 当前验证指标
        
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test seed setting
    set_seed(42)
    
    # Test tensor to image conversion
    test_tensor = torch.rand(1, 3, 256, 256)
    test_image = tensor_to_image(test_tensor)
    print(f"Converted tensor shape: {test_tensor.shape} -> image shape: {test_image.shape}")
    
    print("All tests passed!")
