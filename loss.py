import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from kornia.losses import SSIMLoss as KorniaSSIMLoss

# ============================================
# 1. 特征一致性损失 (Feature Consistency Loss)
# Paper中称为: "Feature Consistency Distillation"
# 作用: 强制 Student 的生成图像在语义特征空间中逼近 Teacher
# ============================================
class FeatureConsistencyLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(FeatureConsistencyLoss, self).__init__()
        # 加载预训练的 VGG16
        vgg = models.vgg16(pretrained=True).features.to(device)
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
            
        # 截取 VGG 的不同层级，提取多尺度特征
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        # relu1_2
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        # relu2_2
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        # relu3_3
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        # relu4_3
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])

        # ImageNet 标准化参数
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, pred, target):
        # 归一化输入
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # 计算多层特征并累加 MSE Loss
        h_pred = self.slice1(pred)
        h_target = self.slice1(target)
        loss = F.mse_loss(h_pred, h_target)
        
        h_pred = self.slice2(h_pred)
        h_target = self.slice2(h_target)
        loss += F.mse_loss(h_pred, h_target)
        
        h_pred = self.slice3(h_pred)
        h_target = self.slice3(h_target)
        loss += F.mse_loss(h_pred, h_target)
        
        h_pred = self.slice4(h_pred)
        h_target = self.slice4(h_target)
        loss += F.mse_loss(h_pred, h_target)
        
        return loss


# ============================================
# 2. 梯度损失 (Gradient Loss)
# 作用: 边缘锐化，解决图像模糊问题
# ============================================
class GradientLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(GradientLoss, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)
        
        self.register_buffer("sobel_x", kernel_x)
        self.register_buffer("sobel_y", kernel_y)

    def compute_gradient(self, img):
        # 转灰度
        if img.shape[1] == 3:
            gray = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
            gray = gray.unsqueeze(1)
        else:
            gray = img
            
        # 自动适配 Device 和 dtype (解决 AMP 混合精度训练报错)
        if self.sobel_x.device != gray.device or self.sobel_x.dtype != gray.dtype:
            self.sobel_x = self.sobel_x.to(device=gray.device, dtype=gray.dtype)
            self.sobel_y = self.sobel_y.to(device=gray.device, dtype=gray.dtype)
            
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        return torch.abs(grad_x) + torch.abs(grad_y)

    def forward(self, pred, target):
        g_pred = self.compute_gradient(pred)
        g_target = self.compute_gradient(target)
        return F.l1_loss(g_pred, g_target)


# ============================================
# 3. 基础 SSIM 损失 (Structural Similarity)
# 作用: 保持局部结构一致性
# ============================================
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, reduction='mean', device='cuda'):
        super(SSIMLoss, self).__init__()
        # 使用 Kornia 的稳定实现
        self.loss_fn = KorniaSSIMLoss(window_size=window_size, reduction=reduction)
        self.device = device

    def forward(self, pred, target):
        # Kornia SSIM Loss 返回的是损失值 (即 1 - SSIM)
        return self.loss_fn(pred, target)


# ============================================
# 4. 总损失函数 (Fusion Loss)
# 作用: 组合所有子损失
# ============================================
class FusionLoss(nn.Module):
    def __init__(self, lambda_pix=1.0, lambda_perc=0.1, lambda_ssim=1.0, lambda_grad=0.5, lambda_feat=0.5, device='cuda'):
        super(FusionLoss, self).__init__()
        self.lambda_pix = lambda_pix
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad
        self.lambda_feat = lambda_feat # 控制 FeatureConsistencyLoss
        
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size=11, reduction='mean', device=device)
        self.grad_loss = GradientLoss(device)
        self.feat_loss = FeatureConsistencyLoss(device)

    def forward(self, pred, target, student_feats=None, teacher_feats=None):
        losses = {}
        
        # 1. 像素损失 (L1)
        if self.lambda_pix > 0:
            losses['pix'] = self.l1_loss(pred, target)
        else:
            losses['pix'] = torch.tensor(0.0).to(pred.device)
        
        # 2. SSIM 损失 (基础结构损失)
        if self.lambda_ssim > 0:
            losses['ssim'] = self.ssim_loss(pred, target)
        else:
            losses['ssim'] = torch.tensor(0.0).to(pred.device)
        
        # 3. 梯度损失 (锐化)
        if self.lambda_grad > 0:
            losses['grad'] = self.grad_loss(pred, target)
        else:
            losses['grad'] = torch.tensor(0.0).to(pred.device)
        
        # 4. 特征一致性损失 (Feature Distillation)
        # 即使传入了 student_feats，我们也使用 VGG 特征作为蒸馏目标
        if self.lambda_feat > 0:
            losses['feat'] = self.feat_loss(pred, target)
        else:
            losses['feat'] = torch.tensor(0.0).to(pred.device)
            
        # lambda_perc 暂时不用，因为我们用 lambda_feat 统一了
        losses['perc'] = torch.tensor(0.0).to(pred.device) 
        
        # 总损失
        total_loss = (self.lambda_pix * losses['pix'] +
                      self.lambda_ssim * losses['ssim'] +
                      self.lambda_grad * losses['grad'] +
                      self.lambda_feat * losses['feat'])
                      
        losses['total'] = total_loss.item()
        return total_loss, losses