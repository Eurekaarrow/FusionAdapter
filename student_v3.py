import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

# ============================================
# 核心创新模块：ConvNeXt V2 Block
# (替代普通的 Conv+BN+ReLU，提升 Novelty 和 性能)
# ============================================

class LayerNorm(nn.Module):
    """ 支持两种数据格式的 LayerNorm: channels_last (default) or channels_first. """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ Global Response Normalization (GRN) layer
    ConvNeXt V2 的核心，增强特征对比度，解决过曝和特征模糊问题的关键。
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtBlock(nn.Module):
    """ ConvNeXt V2 Block """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # Depthwise Conv (7x7) - 大感受野，模仿 Transformer/Diffusion
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = LayerNorm(dim, eps=1e-6)
        # Pointwise Conv 1
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        # GRN (关键组件)
        self.grn = GRN(4 * dim)
        # Pointwise Conv 2
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x) # Apply GRN
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

# ============================================
# 融合模块 (Refined for Robustness)
# ============================================

class RobustFusionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 使用 Sigmoid 而非 Softmax，避免竞争导致的过曝
        self.weight_gen = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, 2, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim) # Depthwise fusion

    def forward(self, ir, vis):
        weights = self.weight_gen(torch.cat([ir, vis], dim=1))
        # 显式分离权重，允许两者都高
        fused = ir * weights[:, 0:1] + vis * weights[:, 1:2]
        # 残差连接，保证底线效果
        return self.proj(fused) + ir + vis

# ============================================
# 主网络架构：ConvNeXt-UNet
# ============================================

class ConvNeXtUNetStudent(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, depths=[2, 2, 4, 2], dims=[48, 96, 192, 384]):
        super().__init__()
        
        # --- Encoder (Downsampling) ---
        self.downsample_layers_ir = nn.ModuleList() 
        self.downsample_layers_vis = nn.ModuleList() 
        
        # Stem layer
        stem_ir = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        stem_vis = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4), # VIS always 3
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers_ir.append(stem_ir)
        self.downsample_layers_vis.append(stem_vis)
        
        # Downsample layers (Stage 2-4)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers_ir.append(downsample_layer)
            self.downsample_layers_vis.append(downsample_layer)

        # Stages (ConvNeXt Blocks)
        self.stages_ir = nn.ModuleList()
        self.stages_vis = nn.ModuleList()
        
        dp_rates = [x.item() for x in torch.linspace(0, 0.1, sum(depths))] 
        cur = 0
        for i in range(4):
            stage_ir = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            stage_vis = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages_ir.append(stage_ir)
            self.stages_vis.append(stage_vis)
            cur += depths[i]

        # --- Fusion ---
        self.fusion_blocks = nn.ModuleList([RobustFusionBlock(dim) for dim in dims])

        # --- Decoder (Upsampling) ---
        self.up_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(3): # 3 upsampling steps
            # Upsample from dims[i+1] -> dims[i]
            self.up_layers.append(nn.Sequential(
                nn.ConvTranspose2d(dims[3-i], dims[2-i], kernel_size=2, stride=2),
                LayerNorm(dims[2-i], eps=1e-6, data_format="channels_first")
            ))
            # Decoder Block (ConvNeXt style but lighter)
            self.decoder_blocks.append(ConvNeXtBlock(dims[2-i]))

        # Final Head
        self.final_up = nn.Sequential(
             nn.ConvTranspose2d(dims[0], dims[0], kernel_size=4, stride=4),
             LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.head = nn.Conv2d(dims[0], out_chans, kernel_size=1)
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x, downsample_layers, stages):
        features = []
        for i in range(4):
            x = downsample_layers[i](x)
            x = stages[i](x)
            features.append(x)
        return features

    def forward(self, ir, vis, return_features=False):
        # 1. Encoder
        ir_feats = self.forward_encoder(ir, self.downsample_layers_ir, self.stages_ir)
        vis_feats = self.forward_encoder(vis, self.downsample_layers_vis, self.stages_vis)
        
        # 2. Fusion
        fused_feats = []
        for i in range(4):
            fused = self.fusion_blocks[i](ir_feats[i], vis_feats[i])
            fused_feats.append(fused)
            
        # 3. Decoder
        # Start from the deepest feature (Stage 4, index 3)
        x = fused_feats[3]
        
        # Up 1: Stage 4 -> Stage 3
        x = self.up_layers[0](x)
        x = x + fused_feats[2] # Skip connection
        x = self.decoder_blocks[0](x)
        
        # Up 2: Stage 3 -> Stage 2
        x = self.up_layers[1](x)
        x = x + fused_feats[1] # Skip connection
        x = self.decoder_blocks[1](x)
        
        # Up 3: Stage 2 -> Stage 1
        x = self.up_layers[2](x)
        x = x + fused_feats[0] # Skip connection
        x = self.decoder_blocks[2](x)
        
        # Final Up
        x = self.final_up(x)
        out = torch.sigmoid(self.head(x))
        
        if return_features:
            return out, fused_feats
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_student_model(in_channels=3, out_channels=3):
    # Dims 和 Depths 可调。这里设置为较轻量级的版本
    # Tiny version: depths=[2, 2, 4, 2], dims=[48, 96, 192, 384]
    model = ConvNeXtUNetStudent(in_chans=in_channels, out_chans=out_channels)
    print(f"ConvNeXt-V2 Student created with {model.count_parameters()/1e6:.2f}M parameters")
    return model

if __name__ == "__main__":
    model = create_student_model()
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)
    out = model(x, y)
    print(out.shape)