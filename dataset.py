import os
import random
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================
# 未配准模拟变换
# ============================================

class MisalignmentAugmentation:
    """
    模拟IR和VIS图像之间的未配准情况
    包括平移、旋转、缩放和仿射变换
    """
    def __init__(self, 
                 max_translate=20,  # 最大平移像素
                 max_rotate=5,      # 最大旋转角度
                 scale_range=(0.95, 1.05)):  # 缩放范围
        self.max_translate = max_translate
        self.max_rotate = max_rotate
        self.scale_range = scale_range
    
    def __call__(self, image):
        """
        对图像应用随机未配准变换
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            transformed: 变换后的图像
        """
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            w, h = image.size
        
        # 随机平移
        tx = random.uniform(-self.max_translate, self.max_translate)
        ty = random.uniform(-self.max_translate, self.max_translate)
        
        # 随机旋转
        angle = random.uniform(-self.max_rotate, self.max_rotate)
        
        # 随机缩放
        scale = random.uniform(*self.scale_range)
        
        # 构建仿射变换矩阵
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        
        # 应用变换
        if isinstance(image, np.ndarray):
            transformed = cv2.warpAffine(image, M, (w, h), 
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT)
        else:
            image = np.array(image)
            transformed = cv2.warpAffine(image, M, (w, h), 
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT)
            transformed = Image.fromarray(transformed)
        
        return transformed


# ============================================
# 增强管线
# ============================================

def get_training_augmentation(input_size=512, misalign_prob=0.5):
    """
    获取训练时的数据增强管线
    
    Args:
        input_size: 输入图像尺寸
        misalign_prob: 应用未配准增强的概率
    
    Returns:
        transform: Albumentations transform
    """
    return A.Compose([
        # 随机裁剪和缩放
        A.RandomResizedCrop(input_size, input_size, scale=(0.8, 1.0)),
        
        # 随机水平翻转
        A.HorizontalFlip(p=0.5),
        
        # 颜色增强（仅对可见光）
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        ], p=0.5),
        
        # 噪声增强（模拟传感器噪声）
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # 模糊增强
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.2),
    ])


def get_validation_augmentation(input_size=512):
    """
    获取验证时的数据增强管线（仅缩放和裁剪）
    """
    return A.Compose([
        A.Resize(input_size, input_size),
    ])


# ============================================
# 主数据集类（适配您的数据结构）
# ============================================

class FusionDataset(Dataset):
    """
    红外-可见光融合数据集
    
    数据组织结构（您的结构）:
    ir_root/          # ./data/ir
    ├── image_001.png
    ├── image_002.png
    └── ...
    
    vis_root/         # ./data/vi
    ├── image_001.png
    ├── image_002.png
    └── ...
    
    teacher_root/     # ./DDAEFuse_MSRS
    ├── image_001.png
    ├── image_002.png
    └── ...
    """
    
    def __init__(self, 
                 ir_root,
                 vis_root,
                 teacher_root,
                 image_list,
                 mode='train',
                 input_size=512,
                 misalign_prob=0.5,
                 ir_channels=1,
                 vis_channels=3):
        """
        Args:
            ir_root: 红外图像根目录
            vis_root: 可见光图像根目录
            teacher_root: Teacher融合图像根目录
            image_list: 图像文件名列表
            mode: 'train' or 'val'
            input_size: 输入图像尺寸
            misalign_prob: 训练时应用未配准增强的概率
            ir_channels: 红外图像通道数 (1 or 3)
            vis_channels: 可见光图像通道数 (3)
        """
        self.ir_root = ir_root
        self.vis_root = vis_root
        self.teacher_root = teacher_root
        self.image_list = image_list
        self.mode = mode
        self.input_size = input_size
        self.misalign_prob = misalign_prob if mode == 'train' else 0.0
        self.ir_channels = ir_channels
        self.vis_channels = vis_channels
        
        # 设置增强
        if mode == 'train':
            self.augmentation = get_training_augmentation(input_size, misalign_prob)
        else:
            self.augmentation = get_validation_augmentation(input_size)
        
        # 未配准增强器
        self.misalign_aug = MisalignmentAugmentation(
            max_translate=20,
            max_rotate=5,
            scale_range=(0.95, 1.05)
        )
        
        print(f"[{mode.upper()}] Loaded {len(self.image_list)} image pairs")
    
    def _load_image(self, path, channels=3):
        """
        加载图像
        
        Args:
            path: 图像路径
            channels: 目标通道数 (1 or 3)
        
        Returns:
            image: numpy array [H, W, C]
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        
        if channels == 1:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {path}")
            image = np.expand_dims(image, axis=-1)  # [H, W, 1]
        else:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to load image: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        Returns:
            ir: IR image tensor [C, H, W]
            vis: VIS image tensor [C, H, W]
            teacher: Teacher fused image tensor [C, H, W]
        """
        # 获取文件名
        filename = self.image_list[idx]
        
        # 构建完整路径
        ir_path = os.path.join(self.ir_root, filename)
        vis_path = os.path.join(self.vis_root, filename)
        teacher_path = os.path.join(self.teacher_root, filename)
        
        # 加载图像
        ir = self._load_image(ir_path, self.ir_channels)
        vis = self._load_image(vis_path, self.vis_channels)
        teacher = self._load_image(teacher_path, 3)
        
        # 如果IR是单通道,复制到3通道以便使用标准增强
        if self.ir_channels == 1:
            ir = np.repeat(ir, 3, axis=-1)
        
        # 应用未配准增强（仅在训练时）
        if self.mode == 'train' and random.random() < self.misalign_prob:
            # 对IR单独应用变换,模拟未配准
            ir = (ir * 255).astype(np.uint8)
            ir = self.misalign_aug(ir).astype(np.float32) / 255.0
        
        # 应用共同增强（裁剪、翻转等）
        # Albumentations需要字典输入
        augmented = self.augmentation(
            image=vis,
            masks=[ir, teacher]
        )
        
        vis = augmented['image']
        ir = augmented['masks'][0]
        teacher = augmented['masks'][1]
        
        # 转换为tensor [C, H, W]
        ir = torch.from_numpy(ir).permute(2, 0, 1).float()
        vis = torch.from_numpy(vis).permute(2, 0, 1).float()
        teacher = torch.from_numpy(teacher).permute(2, 0, 1).float()
        
        # 如果原始IR是单通道,只保留第一个通道
        if self.ir_channels == 1:
            ir = ir[0:1, :, :]
        
        return ir, vis, teacher


# ============================================
# 数据集划分工具
# ============================================

def split_dataset(ir_root, vis_root, teacher_root, 
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                 seed=42):
    """
    自动划分数据集为train/val/test
    
    Args:
        ir_root: 红外图像目录
        vis_root: 可见光图像目录
        teacher_root: Teacher融合图像目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        train_list, val_list, test_list: 文件名列表
    """
    # 获取所有图像文件
    ir_files = set(f for f in os.listdir(ir_root) 
                   if f.endswith(('.png', '.jpg', '.jpeg', '.bmp')))
    vis_files = set(f for f in os.listdir(vis_root) 
                    if f.endswith(('.png', '.jpg', '.jpeg', '.bmp')))
    teacher_files = set(f for f in os.listdir(teacher_root) 
                       if f.endswith(('.png', '.jpg', '.jpeg', '.bmp')))
    
    # 找到三者的交集（确保所有图像都存在）
    common_files = ir_files & vis_files & teacher_files
    
    if len(common_files) == 0:
        raise ValueError("No common files found across IR, VIS, and Teacher directories!")
    
    print(f"\nDataset Statistics:")
    print(f"  IR images: {len(ir_files)}")
    print(f"  VIS images: {len(vis_files)}")
    print(f"  Teacher images: {len(teacher_files)}")
    print(f"  Common images: {len(common_files)}")
    
    # 转换为列表并排序
    all_files = sorted(list(common_files))
    
    # 设置随机种子
    random.seed(seed)
    random.shuffle(all_files)
    
    # 计算划分点
    total = len(all_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_list = all_files[:train_end]
    val_list = all_files[train_end:val_end]
    test_list = all_files[val_end:]
    
    print(f"\nDataset Split:")
    print(f"  Train: {len(train_list)} ({len(train_list)/total*100:.1f}%)")
    print(f"  Val:   {len(val_list)} ({len(val_list)/total*100:.1f}%)")
    print(f"  Test:  {len(test_list)} ({len(test_list)/total*100:.1f}%)")
    
    return train_list, val_list, test_list


# ============================================
# 数据加载器创建函数
# ============================================

def create_dataloaders(ir_root, vis_root, teacher_root,
                       batch_size=8, num_workers=4, 
                       input_size=512, misalign_prob=0.5,
                       train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                       seed=42):
    """
    创建训练和验证数据加载器（自动划分数据集）
    
    Args:
        ir_root: 红外图像根目录
        vis_root: 可见光图像根目录
        teacher_root: Teacher融合图像根目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        input_size: 输入尺寸
        misalign_prob: 未配准增强概率
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 自动划分数据集
    train_list, val_list, test_list = split_dataset(
        ir_root, vis_root, teacher_root,
        train_ratio, val_ratio, test_ratio, seed
    )
    
    # 创建数据集
    train_dataset = FusionDataset(
        ir_root, vis_root, teacher_root,
        image_list=train_list,
        mode='train', 
        input_size=input_size,
        misalign_prob=misalign_prob
    )
    
    val_dataset = FusionDataset(
        ir_root, vis_root, teacher_root,
        image_list=val_list,
        mode='val', 
        input_size=input_size,
        misalign_prob=0.0
    )
    
    test_dataset = FusionDataset(
        ir_root, vis_root, teacher_root,
        image_list=test_list,
        mode='val',  # 使用val模式（不增强）
        input_size=input_size,
        misalign_prob=0.0
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    # 测试数据集划分
    ir_root = './data/ir'
    vis_root = './data/vi'
    teacher_root = './DDAEFuse_MSRS'
    
    if os.path.exists(ir_root) and os.path.exists(vis_root) and os.path.exists(teacher_root):
        print("Testing dataset split...")
        train_list, val_list, test_list = split_dataset(
            ir_root, vis_root, teacher_root
        )
        
        print(f"\nFirst 5 training files:")
        for f in train_list[:5]:
            print(f"  {f}")
    else:
        print("Data directories not found. Creating test data...")
        # 创建测试数据
        for root in [ir_root, vis_root, teacher_root]:
            os.makedirs(root, exist_ok=True)
            for i in range(10):
                img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(root, f'test_{i:03d}.png'), img)
        
        print("Test data created. Run again to test.")
