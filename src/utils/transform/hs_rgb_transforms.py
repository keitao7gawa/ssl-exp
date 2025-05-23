from torchvision import transforms
from .hs_to_rgb import HStoRGB
from typing import Tuple

class HStoRGBSimCLRTransform:
    def __init__(self, n_views: int = 2, gaussian_blur: bool = True, instance_normalization: bool = False,
                 range_wavelength: Tuple[int, int] = (350, 1100),
                 spectrum_stepsize: int = 5,
                 spectrum_bands: int = 151):
        """SimCLRのデータ拡張
        
        Args:
            n_views (int): 生成する拡張画像の数（デフォルト: 2）
        """
        self.n_views = n_views
        transform_list = [
            HStoRGB(lower_limit_wavelength=range_wavelength[0], upper_limit_wavelength=range_wavelength[1], spectrum_stepsize=spectrum_stepsize, spectrum_bands=spectrum_bands),
            transforms.RandomResizedCrop(64, scale=(0.2, 1.0)), # transforms.RandomResizedCrop(32)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        if gaussian_blur:
            transform_list.append(
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            )
        if instance_normalization:
            transform_list.append(
                transforms.Lambda(lambda x: self.instance_normalize(x))
            )
            
        self.transform = transforms.Compose(transform_list)

    def instance_normalize(self, x):
        if x.dim() == 4:
            # (B, C, H, W)
            mean = x.view(x.size(0), x.size(1), -1).mean(dim=2, keepdim=True).view(x.size(0), x.size(1), 1, 1)
            std = x.view(x.size(0), x.size(1), -1).std(dim=2, keepdim=True).view(x.size(0), x.size(1), 1, 1)
            return (x - mean) / (std + 1e-5)
        elif x.dim() == 3:
            # (C, H, W)
            mean = x.view(x.size(0), -1).mean(dim=1, keepdim=True).view(x.size(0), 1, 1)
            std = x.view(x.size(0), -1).std(dim=1, keepdim=True).view(x.size(0), 1, 1)
            return (x - mean) / (std + 1e-5)
        else:
            raise ValueError("Input must be (B, C, H, W) or (C, H, W)")
        
    def __call__(self, x):
        """入力画像にデータ拡張を適用
        
        Args:
            x: 入力画像
            
        Returns:
            tuple: n_views個の拡張画像のタプル
        """
        return tuple(self.transform(x) for _ in range(self.n_views))
    
class HStoRGBMAETransform:
    def __init__(self, size: Tuple[int, int] = (64, 64),
                 range_wavelength: Tuple[int, int] = (350, 1100),
                 spectrum_stepsize: int = 5,
                 spectrum_bands: int = 151):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(size[0], size[1]), scale=(0.2, 1.0)),
            HStoRGB(lower_limit_wavelength=range_wavelength[0], upper_limit_wavelength=range_wavelength[1], spectrum_stepsize=spectrum_stepsize, spectrum_bands=spectrum_bands),
        ])
        
    def __call__(self, x):
        return self.transform(x)