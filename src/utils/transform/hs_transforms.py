from torchvision import transforms
from .hs_colorjitter import RandomHSColorJitter, RandomHSGrayscale

class HSSimCLRTransform:
    def __init__(self, n_views: int = 2, hs_colorjitter: bool = True, gaussian_blur: bool = True, instance_normalization: bool = False):
        """SimCLRのデータ拡張
        
        Args:
            n_views (int): 生成する拡張画像の数（デフォルト: 2）
        """
        self.n_views = n_views
        transform_list = [
            transforms.RandomResizedCrop(64, scale=(0.2, 1.0)), # transforms.RandomResizedCrop(32)
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        if hs_colorjitter:
            transform_list.append(
                transforms.RandomApply([
                    RandomHSColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8)
            )
        transform_list.append(
            transforms.RandomApply([
                RandomHSGrayscale()
            ], p=0.2),
        )
        if gaussian_blur:
            transform_list.append(
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            )
        if instance_normalization:
            transform_list.append(
                transforms.Lambda(lambda x: (x - x.view(x.size(0), -1).mean(dim=1, keepdim=True).view(x.size(0), 1, 1)) / 
                                  (x.view(x.size(0), -1).std(dim=1, keepdim=True).view(x.size(0), 1, 1) + 1e-5))
            )
            
        
        self.transform = transforms.Compose(transform_list)

    
    def __call__(self, x):
        """入力画像にデータ拡張を適用
        
        Args:
            x: 入力画像
            
        Returns:
            tuple: n_views個の拡張画像のタプル
        """
        return tuple(self.transform(x) for _ in range(self.n_views))

class HSMoCoTransform:
    def __init__(self, n_views: int = 2):
        """SimCLRのデータ拡張
        
        Args:
            n_views (int): 生成する拡張画像の数（デフォルト: 2）
        """
        self.n_views = n_views
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.2, 1.0)), # transforms.RandomResizedCrop(32)
            transforms.RandomApply([
                RandomHSGrayscale()
            ], p=0.2),
            RandomHSColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.Normalize([0.4914, 0.4822, 0.4465],
            #                     [0.2470, 0.2435, 0.2616]) # CIFAR-10の平均と標準偏差
        ])

    
    def __call__(self, x):
        """入力画像にデータ拡張を適用
        
        Args:
            x: 入力画像
            
        Returns:
            tuple: n_views個の拡張画像のタプル
        """
        return tuple(self.transform(x) for _ in range(self.n_views))