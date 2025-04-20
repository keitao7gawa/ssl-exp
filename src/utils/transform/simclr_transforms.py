from torchvision import transforms

class SimCLRTransform:
    def __init__(self, n_views: int = 2):
        """SimCLRのデータ拡張
        
        Args:
            n_views (int): 生成する拡張画像の数（デフォルト: 2）
        """
        self.n_views = n_views
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)), # transforms.RandomResizedCrop(32)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                [0.2470, 0.2435, 0.2616]) # CIFAR-10の平均と標準偏差
        ])

    
    def __call__(self, x):
        """入力画像にデータ拡張を適用
        
        Args:
            x: 入力画像
            
        Returns:
            tuple: n_views個の拡張画像のタプル
        """
        return tuple(self.transform(x) for _ in range(self.n_views))