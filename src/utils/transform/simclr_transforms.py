from torchvision import transforms

class SimCLRTransform:
    def __init__(self):
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
        return self.transform(x), self.transform(x)