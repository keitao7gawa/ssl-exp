import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Type, Union, List

class ResNetWrapper(nn.Module):
    """CIFAR-10用にカスタマイズしたResNetラッパー
    
    torchvisionのResNetモデルをベースに，CIFAR-10用に調整したモデルを提供します．
    
    Attributes:
        model (nn.Module): ベースとなるResNetモデル
    """
    
    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = False,
        num_classes: int = 10,
        **kwargs
    ):
        """ResNetWrapperの初期化
        
        Args:
            model_name (str): 使用するResNetモデルの名前（resnet18, resnet34, resnet50, resnet101, resnet152）
            pretrained (bool): 事前学習済みの重みを使用するかどうか
            num_classes (int): 出力クラス数（CIFAR-10の場合は10）
            **kwargs: その他のResNetモデルのパラメータ
        """
        super().__init__()
        
        # モデルの選択
        if model_name == "resnet18":
            self.model = models.resnet18(pretrained=pretrained, **kwargs)
        elif model_name == "resnet34":
            self.model = models.resnet34(pretrained=pretrained, **kwargs)
        elif model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained, **kwargs)
        elif model_name == "resnet101":
            self.model = models.resnet101(pretrained=pretrained, **kwargs)
        elif model_name == "resnet152":
            self.model = models.resnet152(pretrained=pretrained, **kwargs)
        else:
            raise ValueError(f"サポートされていないモデル名: {model_name}")
        
        # CIFAR-10用に最終層を調整
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        # CIFAR-10用に最初の層を調整（32x32の入力に対応）
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # CIFAR-10では不要
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播
        
        Args:
            x (torch.Tensor): 入力テンソル [B, 3, 32, 32]
            
        Returns:
            torch.Tensor: 出力テンソル [B, num_classes]
        """
        return self.model(x) 