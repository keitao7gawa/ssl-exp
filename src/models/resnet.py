import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Type, Union, List

class ResNetWrapper(nn.Module):
    """ImageNet用にカスタマイズしたResNetラッパー
    
    torchvisionのResNetモデルをベースに，入力画像サイズに応じて調整可能なモデルを提供します．
    
    Attributes:
        model (nn.Module): ベースとなるResNetモデル
    """
    
    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = False,
        num_classes: int = 10,
        input_layer_params: Optional[dict] = None,
        **kwargs
    ):
        """ResNetWrapperの初期化
        
        Args:
            model_name (str): 使用するResNetモデルの名前（resnet18, resnet34, resnet50, resnet101, resnet152）
            pretrained (bool): 事前学習済みの重みを使用するかどうか
            num_classes (int): 出力クラス数
            input_layer_params (dict, optional): 入力層のパラメータ
                - in_channels (int): 入力チャンネル数（デフォルト: 3）
                - kernel_size (int): カーネルサイズ（デフォルト: 7）
                - stride (int): ストライド（デフォルト: 2）
                - padding (int): パディング（デフォルト: 3）
                - bias (bool): バイアス項の有無（デフォルト: False）
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
        
        # 入力層のパラメータ設定（ImageNet用デフォルト）
        default_input_params = {
            "in_channels": 3,
            "kernel_size": 7,
            "stride": 2,
            "padding": 3,
            "bias": False
        }
        if input_layer_params is not None:
            default_input_params.update(input_layer_params)
        
        # 入力層の調整
        self.model.conv1 = nn.Conv2d(
            default_input_params["in_channels"], 64,
            kernel_size=default_input_params["kernel_size"],
            stride=default_input_params["stride"],
            padding=default_input_params["padding"],
            bias=default_input_params["bias"]
        )
        
        # 出力層の調整
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播
        
        Args:
            x (torch.Tensor): 入力テンソル [B, 3, 32, 32]
            
        Returns:
            torch.Tensor: 出力テンソル [B, num_classes]
        """
        return self.model(x) 