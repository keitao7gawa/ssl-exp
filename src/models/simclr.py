import torch
import torch.nn as nn
from typing import Optional, Tuple

from .resnet import ResNetWrapper

class SimCLR(nn.Module):
    """SimCLRモデル
    
    ResNetをエンコーダーとして使用し，射影ヘッドを追加したモデル．
    
    Attributes:
        encoder (nn.Module): ベースエンコーダー（ResNet）
        projection (nn.Module): 射影ヘッド（MLP）
    """
    
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        encoder_name: str = "resnet18",
        projection_dim: int = 128,
        hidden_dim: int = 512,
        **kwargs
    ):
        """SimCLRモデルの初期化
        
        Args:
            encoder (Optional[nn.Module]): カスタムエンコーダー．Noneの場合はResNetWrapperを使用
            encoder_name (str): 使用するResNetモデルの名前
            projection_dim (int): 射影ヘッドの出力次元
            hidden_dim (int): 射影ヘッドの中間層の次元
            **kwargs: エンコーダーの追加パラメータ
        """
        super().__init__()
        
        # エンコーダーの設定
        if encoder is None:
            self.encoder = ResNetWrapper(
                model_name=encoder_name,
                pretrained=False,
                num_classes=hidden_dim,  # 射影ヘッドの入力次元に合わせる
                **kwargs
            )
        else:
            self.encoder = encoder
            
        # 射影ヘッドの設定
        # 入力次元はエンコーダーの出力次元
        in_features = self.encoder.model.fc.in_features
        
        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)  # 最終層の正規化
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播
        
        Args:
            x (torch.Tensor): 入力テンソル [B, 3, 32, 32]
            
        Returns:
            torch.Tensor: 射影ヘッドの出力 [B, projection_dim]
        """
        # エンコーダーの出力
        h = self.encoder(x)
        
        # 射影ヘッドの出力
        z = self.projection(h)
        
        return z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """エンコーダーのみを使用して特徴量を抽出
        
        Args:
            x (torch.Tensor): 入力テンソル [B, 3, 32, 32]
            
        Returns:
            torch.Tensor: エンコーダーの出力 [B, hidden_dim]
        """
        return self.encoder(x) 