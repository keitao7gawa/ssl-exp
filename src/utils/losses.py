import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """NT-Xent（Normalized Temperature-scaled Cross Entropy）ロス
    
    SimCLRで使用される対照学習の損失関数．
    温度パラメータτでスケーリングされたコサイン類似度を使用．
    
    Attributes:
        temperature (float): 温度パラメータτ
    """
    
    def __init__(self, temperature: float = 0.07):
        """NT-Xentロスの初期化
        
        Args:
            temperature (float): 温度パラメータ．デフォルトは0.07
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xentロスの計算
        
        Args:
            z1 (torch.Tensor): 1つ目の拡張画像の射影ベクトル [B, D]
            z2 (torch.Tensor): 2つ目の拡張画像の射影ベクトル [B, D]
            
        Returns:
            torch.Tensor: NT-Xentロス
        """
        # バッチサイズ
        batch_size = z1.size(0)
        
        # 正規化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # すべてのペアのコサイン類似度を計算
        # [2B, 2B]の類似度行列を作成
        z = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(z, z.T) / self.temperature
        
        # 正例のインデックス
        # 対角成分（同じ画像のペア）と対角成分の反対側（拡張画像のペア）
        pos_indices = torch.arange(batch_size)
        pos_indices = torch.cat([pos_indices, pos_indices + batch_size], dim=0)
        
        # 正例の類似度を取得
        pos_similarity = similarity_matrix[pos_indices, pos_indices.roll(batch_size)]
        
        # 負例の類似度を取得
        # 対角成分をマスク
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        neg_similarity = similarity_matrix[~mask].view(2 * batch_size, -1)
        
        # 損失の計算
        # log(exp(pos) / (exp(pos) + sum(exp(neg))))
        pos_exp = torch.exp(pos_similarity)
        neg_exp = torch.exp(neg_similarity).sum(dim=1)
        loss = -torch.log(pos_exp / (pos_exp + neg_exp))
        
        return loss.mean() 