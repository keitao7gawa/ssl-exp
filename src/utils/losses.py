import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

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
        
    def forward(self, z_list: List[torch.Tensor]) -> torch.Tensor:
        """NT-Xentロスの計算
        
        Args:
            z_list (List[torch.Tensor]): 拡張画像の射影ベクトルのリスト [B, D]
            
        Returns:
            torch.Tensor: NT-Xentロス
        """
        # バッチサイズ
        batch_size = z_list[0].size(0)
        n_views = len(z_list)
        
        # すべてのビューを正規化
        z_list = [F.normalize(z, dim=1) for z in z_list]
        
        # すべてのビューを連結
        z = torch.cat(z_list, dim=0)
        
        # すべてのペアのコサイン類似度を計算
        # [n_views * B, n_views * B]の類似度行列を作成
        similarity_matrix = torch.matmul(z, z.T) / self.temperature
        
        # 正例のマスクを作成
        # 各ビューに対して，同じ画像の他のビューが正例
        mask = torch.zeros((n_views * batch_size, n_views * batch_size), dtype=torch.bool, device=z.device)
        for i in range(n_views):
            for j in range(n_views):
                if i != j:
                    # 同じ画像の異なるビューを正例としてマーク
                    mask[i * batch_size:(i + 1) * batch_size, j * batch_size:(j + 1) * batch_size] = True
        
        # 正例の類似度を取得
        pos_similarity = similarity_matrix[mask].view(n_views * batch_size, -1)
        
        # 負例の類似度を取得
        # 対角成分をマスク
        neg_mask = torch.eye(n_views * batch_size, dtype=torch.bool, device=z.device)
        neg_similarity = similarity_matrix[~neg_mask].view(n_views * batch_size, -1)
        
        # 損失の計算
        # log(exp(pos) / (exp(pos) + sum(exp(neg))))
        pos_exp = torch.exp(pos_similarity).sum(dim=1)
        neg_exp = torch.exp(neg_similarity).sum(dim=1)
        loss = -torch.log(pos_exp / (pos_exp + neg_exp))
        
        return loss.mean() 