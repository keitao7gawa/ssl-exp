import random
import numpy as np
import torch
from torchvision import transforms


class CutOut:
    """CutOutデータ拡張
    
    CutOut: https://arxiv.org/abs/1708.04552
    画像の一部をランダムにマスクするデータ拡張手法
    
    Args:
        length (int): マスクのサイズ（正方形）
        p (float): 適用確率
    """
    
    def __init__(self, length: int, p: float = 0.5):
        self.length = length
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """画像にCutOutを適用
        
        Args:
            img (torch.Tensor): 入力画像 [C, H, W]
            
        Returns:
            torch.Tensor: 変換後の画像 [C, H, W]
        """
        if random.random() < self.p:
            h, w = img.size(1), img.size(2)
            mask = np.ones((h, w), np.float32)
            
            # マスクの中心位置をランダムに選択
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            # マスクの範囲を計算
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            # マスクを適用
            mask[y1:y2, x1:x2] = 0.
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask
            
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={self.length}, p={self.p})" 