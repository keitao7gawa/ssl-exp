from typing import Dict, Any
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from utils.transform_dict import TRANSFORM_MAP

class MyCIFAR10(Dataset):
    """CIFAR10データセットのラッパークラス
    
    Attributes:
        transform (transforms.Compose): データ変換
        base_dataset (datasets.CIFAR10): ベースとなるCIFAR10データセット
    """
    
    def __init__(
        self,
        train: bool,
        root: str,
        download: bool,
        transform_cfg: Dict[str, Any],
    ) -> None:
        """初期化
        
        Args:
            train (bool): 訓練データかどうか
            root (str): データセットの保存先
            download (bool): ダウンロードするかどうか
            transform_cfg (Dict[str, Any]): 変換の設定辞書
        """
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, os.path.normpath(root))

        if transform_cfg["name"] == "None":
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform_class = TRANSFORM_MAP[transform_cfg["name"]]
            if "params" in transform_cfg:
                self.transform = transform_class(**transform_cfg["params"])
            else:
                self.transform = transform_class()
            
        self.base_dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=download,
            transform=self.transform
        )
    
    def __len__(self) -> int:
        """データセットのサイズを返す
        
        Returns:
            int: データセットのサイズ
        """
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> tuple:
        """データセットからサンプルを取得
        
        Args:
            idx (int): インデックス
            
        Returns:
            tuple: (画像, ラベル)
        """
        return self.base_dataset[idx] 