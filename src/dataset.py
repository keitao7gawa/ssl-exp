from typing import Dict, Any, List
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from utils.transform_dict import TRANSFORM_MAP
import h5py
import os
from ast import literal_eval
import torch

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
    
class HFD100(Dataset):
    """HFD100データセットのラッパークラス
    
    Attributes:
        train (bool): 訓練データかどうか
        dir_path (str): データセットのディレクトリパス
        train_transform (transforms.Compose): 訓練データの変換
        val_transform (transforms.Compose): 検証データの変換
        use_all (bool): 全てのデータを使用するかどうか
        standardization_params (Dict[str, Any]): 標準化のパラメータ
    """
    def __init__(self, train: bool,
                 dir_path: str,
                 train_transform: Dict[str, Any] = {"name": "None"},
                 val_transform: Dict[str, Any] = {"name": "None"},
                 target_transform: Dict[str, Any] = {"name": "None"},
                 data_types = None,
                 use_all: bool = False,
                 standardization_params: Dict[str, Any] = None):
        
        self.dir_path = dir_path
        self.step = "train" if train else "test"

        # set transfomr
        transform_cfg = train_transform if train else val_transform
        if transform_cfg["name"] == "None":
            self.transform = None
        else:
            transform_class = TRANSFORM_MAP[transform_cfg["name"]]
            if "params" in transform_cfg:
                self.transform = transform_class(**transform_cfg["params"])
            else:
                self.transform = transform_class()

        # set target transform
        if target_transform["name"] == "None":
            self.target_transform = None
        else:
            transform_class = TRANSFORM_MAP[target_transform["name"]]
            if "params" in target_transform:
                self.target_transform = transform_class(**target_transform["params"])
            else:
                self.target_transform = transform_class()

        self.use_all = use_all
        if standardization_params is not None:
            standardization_params["name"] = "DatasetStandardization"
            self.standardization_transform = TRANSFORM_MAP[standardization_params["name"]](**standardization_params["params"])
        else:
            self.standardization_transform = None
        
        data_types = data_types if isinstance(data_types, list) else [data_types]
        assert all([data_type in ["scene", "flower", "leaf"] for data_type in data_types]), "data_types must be one of ['scene', 'flower', 'leaf']"

        self.metadata = self.__load_metadata(data_types)
        TYPE_DICT = {"scene": "MatScenes60.h5", "flower": "MatFlower60.h5", "leaf": "MatLeaves60.h5"}
        self.h5_files = {TYPE_DICT[data_type]: h5py.File(os.path.join(self.dir_path, TYPE_DICT[data_type]), "r") for data_type in data_types}
        
        
    def __load_metadata(self, data_types: List[str]):
        """メタデータを読み込む"""
        TYPE_DICT = {"scene": "MatScenes60.h5", "flower": "MatFlower60.h5", "leaf": "MatLeaves60.h5"}
        metadata = []
        for data_type in data_types:
            with h5py.File(os.path.join(self.dir_path, TYPE_DICT[data_type]), "r") as f:
                if self.use_all:
                    for step in ["train", "test"]:
                        meta = f[step]["metadata"][()]
                        meta_dict = literal_eval(meta.decode("utf-8"))
                        for dict in meta_dict:
                            dict['hsi'] = os.path.join(TYPE_DICT[data_type], step, dict['hsi'])
                        metadata += meta_dict
                else:
                    meta = f[self.step]["metadata"][()]
                    meta_dict = literal_eval(meta.decode("utf-8"))
                    for dict in meta_dict:
                        dict['hsi'] = os.path.join(TYPE_DICT[data_type], self.step, dict['hsi'])
                    metadata += meta_dict
        return metadata
    
    def __len__(self):
        """データセットのサイズを返す"""
        return len(self.metadata)
    
    def __getitem__(self, idx: int):
        """データセットからサンプルを取得"""
        image_meta_data = self.metadata[idx]

        image_path = image_meta_data["hsi"].split("/")
        h5_name = image_path[0]
        step_name = image_path[1]
        image_name = image_path[3]

        try:
            image = self.h5_files[h5_name][step_name]['hs'][image_name][()]
            target = self.h5_files[h5_name][step_name]['target'][image_name][()]
        except OSError:
            if h5_name in self.h5_files:
                self.h5_files[h5_name].close()
            self.h5_files[h5_name] = h5py.File(os.path.join(self.dir_path, h5_name), "r")
            image = self.h5_files[h5_name][step_name]['hs'][image_name][()]
            target = self.h5_files[h5_name][step_name]['target'][image_name][()]

        # image (H, W, C) -> (C, H, W), ndarray -> tensor
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
        target = torch.tensor(target, dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.standardization_transform is not None:
            image = self.standardization_transform(image)
        return image, target
    
def __del__(self):
    for h5_file in self.h5_files.values():
        try:
            if h5_file and h5_file.id.valid:
                h5_file.close()
        except:
            pass