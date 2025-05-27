dataset_mean: dict[str, any] = {
    "HFD100_scene": {
        "hs": [0.0330735, 0.04246865, 0.05181025, 0.05674259, 0.06632508,
               0.08207408, 0.10242798, 0.12225091, 0.13196813, 0.13296044,
               0.12752898, 0.12546046, 0.13312519, 0.12969044, 0.13601579,
               0.1381559, 0.13018353, 0.12466665, 0.1144009, 0.11407086,
               0.1429517, 0.14041291, 0.17955197, 0.22456133, 0.19777751,
               0.18449385, 0.18330422, 0.15799843, 0.13109702, 0.09474891,
               0.0957033],
        "rgb": [0.485, 0.456, 0.406],
    }
}
dataset_std: dict[str, any] = {
    "HFD100_scene": {
        "hs": [0.03898594, 0.04962565, 0.06015501, 0.06517923, 0.07352601,
               0.08499902, 0.09651659, 0.11043679, 0.11851531, 0.12076702,
               0.12026139, 0.12052355, 0.12763785, 0.12665579, 0.13505693,
               0.13953114, 0.13555652, 0.13339746, 0.12264947, 0.11176322,
               0.11179046, 0.08983019, 0.09699432, 0.11049844, 0.09698344,
               0.0874173, 0.08631045, 0.07420712, 0.06163858, 0.04498825,
               0.04536467],
        "rgb": [0.1921, 0.1501, 0.0972],
    }
}

import torch
class DatasetStandardization:
    def __init__(self, dataset_name: str, band_type: str) -> None:
        self.mean = torch.tensor(dataset_mean[dataset_name][band_type])
        self.std = torch.tensor(dataset_std[dataset_name][band_type])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        データセットの標準化
        Args:
            x: (C, H, W)
        Returns:
            (C, H, W)
        """
        # x:(C, H, W), mean: (C), std: (C)
        return (x - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)

class DatasetUnstandardization:
    def __init__(self, dataset_name: str, band_type: str) -> None:
        self.mean = torch.tensor(dataset_mean[dataset_name][band_type])
        self.std = torch.tensor(dataset_std[dataset_name][band_type])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        データセットの非標準化
        Args:
            x: (C, H, W)
        Returns:
            (C, H, W)
        """
        return x * self.std.view(-1, 1, 1) + self.mean.view(-1, 1, 1)