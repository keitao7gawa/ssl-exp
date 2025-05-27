from typing import Dict, Any
from .transform.simclr_transforms import SimCLRTransform
from .transform.cutout_transforms import CutOut
from .transform.moco_transforms import MoCoTransform
from .transform.hs_transforms import HSSimCLRTransform, HSMoCoTransform, HSMAETransform
from .transform.hs_rgb_transforms import HStoRGBSimCLRTransform, HStoRGBMAETransform
from .transform.hs_to_rgb import HStoRGB
from torchvision.transforms import RandomResizedCrop
from .transform.dataset_standardization import DatasetStandardization
TRANSFORM_MAP: Dict[str, Any] = {
    "SimCLRTransform": SimCLRTransform,
    "CutOut": CutOut,
    "MoCoTransform": MoCoTransform,
    "HSSimCLRTransform": HSSimCLRTransform,
    "HSMoCoTransform": HSMoCoTransform,
    "HStoRGBSimCLRTransform": HStoRGBSimCLRTransform,
    "HStoRGB": HStoRGB,
    "RandomResizedCrop": RandomResizedCrop,
    "HSMAETransform": HSMAETransform,
    "HStoRGBMAETransform": HStoRGBMAETransform,
    "DatasetStandardization": DatasetStandardization,
}