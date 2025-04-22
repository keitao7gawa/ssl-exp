from typing import Dict, Any
from .transform.simclr_transforms import SimCLRTransform
from .transform.cutout_transforms import CutOut
from .transform.moco_transforms import MoCoTransform
from .transform.hs_transforms import HSSimCLRTransform, HSMoCoTransform
TRANSFORM_MAP: Dict[str, Any] = {
    "SimCLRTransform": SimCLRTransform,
    "CutOut": CutOut,
    "MoCoTransform": MoCoTransform,
    "HSSimCLRTransform": HSSimCLRTransform,
    "HSMoCoTransform": HSMoCoTransform,
}