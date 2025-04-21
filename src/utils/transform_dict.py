from typing import Dict, Any
from .transform.simclr_transforms import SimCLRTransform
from .transform.cutout_transforms import CutOut
from .transform.moco_transforms import MoCoTransform
TRANSFORM_MAP: Dict[str, Any] = {
    "SimCLRTransform": SimCLRTransform,
    "CutOut": CutOut,
    "MoCoTransform": MoCoTransform,
}