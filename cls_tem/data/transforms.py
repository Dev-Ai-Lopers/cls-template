import torch
import torch.nn as nn
import numpy as np
import kornia as K


from dlutils.utils.registry import Registry
from omegaconf import OmegaConf
from PIL import Image


TRANSFORMS_REGISTRY = Registry("transforms")


@TRANSFORMS_REGISTRY.register
class ToTensor(nn.Module):
    def __init__(self, config: OmegaConf | dict) -> None:
        super().__init__()
        
        self.config = config
        
        if "params" not in config:
            self.params = {}
        else:
            self.params = config.params
        
    @torch.no_grad()
    def forward(self, x: np.ndarray | Image.Image) -> torch.Tensor:
        return K.utils.image_to_tensor(
            iamge=x,
            **self.params
        )
