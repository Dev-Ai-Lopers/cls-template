import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from dlutils.utils.registry import Registry


DATASET_REGISTRY = Registry("dataset")


@DATASET_REGISTRY.register
class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        mode: str,
        transform: Optional[Callable] = None,
    ) -> None:
        assert mode in [
            "train",
            "val",
            "test",
        ], "mode must be either train, val, or test"

        data_path = os.path.join(root_dir, mode)
        self.data = ImageFolder(root=data_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def load_img(self, pth: str):
        return Image.open(pth).convert("RGB")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.data[index]
        sample = self.load_img(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target
