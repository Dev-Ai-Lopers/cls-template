from omegaconf import OmegaConf
from cls_tem.data.dataset import DATASET_REGISTRY
from cls_tem.data.transforms import TRANSFORMS_REGISTRY
from pytorch_lightning import LightningDataModule


class ClsDataModule(LightningDataModule):
    def __init__(
        self,
        config: OmegaConf
    ) -> None:
        super().__init__()
        
        self.config = config
        self.dataset_cls = DATASET_REGISTRY.get(config.dataset.type)
        
        self.trainset = self.build_dataset(mode="train")
        self.valset = self.build_dataset(mode="val")
        self.testset = self.build_dataset(mode="test")
    
    def build_dataset(self, mode):
        if mode == "train":
            transform = self.build_transform(self.config.transforms.train)
        else:
            transform = self.build_transform(self.config.transforms.test)
        
        dataset = self.dataset_cls(
            root_dir=self.config.dataset.root_dir,
            mode=mode,
            transform=transform
        )
        
        return dataset
    
    @staticmethod  
    def build_transform(config):
        tf_cls = TRANSFORMS_REGISTRY.get(config.type)
        
        return tf_cls(config)
