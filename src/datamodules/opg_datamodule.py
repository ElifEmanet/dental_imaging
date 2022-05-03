import pytorch_lightning as pl
import numpy as np

import torch

from torch.utils.data import random_split, DataLoader, Dataset
from typing import Optional, Tuple
from torchvision import transforms

from src.dataset.dataset import OPGDataset, DataSubSet, AdjustContrast, Center, NormalizeIntensity, Rotate, RandomNoise, \
    RandomCropAndResize, Sharpen, Resize, Blur, Zoom, RescalePixelDims, ExpandDims, ToTensor


class OPGDataModule(pl.LightningDataModule):
    def __init__(
            self,
            # depending on where you run this project, change the following line:
            # data_dir: str = "data/",
            data_dir: str = "/cluster/project/jbuhmann/dental_imaging/data/",
            train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.data_dir = data_dir

        self.original_height = 415
        self.original_width = 540
        self.dim = 392  # 28*14

        # for rotate: mode = {‘reflect’, ‘constant’, ‘grid-constant’, ‘nearest’, ‘mirror’, ‘grid-wrap’, ‘wrap’}
        self.rotate = Rotate(np.random.uniform(-6, 6), False, 'reflect')

        # for random_noise: mode = {'gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle'}
        self.random_noise = RandomNoise('gaussian', 0.1)

        # for resize mode: {‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’}
        self.resize = Resize(self.dim, self.dim, 'symmetric')

        # for resize mode: {‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’}
        self.random_crop = RandomCropAndResize(np.random.randint(1, self.original_height),
                                               np.random.randint(1, self.original_width),
                                               self.dim, self.dim, 'symmetric')

        self.zoom = Zoom(np.random.uniform(0, 2))

        # data transformations
        self.train_transforms = transforms.Compose(
            [Center(),
             NormalizeIntensity(),
             AdjustContrast(1., 10., 0.),
             Blur(),
             self.rotate,
             self.random_noise,
             self.random_crop,
             self.zoom,
             self.resize,
             ExpandDims(),
             ToTensor()]
        )

        self.test_transforms = transforms.Compose(
            [Center(),
             NormalizeIntensity(),
             AdjustContrast(1., 10., 0.),
             ExpandDims(),
             ToTensor()]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 1

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            data_fit = OPGDataset("data/all_images_train_select.csv", self.data_dir)
            self.length = len(data_fit)
            self.train_len = int(self.length*0.8)
            self.train_set, self.val_set = random_split(data_fit, [self.train_len, self.length - self.train_len])

            self.train_trf_set = DataSubSet(self.train_set, transform=self.train_transform)
            self.val_trf_set = DataSubSet(self.val_set, transform=self.test_transform)

            # self.dims = tuple(self.train_set[0][0].shape)

        if stage == "test" or stage is None:
            self.test_set = OPGDataset("data/all_images_test_aug.csv", self.data_dir, transform=self.test_transforms)

            # self.dims = tuple(self.test_set[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.train_trf_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_trf_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)
