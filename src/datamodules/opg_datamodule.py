import pytorch_lightning as pl
import numpy as np

from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple
from torchvision import transforms

from src.dataset.dataset import OPGDataset, AdjustContrast, NormalizeIntensity, Rotate, RandomNoise, \
    RandomCropAndResize, Resize, Blur, Zoom, ExpandDims, ToTensor


class OPGDataModule(pl.LightningDataModule):
    def __init__(
            self,
            input_pxl,
            # depending on where you run this project, change the following line:
            # data_dir: str = "data/",
            data_dir: str = "/cluster/project/jbuhmann/dental_imaging/data/all_patches",
            batch_size: int = 32,
            test_batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.data_dir = data_dir

        self.dim = input_pxl

        # for rotate: mode = {‘reflect’, ‘constant’, ‘grid-constant’, ‘nearest’, ‘mirror’, ‘grid-wrap’, ‘wrap’}
        self.rotate = Rotate(np.random.uniform(-6, 6), False, 'reflect')

        # for random_noise: mode = {'gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle'}
        self.random_noise = RandomNoise('gaussian', 0.1)

        # for resize mode: {‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’}
        self.resize = Resize(self.dim, self.dim, 'symmetric')

        self.random_crop = RandomCropAndResize(np.random.randint(int(self.dim*0.75), self.dim),
                                               np.random.randint(int(self.dim*0.75), self.dim),
                                               self.dim, self.dim, 'symmetric')

        self.zoom = Zoom(np.random.uniform(0, 2))

        # data transformations
        self.train_transforms = transforms.Compose(
            [self.resize,
             NormalizeIntensity(),
             AdjustContrast(1., 10., 0.),
             Blur(),
             self.rotate,
             self.random_noise,
             self.random_crop,
             ExpandDims(),
             ToTensor()]
        )

        self.test_transforms = transforms.Compose(
            [self.resize,
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
            self.train_set = OPGDataset("/cluster/home/emanete/dental_imaging/data/new_all_images_train.csv",
                                        self.data_dir, transform=self.train_transforms)
            self.val_set = OPGDataset("/cluster/home/emanete/dental_imaging/data/new_all_images_val.csv",
                                      self.data_dir, transform=self.test_transforms)

        if stage == "test" or stage is None:
            self.test_set = OPGDataset("/cluster/home/emanete/dental_imaging/data/new_all_images_test.csv",
                                       self.data_dir, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=True)
