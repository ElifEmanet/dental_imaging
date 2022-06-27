import torch
import numpy as np
import tensorflow as tf

from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from datetime import datetime

from src.models.opg_module_ae import OPGLitModule
from src.datamodules.opg_datamodule import OPGDataModule


def get_threshold(
        path_to_best_model: str
):
    # get the training images
    datamodule = OPGDataModule()
    train_dataloader = datamodule.train_dataloader()

    # load the best model
    loaded_model = OPGLitModule()
    loaded_model.load_state_dict(torch.load(path_to_best_model)["state_dict"])
    loaded_model.eval()

    # compute the average loss for the training images with the best model
    with torch.no_grad():
        for batch_idx, sample in train_dataloader:
            images = sample['image']  # has shape [training_batch_size  1 28 28]
            reconstructed_images = loaded_model(images)

            images_array = images.cpu().numpy()
            rec_images_array = reconstructed_images.cpu().numpy()

            mse_array = mean_squared_error(images_array.transpose(), rec_images_array.transpose(),
                                           multioutput='raw_values')

            d1 = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            np.save('/cluster/home/emanete/dental_imaging/test_results/mse_training' + d1, mse_array)

    average_loss = np.average(mse_array)
    return average_loss




