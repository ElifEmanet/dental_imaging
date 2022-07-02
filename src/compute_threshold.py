import torch
import numpy as np

from sklearn.metrics import mean_squared_error
from datetime import datetime

from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset.dataset import OPGDataset, AdjustContrast, NormalizeIntensity, Rotate, RandomNoise, \
    RandomCropAndResize, Resize, Blur, Zoom, ExpandDims, ToTensor


def get_threshold(
        path_to_best_model: str,
        IS_VAE: bool,
        lat_dim: int,
        batch_size: int = 32
):
    # get the training images
    data_dir: str = "/cluster/project/jbuhmann/dental_imaging/data/all_patches"
    original_height = 415
    original_width = 540
    dim = 28
    """""
    rotate = Rotate(np.random.uniform(-6, 6), False, 'reflect')
    random_noise = RandomNoise('gaussian', 0.1)
    random_crop = RandomCropAndResize(np.random.randint(1, original_height),
                                      np.random.randint(1, original_width),
                                      dim, dim, 'symmetric')
    zoom = Zoom(np.random.uniform(0, 2))
    """""
    resize = Resize(dim, dim, 'symmetric')

    train_transforms = transforms.Compose(
        [NormalizeIntensity(),
         AdjustContrast(1., 10., 0.),
         resize,
         ExpandDims(),
         ToTensor()]
    )

    train_dataset = OPGDataset("/cluster/home/emanete/dental_imaging/data/all_images_train_select.csv",
                               data_dir,
                               transform=train_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    # load the best model
    if IS_VAE:
        import src.models.opg_module_vae
        loaded_model = src.models.opg_module_vae.OPGLitModuleVAE.load_from_checkpoint(checkpoint_path=path_to_best_model)
    else:
        import src.models.opg_module_ae
        loaded_model = src.models.opg_module_ae.OPGLitModule.load_from_checkpoint(checkpoint_path=path_to_best_model)

    loaded_model.eval()

    # compute the average loss for the training images with the best model
    mse_array_to_average = np.zeros(1)
    z_array_start = np.zeros((1, lat_dim))
    train_class_array_start = np.zeros(1)

    with torch.no_grad():
        for batch_idx, sample in enumerate(train_dataloader):
            images = sample['image']  # has shape [training_batch_size  1 28 28]
            if IS_VAE:
                mu, log_var = loaded_model.encoder(images.float())
                z = loaded_model.reparametrize(mu, log_var)
                _, _, reconstructed_images = loaded_model(images)
            else:
                z = loaded_model.encoder(images.float())
                reconstructed_images = loaded_model(images)

            z_array = z.cpu().numpy()
            images_array = images.cpu().numpy()  # numpy.ndarray of size (training_batch_size, 1, 28, 28)
            rec_images_array = reconstructed_images.cpu().numpy()  # same

            images_array = images_array.squeeze()  # numpy.ndarray of size (training_batch_size, 28, 28)
            rec_images_array = rec_images_array.squeeze()  # same

            im_array_red = images_array.reshape(
                (images_array.shape[0], images_array.shape[1] * images_array.shape[2]))  # (training_batch_size, 28*28)
            rec_im_array_red = rec_images_array.reshape(
                (rec_images_array.shape[0], rec_images_array.shape[1] * rec_images_array.shape[2]))  # same

            mse_array = mean_squared_error(im_array_red.transpose(), rec_im_array_red.transpose(),
                                           multioutput='raw_values')  # (training_batch_size,)

            z_array_start = np.concatenate((z_array_start, z_array))
            mse_array_to_average = np.append(mse_array_to_average, mse_array)

            d1 = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            # np.save('/cluster/home/emanete/dental_imaging/test_results/mse_training' + d1, mse_array_to_average)
            # np.save('/cluster/home/emanete/dental_imaging/test_results/latent_repr' + d1, z_array)

            train_class = sample['cl_new']
            train_class_array = train_class.cpu().numpy()
            train_class_array_start = np.append(train_class_array_start, train_class_array)

    z_array_final = z_array_start[1:, :]
    mse_array_final = mse_array_to_average[1:]
    train_class_array_final = train_class_array_start[1:]
    # np.save('/cluster/home/emanete/dental_imaging/test_results/mse_final_array' + d1, mse_array_final)
    average_loss = np.average(mse_array_final)
    # np.save('/cluster/home/emanete/dental_imaging/test_results/lat_repr' + d1, z_array_final)
    np.save('/cluster/home/emanete/dental_imaging/test_results/train_class' + d1, train_class_array_final)

    return average_loss, z_array_final
