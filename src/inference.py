import torch
import numpy as np
import tensorflow as tf
from src.models.opg_module_ae import OPGLitModule
from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, Accuracy, AUROC, ConfusionMatrix

from src.dataset.dataset import OPGDataset, AdjustContrast, NormalizeIntensity, Resize, ExpandDims, ToTensor


def inference(
        PATH,
        threshold,
        test_batch_size
):
    # prepare test data:
    data_dir = "/cluster/project/jbuhmann/dental_imaging/data/all_patches"
    dim = 28
    resize = Resize(dim, dim, 'symmetric')

    test_transforms = transforms.Compose(
        [NormalizeIntensity(),
         AdjustContrast(1., 10., 0.),
         resize,
         ExpandDims(),
         ToTensor()]
    )

    test_set = OPGDataset("/cluster/home/emanete/dental_imaging/data/all_images_test_aug.csv", data_dir,
                          transform=test_transforms)

    test_dataloader = DataLoader(test_set, batch_size=test_batch_size)

    # load and start with the inference
    loaded_model = OPGLitModule()
    loaded_model.load_state_dict(torch.load(PATH)["state_dict"])
    loaded_model.eval()

    # define mse loss:
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    with torch.no_grad():
        for data in test_dataloader:
            images = data['image']  # has shape [test_batch_size  1 28 28]
            bin_class = data['bin_class']
            images_cast = tf.cast(images, dtype=tf.float64)
            print(tf.shape(images_cast))

            # reconstruct the images by running images through the network
            reconstructed_images = loaded_model(images)
            tf.cast(reconstructed_images, tf.float64)
            # print(tf.shape(reconstructed_images))

            # swap axes of tensors
            images_swap = tf.transpose(images, [0, 2, 3, 1])  # has shape [test_batch_size 28 28 1]
            rec_images_swap = tf.transpose(reconstructed_images, [0, 2, 3, 1])
            loss = mse(images_swap, rec_images_swap)  # has shape [test_batch_size 28 28]
            # print(tf.shape(loss))

            # compare with the threshold
            threshold_tensor = tf.constant([threshold])

            bool_result = tf.math.less(threshold_tensor, loss)  # checks if threshold < current loss, if so: anomaly, i.e. 1
            int_result = tf.cast(bool_result, tf.int32)

            # metrics
            accuracy = Accuracy(int_result, bin_class)

            auroc = AUROC()
            auroc(int_result, bin_class)

            confmat = ConfusionMatrix(num_classes=2)
            confmat(int_result, bin_class)

            with open("/cluster/home/emanete/dental_imaging/test_results/results", "a") as f:
                f.write("\n")
                np.savetxt(f, accuracy.numpy())
                np.savetxt(f, auroc.numpy())
                np.savetxt(f, confmat.numpy())
                f.write("\n")
                f.close()
















