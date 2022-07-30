from typing import Any, List

import torch
import linecache
import wandb
import numpy as np

from pytorch_lightning import LightningModule
from torchmetrics import MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MeanSquaredError
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, roc_auc_score
from datetime import datetime

from src.models.components.vae import Encoder, Decoder
from src.compute_threshold import get_threshold, multivariate_gaussian, select_threshold


class OPGLitModuleVAE(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        prob: float,
        beta: float,
        is_resnet: bool,
        latent_dim: int = 30,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        encoded_space_dim: int = 10,
        fc2_input_dim: int = 128,
        stride: int = 2,
        input_pxl: int = 28
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.encoder = Encoder(encoded_space_dim, fc2_input_dim, stride, input_pxl).float()  # VAE
        self.decoder = Decoder(encoded_space_dim, fc2_input_dim, stride, input_pxl).float()  # VAE

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.train_loss = MeanSquaredError()
        self.val_loss = MeanSquaredError()
        self.test_loss = MeanSquaredError()

        # for logging best so far validation loss
        self.val_loss_best = MinMetric()

        self.beta = beta
        self.encoded_space_dim = encoded_space_dim
        self.prob = prob
        self.input_pxl = input_pxl
        self.is_resnet = is_resnet

        self.now = datetime.now()

        self.name = "vae, 1 ep, lat = 5"

        wandb.init(project="dental_imaging",
                   name=self.name,
                   settings=wandb.Settings(start_method='fork'))

    def reparametrize(self, mu, log_var):
        # Reparametrization trick to allow gradients to backpropagate from the stochastic part of the model
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn_like(sigma)
        return mu + sigma * z

    def forward(self, x):
        mu, log_var = self.encoder(x.float())
        hidden = self.reparametrize(mu, log_var)
        x_hat = self.decoder(hidden)
        return mu, log_var, x_hat

    def common_step(self, batch: Any):
        x = batch['image'].float()
        mu, log_var, x_reconstr = self.forward(x.float())
        return x, mu, log_var, x_reconstr

    def training_step(self, batch: Any, batch_idx: int):
        x, mu, log_var, x_hat = self.common_step(batch)

        kl_loss = (-0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss = self.train_loss(x, x_hat)
        loss = kl_loss * self.beta + recon_loss

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch, batch_idx):
        x, mu, log_var, x_hat = self.common_step(batch)

        kl_loss = (-0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss = self.val_loss(x, x_hat)
        loss = kl_loss * self.beta + recon_loss

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        loss = self.val_loss.compute()  # get val accuracy from current epoch
        self.val_loss_best.update(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        y_bin = batch['bin_class']
        y_class = batch['clf']
        y_view_class = batch['view_cl']

        x = batch['image'].float()
        mu, log_var, x_hat = self.forward(x.float())

        kl_loss = (-0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss = self.test_loss(x, x_hat)
        loss = kl_loss * self.beta + recon_loss

        lat_repr = self.reparametrize(mu, log_var)

        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "original_image": x,
                "reconstructed_image": x_hat, "y_bin": y_bin,
                "clf": y_class, "latent_repr": lat_repr, "y_view_class": y_view_class}

    def test_epoch_end(self, outputs: List[Any]):
        # get current time to name the file
        d1 = self.now.strftime("%d-%m-%Y_%H:%M:%S")

        # log parameters:
        self.log("latent dimension", self.encoded_space_dim, on_step=False, on_epoch=True)

        # get original images
        xs = torch.cat([dict['original_image'] for dict in outputs])
        xs_array = xs.cpu().numpy()  # numpy.ndarray of size (# test images, 1, input_pxl, input_pxl)
        xs_array = xs_array.squeeze()  # numpy.ndarray of size (# test images, input_pxl, input_pxl)
        xs_array_red = xs_array.reshape(
            (xs_array.shape[0], xs_array.shape[1] * xs_array.shape[2]))  # (# test images, input_pxl*input_pxl)

        # get reconstructed images
        x_hats = torch.cat([dict['reconstructed_image'] for dict in outputs])
        x_hats_array = x_hats.cpu().numpy()  # numpy.ndarray of size (# test images, 1, input_pxl, input_pxl)
        x_hats_array = x_hats_array.squeeze()  # numpy.ndarray of size (# test images, input_pxl, input_pxl)
        x_hats_array_red = x_hats_array.reshape(
            (x_hats_array.shape[0], x_hats_array.shape[1] * x_hats_array.shape[2]))  # (# test images, input_pxl*input_pxl)

        # compute mse for individual reconstructed images: mse_array has the size (# test images,)
        mse_array = mean_squared_error(xs_array_red.transpose(), x_hats_array_red.transpose(), multioutput='raw_values')
        np.save('/cluster/home/emanete/dental_imaging/test_results/vae_mse' + d1, mse_array)

        # compute MAD for the test set:
        median = np.median(mse_array)
        median_array = np.full(mse_array.shape, float(median))
        diff_array = median_array - mse_array
        absolute_diff = np.absolute(diff_array)
        mad = np.median(absolute_diff)

        # for each test image compute the modified z-score:
        mod_z_array = 0.6745 * (mse_array - median_array) / mad
        np.save('/cluster/home/emanete/dental_imaging/test_results/vae_mod_z_score' + d1, mod_z_array)

        # get the best model path:
        with open(r"/cluster/home/emanete/dental_imaging/checkpoints_and_scores/scores", 'r') as fp:
            num_lines = len(fp.readlines())  # the file score ends with an empty line, hence subtract 2

        trained_path = linecache.getline(r"/cluster/home/emanete/dental_imaging/checkpoints_and_scores/scores",
                                         num_lines - 2).strip()

        # get the threshold and mean and covariance of the training images on the best model
        average_loss, mu, var = get_threshold(trained_path, True, self.encoded_space_dim, self.input_pxl, self.is_resnet)

        # save the latent representations of test images
        lat_reprs = torch.cat([dict['latent_repr'] for dict in outputs])
        lat_reprs_array = lat_reprs.cpu().numpy()
        np.save('/cluster/home/emanete/dental_imaging/test_results/vae_test_lat_repr' + d1, lat_reprs_array)

        # probabilities for test images' latent representations
        p_array = multivariate_gaussian(lat_reprs_array, mu, var)  # has size (test images,)
        np.save('/cluster/home/emanete/dental_imaging/test_results/vae_test_prob' + str(self.prob) + d1, p_array)

        # get true classes:
        ys = torch.cat([dict['y_bin'] for dict in outputs])
        ys_array = ys.cpu().numpy()  # numpy.ndarray of size (# test images,)
        np.save('/cluster/home/emanete/dental_imaging/test_results/vae_test_bin' + d1, ys_array)

        score, ep = select_threshold(p_array, ys_array)

        self.log("score", score, on_step=False, on_epoch=True)

        # depending on which method you want to try out, uncomment the corresponding line

        # mse of test images compared to the average mse of training images:
        # bool_array = mse_array > float(average_loss)

        # modified z-score of the mse of test images:
        # bool_array = np.absolute(mod_z_array) > 3

        # using latent representations of test images, compared to the multivariate distribution of training images:
        bool_array = p_array < float(ep)

        # convert boolean array to int array = predictions
        int_array = [int(elem) for elem in bool_array]  # if True, anomaly, hence 1
        np.save('/cluster/home/emanete/dental_imaging/test_results/vae_pred' + d1, int_array)

        # get accuracy and log
        accuracy = accuracy_score(ys_array, int_array)
        self.log("test/accuracy", accuracy, on_step=False, on_epoch=True)

        wandb.sklearn.plot_confusion_matrix(ys_array, int_array, ["normal", "anomaly"])

        # AUROC
        roc_auc = roc_auc_score(ys_array, p_array)
        self.log("test/roc_auc", roc_auc, on_step=False, on_epoch=True)

        # precision
        precision = precision_score(ys_array, int_array, labels=["normal", "anomaly"], pos_label=1, average='binary')
        self.log("test/precision", precision, on_step=False, on_epoch=True)

        # recall
        recall = recall_score(ys_array, int_array, labels=["normal", "anomaly"], pos_label=1, average='binary')
        self.log("test/recall", recall, on_step=False, on_epoch=True)

        # log beta:
        self.log("beta", self.beta, on_step=False, on_epoch=True)

        # Cohen's kappa
        self.log("test/Cohen's kappa", score, on_step=False, on_epoch=True)

        # get classifications for plotting
        y_class = torch.cat([dict['clf'] for dict in outputs])
        y_class_array = y_class.cpu().numpy()

        np.save('/cluster/home/emanete/dental_imaging/test_results/vae_test_true_class' + d1, y_class_array)

        y_view_class = torch.cat([dict['y_view_class'] for dict in outputs])
        y_view_class_array = y_view_class.cpu().numpy()

        np.save('/cluster/home/emanete/dental_imaging/test_results/vae_test_true_view_class' + d1, y_view_class_array)

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_loss.reset()
        self.test_loss.reset()
        self.val_loss.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
