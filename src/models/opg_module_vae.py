from typing import Any, List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MeanSquaredError

# from src.models.components.autoencoder import Encoder, Decoder
# from src.models.components.conv_encoder_decoder_LR import Encoder, Decoder
from src.models.components.vae import Encoder, Decoder


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
        beta: float,
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

        # self.encoder = Encoder(encoded_space_dim, fc2_input_dim, stride, input_pxl).float()  # LR
        # self.decoder = Decoder(encoded_space_dim, fc2_input_dim, stride, input_pxl).float()  # LR
        # self.encoder = Encoder(latent_dim).float()  # EE
        # self.decoder = Decoder(latent_dim).float()  # EE
        self.encoder = Encoder(encoded_space_dim, fc2_input_dim, stride, input_pxl).float()  # VAE
        self.decoder = Decoder(encoded_space_dim, fc2_input_dim, stride, input_pxl).float()  # VAE

        # Loss function for reconstruction:
        # self.reconstr_loss = nn.MSELoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.train_loss = MeanSquaredError()
        self.val_loss = MeanSquaredError()
        self.test_loss = MeanSquaredError()

        # for logging best so far validation loss
        # self.val_acc_best = MaxMetric()
        self.val_loss_best = MinMetric()

        # Hyperparameter to control the importance of reconstruction loss vs KL-Divergence Loss
        self.beta = beta

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the stochastic part of the model
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
        # loss = self.reconstr_loss(x_reconstr, x)
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
        loss = recon_loss * self.beta + kl_loss

        # self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        loss = self.val_loss.compute()  # get val accuracy from current epoch
        self.val_loss_best.update(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), on_epoch=True, prog_bar=True)
        # pass

    def test_step(self, batch: Any, batch_idx: int):
        x, mu, log_var, x_hat = self.common_step(batch)

        kl_loss = (-0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss = self.test_loss(x, x_hat)
        loss = recon_loss * self.beta + kl_loss

        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        pass

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
