import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(
            self,
            encoded_space_dim: int = 10,
            fc2_input_dim: int = 128,
            stride: int = 2,
            input_pxl: int = 28
    ):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=stride, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=stride, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=stride, padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        input_dim = int(((input_pxl / stride) / stride) / stride)

        self.encoder_lin = nn.Sequential(
            nn.Linear(input_dim * input_dim * 32, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )
        ### mu
        self.hidden2mu = nn.Linear(encoded_space_dim, encoded_space_dim)

        ###
        self.hidden2log_var = nn.Linear(encoded_space_dim, encoded_space_dim)

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)

        latent = self.encoder_lin(x)
        mu = self.hidden2mu(latent)
        log_var = self.hidden2log_var(latent)
        return mu, log_var

# decoder for 28 x 28 images
class Decoder(nn.Module):

    def __init__(
            self,
            encoded_space_dim: int = 10,
            fc2_input_dim: int = 128,
            stride: int = 2,
            input_pxl: int = 28
    ):
        super().__init__()
        input_dim = int(((input_pxl / stride) / stride) / stride)
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, input_dim * input_dim * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, input_dim, input_dim))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=stride, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=stride, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=stride, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x
