from torch import nn


class Encoder(nn.Module):
    def __init__(self,
                 latent_dim: int = 30):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        self.input_dim = int((((((392/2)/2)/2)/2)/2)/2)  # 392 is the #pixels of our images and 64 = (stride)^(#conv layers)
        self.lin_input_dim = self.input_dim * self.input_dim * 512  # 512 = dimension out of the last conv layer
        # self.lin_input_dim = 392 * 64  # 392 is the #pixels of our images (on one axis) and 64 = (stride)^(#conv layers)

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=2),  # 392 -> 196
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),  # 196 -> 98
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # 98 -> 49
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # 49 -> 25
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # 25 -> 13
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=0, stride=2),  # 13 -> 6
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),  # Image grid to single feature vector: will result in
            nn.Linear(self.lin_input_dim, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self,
                 latent_dim: int = 30) -> None:
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = int((((((392/2)/2)/2)/2)/2)/2)  # 392 is the #pixels of our images and 64 = (stride)^(#conv layers)
        self.lin_input_dim = self.input_dim * self.input_dim * 512  # 512 = dimension out of the last conv layer
        # self.lin_input_dim = 392 * 64  # 392 is the #pixels of our images and 64 = (stride)^(#conv layers)
        # 392 * 64 = 25088 = 512 * 49

        self.decoder_input = nn.Linear(latent_dim, self.lin_input_dim)  # 30 -> 25088

        # unflattend size = 512 x 7 x 7
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, self.input_dim, self.input_dim))

        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=0),  # 6 -> 13
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),  # 13 -> 25
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),  # 25 -> 49
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),  # 49 -> 98
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),  # 98 -> 196
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels=1, kernel_size=2, stride=2, padding=0),  # 196 -> 392
            nn.Sigmoid()
        )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder_input(x)
        # x = x.reshape(-1, 512, 2, 2)
        x = self.unflatten(x)
        x = self.net(x)
        x = self.final_layer(x)
        return x

