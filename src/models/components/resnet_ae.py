import torch
import torch.nn as nn
import torchvision


class ResNetEncoder(nn.Module):

    def __init__(
            self,
            encoded_space_dim: int = 10
    ):
        super().__init__()

        self.encoder_layer1 = torchvision.models.resnet50().layer1
        self.encoder_layer2 = torchvision.models.resnet50().layer2
        self.encoder_layer3 = torchvision.models.resnet50().layer3
        self.encoder_layer4 = torchvision.models.resnet50().layer4

        self.fc = nn.Sequential(
            nn.Linear(in_features=8192, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=encoded_space_dim)
        )

        self.relu = torchvision.models.resnet50().relu
        self.bn1 = torchvision.models.resnet50().bn1
        self.conv1 = torchvision.models.resnet50().conv1

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.encoder_layer1(out)
        out = self.encoder_layer2(out)
        out = self.encoder_layer3(out)
        out = self.encoder_layer4(out)
        # x = nn.AdaptiveAvgPool2d(output_size=(1, 1))(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class ResNetDecoder(nn.Module):

    def __init__(
            self,
            encoded_space_dim: int = 10
    ):
        super().__init__()

        self.decoder_fc = nn.Sequential(
            nn.Linear(in_features=encoded_space_dim, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=8192)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(2048, 2, 2))

        self.layer4_1 = nn.Sequential(
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        self.upsample_4 = nn.Sequential(
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(2048, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        )

        self.layer4_0 = nn.Sequential(
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )

        self.layer3_1 = nn.Sequential(
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        self.upsample_3 = nn.Sequential(
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        )
        self.layer3_0 = nn.Sequential(
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )

        self.layer2_1 = nn.Sequential(
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        self.upsample_2 = nn.Sequential(
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        )
        self.layer2_0 = nn.Sequential(
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )

        self.layer1_1 = nn.Sequential(
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        self.upsample_1 = nn.Sequential(
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(256, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
        )
        self.layer1_0 = nn.Sequential(
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )

        self.upsample_0 = nn.Sequential(
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(64, 3, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        )

    def forward(self, x):
        out = self.decoder_fc(x)
        l_4_2_in = self.unflatten(out)

        # layer 4
        out = nn.ReLU(inplace=True)(l_4_2_in)
        out += l_4_2_in
        l_4_1_in = self.layer4_1(out)
        out = nn.ReLU(inplace=True)(l_4_1_in)
        out += l_4_1_in
        l_4_0_in = self.layer4_1(out)
        out = nn.ReLU(inplace=True)(l_4_0_in)
        out += self.upsample_4(l_4_0_in)
        out = self.layer4_0(out)

        # layer 3
        out = nn.ReLU(inplace=True)(out)
        out += x
        out = self.layer3_1(out)
        out = nn.ReLU(inplace=True)(out)
        out += x
        out = self.layer3_1(out)
        out = nn.ReLU(inplace=True)(out)
        out += x
        out = self.layer3_1(out)
        out = nn.ReLU(inplace=True)(out)
        out += x
        out = self.layer3_1(out)
        out = nn.ReLU(inplace=True)(out)
        out += x
        out = self.layer3_1(out)
        out = nn.ReLU(inplace=True)(out)
        out += self.upsample_3(x)
        out = self.layer3_0(out)

        # layer 2
        out = nn.ReLU(inplace=True)(out)
        out += x
        out = self.layer2_1(out)
        out = nn.ReLU(inplace=True)(out)
        out += x
        out = self.layer2_1(out)
        out = nn.ReLU(inplace=True)(out)
        out += x
        out = self.layer2_1(out)
        out = nn.ReLU(inplace=True)(out)
        out += self.upsample_2(x)
        out = self.layer2_0(out)

        # layer 1
        out = nn.ReLU(inplace=True)(out)
        out += x
        out = self.layer1_1(out)
        out = nn.ReLU(inplace=True)(out)
        out += x
        out = self.layer1_1(out)
        out = nn.ReLU(inplace=True)(out)
        out += self.upsample_1(x)
        out = self.layer1_0(x)

        # last
        # x = nn.Identity()(x)
        out = nn.ReLU(inplace=True)(out)
        out = self.upsample_0(out)

        out = torch.sigmoid(out)
        return out

