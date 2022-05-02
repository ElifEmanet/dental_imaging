from torch import nn


class AE(nn.Module):
    def __init__(
        self,
        input_size: int = 392,
        lin1_size: int = 128,
        lin2_size: int = 64,
        output_size: int = 1,
    ):
        """
            Args:
                input_size (int): Size of input layer, namely the number of features.
                lin1_size (int): Output dimension of the first linear layer.
                lin2_size (int): Output dimension of the second linear layer.
                output_size (int): Size of output, namely the number of classes.
            Returns:
                None.
        """
        super().__init__()

        self.flatten = nn.Flatten()

        # Encoder: reduce the dimension of the input to the number of classes
        self.encoder = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(True),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(True),
            nn.Linear(lin2_size, output_size))

        # Activation layer for the output of the encoder:
        if output_size == 1:
            self.act = nn.Sigmoid()
        else:
            self.act = nn.LogSoftmax(dim=1)

            # Activation for the input to the decoder:
            self.softmax = nn.Softmax(dim=1)

            # Decoder: reconstruct the original input:
            self.decoder = nn.Sequential(
                nn.Linear(output_size, lin2_size),
                nn.ReLU(True),
                nn.Linear(lin2_size, lin1_size),
                nn.ReLU(True), nn.Linear(lin1_size, input_size))

    def forward(self, x):
        """
            Computes forward pass through the autoencoder.
            Args:
                x (torch.Tensor): The input of shape [batch_size, feature_dim]
            Returns:
                torch.Tensor: Reconstructed version of the input of shape [batch_size, feature_dim].
                torch.Tensor: Output probabilities of shape [batch_size, num_classes].
        """

        out_en = self.encoder(self.flatten(x))
        out = self.softmax(out_en)
        out = self.decoder(out)

        if self.output_size == 1:
            return out, self.flatten(self.act(out_en))

        return out, self.act(out_en)
