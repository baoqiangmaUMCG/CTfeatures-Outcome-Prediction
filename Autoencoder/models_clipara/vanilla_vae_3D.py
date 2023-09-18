import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


def SeparableTransConv3x3x3(in_planes, out_planes, stride=1, output_pad=[0, 0, 0], bias=False):
    # the separable transpose convolution 
    return nn.Sequential(nn.ConvTranspose3d(in_planes, 1, kernel_size=[1, 3, 3],
                                            stride=[1, stride, stride], padding=[0, 1, 1],
                                            output_padding=[0, output_pad[1], output_pad[2]], bias=bias),
                         nn.ConvTranspose3d(1, out_planes, kernel_size=[3, 1, 1],
                                            stride=[stride, 1, 1], padding=[1, 0, 0],
                                            output_padding=[output_pad[0], 0, 0], bias=bias))


class VanillaVAE(BaseVAE):

    def __init__(self,
                 input_channel: int,
                 z_size: int = 512,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.input_channel = input_channel
        self.z_size = z_size
        self.hidden_dims = hidden_dims
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        modules = []
        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        enc_input_channel = self.input_channel
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(enc_input_channel, 
                              out_channels=h_dim,
                              stride=2,
                              # kernel_size=1, 
                              # padding=1),
                              kernel_size=1, 
                              padding=0),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            enc_input_channel = h_dim

        self.encoder = nn.Sequential(*modules)
        
        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.z_size)
        self.fc_var = nn.Linear(self.hidden_dims[-1], self.z_size)

        # Build Decoder
        modules = []
        # self.decoder_input = nn.Linear(z_size, self.hidden_dims[-1] * 3 * 6 * 6)
        self.decoder_input = nn.Linear(z_size, self.hidden_dims[-1])
        self.Upsample3D = nn.Upsample(scale_factor=(3, 6, 6), mode='trilinear', align_corners=False)
        # self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    # nn.ConvTranspose3d(self.hidden_dims[i],
                                       # self.hidden_dims[i - 1],
                                       # stride=2,
                                       # kernel_size=3,                                       
                                       # padding=1,
                                       # output_padding=1),
                    SeparableTransConv3x3x3(self.hidden_dims[i],
                                       self.hidden_dims[i - 1],
                                       stride=2,
                                       output_pad=[1,1,1]),
                    nn.BatchNorm3d(self.hidden_dims[i - 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)
        self.Upsample3D_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(self.hidden_dims[0],
                               self.hidden_dims[0],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm3d(self.hidden_dims[0]),
            nn.LeakyReLU(),
            # out_channel should actually be input_channel
            # ORIGINAL ONE
            # nn.Conv3d(self.hidden_dims[-1], out_channels=1,
            #           kernel_size=3, padding=1),
            # NEW: this changes shape from (batch_size, hidden_dims[-1], 96, 192, 192) to (batch_size, 1, 92, 180, 180)
            nn.Conv3d(self.hidden_dims[0], out_channels=self.input_channel, kernel_size=(5, 13, 13), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.Sigmoid())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # result = self.maxpool(input)
        result = self.encoder(input)
        print("result.shape (encoder 1):", result.shape)
        # Global Average Pooling
        result = result.mean(dim=(2, 3, 4))  # Shape (B, N)
        print("result.shape (encoder 2):", result.shape)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # result = result.view(-1, self.hidden_dims[0], 3, 6, 6)
        result = result.view(-1, self.hidden_dims[-1], 1, 1, 1)
        result = self.Upsample3D(result)
        print("result.shape (decoder 1)", result.shape) 
        result = self.decoder(result)
        print("result.shape (decoder 2)", result.shape) 
        # result = self.Upsample3D_2(result)
        # print("result.shape (decoder 3)", result.shape) 
        result = self.final_layer(result)
        print("result.shape (decoder 4)", result.shape) 
        
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), input, mu, log_var

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.z_size)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

