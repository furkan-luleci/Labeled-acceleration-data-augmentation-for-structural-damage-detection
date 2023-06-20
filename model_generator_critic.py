"""
Discriminator and Generator implementation from DCGAN paper,
with removed Sigmoid() as output from Discriminator (and therefore
it should be called critic)
"""
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: Nx1x1024
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1), # Nx32x512
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(32, 64, 4, 2, 1), # Nx64x256
            self._block(64, 128, 4, 2, 1), # Nx128x128
            self._block(128, 256, 4, 2, 1), # Nx256x64
            nn.Conv1d(256, 1, kernel_size=64, stride=1, padding=0), # Nx1x1 
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm1d(out_channels,affine=True),
            nn.LeakyReLU(0.2),  
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: Nx256x1
            self._block(256, 256, 64, 1, 0), # Nx256x64
            self._block(256, 128, 4, 2, 1),  # Nx512x128
            self._block(128, 64, 4, 2, 1),  # Nx256x256
            self._block(64, 32, 4, 2, 1),  # Nx128x512
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
            # Output: Nx1x1024
            nn.Tanh())

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
