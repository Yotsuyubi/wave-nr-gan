import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class PixelShuffle1d(nn.Module):

    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x

class PixelUnshuffle1D(nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        long_channel_len = x.shape[1]
        long_width = x.shape[2]

        short_channel_len = long_channel_len * self.downscale_factor
        short_width = long_width // self.downscale_factor

        x = x.contiguous().view([batch_size, long_channel_len, short_width, self.downscale_factor])
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view([batch_size, short_channel_len, short_width])
        return x

class PhaseShift(nn.Module):

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        n = torch.randint(-self.n, self.n, (1,))
        return torch.roll(x, shifts=n.item(), dims=2)

class SamePaddingConv1d(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, padding=int((kernel_size - 1) / 2))

    def forward(self, x):
        return self.conv(x)


class UpConv1d(nn.Module):

    def __init__(self, in_dim, out_dim, scale_factor, kernel_size):
        super().__init__()
        self.pixel_shuffer = PixelShuffle1d(scale_factor)
        self.conv = SamePaddingConv1d(in_dim // scale_factor, out_dim, kernel_size)

    def forward(self, x):
        x = self.pixel_shuffer(x)
        return self.conv(x)

class DownConv1d(nn.Module):

    def __init__(self, in_dim, out_dim, scale_factor, kernel_size):
        super().__init__()
        self.pixel_shuffer = PixelUnshuffle1D(scale_factor)
        self.conv = SamePaddingConv1d(in_dim * scale_factor, out_dim, kernel_size)

    def forward(self, x):
        x = self.pixel_shuffer(x)
        return self.conv(x)

class UpsampleBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.deconv = UpConv1d(in_dim, out_dim, scale_factor=4, kernel_size=1)
        self.conv_1 = UpConv1d(in_dim, out_dim, scale_factor=4, kernel_size=25)
        self.conv_2 = nn.Conv1d(out_dim, out_dim, 25, padding=12)
        self.LReLU_1 = nn.LeakyReLU(0.2)
        self.LReLU_2 = nn.LeakyReLU(0.2)

    def forward(self, input):
        shortcut = self.deconv(input)
        x = input
        x = self.conv_1(x)
        x = self.LReLU_1(x)
        x = self.conv_2(x)
        x = self.LReLU_2(x)
        return x + shortcut

class DownsampleBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.deconv = DownConv1d(in_dim, out_dim, scale_factor=4, kernel_size=1)
        self.conv_1 = DownConv1d(in_dim, out_dim, scale_factor=4, kernel_size=25)
        self.conv_2 = nn.Conv1d(out_dim, out_dim, 25, padding=12)
        self.LReLU_1 = nn.LeakyReLU(0.2)
        self.LReLU_2 = nn.LeakyReLU(0.2)
        self.ps = PhaseShift(2)

    def forward(self, input):
        shortcut = self.deconv(input)
        x = input
        x = self.conv_1(x)
        x = self.LReLU_1(x)
        x = self.conv_2(x)
        x = self.LReLU_2(x)
        return self.ps(x + shortcut)
