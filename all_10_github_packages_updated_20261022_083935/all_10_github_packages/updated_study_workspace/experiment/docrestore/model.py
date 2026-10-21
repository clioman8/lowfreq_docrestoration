"""Shared four-level U-Net. No degradation parameter is an input to this model."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels), nn.ReLU(),
        )


class UNet(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        if len(channels) != 4:
            raise ValueError("Exactly four channel levels are required")
        a, b, c, d = channels
        self.enc0 = DoubleConv(1, a)
        self.enc1 = DoubleConv(a, b)
        self.enc2 = DoubleConv(b, c)
        self.bottom = DoubleConv(c, d)
        self.dec2 = DoubleConv(d + c, c)
        self.dec1 = DoubleConv(c + b, b)
        self.dec0 = DoubleConv(b + a, a)
        self.out = nn.Conv2d(a, 1, 1)

    def forward(self, image):
        x0 = self.enc0(image)
        x1 = self.enc1(F.max_pool2d(x0, 2))
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.bottom(F.max_pool2d(x2, 2))
        # Nearest interpolation avoids nondeterministic CUDA bilinear backward.
        up = lambda x, skip: F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        y2 = self.dec2(torch.cat([up(x3, x2), x2], dim=1))
        y1 = self.dec1(torch.cat([up(y2, x1), x1], dim=1))
        y0 = self.dec0(torch.cat([up(y1, x0), x0], dim=1))
        return self.out(y0).sigmoid()
