"""
model.py - Compact U-Net (Ronneberger et al., 2015) for 64x64 NDWS patches.

Three down-sampling stages (64 -> 32 -> 16 -> 8), a bottleneck with dropout, and three
up-sampling stages with skip connections. Output: one logit per pixel for "burning at
day t+1". With base=24 the network has about 1.1 M parameters.
"""

import torch
from torch import nn


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels, base=24, dropout=0.2):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1, self.enc2, self.enc3 = conv_block(in_channels, c1), conv_block(c1, c2), conv_block(c2, c3)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(conv_block(c3, c4), nn.Dropout2d(dropout))
        self.up3, self.dec3 = nn.ConvTranspose2d(c4, c3, 2, stride=2), conv_block(c3 * 2, c3)
        self.up2, self.dec2 = nn.ConvTranspose2d(c3, c2, 2, stride=2), conv_block(c2 * 2, c2)
        self.up1, self.dec1 = nn.ConvTranspose2d(c2, c1, 2, stride=2), conv_block(c1 * 2, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1).squeeze(1)


def masked_bce(logits, y, pos_weight=None):
    """Mean BCE over pixels with a known label (y >= 0); unknown (-1) pixels contribute nothing."""
    mask = y >= 0
    target = y.clamp(min=0).float()
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    return (loss * mask).sum() / mask.sum().clamp(min=1)
