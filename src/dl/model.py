"""
model.py - Spatial models and masked losses for 64x64 NDWS patches.

UNet: compact U-Net (Ronneberger et al., 2015). Three down-sampling stages (64 -> 32 -> 16 -> 8), a
bottleneck with dropout, and three up-sampling stages joined by skip connections. With base=24 it has
about 1.1 M parameters.

CompactCNN: fully convolutional baseline that keeps the input resolution. Dilated 3x3 convolutions give a
29-pixel receptive field with no down-sampling and no decoder, so comparing it with the U-Net measures what
the multi-scale encoder-decoder adds. Both output one logit per pixel for "burning at day t+1".
"""

import torch
from torch import nn


def conv_block(c_in, c_out, dilation=1):
    pad = dilation
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=pad, dilation=dilation, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=pad, dilation=dilation, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels, base=24, dropout=0.2, skip_connections=True):
        super().__init__()
        self.skip = skip_connections
        merge = 2 if skip_connections else 1
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1, self.enc2, self.enc3 = conv_block(in_channels, c1), conv_block(c1, c2), conv_block(c2, c3)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(conv_block(c3, c4), nn.Dropout2d(dropout))
        self.up3, self.dec3 = nn.ConvTranspose2d(c4, c3, 2, stride=2), conv_block(c3 * merge, c3)
        self.up2, self.dec2 = nn.ConvTranspose2d(c3, c2, 2, stride=2), conv_block(c2 * merge, c2)
        self.up1, self.dec1 = nn.ConvTranspose2d(c2, c1, 2, stride=2), conv_block(c1 * merge, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def _merge(self, up, enc):
        return torch.cat([up, enc], 1) if self.skip else up

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(self._merge(self.up3(b), e3))
        d2 = self.dec2(self._merge(self.up2(d3), e2))
        d1 = self.dec1(self._merge(self.up1(d2), e1))
        return self.head(d1).squeeze(1)


class CompactCNN(nn.Module):
    """Resolution-preserving dilated CNN. Blocks with dilation 1, 2, 4 give a 29-pixel receptive field."""

    def __init__(self, in_channels, base=24, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, base, dilation=1),
            conv_block(base, base * 2, dilation=2),
            conv_block(base * 2, base * 2, dilation=4),
            nn.Dropout2d(dropout),
        )
        self.head = nn.Conv2d(base * 2, 1, 1)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def build_model(cfg, in_channels):
    if cfg["architecture"] == "unet":
        return UNet(in_channels, base=cfg["base"], dropout=cfg["dropout"], skip_connections=cfg.get("skip_connections", True))
    if cfg["architecture"] == "cnn":
        return CompactCNN(in_channels, base=cfg["base"], dropout=cfg["dropout"])
    raise ValueError(f"unknown architecture {cfg['architecture']!r}")


def masked_bce(logits, y, pos_weight=None):
    """Mean BCE over pixels with a known label (y >= 0); unknown (-1) pixels contribute nothing."""
    mask = y >= 0
    target = y.clamp(min=0).float()
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def masked_dice(logits, y, eps=1.0):
    """Soft Dice loss over known pixels, pooled across the batch."""
    mask = (y >= 0).float()
    target = y.clamp(min=0).float() * mask
    prob = torch.sigmoid(logits) * mask
    return 1.0 - (2.0 * (prob * target).sum() + eps) / (prob.sum() + target.sum() + eps)


def make_loss(cfg, device):
    """Loss function chosen by cfg["loss"]."""
    kind = cfg["loss"]
    pos_weight = torch.tensor(cfg["pos_weight"], device=device)
    if kind == "bce":
        return masked_bce
    if kind == "weighted_bce":
        return lambda logits, y: masked_bce(logits, y, pos_weight)
    if kind == "dice":
        return masked_dice
    if kind == "bce_dice":
        return lambda logits, y: masked_bce(logits, y) + masked_dice(logits, y)
    raise ValueError(f"unknown loss {kind!r}")
