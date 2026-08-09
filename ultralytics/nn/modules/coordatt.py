# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    """Hard Sigmoid activation function.

    Implements a computationally efficient approximation of sigmoid using ReLU6.
    """

    def __init__(self, inplace=True):
        """Initialize h_sigmoid with inplace option."""
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        """Apply hard sigmoid: relu(x + 3) / 6."""
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """Hard Swish activation function.

    Implements a computationally efficient approximation of swish using hard sigmoid.
    """

    def __init__(self, inplace=True):
        """Initialize h_swish with inplace option."""
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        """Apply hard swish: x * hard_sigmoid(x)."""
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """Coordinate Attention Module.

    Coordinate Attention captures long-range dependencies with precise positional information
    by decomposing channel attention into two 1D feature encoding processes along horizontal
    and vertical directions.

    Args:
        inp (int): Number of input channels.
        oup (int): Number of output channels.
        reduction (int): Reduction ratio for the intermediate channels. Default: 32.

    References:
        https://arxiv.org/abs/2103.02907
    """

    def __init__(self, inp, oup, reduction=32):
        """Initialize CoordAtt module."""
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Forward pass through CoordAtt module."""
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
