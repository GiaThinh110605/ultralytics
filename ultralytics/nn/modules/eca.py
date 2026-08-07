import math
import torch
import torch.nn as nn


class ECA(nn.Module):
    """
    Efficient Channel Attention (CVPR 2020)
    """

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()

        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k,
            padding=(k - 1) // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.avg_pool(x)

        y = y.squeeze(-1).transpose(-1, -2)

        y = self.conv(y)

        y = y.transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)

        return x * y.expand_as(x)