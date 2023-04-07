from torch import nn
import torch
from axial_attention import AxialAttention, AxialPositionalEmbedding, AxialImageTransformer


class MultiTasAttentionUNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(MultiTasAttentionUNet, self).__init__()
        self.conv1 = DownSample(in_channels, 32)  # 128
        self.down1 = DownSample(32, 16)  # 32
        self.down2 = DownSample(16, 8)  # 8
        self.down3 = DownSample_1(8, 4)  # 4
        self.predict = nn.Sequential(  # 输出层
            nn.Linear(4 * 4 * 4, 4 * 4),
            nn.LeakyReLU(),
            nn.Linear(4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """下采样"""
        x1 = self.conv1(x)  # ===> 1/1 32 512
        # print("x1", x1.size())
        d1 = self.down1(x1)  # ===> 1/2 32 256

        d2 = self.down2(d1)  # ===> 1/4 64 128
        # print(x2.size())

        d3 = self.down3(d2)  # ===> 1/8 128 64
        # print(x3.size())
        return self.predict(d3.flatten(1))


class DownSample_1(nn.Module):

    def __init__(self, in_channel, out_channel):
        """
        最大池化层构成的下采样，池化窗口为2×2
        """
        super(DownSample_1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):

    def __init__(self, in_channel, out_channel):
        """
        最大池化层构成的下采样，池化窗口为2×2
        """
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(4, 4), stride=4),
        )

    def forward(self, x):
        return self.layer(x)


if __name__ == '__main__':
    img1 = torch.randn(1, 1, 512, 512)
    net = MultiTasAttentionUNet(1, 1)
    print(net(img1))
