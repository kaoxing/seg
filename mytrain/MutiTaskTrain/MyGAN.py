from torch import nn
import torch
from axial_attention import AxialAttention, AxialPositionalEmbedding, AxialImageTransformer


class MyGAN(nn.Module):
    """
     in_size = 8*8
     out_size = 512*512
    """

    def __init__(self, in_channels=1, out_channels=1):
        super(MyGAN, self).__init__()
        # ->512*512*1
        # add noise 1->2
        self.convLayer1 = ConvolutionLayer(in_channels+1, 32)  # <-512*512*1
        self.attention1 = AttentionBlock(32, 2, 1)
        self.downLayer1 = DownSample()  # ->256*256*32
        # add noise 32->64
        self.convLayer2 = ConvolutionLayer(32, 128)  # <-256*256*64
        self.attention2 = AttentionBlock(128, 2, 1)
        self.downLayer2 = DownSample()  # ->128*128*128
        # add noise 128->256
        self.convLayer3 = ConvolutionLayer(128, 256)  # <-256*256*128
        self.attention3 = AttentionBlock(256, 2, 1)
        self.upLayer1 = UpSample(256)  # ->256*256*128
        # skip add 128->256
        self.convLayer4 = ConvolutionLayer(256, 128)  # <-256*256*128
        self.attention4 = AttentionBlock(128, 2, 1)
        self.upLayer2 = UpSample(128)  # ->512*512*64
        # skip add 64->96
        self.convLayer5 = ConvolutionLayer(96, 32)  # ->512*512*32

        self.predict = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Sigmoid()
        )

    # def forward(self, x, rand1, rand2, rand3):
    def forward(self, x, rand1):
        in1 = torch.cat((x, rand1), dim=1)
        x1 = self.convLayer1(in1)
        a1 = self.attention1(x1)
        d1 = self.downLayer1(a1)

        # in2 = torch.cat((d1, rand2), dim=1)
        in2 = d1
        x2 = self.convLayer2(in2)
        a2 = self.attention2(x2)
        d2 = self.downLayer2(a2)

        # in3 = torch.cat((d2, rand3), dim=1)
        in3 = d2
        x3 = self.convLayer3(in3)
        a3 = self.attention3(x3)
        u1 = self.upLayer1(a3)

        in4 = torch.cat((u1, a2), dim=1)
        x4 = self.convLayer4(in4)
        a4 = self.attention4(x4)
        u2 = self.upLayer2(a4)

        in5 = torch.cat((u2, a1), dim=1)
        x5 = self.convLayer5(in5)
        out = self.predict(x5)

        return out


class ConvolutionLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        卷积层
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        """
        super(ConvolutionLayer, self).__init__()
        self.layer = nn.Sequential(
            # 卷积层
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):

    def __init__(self, in_channels):
        """
        反卷积，上采样，通道数将会减半，
        :param in_channels: 输入通道数
        """
        super(UpSample, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class AttentionBlock(nn.Module):

    def __init__(self, in_channels, depths, heads):
        """
        注意力模块
        使用的模块已经包括位置编码，一个模块组成:
        一层位置编码，一层注意力，两层卷积
        参数：输入通道数，模块重复数，注意力头个数
        """
        super(AttentionBlock, self).__init__()
        self.transformer = AxialImageTransformer(
            dim=in_channels,
            depth=depths,
            heads=heads,
            reversible=True
        )

    def forward(self, x):
        return self.transformer(x)


class DownSample(nn.Module):

    def __init__(self, ):
        """
        最大池化层构成的下采样，池化窗口为2×2
        """
        super(DownSample, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.layer(x)


if __name__ == '__main__':
    img1 = torch.randn(1, 1, 512, 512)
    # img2 = torch.randn(1, 1, 512, 512)
    net = MyGAN(1, 1)
    out1 = net(
        img1,
        torch.randn(1, 1, 512, 512),
    )
    print(out1)
