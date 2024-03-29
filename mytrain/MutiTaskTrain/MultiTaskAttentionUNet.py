from torch import nn
import torch
from axial_attention import AxialAttention, AxialPositionalEmbedding, AxialImageTransformer


class MultiTasAttentionUNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(MultiTasAttentionUNet, self).__init__()
        self.conv1 = ConvolutionLayer(in_channels, 32)  # out = (1,32,512,512)
        self.attention1 = AttentionBlock(32, 2, 1)
        self.down1 = DownSample(32, 32)  # 下采样至1/2
        # 注意有残差连接，从down1->down2
        self.conv2 = ConvolutionLayer(32, 64)  # out = (1,64,256,256)
        self.attention2 = AttentionBlock(64, 2, 1)
        self.down2 = DownSample(64 + 32, 64)  # 下采样至1/4

        self.conv3 = ConvolutionLayer(64, 128)  # out = (1,128,128,128)
        self.attention3 = AttentionBlock(128, 2, 1)
        self.down3 = DownSample(128 + 64, 128)  # 下采样至1/8

        self.conv4 = ConvolutionLayer(128, 256)  # out = (1,256,64,64)
        self.attention4 = AttentionBlock(256, 2, 1)
        self.down4 = DownSample(256 + 128, 256)  # 下采样至1/16

        self.conv5 = ConvolutionLayer(256, 512)  # out = (1,512,32,32)
        self.up1 = UpSample(512)  # 上采样至1/8

        self.conv6 = ConvolutionLayer(512, 256)  # out = (1,256,64,64)
        self.up2 = UpSample(256)  # 上采样至1/4

        self.conv7 = ConvolutionLayer(256, 128)  # out = (1,128,128,128)
        self.up3 = UpSample(128)  # 上采样至1/2

        self.conv8 = ConvolutionLayer(128, 64)  # out = (1,64,256,256)
        self.up4 = UpSample(64)  # 上采样至1/1

        self.conv9 = ConvolutionLayer(64, 32)  # out = (1,32,512,512)

        self.predict = nn.Sequential(  # 输出层，由sigmoid函数激活
            nn.Conv2d(32, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Sigmoid()
        )

        self.discriminator = DiscriminateHead()

    def forward(self, x):
        """下采样"""
        x1 = self.conv1(x)  # ===> 1/1 32 512
        # print("x1", x1.size())
        a1 = self.attention1(x1)
        d1 = self.down1(a1)  # ===> 1/2 32 256

        x2 = self.conv2(d1)  # ===> 1/2 64 256
        a2 = self.attention2(x2)
        d2 = self.down2(torch.cat((d1, a2), dim=1))  # ===> 1/4 64 128
        # print(x2.size())

        x3 = self.conv3(d2)  # ===> 1/4 128 128
        a3 = self.attention3(x3)
        d3 = self.down3(torch.cat((d2, a3), dim=1))  # ===> 1/8 128 64
        # print(x3.size())

        x4 = self.conv4(d3)  # ===> 1/8 256 64
        a4 = self.attention4(x4)
        d4 = self.down4(torch.cat((d3, a4), dim=1))  # ===> 1/16 256 32
        # print(x4.size())

        """discriminator"""
        fake = self.discriminator(d4)

        """上采样"""
        x5 = self.conv5(d4)  # ===> 1/16 512
        up1 = self.up1(x5)  # ===> 1/8 512
        # print("x4:", x4.size())
        # print("up1", up1.size())
        # print(torch.cat((x4, up1), dim=1).size())
        x6 = self.conv6(torch.cat((a4, up1), dim=1))  # ===> 1/8 256
        up2 = self.up2(x6)  # ===> 1/4 256

        # print("x3:", x3.size())
        # print("up2", up2.size())
        # print(torch.cat((x3, up2), dim=1).size())
        x7 = self.conv7(torch.cat((a3, up2), dim=1))  # ===> 1/4 128
        up3 = self.up3(x7)  # ===> 1/2 128
        # print(torch.cat((x2, up3), dim=1).size())
        x8 = self.conv8(torch.cat((a2, up3), dim=1))  # ===> 1/2 64
        up4 = self.up4(x8)  # ===> 1/1 64

        x9 = self.conv9(torch.cat((a1, up4), dim=1))  # ===> 1/1 32
        mask = self.predict(x9)  # ===> 1/1 out_channels

        return fake, mask


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
            nn.AvgPool2d(kernel_size=(2, 2), stride=2)
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


class DiscriminateHead(nn.Module):
    """
    分类器
    """

    def __init__(self):
        super(DiscriminateHead, self).__init__()
        self.conv1 = ConvolutionLayer(256, 128)
        self.d1 = DownSample(128 + 256, 64)
        self.conv2 = ConvolutionLayer(64, 32)
        self.d2 = DownSample(32 + 64, 16)
        self.d3 = DownSample(16, 4)
        self.predict = nn.Sequential(  # 输出层
            nn.Linear(4 * 4 * 4, 4 * 4),
            nn.LeakyReLU(),
            nn.Linear(4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        d1 = self.d1(torch.cat((x, x1), dim=1))
        x2 = self.conv2(d1)
        d2 = self.d2(torch.cat((d1, x2), dim=1))
        d3 = self.d3(d2)
        return self.predict(d3.flatten(1))


if __name__ == '__main__':
    img1 = torch.randn(1, 1, 512, 512)
    net = MultiTasAttentionUNet(1, 1)
    print(net(img1))
